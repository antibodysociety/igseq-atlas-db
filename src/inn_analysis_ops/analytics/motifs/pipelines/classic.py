"""Classic INN-level Möbius motif pipeline."""

from __future__ import annotations

import math as _math
import re
from dataclasses import dataclass, field
from itertools import combinations
from typing import Dict, List, Optional, Tuple

import networkx as nx
import numpy as np
import pandas as pd
from scipy.stats import binomtest as _binom
from tqdm.auto import tqdm as _tqdm

from inn_analysis_ops.analytics._helpers import build_inn_lookup, tokens_to_mutation_string
from inn_analysis_ops.analytics.motif_schema import MOTIF_INDEX_SCHEMA, MotifStatsCols as C
from inn_analysis_ops.analytics.motifs.bispecific import resolve_bispecific_branches_dep
from inn_analysis_ops.analytics.motifs.subtraction import (
    _add_raw_sub_motifs,
    _subtract_inns_inline,
)


def _mobius_null_freq(
    m: frozenset,
    freq_map: dict,
    freq_indiv: dict,
) -> float:
    """Null frequency for motif m via Möbius log-linear inversion.

    f_null = exp(Σ_{S⊊m, S≠∅} (-1)^(|m|-|S|-1) log f(S))

    For sub-motifs |S|≥2, uses freq_map[S] if observed, otherwise falls back
    to the independence product ∏ freq_indiv[t] for t ∈ S.
    Degenerates to the full-independence null when no sub-motifs are observed.
    """
    tokens = list(m)
    k = len(tokens)
    log_null = 0.0
    for r in range(1, k):  # |S| = 1 … k-1 (all proper non-empty subsets)
        sign = (-1) ** (k - r - 1)
        for subset in combinations(tokens, r):
            s = frozenset(subset)
            if r == 1:
                f = freq_indiv.get(next(iter(s)), 1e-10)
            else:
                f = freq_map.get(s) or _math.prod(freq_indiv.get(t, 1e-10) for t in s)
            log_null += sign * _math.log(max(f, 1e-10))
    return min(_math.exp(log_null), 1.0)


def _fmt_tokens(toks) -> str:
    """Format ``(pos, change)`` token tuples as a sorted mutation string."""
    return tokens_to_mutation_string(toks, sort=True)


_REPASS_ALIASES = {
    "lift": "lift_vs_indep",
    "pvalue": "pvalue_vs_indep",
    "count": "inn_count",
    "size": "num_mutations",
}


def _run_mobius_pass(df, freq_map, N_total, min_lift, max_pvalue):
    """Run one Möbius inversion pass over df using freq_map as the sub-motif pool.

    Returns dicts keyed by motif frozenset:
        cond_exp, cond_lift, cond_pval, cond_pass, cond_subs, cond_new_res
    """
    cond_exp: dict = {}
    cond_lift: dict = {}
    cond_pval: dict = {}
    cond_pass: dict = {}
    cond_subs: dict = {}  # observed sub-motifs (size≥2) present in pool
    cond_new_res: dict = {}  # tokens not covered by any size≥2 sub-motif

    for _, row in _tqdm(
        df.sort_values("num_mutations").iterrows(),
        total=len(df),
        desc="Interaction lift",
    ):
        m = row["motif"]
        freq = row["freq_motif"]
        count = row["inn_count"]
        indiv = row["marginal_freq"]

        if len(m) == 1:
            cond_exp[m] = row["expected_freq_indep"]
            cond_lift[m] = row["lift_vs_indep"]
            cond_pval[m] = row["pvalue_vs_indep"]
            cond_pass[m] = row["passes_indep_filter"]
            cond_subs[m] = []
            cond_new_res[m] = list(m)
            continue

        f_null = _mobius_null_freq(m, freq_map, indiv)
        lft_c = freq / f_null if f_null > 0 else float("inf")
        pval_c = float(
            _binom(count, N_total, min(f_null, 1.0), alternative="greater").pvalue
        )

        cond_exp[m] = f_null
        cond_lift[m] = lft_c
        cond_pval[m] = pval_c
        cond_pass[m] = (lft_c >= min_lift) and (pval_c <= max_pvalue)

        obs_subs = [s for s in freq_map if len(s) >= 2 and s < m]
        covered = set().union(*obs_subs) if obs_subs else set()
        cond_subs[m] = obs_subs
        cond_new_res[m] = [t for t in m if t not in covered]

    return (
        cond_exp,
        cond_lift,
        cond_pval,
        cond_pass,
        cond_subs,
        cond_new_res,
    )


def _dedup_by_constant_chain_mutations(
    motifs_chains: pd.DataFrame,
    inn_col: str,
) -> pd.DataFrame:
    """Drop rows that are exact duplicates of another row for the same INN.

    Two rows are considered duplicates when they share the same *inn_col*
    value **and** produce an identical frozenset of mutation tokens (derived
    from ``EU``, ``mutation_type``, ``germline_seq``, ``mutated_seq``).
    Keeping the first occurrence preserves row order; the returned DataFrame
    has the original index reset.
    """

    def _token_key(row):
        tokens = frozenset(
            (
                eu,
                (f"{gl}{ms}" if str(mt).lower() == "insertion" else f"{gl}->{ms}"),
            )
            for eu, mt, gl, ms in zip(
                row["EU"],
                row["mutation_type"],
                row["germline_seq"],
                row["mutated_seq"],
            )
        )
        return (row[inn_col], tokens)

    df = motifs_chains.copy()
    df["_dedup_key"] = df.apply(_token_key, axis=1)
    df = df.drop_duplicates(subset=["_dedup_key"]).drop(columns=["_dedup_key"])
    return df.reset_index(drop=True)


def _build_motif_table_single_group(
    collapsed: pd.DataFrame,
    min_count: int,
    min_lift: float,
    max_pvalue: float,
    min_size: int,
    max_size: int,
    marginals_table: Optional[pd.DataFrame],
    repass: Optional[dict],
) -> pd.DataFrame:
    """Core single-group algorithm: enumerate sub-combos from collapsed profiles, score, annotate.

    Accepts the output of :func:`_collapse_single_group` — one row per unique
    mutation profile with ``motif`` (frozenset) and ``inns`` (set) columns.
    Sub-combinations of each profile are enumerated and scored; a Möbius
    inversion pass then computes conditional lift / p-value against the pool
    of known lower-order sub-motifs.

    Parameters
    ----------
    collapsed:
        Output of :func:`_collapse_single_group` — must have ``motif``,
        ``inns``, and ``inn_total`` columns.
    min_count:
        Minimum number of distinct INNs carrying a combination.
    min_lift:
        Minimum lift for the Möbius conditional-significance test.
    max_pvalue:
        Maximum p-value for the Möbius conditional-significance test.
    min_size, max_size:
        Combination size range to enumerate.
    marginals_table:
        Optional per-token artifact-fraction lookup table.
    repass:
        Optional filter dict restricting which sub-motifs enter the Möbius
        null (e.g. ``{'lift': ('min', 1.0), 'pvalue': ('max', 0.05)}``).
    group_by:
        Optional list of group column names present in *collapsed*.  When
        provided, the missing-profile check uses a composite key of
        ``(mutation_string, group_val_1, ...)`` so that the same frozenset
        appearing in different groups is treated as distinct.

    Returns
    -------
    Validated ``motif_df`` with the full column schema defined by
    ``MOTIF_INDEX_SCHEMA``, sorted (num_mutations desc, inn_count desc).
    """
    _cols = [
        "mutation_string",
        "motif_id",
        "num_mutations",
        "sub_motifs",
        "novel_mutations",
        "inn_count",
        "inn_total",
        "freq_motif",
        "marginal_freq",
        "marginal_count",
        "expected_freq_indep",
        "lift_vs_indep",
        "log_lift_vs_indep",
        "pvalue_vs_indep",
        "neg_log10_pvalue_vs_indep",
        "lift_passes_vs_indep",
        "pvalue_passes_vs_indep",
        "passes_indep_filter",
        "expected_freq_mobius",
        "lift_vs_submotifs",
        "log_lift_vs_submotifs",
        "pvalue_vs_submotifs",
        "neg_log10_pvalue_vs_submotifs",
        "lift_passes_vs_submotifs",
        "pvalue_passes_vs_submotifs",
        "passes_submotif_filter",
        "pairwise_cohesion",
        "pairwise_lift",
        "explained_by_submotifs",
        "most_sig_sub_motif",
        "motif_score",
        "artifact_fraction",
        "motif_score_artifact_adjusted",
        "longest_motif_in_group",
        "no_larger_passing_motif_in_group",
        "is_fully_described_by_larger_motif",
        "larger_motif_with_same_inns",
        "inns",
        "mutation_list",
        "motif",
    ]

    # N_total is consistent within a single group's collapsed output
    N_total = int(collapsed["inn_total"].iloc[0])
    combo_inns: dict[frozenset, set] = {}
    marginal_inns: dict[tuple, set] = {}

    for _, row in _tqdm(
        collapsed.iterrows(),
        total=len(collapsed),
        desc="Enumerating combos",
    ):
        profile: frozenset = row["motif"]
        inns: set = row["inns"]
        for t in profile:
            marginal_inns.setdefault(t, set()).update(inns)
        tokens = list(profile)
        _cap = min(len(tokens), max_size)
        for size in range(min_size, _cap + 1):
            for combo in combinations(tokens, size):
                combo_inns.setdefault(frozenset(combo), set()).update(inns)

    records = []
    for motif, inns in _tqdm(combo_inns.items(), desc="Scoring motifs"):
        count = len(inns)
        if count < min_count:
            continue
        freq = count / N_total
        count_indiv = {t: len(marginal_inns[t]) for t in motif}
        indiv = {t: c / N_total for t, c in count_indiv.items()}
        expected = float(np.prod(list(indiv.values())))
        if expected <= 0:
            continue
        lift = freq / expected
        pvalue = _binom(
            count, N_total, min(expected, 1.0), alternative="greater"
        ).pvalue
        _lift_pass = len(motif) == 1 or lift >= min_lift
        _pvalue_pass = float(pvalue) <= max_pvalue
        records.append(
            {
                "motif": motif,
                "num_mutations": len(motif),
                "inn_count": count,
                "inns": inns,
                "freq_motif": round(freq, 5),
                "marginal_freq": indiv,
                "marginal_count": count_indiv,
                "expected_freq_indep": round(expected, 6),
                "lift_vs_indep": round(lift, 3),
                "pvalue_vs_indep": round(float(pvalue), 10),
                "lift_passes_vs_indep": _lift_pass,
                "pvalue_passes_vs_indep": _pvalue_pass,
                "passes_indep_filter": _lift_pass and _pvalue_pass,
            }
        )

    if not records:
        return pd.DataFrame(columns=_cols)

    df = pd.DataFrame(records)
    if df.empty:
        return pd.DataFrame(columns=_cols)

    # ── Interaction lift via Möbius inversion (second pass, size-ascending) ───
    freq_map: dict = dict(zip(df["motif"], df["freq_motif"]))
    if repass:
        _mask = pd.Series(True, index=df.index)
        for _key, (_op, _thresh) in repass.items():
            _col = _REPASS_ALIASES.get(_key, _key)
            _mask &= df[_col] >= _thresh if _op == "min" else df[_col] <= _thresh
        freq_map = dict(zip(df.loc[_mask, "motif"], df.loc[_mask, "freq_motif"]))
    cond_exp, cond_lift, cond_pval, cond_pass, cond_subs, cond_new_res = (
        _run_mobius_pass(df, freq_map, N_total, min_lift, max_pvalue)
    )

    df["expected_freq_mobius"] = df["motif"].map(cond_exp).round(6)
    df["lift_vs_submotifs"] = df["motif"].map(cond_lift).round(3)
    df["pvalue_vs_submotifs"] = df["motif"].map(cond_pval).round(5)
    df["lift_passes_vs_submotifs"] = df["motif"].apply(
        lambda m: len(m) == 1 or cond_lift[m] >= min_lift
    )
    df["pvalue_passes_vs_submotifs"] = df["motif"].apply(
        lambda m: cond_pval[m] <= max_pvalue
    )
    df["passes_submotif_filter"] = df["motif"].map(cond_pass)

    df["explained_by_submotifs"] = (
        df["passes_indep_filter"] & ~df["passes_submotif_filter"]
    )

    # ── Maximality flags ──────────────────────────────────────────────────────
    all_motifs = set(df["motif"])
    passing_motifs_cond = set(df.loc[df["passes_submotif_filter"], "motif"])
    df["longest_motif_in_group"] = df["motif"].apply(
        lambda m: not any(m < s for s in all_motifs)
    )
    df["no_larger_passing_motif_in_group"] = df["motif"].apply(
        lambda m: not any(m < s for s in passing_motifs_cond)
    )

    # ── Human-readable expansion ──────────────────────────────────────────────
    def _sorted_tokens(m):
        return sorted(m, key=lambda t: (str(t[0]), t[1]))

    df["mutation_list"] = df["motif"].apply(_sorted_tokens)
    df["num_mutations"] = df["motif"].apply(len)
    df["mutation_string"] = df["mutation_list"].apply(
        lambda toks: " / ".join(f"{eu}:{change}" for eu, change in toks)
    )

    motif_to_str = dict(zip(df["motif"], df["mutation_string"]))

    df["sub_motifs"] = df["motif"].apply(
        lambda m: [motif_to_str[s] for s in cond_subs.get(m, []) if s in motif_to_str]
    )
    df["novel_mutations"] = df["motif"].apply(
        lambda m: _fmt_tokens(cond_new_res.get(m, list(m)))
    )

    consumed_by_map: dict[frozenset, str] = {}
    _seen_inn_sets: dict[frozenset, str] = {}
    for _, row in (
        df[df["passes_indep_filter"]]
        .sort_values("num_mutations", ascending=False)
        .iterrows()
    ):
        m = row["motif"]
        inn_key = frozenset(row["inns"])
        if inn_key in _seen_inn_sets:
            consumed_by_map[m] = _seen_inn_sets[inn_key]
        else:
            _seen_inn_sets[inn_key] = motif_to_str[m]
    df["is_fully_described_by_larger_motif"] = df["motif"].apply(
        lambda m: m in consumed_by_map
    )
    df["larger_motif_with_same_inns"] = df["motif"].apply(
        lambda m: consumed_by_map.get(m, None)
    )

    # ── Prune consumed sub-motifs & recompute novel_mutations ─────────────────
    _consumed_frozensets: set = set(
        df.loc[df["is_fully_described_by_larger_motif"], "motif"]
    )

    def _prune_and_recompute(row):
        m = row["motif"]
        pruned_subs = [s for s in cond_subs.get(m, []) if s not in _consumed_frozensets]
        covered: set = set().union(*pruned_subs) if pruned_subs else set()
        novel_toks = sorted(
            (t for t in m if t not in covered),
            key=lambda t: (str(t[0]), t[1]),
        )
        assert covered | set(novel_toks) == set(m), (
            f"sub_motifs + novel_mutations does not reconstruct '{row['mutation_string']}': "
            f"covered={covered}, novel={set(novel_toks)}"
        )
        assert not (covered & set(novel_toks)), (
            f"novel_mutations overlaps sub_motifs for '{row['mutation_string']}'"
        )
        return (
            [motif_to_str[s] for s in pruned_subs if s in motif_to_str],
            _fmt_tokens(novel_toks),
        )

    _pruned = df.apply(_prune_and_recompute, axis=1)
    df["sub_motifs"] = _pruned.apply(lambda x: x[0])
    df["novel_mutations"] = _pruned.apply(lambda x: x[1])

    _pass_lift_cond = {
        row["motif"]: row["lift_vs_submotifs"]
        for _, row in df[df["passes_submotif_filter"]].iterrows()
    }

    def _most_sig_sub_motif(row):
        if not row["explained_by_submotifs"]:
            return None
        m = row["motif"]
        candidates = [(s, _pass_lift_cond[s]) for s in _pass_lift_cond if s < m]
        if not candidates:
            return None
        best = max(candidates, key=lambda x: x[1])[0]
        return motif_to_str.get(best)

    df["most_sig_sub_motif"] = df.apply(_most_sig_sub_motif, axis=1)

    # ── Log-lift & -log10(pvalue) ─────────────────────────────────────────────
    df["log_lift_vs_indep"] = np.log(df["lift_vs_indep"].clip(lower=1e-10)).round(4)
    df["log_lift_vs_submotifs"] = np.log(
        df["lift_vs_submotifs"].clip(lower=1e-10)
    ).round(4)
    df["neg_log10_pvalue_vs_indep"] = (
        -np.log10(df["pvalue_vs_indep"].clip(lower=1e-300))
    ).round(4)
    df["neg_log10_pvalue_vs_submotifs"] = (
        -np.log10(df["pvalue_vs_submotifs"].clip(lower=1e-300))
    ).round(4)

    # ── Composite motif score ─────────────────────────────────────────────────
    df["motif_score"] = (
        df["log_lift_vs_submotifs"].clip(lower=0)
        * df["neg_log10_pvalue_vs_submotifs"].clip(lower=0)
        * np.log1p(df["inn_count"])
    ).round(4)

    # ── Artifact fraction (from marginals_table) ──────────────────────────────
    if marginals_table is not None:
        _marg_iter = marginals_table.reset_index()
        _marg_lookup: dict[tuple, float] = {}
        for _, mr in _marg_iter.iterrows():
            if mr.get("mutation_type") == "insertion":
                change = f"{mr['germline_seq']}{mr['mutated_seq']}"
            else:
                change = f"{mr['germline_seq']}->{mr['mutated_seq']}"
            key = (mr["_pos"], change)
            n = max(int(mr["n_inn"]), 1)
            frac = (
                int(mr.get("n_inn_multi_chain", 0))
                + int(mr.get("n_inn_intra_repeat", 0))
            ) / n
            _marg_lookup[key] = min(frac, 1.0)

        def _artifact_frac(motif):
            fracs = [_marg_lookup.get(t, 0.0) for t in motif]
            return round(float(np.mean(fracs)), 4) if fracs else 0.0

        df["artifact_fraction"] = df["motif"].apply(_artifact_frac)
    else:
        df["artifact_fraction"] = np.nan

    df["motif_score_artifact_adjusted"] = (
        df["motif_score"] * (1.0 - df["artifact_fraction"].fillna(0.0))
    ).round(4)

    # ── Pairwise co_df metrics ────────────────────────────────────────────────
    def _pair_metrics(motif):
        tokens = list(motif)
        pairs = list(combinations(tokens, 2))
        if not pairs:
            return np.nan, np.nan
        max_confs, lifts = [], []
        for a, b in pairs:
            sa, sb = marginal_inns.get(a, set()), marginal_inns.get(b, set())
            if not sa or not sb:
                continue
            n_ab = len(sa & sb)
            max_confs.append(max(n_ab / len(sa), n_ab / len(sb)))
            lifts.append((n_ab / N_total) / ((len(sa) / N_total) * (len(sb) / N_total)))
        return (
            round(float(np.mean(max_confs)), 4) if max_confs else np.nan,
            round(float(np.mean(lifts)), 4) if lifts else np.nan,
        )

    metrics = df["motif"].apply(_pair_metrics)
    df["pairwise_cohesion"] = metrics.apply(lambda x: x[0])
    df["pairwise_lift"] = metrics.apply(lambda x: x[1])
    df["inn_total"] = N_total

    # ── passes_filters ────────────────────────────────────────────────────────
    # Uses the Möbius conditional significance — the strictest test — which
    # checks whether the motif is significant beyond what its sub-motifs predict.
    # For singletons (size == 1), passes_submotif_filter is always True.
    df["passes_filters"] = (df["inn_count"] >= min_count) & df["passes_submotif_filter"]

    df = df.sort_values(
        ["num_mutations", "inn_count"], ascending=[False, False]
    ).reset_index(drop=True)
    df["motif_id"] = df.index
    df = _annotate_is_seed(df)

    # ── Column ordering ───────────────────────────────────────────────────────
    _front = [
        "mutation_string",
        "motif_id",
        "num_mutations",
        "sub_motifs",
        "is_seed",
        "novel_mutations",
        "inn_count",
        "inn_total",
        "freq_motif",
        "marginal_freq",
        "marginal_count",
        "expected_freq_indep",
        "lift_vs_indep",
        "log_lift_vs_indep",
        "expected_freq_mobius",
        "lift_vs_submotifs",
        "log_lift_vs_submotifs",
        "pairwise_cohesion",
        "pairwise_lift",
        "pvalue_vs_indep",
        "neg_log10_pvalue_vs_indep",
        "pvalue_vs_submotifs",
        "neg_log10_pvalue_vs_submotifs",
        "lift_passes_vs_indep",
        "pvalue_passes_vs_indep",
        "passes_indep_filter",
        "lift_passes_vs_submotifs",
        "pvalue_passes_vs_submotifs",
        "passes_submotif_filter",
        "explained_by_submotifs",
        "most_sig_sub_motif",
        "passes_filters",
        "motif_score",
        "artifact_fraction",
        "motif_score_artifact_adjusted",
        "longest_motif_in_group",
        "no_larger_passing_motif_in_group",
        "is_fully_described_by_larger_motif",
        "larger_motif_with_same_inns",
        "inns",
        "mutation_list",
        "motif",
    ]
    _rest = [c for c in df.columns if c not in _front]
    df = df[_front + _rest]

    for _col in ("most_sig_sub_motif", "larger_motif_with_same_inns"):
        if _col in df.columns:
            df[_col] = df[_col].astype(object)

    return MOTIF_INDEX_SCHEMA.validate(df, lazy=True)


def _apply_group_by_split(
    motifs_chains: pd.DataFrame,
    group_by: list,
    single_group_fn,
    **kwargs,
) -> pd.DataFrame:
    """Run *single_group_fn* independently per group and reassign motif_ids.

    Parameters
    ----------
    motifs_chains:
        Input DataFrame (pre-filtered or raw).
    group_by:
        Column names to split on; columns absent from *motifs_chains* are
        silently ignored.
    single_group_fn:
        Callable with signature ``(subset, **kwargs) -> pd.DataFrame``.
    **kwargs:
        Forwarded verbatim to *single_group_fn*.

    Returns
    -------
    Concatenated ``motif_df`` with globally unique ``motif_id`` values, or an
    empty ``pd.DataFrame`` if no group produced results.
    """
    available = [c for c in group_by if c in motifs_chains.columns]
    if not available:
        return single_group_fn(motifs_chains, **kwargs)
    parts = []
    for keys, subset in motifs_chains.groupby(available, sort=True):
        part = single_group_fn(subset, **kwargs)
        if not part.empty:
            keys_iter = keys if isinstance(keys, tuple) else (keys,)
            for col, val in zip(available, keys_iter):
                part.insert(0, col, val)
            parts.append(part)
    if not parts:
        return pd.DataFrame()
    combined = pd.concat(parts, ignore_index=True)
    combined["motif_id"] = combined.index
    assert combined["motif_id"].is_unique, "motif_id not unique after group concat"
    return combined


def _collapse_single_group(
    motifs_chains: pd.DataFrame,
    inn_col: str,
    marginals_table: Optional[pd.DataFrame],
) -> pd.DataFrame:
    """Single-group core for :func:`collapse_motifs_chains`.

    Groups rows by their exact full mutation token set (frozenset of all
    ``(EU, change)`` tokens in the row), counts distinct INNs per group, and
    computes marginal frequencies, independence-expected frequency, lift, and
    binomial p-value.  No sub-combination enumeration; no Möbius pass.
    """

    def _row_tokens(row):
        return frozenset(
            (
                eu,
                (f"{gl}{ms}" if str(mt).lower() == "insertion" else f"{gl}->{ms}"),
            )
            for eu, mt, gl, ms in zip(
                row["EU"],
                row["mutation_type"],
                row["germline_seq"],
                row["mutated_seq"],
            )
        )

    N_total = motifs_chains[inn_col].nunique()

    # ── Per-row token sets and marginal INN sets ──────────────────────────────
    motif_inns: dict[frozenset, set] = {}
    marginal_inns: dict[tuple, set] = {}

    for _, row in _tqdm(
        motifs_chains.iterrows(),
        total=len(motifs_chains),
        desc="Collapsing motifs",
    ):
        inn = row[inn_col]
        tokens = _row_tokens(row)
        motif_inns.setdefault(tokens, set()).add(inn)
        for t in tokens:
            marginal_inns.setdefault(t, set()).add(inn)

    # ── Score each unique motif ───────────────────────────────────────────────
    records = []
    for motif, inns in motif_inns.items():
        count = len(inns)
        freq = count / N_total
        count_indiv = {t: len(marginal_inns[t]) for t in motif}
        indiv = {t: c / N_total for t, c in count_indiv.items()}
        expected = float(np.prod(list(indiv.values()))) if indiv else 0.0
        if expected <= 0:
            continue
        lift = freq / expected
        pvalue = float(
            _binom(count, N_total, min(expected, 1.0), alternative="greater").pvalue
        )
        records.append(
            {
                "motif": motif,
                "num_mutations": len(motif),
                "inn_count": count,
                "inn_total": N_total,
                "inns": inns,
                "freq_motif": round(freq, 5),
                "marginal_freq": indiv,
                "marginal_count": count_indiv,
                "expected_freq_indep": round(expected, 6),
                "lift_vs_indep": round(lift, 3),
                "pvalue_vs_indep": round(pvalue, 10),
            }
        )

    if not records:
        return pd.DataFrame()

    df = pd.DataFrame(records)

    def _sorted_tokens(m):
        return sorted(m, key=lambda t: (str(t[0]), t[1]))

    df["mutation_list"] = df["motif"].apply(_sorted_tokens)
    df["mutation_string"] = df["mutation_list"].apply(
        lambda toks: " / ".join(f"{eu}:{change}" for eu, change in toks)
    )
    df["log_lift_vs_indep"] = np.log(df["lift_vs_indep"].clip(lower=1e-10)).round(4)
    df["neg_log10_pvalue_vs_indep"] = (
        -np.log10(df["pvalue_vs_indep"].clip(lower=1e-300))
    ).round(4)

    # ── Artifact fraction (from marginals_table) ──────────────────────────────
    if marginals_table is not None:
        # marginals_table may have its key columns in the index (new format from
        # build_marginals_table) — reset so all columns are accessible by name.
        _marg_iter = marginals_table.reset_index()
        _marg_lookup: dict[tuple, float] = {}
        for _, mr in _marg_iter.iterrows():
            change = (
                f"{mr['germline_seq']}{mr['mutated_seq']}"
                if mr.get("mutation_type") == "insertion"
                else f"{mr['germline_seq']}->{mr['mutated_seq']}"
            )
            n = max(int(mr["n_inn"]), 1)
            frac = (
                int(mr.get("n_inn_multi_chain", 0))
                + int(mr.get("n_inn_intra_repeat", 0))
            ) / n
            _marg_lookup[(mr["_pos"], change)] = min(frac, 1.0)

        df["artifact_fraction"] = df["motif"].apply(
            lambda m: (
                round(
                    float(np.mean([_marg_lookup.get(t, 0.0) for t in m])),
                    4,
                )
                if m
                else 0.0
            )
        )
    else:
        df["artifact_fraction"] = np.nan

    df = df.sort_values(
        ["num_mutations", "inn_count"], ascending=[False, False]
    ).reset_index(drop=True)
    df.insert(0, "motif_id", df.index)

    _front = [
        "motif_id",
        "mutation_string",
        "num_mutations",
        "inn_count",
        "inn_total",
        "freq_motif",
        "marginal_freq",
        "marginal_count",
        "expected_freq_indep",
        "lift_vs_indep",
        "log_lift_vs_indep",
        "pvalue_vs_indep",
        "neg_log10_pvalue_vs_indep",
        "artifact_fraction",
        "inns",
        "mutation_list",
        "motif",
    ]
    _rest = [c for c in df.columns if c not in _front]
    return df[_front + _rest]


def collapse_motifs_chains(
    motifs_chains: pd.DataFrame,
    inn_col: str = "inn",
    marginals_table: Optional[pd.DataFrame] = None,
    group_by: Optional[list] = None,
) -> pd.DataFrame:
    """Collapse *motifs_chains* to a table of unique observed mutation profiles.

    Each row in *motifs_chains* represents one INN × region entry with a
    list-valued set of mutations.  This function groups rows by their **exact
    full mutation token set** (the frozenset of all ``(EU, change)`` tokens in
    the row) and counts how many distinct INNs carry each profile.

    No sub-combination enumeration and no Möbius pass are performed.  The
    output is a simple frequency table — one row per unique mutation profile —
    with the independence-expected frequency, lift, and binomial p-value
    computed for that exact profile.  Use :func:`build_motif_index` for
    sub-motif discovery and significance-filtered results.

    Parameters
    ----------
    motifs_chains:
        DataFrame with list-valued columns ``EU``, ``mutation_type``,
        ``germline_seq``, ``mutated_seq`` and an INN identifier column.
        Typically the ``ConstantChainMotifsByRegion`` sheet — one row per
        INN × region.
    inn_col:
        Column name for the INN / antibody identifier (default ``"inn"``).
    marginals_table:
        Optional per-token artifact-fraction lookup for the
        ``artifact_fraction`` column.
    group_by:
        Optional list of column names to split on.  Each group is processed
        independently and results are concatenated with globally unique
        ``motif_id`` values.

    Returns
    -------
    DataFrame sorted (num_mutations desc, inn_count desc) with columns:

    * ``motif_id``                          – unique row identifier
    * ``mutation_string``                   – human-readable token list
    * ``num_mutations``                     – number of mutations in the profile
    * ``inn_count``                         – distinct INNs carrying this profile
    * ``inn_total``                         – total INN population in this group
    * ``freq_motif``                        – inn_count / inn_total
    * ``marginal_freq``                     – per-token marginal frequency dict
    * ``marginal_count``                    – per-token marginal INN count dict
    * ``expected_freq_indep`` – product of marginal frequencies
    * ``lift_vs_indep``          – freq_motif / expected
    * ``log_lift_vs_indep``
    * ``pvalue_vs_indep``        – one-sided binomial p-value
    * ``neg_log10_pvalue_vs_indep``
    * ``artifact_fraction``                 – mean per-token artifact fraction
    * ``inns``                              – set of INN identifiers
    * ``mutation_list``                     – sorted token list
    * ``motif``                             – frozenset of ``(EU, change)`` tokens
    """
    _fn = lambda subset, **kw: _collapse_single_group(  # noqa: E731
        subset, inn_col=inn_col, marginals_table=marginals_table
    )
    if group_by:
        return _apply_group_by_split(motifs_chains, group_by, _fn)
    return _fn(motifs_chains)


# ---------------------------------------------------------------------------
# MotifIndexResult: pipeline result object for build_motif_index
# ---------------------------------------------------------------------------


@dataclass
class MotifIndexResult:
    """Result of :func:`build_motif_index` containing each pipeline step.

    Attributes
    ----------
    scored_df : DataFrame
        Full catalog of all enumerated combinations that passed
        ``min_count``/``min_size``/``max_size``, with all pipeline columns
        appended: independence stats, ``raw_sub_motifs`` (unpruned ancestry),
        ``adjusted_inns``/``adjusted_count``/``adjusted_freq`` after INN
        subtraction, and ``is_fully_described_by_larger_motif``
        (= ``adjusted_count == 0``).
    removed_df : DataFrame
        Subset of ``scored_df`` where ``adjusted_count == 0`` — motifs whose
        INNs were fully absorbed by larger motifs.
    motif_df_pre_recompute : DataFrame
        Subset of ``scored_df`` where ``adjusted_count > 0`` — surviving motifs
        before Möbius statistics are recomputed.
    motif_df : DataFrame
        Primary analysis output.  After the optional cohesion refinement loop
        this is the final recomputed table: surviving motifs with all Möbius
        statistics, INN subtraction, and cohesion columns
        (``is_cohesive``, ``min_frac_pair_in_full``, ``candidate_sub_motifs``,
        etc.) recomputed on the refined set.
    filters : dict
        The filter parameters used to produce this result.
    group_by : list or None
        Grouping columns used during computation.
    inn_lookup : dict
        Pre-built ``mutation_string → set[INN]`` dict built from
        ``scored_df``.  Shared by all downstream debug/audit functions —
        pass this instead of rebuilding from the DataFrame each time.
    cohesion_frac_threshold : float or None
        The ``frac_pair_in_full`` threshold used for the refinement loop.
        ``None`` if refinement was not run.
    cohesion_iterations : int
        Number of refinement iterations that actually ran (0 if skipped).
    refinement_log : list or None
        Per-iteration log from the cohesion loop.  Each entry is a dict with
        ``iteration``, ``n_leaky``, ``split`` (list of
        ``{source, replaced_by}``), and ``converged`` flag.
    motif_df_pre_refinement : DataFrame or None
        Snapshot of ``motif_df`` before the cohesion refinement loop — useful
        for comparing what changed.  ``None`` if refinement was not run.
    """

    scored_df: pd.DataFrame
    removed_df: pd.DataFrame
    motif_df_pre_recompute: pd.DataFrame
    motif_df: pd.DataFrame
    filters: dict
    group_by: Optional[list] = None
    inn_lookup: Optional[dict] = None
    cohesion_frac_threshold: Optional[float] = None
    cohesion_iterations: int = 0
    refinement_log: Optional[list] = None
    motif_df_pre_refinement: Optional[pd.DataFrame] = None
    step_changes: Dict[str, Dict[str, pd.DataFrame]] = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.motif_df)

    @staticmethod
    def _count_unique_inns(df: Optional[pd.DataFrame]) -> int:
        """Count unique INNs represented by a motif table."""
        if not isinstance(df, pd.DataFrame) or df.empty:
            return 0
        if "inns" in df.columns:
            acc: set[str] = set()
            for val in df["inns"]:
                if isinstance(val, (set, frozenset, list, tuple)):
                    acc.update(str(x) for x in val if str(x))
            if acc:
                return len(acc)
        if "inn_count" in df.columns:
            return int(pd.to_numeric(df["inn_count"], errors="coerce").fillna(0).sum())
        return 0

    @staticmethod
    def _shape(df: Optional[pd.DataFrame]) -> str:
        if not isinstance(df, pd.DataFrame):
            return "n/a"
        return f"{len(df)}x{len(df.columns)}"

    @staticmethod
    def _count_unique_mutation_string_pe(
        df: Optional[pd.DataFrame],
    ) -> int:
        """Count unique mutation_string_pe values in motif table."""
        if (
            not isinstance(df, pd.DataFrame)
            or df.empty
            or "mutation_string_pe" not in df.columns
        ):
            return 0
        return int(df["mutation_string_pe"].nunique())

    @staticmethod
    def _count_unique_seed_strings(df: Optional[pd.DataFrame]) -> int:
        """Count unique mutation_string_pe for seed motifs (is_seed=True)."""
        if not isinstance(df, pd.DataFrame) or df.empty:
            return 0
        if "is_seed" not in df.columns or "mutation_string_pe" not in df.columns:
            return 0
        seeds = df[df["is_seed"] == True]
        if seeds.empty:
            return 0
        return int(seeds["mutation_string_pe"].nunique())

    def __repr__(self) -> str:
        # Tables in generation order: scored_df → motif_df_pre_recompute → removed_df →
        # motif_df_pre_refinement (optional) → motif_df
        table_rows = [
            ("scored_df", self.scored_df),
            ("motif_df_pre_recompute", self.motif_df_pre_recompute),
            ("removed_df", self.removed_df),
            ("motif_df_pre_refinement", self.motif_df_pre_refinement),
            ("motif_df", self.motif_df),
        ]
        table_lines = [
            f"  - {name}: shape={self._shape(df)}, unique_inns={self._count_unique_inns(df)}, "
            f"unique_motif_strings_pe={self._count_unique_mutation_string_pe(df)}, "
            f"unique_seed_strings={self._count_unique_seed_strings(df)}"
            for name, df in table_rows
            if isinstance(df, pd.DataFrame)
        ]

        step_lines: list[str] = []
        for step_name, payload in (self.step_changes or {}).items():
            if not isinstance(payload, dict):
                continue
            # Handle count-based metadata steps (initial_unique_substrings, generated_submotifs)
            if "description" in payload and "count" in payload:
                desc = payload.get("description", "")
                count = payload.get("count", 0)
                unique_inns = payload.get("unique_inns", 0)
                step_lines.append(f"  - {step_name}: {desc} (inns={unique_inns})")
            # Handle transformation steps (added/removed DataFrames)
            else:
                added_df = payload.get("added")
                removed_df = payload.get("removed")
                added_rows = len(added_df) if isinstance(added_df, pd.DataFrame) else 0
                removed_rows = (
                    len(removed_df) if isinstance(removed_df, pd.DataFrame) else 0
                )
                added_inns = self._count_unique_inns(added_df)
                removed_inns = self._count_unique_inns(removed_df)
                step_lines.append(
                    f"  - {step_name}: added_rows={added_rows}, added_inns={added_inns}, "
                    f"filtered_removed_rows={removed_rows}, filtered_removed_inns={removed_inns}"
                )

        ref = f"\nSummary: cohesion_refinement_iterations={self.cohesion_iterations}"
        step_block = (
            "\nSteps:\n" + "\n".join(step_lines) if step_lines else "\nSteps:\n  - none"
        )
        return "MotifIndexResult\nTables:\n" + "\n".join(table_lines) + step_block + ref

    def search_motif(self, mutation_string_pe: str) -> str:
        """Search for a mutation string across all pipeline tables and steps.

        Provides comprehensive audit trail showing:
        - Which tables contain the exact motif
        - If filtered, at which step and why
        - INNs/chains that had this motif before filtering
        - Substrings of this motif found in the final motif_df

        Parameters
        ----------
        mutation_string_pe : str
            Mutation string in PE notation (e.g., 'S267K/Q295E/L368D/K370S/N384D/Q418E/N421D')

        Returns
        -------
        str
            Formatted report showing search results across pipeline
        """
        from inn_analysis_ops.analytics._helpers import motif_is_subset, split_motif_tokens

        report_lines = [f"Search Results for: {mutation_string_pe}\n"]

        # Step 1: Check which tables contain this motif
        report_lines.append("Tables containing this motif:")
        tables_with_motif = []
        tables_data = [
            ("scored_df", self.scored_df),
            ("motif_df_pre_recompute", self.motif_df_pre_recompute),
            ("motif_df_pre_refinement", self.motif_df_pre_refinement),
            ("motif_df", self.motif_df),
        ]

        for table_name, df in tables_data:
            if (
                not isinstance(df, pd.DataFrame)
                or df.empty
                or "mutation_string_pe" not in df.columns
            ):
                continue
            mask = df["mutation_string_pe"] == mutation_string_pe
            if mask.any():
                count = mask.sum()
                tables_with_motif.append(table_name)
                report_lines.append(f"  ✓ {table_name}: {count} row(s)")
                # Show INN count for this motif
                rows = df[mask]
                if "inns" in rows.columns:
                    inn_set = set()
                    for inns_val in rows["inns"]:
                        if isinstance(inns_val, (set, frozenset, list)):
                            inn_set.update(inns_val)
                    if inn_set:
                        report_lines.append(
                            f"    └─ INNs: {len(inn_set)} unique ({', '.join(sorted(str(x) for x in inn_set)[:5])}{'...' if len(inn_set) > 5 else ''})"
                        )

        if not tables_with_motif:
            report_lines.append("  ✗ Not found in any table")

        # Step 2: Check if motif was filtered and track through steps
        report_lines.append("\nFiltering history:")
        found_in_step = False
        if self.step_changes:
            for step_name, payload in self.step_changes.items():
                if not isinstance(payload, dict):
                    continue
                removed_df = payload.get("removed")
                if (
                    isinstance(removed_df, pd.DataFrame)
                    and not removed_df.empty
                    and "mutation_string_pe" in removed_df.columns
                ):
                    mask = removed_df["mutation_string_pe"] == mutation_string_pe
                    if mask.any():
                        found_in_step = True
                        rows = removed_df[mask]
                        count = len(rows)
                        report_lines.append(
                            f"  🗑  {step_name}: removed in {count} row(s)"
                        )
                        # Report INNs that had this motif
                        if "inns" in rows.columns:
                            inn_set = set()
                            for inns_val in rows["inns"]:
                                if isinstance(inns_val, (set, frozenset, list)):
                                    inn_set.update(inns_val)
                            if inn_set:
                                report_lines.append(
                                    f"    └─ INNs removed: {', '.join(sorted(str(x) for x in inn_set)[:5])}{'...' if len(inn_set) > 5 else ''}"
                                )

        if not found_in_step and "motif_df" in tables_with_motif:
            report_lines.append("  ✓ Survived all filtering steps")
        elif not found_in_step and "scored_df" in tables_with_motif:
            report_lines.append(
                "  ✗ Filtered out (filtered at initial_filter or inn_subtraction)"
            )

        # Step 3: Find substrings of this motif in final motif_df
        report_lines.append("\nSubstrings in final motif_df:")
        if (
            isinstance(self.motif_df, pd.DataFrame)
            and "mutation_string_pe" in self.motif_df.columns
        ):
            query_tokens = split_motif_tokens(mutation_string_pe)
            substrings = []

            for motif in self.motif_df["mutation_string_pe"].unique():
                if motif == mutation_string_pe:
                    continue  # Skip the exact match
                motif_tokens = split_motif_tokens(motif)
                # Check if motif is a subset of the query string
                if motif_is_subset(motif, mutation_string_pe):
                    rows = self.motif_df[self.motif_df["mutation_string_pe"] == motif]
                    is_seed = (
                        rows["is_seed"].iloc[0] if "is_seed" in rows.columns else None
                    )
                    seed_marker = " [SEED]" if is_seed else ""
                    inn_count = self._count_unique_inns(rows)
                    substrings.append(f"  ├─ {motif}{seed_marker} (inns={inn_count})")

            if substrings:
                report_lines.extend(substrings)
            else:
                report_lines.append("  (none)")

        return "\n".join(report_lines)


def _compute_is_seed(sub_motifs_value: object) -> bool:
    """Return True when a motif has no discovered sub-motifs."""
    if isinstance(sub_motifs_value, (set, list, tuple, frozenset)):
        return len(sub_motifs_value) == 0
    if sub_motifs_value is None or (
        isinstance(sub_motifs_value, float) and np.isnan(sub_motifs_value)
    ):
        return True
    return False


def _annotate_is_seed(
    df: pd.DataFrame,
    sub_motifs_col: str = "sub_motifs",
) -> pd.DataFrame:
    """Add/refresh ``is_seed`` flag from sub-motif ancestry."""
    out = df.copy()
    if sub_motifs_col in out.columns:
        out["is_seed"] = out[sub_motifs_col].apply(_compute_is_seed)
    else:
        out["is_seed"] = False
    return out


def _recompute_mobius_stats(
    df: pd.DataFrame,
    group_by: Optional[list],
    min_lift: float,
    max_pvalue: float,
    repass: Optional[dict],
) -> pd.DataFrame:
    """Recompute Möbius sub-motif statistics on *df* (the pruned table).

    Rebuilds ``freq_map`` from surviving motifs, re-runs the Möbius pass, and
    overwrites the ``sub_motifs``, ``novel_mutations``, and all
    ``*_vs_submotifs`` columns with fresh values scoped to the pruned set.
    """
    _GROUP_COLS = ["gene", "chain_type", "gene_species"]
    group_cols = [c for c in (group_by or []) if c in df.columns] or [
        c for c in _GROUP_COLS if c in df.columns
    ]

    def _process_group(gdf: pd.DataFrame) -> pd.DataFrame:
        gdf = gdf.copy()
        if gdf.empty:
            return gdf

        N_total_vals = gdf["inn_total"].dropna()
        N_total = int(N_total_vals.iloc[0]) if not N_total_vals.empty else 1

        freq_map: dict = dict(zip(gdf["motif"], gdf["freq_motif"]))
        if repass:
            _mask = pd.Series(True, index=gdf.index)
            for _key, (_op, _thresh) in repass.items():
                _col = _REPASS_ALIASES.get(_key, _key)
                if _col not in gdf.columns:
                    continue
                _mask &= gdf[_col] >= _thresh if _op == "min" else gdf[_col] <= _thresh
            freq_map = dict(zip(gdf.loc[_mask, "motif"], gdf.loc[_mask, "freq_motif"]))

        (
            cond_exp,
            cond_lift,
            cond_pval,
            cond_pass,
            cond_subs,
            cond_new_res,
        ) = _run_mobius_pass(gdf, freq_map, N_total, min_lift, max_pvalue)

        motif_to_str = dict(zip(gdf["motif"], gdf["mutation_string"]))

        gdf["expected_freq_mobius"] = gdf["motif"].map(cond_exp).round(6)
        gdf["lift_vs_submotifs"] = gdf["motif"].map(cond_lift).round(3)
        gdf["pvalue_vs_submotifs"] = gdf["motif"].map(cond_pval).round(5)
        gdf["lift_passes_vs_submotifs"] = gdf["motif"].apply(
            lambda m: len(m) == 1 or cond_lift.get(m, 0) >= min_lift
        )
        gdf["pvalue_passes_vs_submotifs"] = gdf["motif"].apply(
            lambda m: cond_pval.get(m, 1.0) <= max_pvalue
        )
        gdf["passes_submotif_filter"] = (
            gdf["lift_passes_vs_submotifs"] & gdf["pvalue_passes_vs_submotifs"]
        )
        gdf["explained_by_submotifs"] = ~gdf["passes_indep_filter"].fillna(
            False
        ).astype(bool) & ~gdf["passes_submotif_filter"].fillna(False).astype(bool)

        # Rebuild sub_motifs (pruned to surviving motifs only — no consumed motifs remain)
        surviving = set(gdf["motif"])
        gdf["sub_motifs"] = gdf["motif"].apply(
            lambda m: [
                motif_to_str[s]
                for s in cond_subs.get(m, [])
                if s in motif_to_str and s in surviving
            ]
        )
        gdf["novel_mutations"] = gdf["motif"].apply(
            lambda m: _fmt_tokens(cond_new_res.get(m, list(m)))
        )
        return gdf

    if group_cols:
        parts = []
        for _, gdf in df.groupby(group_cols, sort=False):
            parts.append(_process_group(gdf))
        return pd.concat(parts, ignore_index=True)
    return _process_group(df)


def _add_cohesion_columns(
    motif_df: pd.DataFrame,
    inn_lookup: dict[str, set],
    inn_total: int,
    frac_threshold: float = 1.0,
) -> pd.DataFrame:
    """Add pairwise-cohesion columns to *motif_df* in a single pass.

    For every motif with ≥ 2 tokens, computes all C(n,2) tight-pair statistics
    against the pre-built *inn_lookup*, then classifies each motif as cohesive
    or leaky and records the maximal tight-clique decomposition.

    All lookups go through *inn_lookup* (a plain dict) — no DataFrame scanning.

    Columns added
    -------------
    ``is_cohesive`` : bool
        True when every pair has ``frac_pair_in_full >= frac_threshold``.
        A cohesive motif is a genuine seed; a leaky one may be over-greedy.
    ``min_frac_pair_in_full`` : float
        Worst (lowest) ``frac_pair_in_full`` across all C(n,2) pairs.
    ``mean_frac_pair_in_full`` : float
        Mean ``frac_pair_in_full`` across all pairs.
    ``n_leaky_pairs`` : int
        Number of pairs below *frac_threshold*.
    ``leakiest_pair`` : str or None
        Canonical mutation_string of the pair with the lowest
        ``frac_pair_in_full``.
    ``leakiest_pair_inn_count`` : int or None
        INN count of that leakiest pair.
    ``candidate_sub_motifs`` : list[dict] or None
        For leaky motifs: one entry per maximal clique (size ≥ 2) of the
        tight-pair graph, sorted by size then inn_count descending.
        Each dict has ``mutation_string``, ``size``, ``inn_count``,
        ``inn_freq``, ``in_scored_df``.
        ``None`` for cohesive motifs (no decomposition needed).
    """
    motif_df = motif_df.copy()

    # Extract column arrays — one pass, no iterrows
    ms_list: list[str] = motif_df["mutation_string"].tolist()

    is_cohesive_col: list = []
    min_frac_col: list = []
    mean_frac_col: list = []
    n_leaky_col: list = []
    leakiest_pair_col: list = []
    leakiest_pair_count_col: list = []
    leakiest_pair_outside_col: list = []
    candidate_col: list = []

    for ms in ms_list:
        if not isinstance(ms, str) or not ms.strip():
            is_cohesive_col.append(None)
            min_frac_col.append(float("nan"))
            mean_frac_col.append(float("nan"))
            n_leaky_col.append(None)
            leakiest_pair_col.append(None)
            leakiest_pair_count_col.append(None)
            leakiest_pair_outside_col.append(None)
            candidate_col.append(None)
            continue

        tokens = _parse_tokens(ms)
        if len(tokens) < 2:
            is_cohesive_col.append(True)
            min_frac_col.append(1.0)
            mean_frac_col.append(1.0)
            n_leaky_col.append(0)
            leakiest_pair_col.append(None)
            leakiest_pair_count_col.append(None)
            leakiest_pair_outside_col.append(None)
            candidate_col.append(None)
            continue

        full_inns: set = (
            inn_lookup.get(_canonical_str(tokens))
            or inn_lookup.get(ms.strip())
            or set()
        )

        # ── Score all C(n,2) pairs in one tight loop ──────────────────────────
        tight_edges: list[tuple[str, str]] = []
        leaky_count = 0
        min_frac = 1.0
        mean_acc = 0.0
        n_valid = 0
        worst_key = None
        worst_count = 0
        worst_frac = 1.0
        worst_outside = 0  # INNs in leakiest pair but NOT in this seed

        for tok_a, tok_b in combinations(tokens, 2):
            pair_key = _canonical_str([tok_a, tok_b])
            pair_inns = inn_lookup.get(pair_key, set())
            pair_count = len(pair_inns)
            if pair_count == 0:
                continue
            overlap = len(pair_inns & full_inns)
            frac = overlap / pair_count
            outside = pair_count - overlap

            mean_acc += frac
            n_valid += 1
            if frac < min_frac:
                min_frac = frac
            if frac < frac_threshold:
                leaky_count += 1
                if frac < worst_frac:
                    worst_frac = frac
                    worst_key = pair_key
                    worst_count = pair_count
                    worst_outside = outside
            else:
                tight_edges.append((tok_a, tok_b))

        mean_frac = mean_acc / n_valid if n_valid else float("nan")
        cohesive = leaky_count == 0

        is_cohesive_col.append(cohesive)
        min_frac_col.append(round(min_frac, 4))
        mean_frac_col.append(
            round(mean_frac, 4) if not np.isnan(mean_frac) else float("nan")
        )
        n_leaky_col.append(leaky_count)
        leakiest_pair_col.append(worst_key)
        leakiest_pair_count_col.append(worst_count if worst_key else None)
        leakiest_pair_outside_col.append(worst_outside if worst_key else None)

        if cohesive:
            candidate_col.append(None)
            continue

        # ── Maximal cliques of the tight-pair graph ───────────────────────────
        G = nx.Graph()
        G.add_nodes_from(tokens)
        G.add_edges_from(tight_edges)

        raw_cliques = list(nx.find_cliques(G)) if G.number_of_edges() else []
        cliques = [sorted(c) for c in raw_cliques if len(c) >= 2]

        candidates: list[dict] = []
        for clique_toks in cliques:
            clique_str = _canonical_str(clique_toks)
            clique_inns = inn_lookup.get(clique_str, set())
            clique_count = len(clique_inns) if clique_inns else None
            candidates.append(
                {
                    "mutation_string": clique_str,
                    "size": len(clique_toks),
                    "inn_count": clique_count,
                    "inn_freq": (
                        round(clique_count / inn_total, 5) if clique_count else None
                    ),
                    "in_scored_df": bool(clique_inns),
                }
            )
        candidates.sort(key=lambda d: (d["size"], d["inn_count"] or 0), reverse=True)
        candidate_col.append(candidates if candidates else None)

    motif_df["is_cohesive"] = is_cohesive_col
    motif_df["min_frac_pair_in_full"] = min_frac_col
    motif_df["mean_frac_pair_in_full"] = mean_frac_col
    motif_df["n_leaky_pairs"] = n_leaky_col
    motif_df["leakiest_pair"] = leakiest_pair_col
    motif_df["leakiest_pair_inn_count"] = leakiest_pair_count_col
    motif_df["leakiest_pair_inns_outside_seed"] = leakiest_pair_outside_col
    motif_df["candidate_sub_motifs"] = candidate_col
    return motif_df


def _refine_by_cohesion(
    motif_df: pd.DataFrame,
    inn_lookup: dict[str, set],
    inn_total: int,
    frac_threshold: float,
    max_iterations: Optional[int],
) -> tuple[set, list, int]:
    """Iteratively replace leaky motifs with their maximal tight-pair cliques.

    On each iteration every motif with ≥ 1 pair below *frac_threshold* is
    removed from the working set and replaced by the maximal cliques of its
    tight-pair graph.  Cliques are only admitted if their canonical string is
    already present in *inn_lookup* (i.e. they were enumerated by the scoring
    step and have a known INN set).

    Iteration stops when the working set is fully cohesive, when no progress
    is made (working set unchanged), or when *max_iterations* is reached.

    Parameters
    ----------
    motif_df : DataFrame
        Current surviving motifs — must have a ``"mutation_string"`` column.
    inn_lookup : dict
        Pre-built ``mutation_string → set[INN]`` from :func:`build_inn_lookup`.
    inn_total : int
        Total INN population (denominator).
    frac_threshold : float
        Pairs with ``frac_pair_in_full < frac_threshold`` are considered leaky.
    max_iterations : int or None
        Hard cap on iterations.  ``None`` runs until convergence.

    Returns
    -------
    (surviving_strings, refinement_log, n_iterations_run)
    """
    working: set[str] = set(motif_df["mutation_string"].tolist())
    log: list[dict] = []
    _max = max_iterations if max_iterations is not None else 100_000

    pbar = _tqdm(
        total=_max,
        desc="Cohesion refinement",
        unit="iter",
        dynamic_ncols=True,
    )
    iteration = 0
    try:
        for iteration in range(_max):
            leaky: dict[str, tuple[list, list]] = {}  # ms -> (leaky_edges, token_list)

            for ms in working:
                tokens = _parse_tokens(ms)
                if len(tokens) < 2:
                    continue
                full_inns = inn_lookup.get(_canonical_str(tokens)) or inn_lookup.get(
                    ms, set()
                )
                leaky_edges: list[tuple[str, str]] = []
                for tok_a, tok_b in combinations(tokens, 2):
                    pair_key = _canonical_str([tok_a, tok_b])
                    pair_inns = inn_lookup.get(pair_key, set())
                    if not pair_inns:
                        continue
                    frac = len(pair_inns & full_inns) / len(pair_inns)
                    if frac < frac_threshold:
                        leaky_edges.append((tok_a, tok_b))
                if leaky_edges:
                    leaky[ms] = (leaky_edges, tokens)

            n_leaky = len(leaky)
            pbar.set_postfix({"leaky": n_leaky, "motifs": len(working)})
            pbar.update(1)

            if not leaky:
                log.append(
                    {
                        "iteration": iteration,
                        "n_leaky": 0,
                        "split": [],
                        "converged": True,
                    }
                )
                break

            new_working = working - set(leaky)
            split_details: list[dict] = []

            for ms, (leaky_edges, tokens) in leaky.items():
                # ── Build connected components of leaky-pair graph ────────────
                # Each component = a group of tokens that co-occur OUTSIDE the
                # full motif → they form an independent sub-pattern.
                # Tokens not touched by any leaky edge = the "tight core".
                G_leaky = nx.Graph()
                G_leaky.add_nodes_from(tokens)
                G_leaky.add_edges_from(leaky_edges)

                # Connected components of size ≥ 2 are the leaky sub-groups;
                # isolated nodes are in the tight core.
                components: list[list[str]] = [
                    sorted(c) for c in nx.connected_components(G_leaky)
                ]
                tight_core: list[str] = sorted(
                    t for t in tokens if G_leaky.degree(t) == 0
                )

                # Candidate replacement strings: each component + tight core
                candidates: list[list[str]] = [c for c in components if len(c) >= 2]
                if len(tight_core) >= 2:
                    candidates.append(tight_core)
                elif len(tight_core) == 1:
                    # Single isolated token: keep as orphan (added back alone)
                    candidates.append(tight_core)

                # Only replace if ALL candidates exist in inn_lookup —
                # otherwise keep the original motif (better than losing it).
                candidate_strs = [_canonical_str(c) for c in candidates]
                all_known = all(
                    cs in inn_lookup
                    for cs in candidate_strs
                    if len(cs.split(" / ")) >= 2
                )

                added: list[str] = []
                if all_known and candidate_strs:
                    for cs in candidate_strs:
                        if cs in inn_lookup:
                            new_working.add(cs)
                            added.append(cs)
                else:
                    # Can't fully decompose — keep the original motif as-is
                    new_working.add(ms)

                split_details.append(
                    {
                        "source": ms,
                        "replaced_by": added,
                        "kept_original": not bool(added),
                    }
                )

            log.append(
                {
                    "iteration": iteration,
                    "n_leaky": n_leaky,
                    "split": split_details,
                    "converged": False,
                }
            )

            if new_working == working:
                break  # no progress — stable
            working = new_working

    finally:
        pbar.close()

    return working, log, iteration + 1


def _build_refined_motif_df(
    surviving_strings: set,
    scored_df: pd.DataFrame,
    group_by: Optional[list],
    min_lift: float,
    max_pvalue: float,
    repass: Optional[dict],
    inn_lookup: dict[str, set],
    inn_total: int,
) -> pd.DataFrame:
    """Rebuild motif_df from a refined surviving string set.

    Pulls base rows from *scored_df* (using the original pre-subtraction INN
    sets), recomputes ``raw_sub_motifs`` within the refined set, re-runs INN
    subtraction, re-runs Möbius statistics, and adds cohesion columns.

    Any string in *surviving_strings* absent from *scored_df* is silently
    skipped (it has no enumerated base row to pull stats from).
    """
    mask = scored_df["mutation_string"].isin(surviving_strings)
    refined_base = scored_df[mask].copy().reset_index(drop=True)

    if refined_base.empty:
        return pd.DataFrame()

    # Use original INN sets (scored_df["inns"] is pre-subtraction)
    refined_base["inn_count"] = refined_base["inns"].apply(len)
    refined_base["motif_id"] = refined_base.index

    # Recompute ancestry within the refined set only
    refined_base = _add_raw_sub_motifs(refined_base)

    # Re-run INN subtraction on the refined set
    refined_base = _subtract_inns_inline(refined_base, ancestry_col="raw_sub_motifs")

    refined_surviving = (
        refined_base[refined_base["adjusted_count"] > 0].copy().reset_index(drop=True)
    )
    refined_surviving["motif_id"] = refined_surviving.index

    if refined_surviving.empty:
        return pd.DataFrame()

    # Recompute Möbius stats on the refined surviving set
    refined_df = _recompute_mobius_stats(
        refined_surviving,
        group_by=group_by,
        min_lift=min_lift,
        max_pvalue=max_pvalue,
        repass=repass,
    )
    refined_df = refined_df.sort_values(
        ["num_mutations", "inn_count"], ascending=[False, False]
    ).reset_index(drop=True)
    refined_df["motif_id"] = refined_df.index

    # Final cohesion annotation on the refined set
    refined_df = _add_cohesion_columns(refined_df, inn_lookup, inn_total)

    return refined_df


def _recompute_stats_post_bispecific(
    motif_df: pd.DataFrame,
    group_by: Optional[list],
    min_lift: float,
    max_pvalue: float,
    repass: Optional[dict],
) -> pd.DataFrame:
    """Recompute all inn_count-derived statistics after bispecific pruning.

    Bispecific pruning changes ``inn_count`` and ``adjusted_count`` on branch
    and anchor rows.  Every statistic derived from those counts must be
    refreshed.  Token-level marginal frequencies (``marginal_freq``,
    ``marginal_count``, ``expected_freq_indep``) are unchanged
    because bispecific pruning reassigns INN *attribution*, not INN *biology* —
    an INN that physically carries token X still carries it.

    Per-row ``inn_total`` from the DataFrame is used for all frequency
    calculations so that multi-group DataFrames (stratified by gene × gene_species)
    are handled correctly — each group has its own ``inn_total``.

    Updates in order:

    1. ``freq_motif``, ``adjusted_freq`` — from new counts / per-row inn_total.
    2. Independence stats — ``lift_vs_indep``,
       ``pvalue_vs_indep`` and their sig/log variants — recomputed
       from the new ``inn_count`` against the stored ``expected_freq``.
    3. Möbius sub-motif stats — ``lift_vs_submotifs``,
       ``pvalue_vs_submotifs``, ``sub_motifs``, ``novel_mutations``, and
       their sig/log variants — via :func:`_recompute_mobius_stats`.
    4. ``motif_score``, ``motif_score_artifact_adjusted`` — derived from the refreshed
       Möbius stats and ``inn_count``.
    5. ``longest_motif_in_group`` — recomputed from updated ``adjusted_count``.
    """
    df = motif_df.copy()

    # Per-row inn_total (each stratification group has its own value)
    row_totals = df["inn_total"].fillna(1).astype(int).clip(lower=1)

    # 1. Frequency columns from updated counts / per-row inn_total
    df["freq_motif"] = (df["inn_count"] / row_totals).round(5)
    if "adjusted_count" in df.columns:
        df["adjusted_freq"] = (df["adjusted_count"] / row_totals).round(5)

    # 2. Independence stats — recompute lift + pvalue row-by-row using stored
    #    expected_freq (product of per-token marginals, biologically unchanged).
    if "expected_freq_indep" in df.columns:
        new_lifts, new_pvals = [], []
        for (_, row), n_total in zip(df.iterrows(), row_totals):
            expected = row["expected_freq_indep"]
            count = int(row["inn_count"])
            freq = row["freq_motif"]
            if expected > 0 and count <= n_total:
                lift = round(freq / expected, 3)
                pval = round(
                    float(
                        _binom(
                            count,
                            n_total,
                            min(expected, 1.0),
                            alternative="greater",
                        ).pvalue
                    ),
                    10,
                )
            else:
                lift = round(freq / expected, 3) if expected > 0 else 0.0
                pval = 1.0
            new_lifts.append(lift)
            new_pvals.append(pval)

        df["lift_vs_indep"] = new_lifts
        df["pvalue_vs_indep"] = new_pvals

        _lift_pass = df["motif"].apply(lambda m: len(m) == 1) | (
            df["lift_vs_indep"] >= min_lift
        )
        _pval_pass = df["pvalue_vs_indep"] <= max_pvalue
        df["lift_passes_vs_indep"] = _lift_pass
        df["pvalue_passes_vs_indep"] = _pval_pass
        df["passes_indep_filter"] = _lift_pass & _pval_pass
        df["log_lift_vs_indep"] = np.log(df["lift_vs_indep"].clip(lower=1e-10)).round(4)
        df["neg_log10_pvalue_vs_indep"] = (
            -np.log10(df["pvalue_vs_indep"].clip(lower=1e-300))
        ).round(4)

    # 3. Möbius sub-motif stats — depends on updated freq_motif
    df = _recompute_mobius_stats(
        df,
        group_by=group_by,
        min_lift=min_lift,
        max_pvalue=max_pvalue,
        repass=repass,
    )
    if "lift_vs_submotifs" in df.columns:
        df["log_lift_vs_submotifs"] = np.log(
            df["lift_vs_submotifs"].clip(lower=1e-10)
        ).round(4)
    if "pvalue_vs_submotifs" in df.columns:
        df["neg_log10_pvalue_vs_submotifs"] = (
            -np.log10(df["pvalue_vs_submotifs"].clip(lower=1e-300))
        ).round(4)

    # 4. Composite motif score
    if (
        "log_lift_vs_submotifs" in df.columns
        and "neg_log10_pvalue_vs_submotifs" in df.columns
    ):
        df["motif_score"] = (
            df["log_lift_vs_submotifs"].clip(lower=0)
            * df["neg_log10_pvalue_vs_submotifs"].clip(lower=0)
            * np.log1p(df["inn_count"])
        ).round(4)
        df["motif_score_artifact_adjusted"] = (
            df["motif_score"] * (1.0 - df["artifact_fraction"].fillna(0.0))
        ).round(4)

    # 5. longest_motif_in_group from updated adjusted_count
    if "adjusted_count" in df.columns and "motif" in df.columns:
        _motifs_with_inns = set(df.loc[df["adjusted_count"] > 0, "motif"])
        df["longest_motif_in_group"] = df["motif"].apply(
            lambda m: not any(m < other for other in _motifs_with_inns)
        )

    return df


def build_motif_index(
    collapsed: pd.DataFrame,
    min_count: int = 2,
    max_pvalue: float = 0.05,
    min_lift: float = 1.5,
    min_size: int = 2,
    max_size: int = 5,
    marginals_table: Optional[pd.DataFrame] = None,
    group_by: Optional[list] = None,
    repass: Optional[dict] = None,
    cohesion_frac_threshold: Optional[float] = None,
    max_cohesion_iterations: Optional[int] = None,
    prune_bispecific_branches: bool = True,
) -> "MotifIndexResult":
    """Build a significance-filtered sub-motif index from a collapsed mutation catalog.

    Returns a :class:`MotifIndexResult` whose ``.motif_df`` attribute is the
    primary analysis-ready output.  Intermediate pipeline steps are exposed as
    ``.scored_df``, ``.sub_motif_df``, ``.subtracted_df``, and ``.pruned_df``.

    Parameters
    ----------
    collapsed:
        Output of :func:`collapse_motifs_chains`.  Must have ``motif``,
        ``inns``, and ``inn_total`` columns.
    min_count:
        Minimum number of distinct INNs carrying a combination.
    max_pvalue:
        One-sided binomial p-value threshold.
    min_lift:
        Minimum lift threshold (observed / expected under independence).
    min_size:
        Minimum combination size to enumerate (default 2).
    max_size:
        Maximum combination size to enumerate (default 5).
    marginals_table:
        Optional per-token artifact-fraction lookup table.
    group_by:
        Optional list of column names to split on (must match those used in
        :func:`collapse_motifs_chains`).  Each group is processed
        independently and results are concatenated with globally unique
        ``motif_id`` values.
    repass:
        Optional filter restricting which sub-motifs enter the Möbius null
        (e.g. ``{'lift': ('min', 1.0), 'pvalue': ('max', 0.05)}``).
    cohesion_frac_threshold:
        When set, enables the cohesion refinement loop (steps 8–9).  Any
        motif with at least one pair whose ``frac_pair_in_full`` is below
        this threshold is replaced by the maximal cliques of its tight-pair
        graph.  The loop repeats until convergence.  ``None`` skips
        refinement (default).
    max_cohesion_iterations:
        Hard cap on refinement iterations.  ``None`` runs until convergence
        (default).  Ignored when *cohesion_frac_threshold* is ``None``.
    """
    _step_changes: Dict[str, Dict[str, pd.DataFrame]] = {}

    # ── Step 0: Count initial unique substrings ────────────────────────────────
    _initial_unique_count = len(collapsed) if isinstance(collapsed, pd.DataFrame) else 0
    _initial_unique_inns = MotifIndexResult._count_unique_inns(collapsed)
    _step_changes["initial_unique_substrings"] = {
        "description": f"Found {_initial_unique_count} unique mutation strings",
        "count": _initial_unique_count,
        "unique_inns": _initial_unique_inns,
    }

    _fn = lambda subset, **kw: _build_motif_table_single_group(  # noqa: E731
        subset,
        min_count=min_count,
        min_lift=min_lift,
        max_pvalue=max_pvalue,
        min_size=min_size,
        max_size=max_size,
        marginals_table=marginals_table,
        repass=repass,
    )
    if group_by:
        scored_df = _apply_group_by_split(collapsed, group_by, _fn)
    else:
        scored_df = _fn(collapsed)

    # ── Step 1: Count generated sub-motifs after enumeration ──────────────────────
    _generated_motif_count = (
        len(scored_df) if isinstance(scored_df, pd.DataFrame) else 0
    )
    _generated_motif_inns = MotifIndexResult._count_unique_inns(scored_df)
    _step_changes["generated_submotifs"] = {
        "description": f"Generated {_generated_motif_count} sub-motif combinations",
        "count": _generated_motif_count,
        "unique_inns": _generated_motif_inns,
    }

    _filters = {
        "min_count": min_count,
        "max_pvalue": max_pvalue,
        "min_lift": min_lift,
        "min_size": min_size,
        "max_size": max_size,
        "repass": repass,
    }

    if scored_df.empty:
        _empty = pd.DataFrame()
        return MotifIndexResult(
            scored_df=_empty,
            removed_df=_empty,
            motif_df_pre_recompute=_empty,
            motif_df=_empty,
            filters=_filters,
            group_by=group_by,
            step_changes={},
        )

    # ── Composite-key helpers ─────────────────────────────────────────────────
    _gcols = [c for c in (group_by or []) if c in collapsed.columns]

    def _key(row):
        base = row["mutation_string"]
        return (base,) + tuple(row[c] for c in _gcols) if _gcols else base

    _collapsed_lookup: dict = {
        _key(row): row["inns"] for _, row in collapsed.iterrows()
    }

    # ── sequences_representing_full_motif ─────────────────────────────────────
    scored_df["sequences_representing_full_motif"] = scored_df.apply(
        lambda row: _collapsed_lookup.get(_key(row)) or None, axis=1
    )

    # ── Append full profiles from collapsed absent from scored_df ─────────────
    _existing_keys = set(scored_df.apply(_key, axis=1))
    _missing = collapsed[~collapsed.apply(lambda r: _key(r) in _existing_keys, axis=1)]
    if not _missing.empty:
        extra_records = []
        for _, row in _missing.iterrows():
            record = {
                "motif": row["motif"],
                "mutation_string": row["mutation_string"],
                "mutation_list": row["mutation_list"],
                "num_mutations": int(row["num_mutations"]),
                "inn_count": int(row["inn_count"]),
                "inn_total": int(row["inn_total"]),
                "inns": row["inns"],
                "freq_motif": row["freq_motif"],
                "marginal_freq": row["marginal_freq"],
                "marginal_count": row["marginal_count"],
                "expected_freq_indep": row["expected_freq_indep"],
                "lift_vs_indep": row["lift_vs_indep"],
                "log_lift_vs_indep": row.get("log_lift_vs_indep", np.nan),
                "pvalue_vs_indep": row["pvalue_vs_indep"],
                "neg_log10_pvalue_vs_indep": row.get(
                    "neg_log10_pvalue_vs_indep", np.nan
                ),
                "artifact_fraction": row.get("artifact_fraction", np.nan),
                "passes_filters": False,
                "no_larger_passing_motif_in_group": True,
                "is_fully_described_by_larger_motif": False,
                "sequences_representing_full_motif": row["inns"],
            }
            for c in _gcols:
                record[c] = row[c]
            extra_records.append(record)

        scored_df = pd.concat(
            [scored_df, pd.DataFrame(extra_records)], ignore_index=True
        )
        _added_cols = [
            c
            for c in (
                "mutation_string",
                "gene",
                "gene_species",
                "inn_count",
                "inns",
            )
            if c in scored_df.columns
        ]
        _added_df = scored_df[
            scored_df["mutation_string"].isin(set(_missing["mutation_string"]))
        ][_added_cols].copy()
        _added_df["reason"] = "below_filter_thresholds"
        _step_changes["initial_filter"] = {
            "added": pd.DataFrame(columns=[*_added_cols, "reason"]),
            "removed": _added_df.reset_index(drop=True),
        }
    scored_df["motif_id"] = scored_df.index

    # ── Steps 2–3: bispecific pruning → raw_sub_motifs → INN subtraction ────────
    # Bispecific pruning runs first (before subtraction) so that shared INNs
    # are correctly attributed to the stem before the subtraction step assigns
    # them to branch motifs.  scored_df already has sub_motifs from the Möbius
    # pass so anchor-grouped detection works correctly here.
    if prune_bispecific_branches and not scored_df.empty:
        _before_prune = scored_df.copy()
        scored_df = resolve_bispecific_branches_dep(scored_df, group_by=group_by)
        scored_df = scored_df[scored_df["inn_count"] > 0].copy().reset_index(drop=True)
        _removed_prune = _before_prune[
            ~_before_prune["mutation_string"].isin(set(scored_df["mutation_string"]))
        ].copy()
        _prune_cols = [
            c
            for c in (
                "mutation_string",
                "gene",
                "gene_species",
                "inn_count",
                "inns",
            )
            if c in _removed_prune.columns
        ]
        _removed_prune = (
            _removed_prune[_prune_cols]
            if len(_removed_prune)
            else pd.DataFrame(columns=_prune_cols)
        )
        if len(_removed_prune):
            _removed_prune["reason"] = "bispecific_branch_pruned_or_zero_inn_count"
        _step_changes["prune_bispecific_branches"] = {
            "added": pd.DataFrame(columns=[*_prune_cols, "reason"]),
            "removed": _removed_prune.reset_index(drop=True),
        }
    scored_df = _add_raw_sub_motifs(scored_df)
    scored_df = _subtract_inns_inline(scored_df, ancestry_col="raw_sub_motifs")
    # scored_df now has all columns including adjusted_inns/count/freq and
    # is_fully_described_by_larger_motif (= adjusted_count == 0)

    # ── Step 5: INN subtraction complete — separate by unique adjusted INNs ──────
    removed_df = (
        scored_df[scored_df["adjusted_count"] == 0].copy().reset_index(drop=True)
    )
    motif_df_pre_recompute = scored_df[scored_df["adjusted_count"] > 0].copy()
    motif_df_pre_recompute = motif_df_pre_recompute.reset_index(drop=True)
    motif_df_pre_recompute["motif_id"] = motif_df_pre_recompute.index

    # Track motifs with zero adjusted count (consumed by larger motifs)
    _removed_cols = [
        c
        for c in (
            "mutation_string",
            "gene",
            "gene_species",
            "inn_count",
            "adjusted_count",
            "inns",
        )
        if c in removed_df.columns
    ]
    _removed_df_tracked = (
        removed_df[_removed_cols].copy()
        if len(removed_df)
        else pd.DataFrame(columns=_removed_cols)
    )
    if len(_removed_df_tracked):
        _removed_df_tracked["reason"] = "adjusted_count_zero_after_subtraction"
    _step_changes["inn_subtraction"] = {
        "added": pd.DataFrame(columns=[*_removed_cols, "reason"]),
        "removed": _removed_df_tracked.reset_index(drop=True),
    }

    # ── Step 6: recompute Möbius stats on pruned table ────────────────────────
    motif_df = _recompute_mobius_stats(
        motif_df_pre_recompute,
        group_by=group_by,
        min_lift=min_lift,
        max_pvalue=max_pvalue,
        repass=repass,
    )
    motif_df = motif_df.sort_values(
        ["num_mutations", "inn_count"], ascending=[False, False]
    ).reset_index(drop=True)
    motif_df["motif_id"] = motif_df.index
    motif_df = _annotate_is_seed(motif_df)

    # ── Step 7: build inn_lookup once; annotate cohesion on initial motif_df ──
    _inn_lookup = build_inn_lookup(scored_df)
    _inn_total_val = (
        int(scored_df["inn_total"].dropna().iloc[0])
        if "inn_total" in scored_df.columns
        else 1
    )
    motif_df = _add_cohesion_columns(motif_df, _inn_lookup, _inn_total_val)

    # ── Steps 8–9: cohesion refinement loop + final full recompute ────────────
    _motif_df_pre_refinement: Optional[pd.DataFrame] = None
    _refinement_log: Optional[list] = None
    _cohesion_iters: int = 0

    if cohesion_frac_threshold is not None:
        _motif_df_pre_refinement = motif_df.copy()

        # Step 8: iteratively replace leaky motifs with tight-clique sub-motifs
        surviving_strings, _refinement_log, _cohesion_iters = _refine_by_cohesion(
            motif_df=motif_df,
            inn_lookup=_inn_lookup,
            inn_total=_inn_total_val,
            frac_threshold=cohesion_frac_threshold,
            max_iterations=max_cohesion_iterations,
        )

        # Step 9: final full recompute — INN subtraction + Möbius + cohesion
        # on the refined surviving set, using original (pre-subtraction) INNs
        # from scored_df as the clean baseline.
        motif_df = _build_refined_motif_df(
            surviving_strings=surviving_strings,
            scored_df=scored_df,
            group_by=group_by,
            min_lift=min_lift,
            max_pvalue=max_pvalue,
            repass=repass,
            inn_lookup=_inn_lookup,
            inn_total=_inn_total_val,
        )
        motif_df = _annotate_is_seed(motif_df)

    # ── Step 10: final adjusted_count recompute on the surviving set only ────
    # adjusted_count was computed against all motifs (including those with
    # adjusted_count=0 that were filtered out in step 5).  Recomputing here
    # against only the surviving motifs ensures:
    #   - longest_motif_in_group == True  →  adjusted_count == inn_count
    #   - adjusted_count reflects only subtractions by motifs that actually survive
    # Reset inns from scored_df (pre-subtraction baseline) so we start clean.
    if not motif_df.empty and "mutation_string" in motif_df.columns:
        if "inns" in scored_df.columns:
            # Key by (mutation_string, gene, gene_species) to avoid overwriting
            # duplicate mutation strings that exist in multiple gene groups.
            _grp_key_cols = [
                c
                for c in ("mutation_string", "gene", "gene_species")
                if c in scored_df.columns
            ]
            _orig_inns_map = {
                tuple(str(r[c]) for c in _grp_key_cols): r["inns"]
                for _, r in scored_df[_grp_key_cols + ["inns"]].iterrows()
            }

            def _restore_inns(row):
                key = tuple(
                    str(row[c]) if c in row.index else "" for c in _grp_key_cols
                )
                return _orig_inns_map.get(key, row.get("inns", set()))

            motif_df = motif_df.copy()
            motif_df["inns"] = motif_df.apply(_restore_inns, axis=1)
            motif_df["inn_count"] = motif_df["inns"].apply(
                lambda s: len(s) if isinstance(s, (set, frozenset)) else 0
            )

        # Re-run subtraction relative to surviving motifs only
        motif_df = _subtract_inns_inline(motif_df, ancestry_col="raw_sub_motifs")

        # Recompute longest_motif_in_group relative to surviving set
        if "motif" in motif_df.columns:
            _surviving_motifs = set(motif_df["motif"])
            motif_df["longest_motif_in_group"] = motif_df["motif"].apply(
                lambda m: not any(m < other for other in _surviving_motifs)
            )
        motif_df = motif_df.sort_values(
            ["num_mutations", "inn_count"], ascending=[False, False]
        ).reset_index(drop=True)
        motif_df["motif_id"] = motif_df.index
        motif_df = _annotate_is_seed(motif_df)

    # ── Add mutation_string_pe ─────────────────────────────────────────────────
    try:
        from .position_labeling import format_motif_label as _fmt_motif
    except ImportError:
        from inn_analysis_ops.analytics._helpers import (
            format_motif_label as _fmt_motif,
        )

    def _safe_fmt_pe(x: object) -> object:
        if not isinstance(x, str) or not x:
            return x
        try:
            return _fmt_motif(x).replace(" ", "")
        except Exception:
            return x

    _pe_targets = [scored_df, removed_df, motif_df_pre_recompute, motif_df]
    if _motif_df_pre_refinement is not None:
        _pe_targets.append(_motif_df_pre_refinement)
    for _df in _pe_targets:
        if "mutation_string" in _df.columns and "mutation_string_pe" not in _df.columns:
            _df["mutation_string_pe"] = _df["mutation_string"].map(_safe_fmt_pe)
        if "is_seed" not in _df.columns:
            _df["is_seed"] = _df.get("sub_motifs", pd.Series([[]] * len(_df))).apply(
                _compute_is_seed
            )

    return MotifIndexResult(
        scored_df=scored_df,
        removed_df=removed_df,
        motif_df_pre_recompute=motif_df_pre_recompute,
        motif_df=motif_df,
        filters=_filters,
        group_by=group_by,
        inn_lookup=_inn_lookup,
        cohesion_frac_threshold=cohesion_frac_threshold,
        cohesion_iterations=_cohesion_iters,
        refinement_log=_refinement_log,
        motif_df_pre_refinement=_motif_df_pre_refinement,
        step_changes=_step_changes,
    )


# ---------------------------------------------------------------------------
# Debug: pairwise INN overlap for a given mutation string
# ---------------------------------------------------------------------------


__all__ = [
    "MotifIndexResult",
    "collapse_motifs_chains",
    "build_motif_index",
    "build_inn_lookup",
    "_apply_group_by_split",
    "_build_motif_table_single_group",
    "_run_mobius_pass",
    "_mobius_null_freq",
    "_fmt_tokens",
    "_REPASS_ALIASES",
    "_dedup_by_constant_chain_mutations",
    "_add_cohesion_columns",
    "_refine_by_cohesion",
    "_build_refined_motif_df",
    "_recompute_stats_post_bispecific",
    "_collapse_single_group",
    "_compute_is_seed",
    "_annotate_is_seed",
    "_recompute_mobius_stats",
]
