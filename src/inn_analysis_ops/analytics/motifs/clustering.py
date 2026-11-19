"""Motif clustering: seed identification, INN attribution, statistics, and cluster info.

Functions
---------

step_compute_inn_attribution
    Compute adjusted_inns by attributing each INN to the most specific motif.
step_add_statistics
    Möbius conditional stats, independence baseline, and pairwise co-mutation metrics.
step_filter_by_statistics
    Drop motifs that fail a caller-supplied dict of ``{column: (operator, threshold)}`` filters.
step_build_cluster_info
    Assign cluster_seeds, seed_group_reps, primary_cluster_seeds, primary_cluster_rep.


All functions operate independently within strata defined by ``group_cols`` on the
``ChainMotifStep`` input model.  They have no knowledge of pipeline orchestration —
that belongs in :mod:`analytics.motifs.pipelines.chain`.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from copy import deepcopy
from typing import TYPE_CHECKING, Optional

import numpy as np
import pandas as pd
from scipy.stats import binomtest as _binom
from tqdm.auto import tqdm as _tqdm

from inn_analysis_ops.analytics.motif_schema import MotifStatsCols as C
from inn_analysis_ops.analytics.motifs.models import (
    ChainMotifStep,
    ChainMotifStepResult,
    _ComputeInnAttributionIn,
)
from inn_analysis_ops.analytics.motifs.stat_functions import (
    ChildLiftInput,
    CompositeInput,
    MarginalInput,
    MobiusInput,
    PairwiseInput,
    StratumContext,
    compute_child_lift_ratio,
    compute_composite_score,
    compute_marginals,
    compute_mobius_baseline,
    compute_pairwise,
)

_log = logging.getLogger(__name__)


def step_add_statistics(
    step_in: ChainMotifStep,
    group_counts_map: dict[tuple, int],
    marginals_table: Optional[pd.DataFrame] = None,
    min_lift_pass_condition: float = 1.0,
    max_pvalue_pass_condition: float = 0.05,
) -> ChainMotifStepResult:
    """Compute all per-motif statistics within each stratum.

    Delegates to the typed functions in :mod:`analytics.motifs.stat_functions`:

    1. :func:`compute_marginals` — ``inn_total``, ``freq_motif``, marginals,
       independence null and threshold booleans.
    2. :func:`compute_mobius_baseline` — Möbius null, ``lift_vs_submotifs``,
       ``pvalue_vs_submotifs``, threshold booleans, ``explained_by_submotifs``.
    3. :func:`compute_pairwise` — ``pairwise_cohesion``, ``pairwise_lift``.
    4. :func:`compute_child_lift_ratio` — ``n_children``,
       ``lift_ratio_vs_child_min/max``.
    5. :func:`compute_composite_score` — ``motif_score``, ``artifact_fraction``,
       ``motif_score_artifact_adjusted``.
    """
    df = step_in.motif_df.copy().reset_index(drop=True)
    group_cols = [c for c in step_in.group_cols if c in df.columns]

    support_inns_col = "inns"
    support_count_col = "inn_count"

    if not 0 < max_pvalue_pass_condition <= 1.0:
        raise ValueError(
            "max_pvalue_pass_condition must be in the range (0, 1], "
            f"got {max_pvalue_pass_condition}"
        )

    marginals_lookup: Optional[dict[str, float]] = None
    if marginals_table is not None:
        marginals_lookup = {
            str(mr.get("mutation_string_pe", "")): float(
                mr.get("artifact_fraction", 0.0)
            )
            for _, mr in marginals_table.reset_index().iterrows()
        }

    def _compute_stratum_stats(
        sdf: pd.DataFrame, total_group_counts: int
    ) -> pd.DataFrame:
        sdf = sdf.copy().reset_index(drop=True)

        # all_stratum_inns = []
        # _unique_checks = 10000
        # for idx, i in enumerate(sdf["inns"].apply(lambda x: list(x))):
        #     all_stratum_inns.extend(i)
        #     if idx % _unique_checks == 0:
        #         all_stratum_inns = sorted(set(all_stratum_inns))
        # all_stratum_inns = sorted(set(all_stratum_inns))
        # # ── Build stratum context ─────────────────────────────────────────────
        # all_stratum_inns: set = set().union(
        #     *[
        #         s
        #         for s in sdf[support_inns_col]
        #         if isinstance(s, (set, frozenset))
        #     ]
        # )
        # N_total = max(len(all_stratum_inns), 1)
        # assert total_group_counts == N_total, (
        #     total_group_counts,
        #     N_total,
        #     sdf[["gene", "gene_species"]].value_counts(),
        # )
        N_total = max(total_group_counts, 1)

        token_to_inns: dict[str, set] = defaultdict(set)
        for _, row in sdf.iterrows():
            inns = row[support_inns_col]
            if not isinstance(inns, (set, frozenset)):
                continue
            for tok in (row["mutation_string_pe"] or "").split("/"):
                tok = tok.strip()
                if tok:
                    token_to_inns[tok] |= set(inns)

        token_marginal_freq = {
            tok: len(v) / N_total for tok, v in token_to_inns.items()
        }
        pe_to_tokens: dict[str, frozenset] = {
            pe: frozenset(t.strip() for t in pe.split("/") if t.strip())
            for pe in sdf["mutation_string_pe"]
        }

        ctx = StratumContext(
            N_total=N_total,
            token_to_inns=dict(token_to_inns),
            token_marginal_freq=token_marginal_freq,
            pe_to_freq={},
            pe_to_tokens=pe_to_tokens,
        )

        # ── Pass 1: marginals + independence baseline ─────────────────────────
        sdf[C.inn_total] = N_total
        marginal_results = [
            compute_marginals(
                MarginalInput(
                    mutation_string_pe=row["mutation_string_pe"],
                    inn_count=int(row[support_count_col]),
                    inns=row[support_inns_col],
                ),
                ctx,
                min_lift_pass_condition,
                max_pvalue_pass_condition,
            )
            for _, row in sdf.iterrows()
        ]
        for col in [
            C.freq_motif,
            C.marginal_freq,
            C.marginal_count,
            C.expected_freq_indep,
            C.lift_vs_indep,
            C.pvalue_vs_indep,
            C.log_lift_vs_indep,
            C.neg_log10_pvalue_vs_indep,
            C.lift_passes_vs_indep,
            C.pvalue_passes_vs_indep,
            C.passes_indep_filter,
        ]:
            sdf[col] = [r.values[col] for r in marginal_results]

        ctx.pe_to_freq = dict(
            zip(sdf["mutation_string_pe"], sdf[C.freq_motif])
        )

        # ── Pass 2: Möbius baseline ───────────────────────────────────────────
        mobius_results = [
            compute_mobius_baseline(
                MobiusInput(
                    mutation_string_pe=row["mutation_string_pe"],
                    inn_count=int(row[support_count_col]),
                    freq_motif=float(row[C.freq_motif]),
                    passes_indep_filter=bool(row[C.passes_indep_filter]),
                ),
                ctx,
                min_lift_pass_condition,
                max_pvalue_pass_condition,
            )
            for _, row in sdf.iterrows()
        ]
        for col in [
            C.expected_freq_mobius,
            C.lift_vs_submotifs,
            C.pvalue_vs_submotifs,
            C.log_lift_vs_submotifs,
            C.neg_log10_pvalue_vs_submotifs,
            C.lift_passes_vs_submotifs,
            C.pvalue_passes_vs_submotifs,
            C.passes_submotif_filter,
            C.explained_by_submotifs,
        ]:
            sdf[col] = [r.values[col] for r in mobius_results]

        # ── Pass 3: pairwise co-occurrence ────────────────────────────────────
        pairwise_results = [
            compute_pairwise(
                PairwiseInput(
                    mutation_string_pe=row["mutation_string_pe"]
                ),
                ctx,
            )
            for _, row in sdf.iterrows()
        ]
        sdf[C.pairwise_cohesion] = [
            r.values[C.pairwise_cohesion] for r in pairwise_results
        ]
        sdf[C.pairwise_lift] = [
            r.values[C.pairwise_lift] for r in pairwise_results
        ]

        # ── Pass 4: child lift ratio ──────────────────────────────────────────
        pe_to_lift = dict(
            zip(sdf["mutation_string_pe"], sdf[C.lift_vs_submotifs])
        )
        children_map: dict[str, list[float]] = {
            pe: [] for pe in sdf["mutation_string_pe"]
        }
        for _, row in sdf.iterrows():
            child_lift = pe_to_lift.get(
                row["mutation_string_pe"], float("nan")
            )
            for ancestor_pe in row.get("super_motifs") or []:
                if ancestor_pe in children_map:
                    children_map[ancestor_pe].append(child_lift)

        child_results = [
            compute_child_lift_ratio(
                ChildLiftInput(
                    mutation_string_pe=row["mutation_string_pe"],
                    lift_vs_submotifs=float(row[C.lift_vs_submotifs]),
                    super_motifs=list(row.get("super_motifs") or []),
                ),
                children_map,
            )
            for _, row in sdf.iterrows()
        ]
        sdf[C.n_children] = [r.values[C.n_children] for r in child_results]
        sdf[C.lift_ratio_vs_child_min] = [
            r.values[C.lift_ratio_vs_child_min] for r in child_results
        ]
        sdf[C.lift_ratio_vs_child_max] = [
            r.values[C.lift_ratio_vs_child_max] for r in child_results
        ]

        # ── Pass 5: composite score ───────────────────────────────────────────
        composite_results = [
            compute_composite_score(
                CompositeInput(
                    mutation_string_pe=row["mutation_string_pe"],
                    log_lift_vs_submotifs=float(
                        row[C.log_lift_vs_submotifs]
                    ),
                    neg_log10_pvalue_vs_submotifs=float(
                        row[C.neg_log10_pvalue_vs_submotifs]
                    ),
                    inn_count=int(row[support_count_col]),
                ),
                inn_count=int(row[support_count_col]),
                marginals_lookup=marginals_lookup,
            )
            for _, row in sdf.iterrows()
        ]
        sdf[C.motif_score] = [
            r.values[C.motif_score] for r in composite_results
        ]
        sdf[C.artifact_fraction] = [
            r.values[C.artifact_fraction] for r in composite_results
        ]
        sdf[C.motif_score_artifact_adjusted] = [
            r.values[C.motif_score_artifact_adjusted]
            for r in composite_results
        ]

        return sdf

    init_rows = df.shape[0]

    if not group_cols:
        total_group_counts = sum(group_counts_map.values())
        df = _compute_stratum_stats(df, total_group_counts)
    else:
        parts: list[pd.DataFrame] = []
        for grp_name, sdf in df.groupby(
            group_cols, sort=False, dropna=False
        ):
            total_group_counts = group_counts_map[grp_name]
            parts.append(
                _compute_stratum_stats(
                    sdf, total_group_counts
                ).reset_index(drop=True)
            )
        df = pd.concat(parts, ignore_index=True)

    output_rows = df.shape[0]
    return ChainMotifStepResult(
        motif_df=df,
        step_name="add_statistics",
        rows_added=0,
        rows_dropped=init_rows - output_rows,
        group_cols=step_in.group_cols,
        detail=f"Statistics computed for {len(df)} motifs",
    )


def step_filter_by_statistics(
    step_in: ChainMotifStep,
    filters: dict[str, tuple[str, float]],
    allowed_motifs_by_select: list[str] | None = None,
) -> ChainMotifStepResult:
    """Drop motifs that fail a caller-supplied set of column filters.

    Must be called after :func:`step_add_statistics`.

    Parameters
    ----------
    step_in:
        Input step with statistics columns already present.
    filters:
        ``{column_name: (operator, threshold)}`` mapping.  Every key must
        exist as a column in the DataFrame — a missing column raises
        ``KeyError`` immediately (the DataFrame does not match expectations).

        Supported operators:

        - ``"min"``  — keep rows where ``col >= threshold``
        - ``"max"``  — keep rows where ``col <= threshold``
        - ``"eq"``   — keep rows where ``col == threshold``

        For nullable float columns (e.g. ``C.pairwise_cohesion``) rows with
        NaN are treated as **passing** the filter — they are not penalised for
        not having a value (e.g. single-token motifs have no pairwise score).

        Defaults to the standard independence + cohesion gates when ``None``:

        .. code-block:: python

            {
                C.lift_vs_indep:    ("min", 1.0),
                C.pvalue_vs_indep:  ("max", 0.05),
                C.pairwise_cohesion: ("min", 1.0),
            }

    allowed_motifs_by_select:
        Rows whose ``motif_selection_reason`` is in this list are kept
        unconditionally, bypassing all filters.
    """

    df = step_in.motif_df.copy().reset_index(drop=True)

    missing = set(filters) - set(df.columns)
    if missing:
        raise KeyError(
            f"step_filter_by_statistics: filter column(s) {sorted(missing)} "
            "not found in motif_df — call step_add_statistics first or "
            "remove the column from the filters dict."
        )

    _is_protected = (
        df["motif_selection_reason"].isin(allowed_motifs_by_select)
        if allowed_motifs_by_select is not None
        else pd.Series(False, index=df.index)
    )

    keep = pd.Series(True, index=df.index)
    for col, (op, threshold) in filters.items():
        series = df[col]
        if op == "min":
            gate = series.isna() | (series >= threshold)
        elif op == "max":
            gate = series.isna() | (series <= threshold)
        elif op == "eq":
            gate = series.isna() | (series == threshold)
        else:
            raise ValueError(
                f"step_filter_by_statistics: unknown operator {op!r} for column {col!r}. "
                "Use 'min', 'max', or 'eq'."
            )
        keep = keep & gate

    keep = keep | _is_protected

    n_dropped = int((~keep).sum())
    df = df[keep].reset_index(drop=True)

    filter_parts = [
        f"{col} {op} {threshold}"
        for col, (op, threshold) in filters.items()
    ]
    return ChainMotifStepResult(
        motif_df=df,
        step_name="filter_by_statistics",
        rows_added=0,
        rows_dropped=n_dropped,
        group_cols=step_in.group_cols,
        detail=(
            f"{n_dropped} motifs dropped ({', '.join(filter_parts)}); "
            f"{len(df)} remaining"
        ),
    )


def _find_seeds(
    sdf: pd.DataFrame,
) -> tuple[dict[str, list[str]], list[str]]:
    sdf = sdf.sort_values(by="mutation_string_pe").query("~remove_motif")
    seeds: dict[str, list[str]] = {}
    token_map: dict[str, set[str]] = sdf.set_index("mutation_string_pe")[
        "tokens"
    ].to_dict()

    # Build the seed dictionary from rows that meet the minimum size.
    for _, row in sdf.sort_values(by=["motif_length"]).iterrows():
        seed_names = list(seeds.keys())
        found = 0
        for current_seed in seed_names:
            token_seed = token_map[current_seed]
            if token_seed.issubset(row["tokens"]):
                seeds[current_seed].append(row["mutation_string_pe"])
                found += 1

        if found == 0:
            seeds[row["mutation_string_pe"]] = [row["mutation_string_pe"]]

    paired_seeds = []
    for seed1 in list(seeds.keys()):
        for seed2 in list(seeds.keys()):
            if seed1 == seed2:
                continue
            if token_map[seed1].issubset(token_map[seed2]) or token_map[
                seed2
            ].issubset(token_map[seed1]):
                paired_seeds.append((seed1, seed2))
            elif seed1 in seed2 or seed2 in seed1:
                paired_seeds.append((seed1, seed2))

    if paired_seeds:
        raise AssertionError(f"Paired seeds found: {paired_seeds}")
    return seeds


def _find_supersets(sdf: pd.DataFrame):
    super_sets = {}
    token_map = sdf.set_index("mutation_string_pe")["tokens"].to_dict()
    groups_by_size: dict[int, list[str]] = {}
    for _, row in sdf.iterrows():
        size = len(row["tokens"])
        if size not in groups_by_size:
            groups_by_size[size] = []
        groups_by_size[size].append(row["mutation_string_pe"])

    for _, row in sdf.iterrows():
        super_sets[row["mutation_string_pe"]] = []
        current_size = len(row["tokens"])
        for size, members in groups_by_size.items():
            if size <= current_size:
                continue
            for member in members:
                if token_map[row["mutation_string_pe"]].issubset(
                    token_map[member]
                ):
                    super_sets[row["mutation_string_pe"]].append(member)

    return super_sets


def _get_adjusted_inns(sdf, super_sets) -> dict[str, set[str]]:
    all_inn_sets = {}
    for _, row in sdf.iterrows():
        all_inn_sets[row["mutation_string_pe"]] = row["inns"]

    updated_inn_counts = {}
    for _, row in sdf.iterrows():
        current_inn_counts = deepcopy(
            all_inn_sets[row["mutation_string_pe"]]
        )
        for super_motif in super_sets[row["mutation_string_pe"]]:
            inns_super = all_inn_sets[super_motif]
            current_inn_counts -= inns_super
        updated_inn_counts[row["mutation_string_pe"]] = current_inn_counts

    return updated_inn_counts


def _re_query_inn_map(
    mutation_string_pe_token: set[str],
    database_tokens: dict[tuple[str, str], set[str]],
) -> dict[str, list[str]]:
    result_inn_map = {}
    for (key, chain), tokens in database_tokens.items():
        if mutation_string_pe_token.issubset(tokens):
            if key not in result_inn_map:
                result_inn_map[key] = []
            result_inn_map[key].append(chain)
    return result_inn_map


def num_unique_super_motifs(row, full_df) -> int:
    super_motifs = (
        row.get("super_motifs")
        if hasattr(row, "get")
        else row["super_motifs"]
    )
    if not super_motifs:
        return 0
    subset_df = full_df[
        full_df["mutation_string_pe"].isin(super_motifs)
    ].sort_values(by="motif_length", ascending=False)

    unique_hits = {}
    for _, r in subset_df.iterrows():
        found = False
        for hit in unique_hits:
            if r["tokens"].issubset(hit):
                found = True
                break
        if not found:
            unique_hits[r["tokens"]] = (r["mutation_string_pe"], r["inns"])

    unique_hits_by_inn = {
        mut_string_pe: [] for (mut_string_pe, _) in unique_hits.values()
    }
    for mut_string_pe, inns_x in unique_hits.values():
        for _, inns_y in unique_hits.values():
            unique_hits_by_inn[mut_string_pe].append(inns_x != inns_y)

    return sum(any(x) for x in unique_hits_by_inn.values())


def step_build_cluster_info(
    step_in: ChainMotifStep,
    initial_database_tokens: dict[
        tuple[str, ...], dict[tuple[str, str], set[str]]
    ],
) -> tuple[ChainMotifStepResult, pd.DataFrame]:
    """Assign each motif to its seed group(s) and elect per-group representatives.

    Requires ``step_annotate_seeds`` (for ``is_seed``) and
    ``step_compute_inn_attribution`` (for ``adjusted_count``) to have run first.

    For each motif M within its stratum:

    - ``cluster_seeds``         — set of seed names S where tokens(S) ⊆ tokens(M)
    - ``n_clusters``            — number of seed groups M belongs to
    - ``seed_group_reps``       — ``{seed_name: rep_mutation_string_pe}`` dict for
                                  every seed group M belongs to; representative =
                                  member with highest adjusted_count, tiebreak inn_count
    - ``primary_cluster_seeds`` — list with the primary seed(s): the smallest-token
                                  seed(s) among those whose group has the most members
    - ``primary_cluster_rep``   — representative of the primary seed's group
    """
    df = step_in.motif_df.copy().reset_index(drop=True)
    group_cols = [c for c in step_in.group_cols if c in df.columns]
    min_seed_size = step_in.min_seed_size
    allowed_motifs_by_select = step_in.allowed_motifs_by_select
    adj_col = "adjusted_count"
    raw_col = "inn_count"
    rows_init = df.shape[0]

    # Compute tokens from mutation_string_pe if missing
    if "tokens" not in df.columns and "mutation_string_pe" in df.columns:
        df["tokens"] = df["mutation_string_pe"].apply(
            lambda pe: frozenset(
                t.strip() for t in str(pe).split("/") if t.strip()
            )
        )
    orig_copy_df = df.copy()

    required_columns = [raw_col, "tokens", "mutation_string_pe", "inns"]
    for col in required_columns:
        if col not in df.columns:
            raise AssertionError(
                f"Required column '{col}' not found in DataFrame"
            )

    def _process_stratum(
        sdf: pd.DataFrame,
        initial_database_tokens: dict[tuple[str, str], set[str]],
    ) -> pd.DataFrame:
        if sdf["mutation_string_pe"].value_counts().max() > 1:
            raise AssertionError(
                f"Duplicate mutation_string_pe found in stratum: {sdf['mutation_string_pe'].value_counts().loc[lambda x: x > 1].index.tolist()}."
                "Consider updating group by to ensure that each group has a unique mutation string within it (i.e. gene/allele/species)"
            )

        sdf = sdf.copy().reset_index(drop=True)
        orig_shape = sdf.shape[0]
        # just in case we lost data when perfomrning clustering tha twe cannot re trace, research the original
        # chain database for any inns ocntaining substrings
        sdf["unique_inn_chains"] = sdf["tokens"].apply(
            lambda x: _re_query_inn_map(x, initial_database_tokens)
        )
        sdf["inns"] = sdf["unique_inn_chains"].apply(
            lambda x: set(x.keys())
        )
        sdf[raw_col] = sdf["inns"].apply(len)
        if any(sdf[raw_col] == 0):
            raise AssertionError(
                f"Found rows with zero {raw_col}: {sdf.loc[sdf[raw_col] == 0, 'mutation_string_pe'].tolist()}"
            )
        super_sets = _find_supersets(sdf)
        adjusted_inn_map = _get_adjusted_inns(sdf, super_sets)
        sdf["adjusted_inns"] = sdf["mutation_string_pe"].map(
            adjusted_inn_map
        )
        sdf[raw_col] = sdf["inns"].apply(len)
        sdf[adj_col] = sdf["adjusted_inns"].apply(len)

        sdf["num_unique_super_motifs"] = sdf.apply(
            lambda x: num_unique_super_motifs(x, orig_copy_df), axis=1
        )

        sdf["zero_adjusted_counts"] = (sdf[adj_col] == 0) & (
            sdf["num_unique_super_motifs"] <= 1
        )
        _is_protected = (
            sdf["motif_selection_reason"].isin(allowed_motifs_by_select)
            if allowed_motifs_by_select is not None
            else pd.Series(False, index=sdf.index)
        )
        failed_tests = sdf["zero_adjusted_counts"] & ~_is_protected

        sdf.loc[failed_tests, "remove_motif"] = True

        sdf.loc[failed_tests, "motif_selection_reason"] = (
            "No adjusted counts and no unique groups of super motifs"
        )

        # rows_to_be_removed = df[failed_tests].copy()
        # all_rows_removed.append(rows_to_be_removed)
        # df = df[~failed_tests].copy()

        # first remove rows by adjusted count of 0 as long as they are not part of set of unique bifurcating motifs
        # so ab ad ac are unique evnts with common a so suggets a should still be a seed

        seeds = _find_seeds(sdf)
        seed_counts = {
            seed: (
                sum(
                    -1 * len(adjusted_inn_map[member])
                    for member in members
                ),
                seed,
            )
            for seed, members in seeds.items()
        }

        sdf["is_seed"] = sdf["mutation_string_pe"].apply(
            lambda x: x in seeds
        )

        sdf["cluster_seeds"] = sdf["mutation_string_pe"].apply(
            lambda x: [
                seed for seed, members in seeds.items() if x in members
            ]
        )
        sdf["n_clusters"] = sdf["cluster_seeds"].apply(lambda x: len(x))
        sdf["longest_motif_in_group"] = sdf[
            ["cluster_seeds", "mutation_string_pe"]
        ].apply(
            lambda x: (
                len(x["mutation_string_pe"].split("/"))
                in [
                    max([len(member.split("/")) for member in seeds[seed]])
                    for seed in x["cluster_seeds"]
                ]
                if len(x["cluster_seeds"])
                else False
            ),
            axis=1,
        )

        max_seed_rep = {
            seed: sorted(
                [
                    (-1 * len(adjusted_inn_map[member]), member)
                    for member in members
                ]
            )[0]
            for seed, members in seeds.items()
        }

        sdf["primary_cluster_seeds"] = sdf["cluster_seeds"].apply(
            lambda x: (
                sorted([seed_counts[seed] for seed in x])[0][1]
                if len(x)
                else None
            )
        )
        sdf["primary_cluster_rep"] = sdf["primary_cluster_seeds"].apply(
            lambda x: max_seed_rep[x][1] if not pd.isnull(x) else None
        )
        assert sdf.shape[0] == orig_shape, sdf[
            ["gene", "gene_species"]
        ].value_counts()
        return sdf

    all_rows_removed = []
    iterations = 0

    while True:
        init_rows_to_be_removed = df["remove_motif"].sum()

        if not group_cols:
            assert (
                len(initial_database_tokens) == 1
            ), initial_database_tokens.keys()
            db_tokens_per_group = next(
                iter(initial_database_tokens.values())
            )
            df = _process_stratum(df, db_tokens_per_group)
        else:
            parts: list[pd.DataFrame] = []
            for grp_name, sdf in df.groupby(
                group_cols, sort=False, dropna=False
            ):
                db_tokens_per_group = initial_database_tokens[grp_name]
                parts.append(_process_stratum(sdf, db_tokens_per_group))
            df = pd.concat(parts, ignore_index=True)

        if df.empty:
            break

        rows_to_be_removed = df["remove_motif"].sum()
        all_rows_removed.append(df[df["remove_motif"]].copy())

        df = df[~df["remove_motif"]].copy()

        if init_rows_to_be_removed >= rows_to_be_removed:
            break
        iterations += 1
        if iterations > 2:
            raise AssertionError(
                f"Check whats happening here for :{step_in}"
            )
        # re run the loop if we have filtered rows because now seeds need to be recalculated if here were filtered

    # if allowed_motifs_by_select is not None:

    if all_rows_removed:
        all_rows_removed_df = pd.concat(all_rows_removed)
        rows_dropped = all_rows_removed_df.shape[0]
    else:
        all_rows_removed_df = None
        rows_dropped = 0

    n_seeds = int(df.get("is_seed", pd.Series(dtype=bool)).sum())
    assert df["remove_motif"].sum() == 0, df[df["remove_motif"]]
    return (
        ChainMotifStepResult(
            motif_df=df,
            step_name="build_cluster_info",
            rows_added=0,
            rows_dropped=rows_dropped,
            group_cols=step_in.group_cols,
            detail=f"Cluster info built: {n_seeds} seeds across {len(df)} motifs",
        ),
        all_rows_removed_df,
    )
