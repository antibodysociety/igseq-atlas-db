"""Bispecific branch detection and pruning for motif DataFrames.

Algorithm
---------
Uses the pre-computed ``sub_motifs`` column (Möbius ancestry) for grouping:

  For each anchor motif S that appears in any row's ``sub_motifs`` list:
    1. Collect all rows where S ∈ sub_motifs → the branch candidates for S.
    2. Sort candidates largest-first (``num_mutations`` desc).
    3. Sequential maximal scan within the group: a candidate becomes a **node**
       if no already-known node in this group is a superset of it.
    4. INNs appearing in ≥2 nodes are **bispecific** relative to anchor S.
    5. Common stem = closest existing motif to the token-set intersection of
       the sharing nodes.
    6. Bispecific INNs are removed from the node rows and attributed to the
       stem row's ``adjusted_inns``.

Grouping by ``sub_motifs`` scopes each sequential scan within the correct
ancestry group, preventing intermediate motifs from unrelated branches from
absorbing node candidates before they can be compared.

New columns
-----------
``bispecific_inns``
    On stem rows: set of INNs attributed here via bispecific pruning.

``bispecific_inn_count``
    On stem rows: ``len(bispecific_inns)``.

``bispecific_inn_branch_records``
    On stem rows: list of dicts — **one record per bispecific INN**::

        {
            "inn":             str,
            "branches":        list[str],    # node mutation_strings (sorted)
            "branch_deltas":   list[str],    # arm-specific tokens per branch,
                                             #   parallel to "branches"
            "common_stem":     str | None,   # closest existing motif to the
                                             #   token-set intersection
            "combination_key": str,          # canonical sorted join of deltas
            "n_branches":      int,
        }

``bispecific_branch_tokens``
    Human-readable summary of unique arm-delta strings on this stem.

``bispecific_dropped_branches``
    Node mutation_strings fully emptied (``inn_count → 0``) by pruning.

``is_bispecific_anchor``
    True on stem rows that received ≥1 bispecific INN.

``bispecific_collapsed_into``
    On node rows that lost all INNs: the stem mutation_string.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from typing import Optional

import pandas as pd

from inn_analysis_ops.analytics._helpers import build_token_ignore_set as _build_ignore_set
from inn_analysis_ops.analytics._helpers import (
    find_best_subset_motif,
    find_multi_chain_inns,
    motif_tokens_set,
    tokens_to_mutation_string,
)
from inn_analysis_ops.analytics._helpers import (
    token_matches_ignore_set as _should_ignore_token,
)

_log = logging.getLogger(__name__)


def resolve_bispecific_branches_dep(
    motif_df: pd.DataFrame,
    mutation_col: str = "mutation_string",
    inn_count_col: str = "inn_count",
    adjusted_count_col: str = "adjusted_count",
    group_by: Optional[list[str]] = None,
) -> pd.DataFrame:
    """.. deprecated::
        Superseded by :func:`~analytics.motifs.pipelines.chain.build_chain_motif_index`
        which handles bispecific detection at chain level.  Retained for the
        classical ``build_motif_index`` pipeline only.

    Parameters
    ----------
    motif_df:
        Output of ``build_motif_index``.  Must have ``mutation_string``,
        ``sub_motifs`` (list), ``inns`` (set), ``adjusted_inns`` (set),
        ``num_mutations``.
    mutation_col:
        Column with canonical mutation strings (default ``"mutation_string"``).
    inn_count_col, adjusted_count_col:
        Column names to update after removing bispecific INNs.
    group_by:
        Optional stratification columns (e.g. ``["gene", "gene_species"]``).
        Detection is scoped within each stratum.

    Returns
    -------
    Updated copy of *motif_df* with new columns as described in the module
    docstring.
    """
    df = motif_df.copy().reset_index(drop=True)

    df["bispecific_inns"] = [set() for _ in range(len(df))]
    df["bispecific_inn_count"] = 0
    df["bispecific_inn_branch_records"] = [[] for _ in range(len(df))]
    df["bispecific_branch_tokens"] = ""
    df["bispecific_dropped_branches"] = [[] for _ in range(len(df))]
    df["is_bispecific_anchor"] = False
    df["bispecific_collapsed_into"] = None

    _GROUP_COLS = ["gene", "chain_type", "gene_species"]
    strat_cols = [c for c in (group_by or []) if c in df.columns] or [
        c for c in _GROUP_COLS if c in df.columns
    ]

    def _process_stratum(sdf: pd.DataFrame) -> pd.DataFrame:
        sdf = sdf.copy()

        ms_to_idx: dict[str, int] = {
            row[mutation_col]: i
            for i, row in sdf.iterrows()
            if isinstance(row.get(mutation_col), str)
        }
        token_sets: dict[str, frozenset] = {
            ms: motif_tokens_set(ms) for ms in ms_to_idx
        }

        # ── Group candidates by anchor via sub_motifs ─────────────────────────
        # For each motif S that appears in any row's sub_motifs list, collect
        # all rows that list it. This scopes the sequential scan to the correct
        # ancestry group, preventing cross-branch interference.
        anchor_to_cand_idxs: dict[str, list[int]] = defaultdict(list)
        for i, row in sdf.iterrows():
            sub_motifs = row.get("sub_motifs")
            if isinstance(sub_motifs, list):
                for sm in sub_motifs:
                    if isinstance(sm, str) and sm in ms_to_idx:
                        anchor_to_cand_idxs[sm].append(i)

        for anchor_ms, cand_idxs in anchor_to_cand_idxs.items():
            if len(cand_idxs) < 2:
                continue

            # ── Sequential maximal scan within this anchor's group ────────────
            # Sort largest-first (num_mutations desc, then ms for reproducibility)
            cand_sorted: list[tuple[int, str]] = sorted(
                [
                    (i, sdf.at[i, mutation_col])
                    for i in cand_idxs
                    if isinstance(sdf.at[i, mutation_col], str)
                ],
                key=lambda x: (-sdf.at[x[0], "num_mutations"], x[1]),
            )

            # node_ms → [child_ms, ...]
            substring_groups: dict[str, list[str]] = {}

            for i, ms in cand_sorted:
                ms_s = token_sets[ms]
                has_super_str = False
                for node_ms in substring_groups:
                    if ms_s.issubset(token_sets[node_ms]):
                        has_super_str = True
                        substring_groups[node_ms].append(ms)
                if not has_super_str:
                    substring_groups[ms] = []

            nodes: list[str] = list(substring_groups.keys())
            if len(nodes) < 2:
                continue

            # ── INN sharing across nodes ──────────────────────────────────────
            inn_to_nodes: dict[str, list[str]] = defaultdict(list)
            for node_ms in nodes:
                idx = ms_to_idx.get(node_ms)
                if idx is None:
                    continue
                node_inns = sdf.at[idx, "inns"]
                if isinstance(node_inns, (set, frozenset)):
                    for inn in node_inns:
                        inn_to_nodes[inn].append(node_ms)

            bispecific_inns: set[str] = {
                inn for inn, nl in inn_to_nodes.items() if len(nl) >= 2
            }
            if not bispecific_inns:
                continue

            # ── Per-INN records: common stem + branch deltas ──────────────────
            inn_records: list[dict] = []
            unique_deltas: set[str] = set()

            for inn in sorted(bispecific_inns):
                branch_motifs = sorted(inn_to_nodes[inn])
                branch_tok_sets = [token_sets[ms] for ms in branch_motifs]

                common_tokens: frozenset = branch_tok_sets[0]
                for ts in branch_tok_sets[1:]:
                    common_tokens = common_tokens & ts

                # Closest existing motif with tokens ⊆ common_tokens
                common_stem_ms: Optional[str] = find_best_subset_motif(
                    common_tokens, token_sets
                )

                stem_tokens: frozenset = (
                    token_sets[common_stem_ms] if common_stem_ms else common_tokens
                )

                branch_deltas: list[str] = []
                for branch_ms in branch_motifs:
                    delta = token_sets[branch_ms] - stem_tokens
                    d = " / ".join(sorted(delta)) if delta else "(stem only)"
                    branch_deltas.append(d)
                    unique_deltas.add(d)

                inn_records.append(
                    {
                        "inn": inn,
                        "branches": branch_motifs,
                        "branch_deltas": branch_deltas,
                        "common_stem": common_stem_ms,
                        "combination_key": " | ".join(sorted(branch_deltas)),
                        "n_branches": len(branch_motifs),
                    }
                )

            # ── Group by stem and annotate ────────────────────────────────────
            stem_to_records: dict[Optional[str], list[dict]] = defaultdict(list)
            for rec in inn_records:
                stem_to_records[rec["common_stem"]].append(rec)

            for stem_ms, records in stem_to_records.items():
                stem_idx = ms_to_idx.get(stem_ms) if stem_ms else None
                bi_inns_for_stem: set[str] = {rec["inn"] for rec in records}

                if stem_idx is not None:
                    existing_bi: set = set(sdf.at[stem_idx, "bispecific_inns"] or set())
                    sdf.at[stem_idx, "bispecific_inns"] = existing_bi | bi_inns_for_stem
                    sdf.at[stem_idx, "bispecific_inn_count"] = len(
                        sdf.at[stem_idx, "bispecific_inns"]
                    )
                    sdf.at[stem_idx, "is_bispecific_anchor"] = True

                    existing_recs: list = list(
                        sdf.at[stem_idx, "bispecific_inn_branch_records"] or []
                    )
                    sdf.at[stem_idx, "bispecific_inn_branch_records"] = (
                        existing_recs + records
                    )

                    all_deltas: set[str] = {
                        d for rec in records for d in rec["branch_deltas"]
                    }
                    existing_bt: str = (
                        sdf.at[stem_idx, "bispecific_branch_tokens"] or ""
                    )
                    sdf.at[stem_idx, "bispecific_branch_tokens"] = " | ".join(
                        filter(
                            None,
                            [
                                existing_bt,
                                " | ".join(sorted(all_deltas)),
                            ],
                        )
                    )

                    adj = sdf.at[stem_idx, "adjusted_inns"]
                    existing_adj = (
                        set(adj) if isinstance(adj, (set, frozenset)) else set()
                    )
                    sdf.at[stem_idx, "adjusted_inns"] = existing_adj | bi_inns_for_stem
                    sdf.at[stem_idx, adjusted_count_col] = len(
                        sdf.at[stem_idx, "adjusted_inns"]
                    )
                    stem_raw = set(sdf.at[stem_idx, "inns"] or set())
                    sdf.at[stem_idx, "inns"] = stem_raw | bi_inns_for_stem
                    sdf.at[stem_idx, inn_count_col] = len(sdf.at[stem_idx, "inns"])

                all_branches: set[str] = {
                    ms for rec in records for ms in rec["branches"]
                }
                dropped: list[str] = []
                for branch_ms in sorted(all_branches):
                    branch_idx = ms_to_idx.get(branch_ms)
                    if branch_idx is None:
                        continue
                    branch_bi = bi_inns_for_stem & set(
                        sdf.at[branch_idx, "inns"] or set()
                    )
                    if not branch_bi:
                        continue
                    new_inns = set(sdf.at[branch_idx, "inns"]) - bi_inns_for_stem
                    sdf.at[branch_idx, "inns"] = new_inns
                    sdf.at[branch_idx, inn_count_col] = len(new_inns)
                    adj = sdf.at[branch_idx, "adjusted_inns"]
                    if isinstance(adj, (set, frozenset)):
                        new_adj = set(adj) - bi_inns_for_stem
                        sdf.at[branch_idx, "adjusted_inns"] = new_adj
                        sdf.at[branch_idx, adjusted_count_col] = len(new_adj)
                    if len(new_inns) == 0 and stem_ms is not None:
                        dropped.append(branch_ms)
                        sdf.at[branch_idx, "bispecific_collapsed_into"] = stem_ms

                if dropped and stem_idx is not None:
                    existing_dropped: list = list(
                        sdf.at[stem_idx, "bispecific_dropped_branches"] or []
                    )
                    sdf.at[stem_idx, "bispecific_dropped_branches"] = (
                        existing_dropped + dropped
                    )

        return sdf

    if strat_cols:
        parts = []
        for _, gdf in df.groupby(strat_cols, dropna=False):
            parts.append(_process_stratum(gdf))
        return pd.concat(parts, ignore_index=True)
    return _process_stratum(df)


from inn_analysis_ops.analytics._helpers import (
    eu_mutations_to_tokens as _eu_mutations_to_tokens,
)
from inn_analysis_ops.analytics._helpers import (
    full_length_chain_eu_to_token as _full_length_chain_eu_to_token,
)


def _derive_sub_motifs(
    tok: frozenset,
    all_token_sets: dict,
) -> list[str]:
    """Return mutation strings of all existing motifs that are strict subsets of *tok*.

    Mirrors the logic used in ``_recompute_mobius_stats`` / ``_run_mobius_pass``
    but operates on the pre-built ``{mutation_string: frozenset}`` lookup rather
    than running a full Möbius pass.  Used to populate ``sub_motifs`` on
    synthetic bispecific rows so that downstream ``len(sub_motifs) == 0``
    queries correctly identify truly "atomic" motifs.

    Parameters
    ----------
    tok :
        Frozenset of ``(pos, change)`` tuples for the synthetic row.
    all_token_sets :
        ``{mutation_string: frozenset}`` built from the motif table at the
        point of synthesis.

    Returns
    -------
    Sorted list of mutation strings whose token set is a strict proper subset
    of *tok* (i.e. ``ts < tok``).
    """
    return sorted(ms for ms, ts in all_token_sets.items() if ts < tok)


def _parse_tok(tok: str) -> tuple:
    """Parse a colon-notation token ``"234:L->A"`` to ``(pos, change)``.

    Position is converted to ``int`` for numeric values (e.g. ``234``) and
    kept as ``str`` for insertion positions (e.g. ``"193.i3"``).  This matches
    the type convention used throughout the motif pipeline's frozensets.
    """
    parts = tok.split(":", 1)
    if len(parts) != 2:
        raise ValueError(
            f"_parse_tok: malformed token '{tok}' — expected 'POS:CHANGE'."
        )
    pos_str = parts[0].strip()
    try:
        pos: int | str = int(pos_str)
    except ValueError:
        pos = pos_str  # insertion position like "193.i3"
    return (pos, parts[1].strip())


def _derive_mutation_string_pe(mutation_string: str) -> str:
    """Convert colon-notation *mutation_string* to compact PE format.

    Delegates to :func:`format_motif_label` via a
    lazy import (same pattern as ``co_mutation_analytics.build_motif_index``).
    Falls back to the original string on ImportError or any formatting error.
    """
    try:
        try:
            from inn_analysis_ops.analytics.position_labeling import (
                format_motif_label as _fmt,  # type: ignore[import]
            )
        except ImportError:
            from inn_analysis_ops.analytics._helpers import (
                format_motif_label as _fmt,
            )
        return _fmt(mutation_string).replace(" ", "")
    except Exception:
        return mutation_string


def _match_motif_to_chains(
    inn_ref: str,
    inn_col: Literal["inn", "inn_id"],
    motif_tokens: frozenset,
    chain_df: "pd.DataFrame",
    mutation_col: str = "constant_chain_mutations_EU",
) -> "pd.DataFrame":
    """Find the chains of *inn* that carry all of *motif_tokens*.

    Parameters
    ----------
    inn_ref : INN name.
    motif_tokens : frozenset of ``"pos:change"`` strings for the motif.
    chain_df : ``ds.full_length_chain_collection.to_dataframe()`` — one row
        per chain with columns ``inn``, ``chain_id``, ``chain_type``, and
        ``constant_chain_mutations_EU``.
    mutation_col : column in *chain_df* holding EU mutation strings.

    Returns
    -------
    DataFrame of rows where ``motif_tokens.issubset(chain_tokens)``.

    Raises
    ------
    ValueError
        Hard fail if no row for *inn* carries all of *motif_tokens*.
    """
    if mutation_col not in chain_df.columns:
        raise ValueError(
            f"_match_motif_to_chains: column '{mutation_col}' not found in "
            f"chain_df. Available columns: {list(chain_df.columns)}"
        )
    if inn_col not in ["inn", "inn_id"]:
        raise ValueError(
            f"_match_motif_to_chains: inn_col '{inn_col}' is not valid. "
            f"Expected one of ['inn', 'inn_id']"
        )
    inn_rows = chain_df[chain_df[inn_col] == inn_ref]
    if inn_rows.empty:
        raise ValueError(
            f"_match_motif_to_chains: INN '{inn_ref}' not found in chain_df."
        )
    match_mask = inn_rows[mutation_col].apply(
        lambda x: motif_tokens.issubset(_eu_mutations_to_tokens(x))
    )
    matched = inn_rows[match_mask]
    if matched.empty:
        raise ValueError(
            f"_match_motif_to_chains: INN '{inn_ref}' has no chain where "
            f"motif tokens {sorted(motif_tokens)} are a subset of the chain's "
            f"{mutation_col}. "
            f"Chain mutation strings for this INN:\n"
            + "\n".join(
                f"  chain_id={r['chain_id']} chain_type={r['chain_type']}: "
                f"{r[mutation_col]}"
                for _, r in inn_rows.iterrows()
            )
        )
    return matched


def expand_bispecific_branches(
    motif_df: pd.DataFrame,
    constant_chains: "pd.DataFrame",
    mutation_col: str = "mutation_string",
    inn_count_col: str = "inn_count",
    adjusted_count_col: str = "adjusted_count",
    group_by: Optional[list[str]] = None,
    min_count: int = 1,
    mutations_to_ignore: Optional[dict] = None,
) -> pd.DataFrame:
    """.. deprecated::
        ``expand_bispecific_branches`` is superseded by the native cross-chain
        pairing step (``expand_motifs_by_cross_chain_pairing``, pipeline step
        2/15) inside ``build_chain_motif_index``, which runs automatically as
        part of ``build_constant_chain_analysis``.  The result is stored on
        ``result.motif_result.cross_chain_df`` and its rows are identifiable in
        ``result.motif_df_clustered_post_filter`` via
        ``motif_selection_reason in {"bispecific independent arm",
        "bispecific shared anchor"}``.

        This function requires a ``chain_id`` column that is no longer present
        in the post-collapse motif tables and will raise ``ValueError`` when
        called on them.  Do not use.

    Expand motif rows to expose bispecific arm-specific patterns.

    **Primary detection via ``constant_chains``** (= ``ds.full_length_chain_collection.to_dataframe()``):

    For each motif M and each INN X in M.inns:

    1. Call :func:`_match_motif_to_chains` to find which chains of X carry M's
       tokens (hard fail if none found).
    2. From the matched chains, get ``chain_type``.
    3. Find all OTHER chains of X with the same ``chain_type`` and different
       mutation profiles.
    4. Filter the other chain's tokens through *mutations_to_ignore* (same
       rules as ``MotifIndexConfig.mutations_to_ignore_for_motif_ops``).
    5. Compute ``common = M.tokens & filtered_other_tokens``.  No common
       tokens → skip (separate motifs / fully ignored profile).
    6. Find the closest existing motif for the common stem.
    7. For new arm rows: search *constant_chains* for **all** INNs whose chains
       carry the filtered arm tokens (not just the triggering INN).
       Only add the row if ``len(arm_inns) >= min_count``.

    **Debug assertion**: logs whether the INN also appears in multiple motifs in
    the motif table (confirms the bispecific signal is reflected upstream).

    Parameters
    ----------
    motif_df : Clustered motif table. Must have ``mutation_string``, ``inns``,
        ``adjusted_inns``.
    constant_chains : ``ds.full_length_chain_collection.to_dataframe()`` —
        one row per chain with ``inn``, ``chain_id``, ``chain_type``,
        ``constant_chain_mutations_EU``.
    mutation_col, inn_count_col, adjusted_count_col : column overrides.
    group_by : stratification columns (e.g. ``["gene_species", "gene"]``).
    min_count : minimum ``inn_count`` required for a new arm row to be added.
        Arm patterns seen in fewer than *min_count* INNs are skipped.
    mutations_to_ignore : ``MotifIndexConfig.mutations_to_ignore_for_motif_ops``
        dict.  Tokens whose (mutation_type, position) match any rule are
        stripped from the arm profile before ``arm_ms`` is constructed and
        before ``arm_inns`` is computed.
    """
    import warnings

    warnings.warn(
        "expand_bispecific_branches is deprecated and will be removed in a future "
        "release. Bispecific cross-chain pairing is now handled natively by "
        "expand_motifs_by_cross_chain_pairing (pipeline step 2/15) inside "
        "build_chain_motif_index, which runs automatically via "
        "build_constant_chain_analysis. Access results via "
        "result.motif_result.cross_chain_df or filter "
        "result.motif_df_clustered_post_filter on motif_selection_reason.",
        DeprecationWarning,
        stacklevel=2,
    )

    df = motif_df.copy().reset_index(drop=True)

    for _req in (mutation_col, "inns", "chain_id", "adjusted_inns"):
        if _req not in df.columns:
            raise ValueError(
                f"expand_bispecific_branches: motif_df missing required column '{_req}'."
            )

    df["bispecific_pair"] = False
    df["bispecific_anchor"] = False
    df["bispecific_source"] = None

    # ── Build token-ignore set ────────────────────────────────────────────
    _ignore_set = _build_ignore_set(mutations_to_ignore or {})
    if _ignore_set:
        _log.info(
            "expand_bispecific_branches: %d ignore rules loaded from mutations_to_ignore.",
            len(_ignore_set),
        )

    # ── Validate chain_df ──────────────────────────────────────────────────
    _EU_COL = "constant_chain_mutations_EU"
    for _req_col in ("inn", "inn_id", "chain_id", "chain_type", _EU_COL):
        if _req_col not in constant_chains.columns:
            raise ValueError(
                f"expand_bispecific_branches: constant_chains must have "
                f"column '{_req_col}'. Found: {list(constant_chains.columns[:10])}"
            )

    # ── Multi-chain INNs: any INN with ≥2 chains of the same chain_type ────
    multi_chain_inns: set[str] = set(
        find_multi_chain_inns(constant_chains, multi_chain_inn_key="by_inn_id").keys()
    )
    inn_id_to_inn_map = constant_chains.groupby(by=["inn_id"])["inn"].first().to_dict()
    # inn_to_inn_ids_map = (
    #     constant_chains.groupby(by=["inn"])["inn_id"].first().to_dict()
    # )

    _log.info(
        "expand_bispecific_branches: %d multi-chain INNs (same chain_type, ≥2 chains).",
        len(multi_chain_inns),
    )
    if not multi_chain_inns:
        _log.info("expand_bispecific_branches: no multi-chain INNs — table unchanged.")
        return df

    # ── Motif-table lookups ────────────────────────────────────────────────
    ms_to_idx: dict[str, int] = {}
    for i, row in df.iterrows():
        ms = row[mutation_col]
        if not isinstance(ms, str):
            raise ValueError(
                f"expand_bispecific_branches: row {i} has non-string "
                f"'{mutation_col}': {ms!r}"
            )
        ms_to_idx[ms] = i

    all_motif_tokens: dict[str, frozenset] = {
        ms: motif_tokens_set(ms) for ms in ms_to_idx
    }

    # INN → motifs in table (for debug assertion)
    inn_to_motifs: dict[str, set[str]] = defaultdict(set)
    for _, row in df.iterrows():
        _inns = row.get("inns")
        if not isinstance(_inns, (set, frozenset)):
            raise ValueError(
                f"expand_bispecific_branches: motif '{row[mutation_col]}' "
                f"has non-set 'inns': {_inns!r}"
            )
        for _inn in _inns:
            inn_to_motifs[_inn].add(row[mutation_col])

    # group context for stem row placement
    _grp_cols = [c for c in (group_by or []) if c in df.columns]
    inn_to_group_rows: dict[str, list[dict]] = defaultdict(list)
    for _, row in df.iterrows():
        _inns = row.get("inns", set())
        if not isinstance(_inns, (set, frozenset)):
            continue
        _gk = tuple(row[c] for c in _grp_cols)
        for _inn in _inns:
            if not any(
                tuple(r.get(c) for c in _grp_cols) == _gk
                for r in inn_to_group_rows[_inn]
            ):
                inn_to_group_rows[_inn].append({c: row[c] for c in df.columns})

    # ── Main loop ──────────────────────────────────────────────────────────
    new_rows: list[dict] = []
    # Tracks arm_ms strings already added as new rows — avoids duplicate rows
    # when the same arm pattern is triggered by multiple bispecific INNs or
    # multiple source motifs.
    _seen_new_arm_ms: set[str] = set()

    for i, row in df.iterrows():
        motif_ms: str = row[mutation_col]
        motif_tokens = all_motif_tokens[motif_ms]
        motif_inns = row["inns"]
        motif_inn_ids = [chain_id.split("_")[0] for chain_id in row["chain_id"]]

        candidate_inns = set(motif_inn_ids) & set(multi_chain_inns)
        if not candidate_inns:
            continue

        for inn_id in sorted(candidate_inns):
            # Step 1: find chains of this INN that carry the motif — HARD FAIL if none
            inn = inn_id_to_inn_map[inn_id]
            try:
                matched_chains = _match_motif_to_chains(
                    inn, "inn", motif_tokens, constant_chains, _EU_COL
                )
            except ValueError as exc:
                raise ValueError(
                    f"expand_bispecific_branches: motif '{motif_ms}', INN '{inn_id}': "
                    f"{exc}"
                ) from exc

            # Step 2: for each matched chain, look for other chains with the same
            # chain_type but different mutation profile
            for chain_type, ct_group in matched_chains.groupby("chain_type"):
                matched_token_sets = [
                    _eu_mutations_to_tokens(r[_EU_COL]) for _, r in ct_group.iterrows()
                ]

                # All chains of this INN with the same chain_type
                all_same_type = constant_chains[
                    (constant_chains["inn"] == inn)
                    & (constant_chains["chain_type"] == chain_type)
                ]

                other_chains = all_same_type[
                    ~all_same_type["chain_id"].isin(ct_group["chain_id"])
                ]
                if other_chains.empty:
                    _log.debug(
                        "expand: INN '%s' chain_type '%s' — no other chains; skip.",
                        inn_id,
                        chain_type,
                    )
                    continue

                for _, other_row in other_chains.iterrows():
                    other_tokens_raw = _eu_mutations_to_tokens(other_row[_EU_COL])
                    if not other_tokens_raw:
                        _log.debug(
                            "expand: INN '%s' other chain '%s' is germline; skip.",
                            inn_id,
                            other_row["chain_id"],
                        )
                        continue
                    if other_tokens_raw in matched_token_sets:
                        _log.debug(
                            "expand: INN '%s' other chain '%s' same profile as matched; skip.",
                            inn_id,
                            other_row["chain_id"],
                        )
                        continue

                    # ── Apply mutations_to_ignore filter ──────────────────────
                    other_tokens = frozenset(
                        t
                        for t in other_tokens_raw
                        if not _should_ignore_token(t, _ignore_set)
                    )
                    if not other_tokens:
                        _log.debug(
                            "expand: INN '%s' other chain '%s' has no tokens after "
                            "ignore-filter; skip.",
                            inn,
                            other_row["chain_id"],
                        )
                        continue

                    common = motif_tokens & other_tokens
                    if not common:
                        _log.debug(
                            "expand: INN '%s' motif '%s' vs chain '%s' %s: "
                            "no common tokens after ignore-filter; skip.",
                            inn_id,
                            motif_ms,
                            other_row["chain_id"],
                            sorted(other_tokens),
                        )
                        continue

                    arm_ms = " / ".join(sorted(other_tokens))
                    stem_ms: Optional[str] = find_best_subset_motif(
                        common, all_motif_tokens
                    )

                    _log.info(
                        "expand: BISPECIFIC PAIR — motif '%s' | INN '%s' "
                        "chain_type '%s' | other chain '%s'\n"
                        "  other arm: %s  [%s]\n"
                        "  stem:      %s",
                        motif_ms,
                        inn,
                        chain_type,
                        other_row["chain_id"],
                        arm_ms,
                        "NEW" if arm_ms not in ms_to_idx else "existing",
                        stem_ms or "(none)",
                    )

                    # Debug assertion
                    sharing = inn_to_motifs.get(inn, set()) - {motif_ms}
                    if sharing:
                        _log.info(
                            "  ✓ INN '%s' also shared in table motifs: %s",
                            inn,
                            sorted(sharing),
                        )
                    else:
                        _log.warning(
                            "  ⚠ INN '%s' bispecific in constant_chains but "
                            "NOT shared across motifs in table.",
                            inn,
                        )

                    # ── Arm row ─────────────────────────────────────────────
                    if arm_ms in ms_to_idx:
                        df.at[ms_to_idx[arm_ms], "bispecific_pair"] = True
                        df.at[ms_to_idx[arm_ms], "bispecific_source"] = motif_ms
                        _log.info("  → arm '%s' flagged bispecific_pair", arm_ms)
                    elif arm_ms not in _seen_new_arm_ms:
                        # ── Find all INNs with this arm profile in constant_chains ──
                        # Any chain whose filtered-token set contains all arm tokens
                        # qualifies.  This gives the true population for this arm
                        # pattern — not just the triggering bispecific INN.
                        arm_inns: set[str] = set(
                            constant_chains[
                                constant_chains[_EU_COL].apply(
                                    lambda eu, _ft=other_tokens: _ft.issubset(
                                        frozenset(
                                            t
                                            for t in _eu_mutations_to_tokens(eu)
                                            if not _should_ignore_token(t, _ignore_set)
                                        )
                                    )
                                )
                            ]["inn"].unique()
                        )

                        if len(arm_inns) < min_count:
                            _log.debug(
                                "  → arm '%s' SKIPPED (inn_count=%d < min_count=%d)",
                                arm_ms,
                                len(arm_inns),
                                min_count,
                            )
                            continue

                        _seen_new_arm_ms.add(arm_ms)

                        arm_tok = frozenset(_parse_tok(t) for t in other_tokens)
                        _arm_sorted = sorted(arm_tok, key=lambda t: (str(t[0]), t[1]))
                        syn: dict = {c: row[c] for c in df.columns}
                        syn[mutation_col] = arm_ms
                        syn["mutation_string_pe"] = _derive_mutation_string_pe(arm_ms)
                        syn["motif"] = arm_tok
                        syn["mutation_list"] = _arm_sorted
                        syn["num_mutations"] = len(arm_tok)
                        syn["inns"] = arm_inns
                        syn[inn_count_col] = len(arm_inns)
                        syn["adjusted_inns"] = arm_inns
                        syn[adjusted_count_col] = len(arm_inns)
                        syn["bispecific_pair"] = True
                        syn["bispecific_anchor"] = False
                        syn["bispecific_source"] = motif_ms
                        # Populate sub_motifs from existing table subset lookup
                        _arm_subs = _derive_sub_motifs(arm_tok, all_motif_tokens)
                        syn["sub_motifs"] = _arm_subs
                        if "raw_sub_motifs" in syn:
                            syn["raw_sub_motifs"] = _arm_subs
                        for _s in (
                            "motif_score",
                            "motif_score_artifact_adjusted",
                            "motif_id",
                            "pvalue_vs_submotifs",
                            "lift_vs_submotifs",
                        ):
                            if _s in syn:
                                syn[_s] = None
                        new_rows.append(syn)
                        _log.info("  → arm '%s' ADDED", arm_ms)

                    # ── Stem row: one per gene/species group ─────────────────
                    if stem_ms is not None:
                        if stem_ms in ms_to_idx:
                            df.at[ms_to_idx[stem_ms], "bispecific_anchor"] = True
                            _log.info(
                                "  → stem '%s' flagged bispecific_anchor",
                                stem_ms,
                            )
                        else:
                            stem_tok = frozenset(_parse_tok(t) for t in _tok(stem_ms))
                            _stem_sorted = sorted(
                                stem_tok, key=lambda t: (str(t[0]), t[1])
                            )
                            _stem_subs = _derive_sub_motifs(stem_tok, all_motif_tokens)
                            _seen: set[tuple] = set()
                            for _grp_row in inn_to_group_rows.get(inn, [row]):
                                _gk = tuple(_grp_row.get(c) for c in _grp_cols)
                                if _gk in _seen:
                                    continue
                                _seen.add(_gk)
                                syn_stem: dict = {
                                    c: _grp_row.get(c) for c in df.columns
                                }
                                syn_stem[mutation_col] = stem_ms
                                syn_stem["mutation_string_pe"] = (
                                    _derive_mutation_string_pe(stem_ms)
                                )
                                syn_stem["motif"] = stem_tok
                                syn_stem["mutation_list"] = _stem_sorted
                                syn_stem["num_mutations"] = len(stem_tok)
                                syn_stem["inns"] = {inn}
                                syn_stem[inn_count_col] = 1
                                syn_stem["adjusted_inns"] = {inn}
                                syn_stem[adjusted_count_col] = 1
                                syn_stem["bispecific_pair"] = False
                                syn_stem["bispecific_anchor"] = True
                                syn_stem["bispecific_source"] = motif_ms
                                syn_stem["sub_motifs"] = _stem_subs
                                if "raw_sub_motifs" in syn_stem:
                                    syn_stem["raw_sub_motifs"] = _stem_subs
                                for _s in (
                                    "motif_score",
                                    "motif_score_artifact_adjusted",
                                    "motif_id",
                                    "pvalue_vs_submotifs",
                                    "lift_vs_submotifs",
                                ):
                                    if _s in syn_stem:
                                        syn_stem[_s] = None
                                new_rows.append(syn_stem)
                                _log.info(
                                    "  → stem '%s' ADDED (group %s)",
                                    stem_ms,
                                    dict(zip(_grp_cols, _gk)),
                                )

    n_new = len(new_rows)
    if n_new:
        _log.info(
            "expand_bispecific_branches: %d new row(s) added (%d arm, %d anchor).",
            n_new,
            sum(1 for r in new_rows if r.get("bispecific_pair")),
            sum(1 for r in new_rows if r.get("bispecific_anchor")),
        )
        new_df = pd.DataFrame(new_rows, columns=df.columns)
        df = pd.concat([df, new_df], ignore_index=True)

        # Re-derive sub_motifs for all new bispecific rows now that the full
        # table is available.  Construction-time lookup missed sibling new rows
        # added in the same pass (they weren't in all_motif_tokens yet).
        _final_tok: dict[str, frozenset] = {
            row[mutation_col]: row["motif"]
            for _, row in df.iterrows()
            if isinstance(row.get("motif"), frozenset)
        }
        _new_mask = df["bispecific_pair"].fillna(False).astype(bool) | df[
            "bispecific_anchor"
        ].fillna(False).astype(bool)
        for idx in df.index[_new_mask]:
            _tok = df.at[idx, "motif"]
            if not isinstance(_tok, frozenset):
                continue
            _ms = df.at[idx, mutation_col]
            _subs = _derive_sub_motifs(
                _tok,
                {ms: ts for ms, ts in _final_tok.items() if ms != _ms},
            )
            df.at[idx, "sub_motifs"] = _subs
            if "raw_sub_motifs" in df.columns:
                df.at[idx, "raw_sub_motifs"] = _subs
    else:
        _log.info(
            "expand_bispecific_branches: no bispecific pairs found — table unchanged."
        )

    return df


# ── Always-on lateral-sharing annotation ──────────────────────────────────────

_STALE_COLS = (
    "motif_score",
    "motif_score_artifact_adjusted",
    "motif_id",
    "pvalue_vs_submotifs",
    "lift_vs_submotifs",
)


def annotate_bispecific_lateral_sharing(
    motif_df: pd.DataFrame,
    mutation_col: str = "mutation_string",
    inn_col: str = "adjusted_inns",
    inn_count_col: str = "inn_count",
    adjusted_count_col: str = "adjusted_count",
    group_by: Optional[list[str]] = None,
    anchor_min_mutations: Optional[int] = None,
) -> pd.DataFrame:
    """Always-on non-destructive annotation of lateral INN sharing across motifs.

    After INN subtraction, any INN that remains in ``adjusted_inns`` of ≥ 2
    *unrelated* motifs signals a bispecific / multi-arm antibody — those motifs
    co-occur in the same INN's sequence profile rather than being explained by
    a parent-child ancestry relationship.

    This function is **non-destructive**: it does not change ``inns``,
    ``adjusted_inns``, ``inn_count``, or ``adjusted_count`` on existing rows.

    Algorithm
    ---------
    1. Build INN → [row indices] from ``adjusted_inns``; keep only INNs that
       appear in ≥ 2 distinct rows (lateral sharing).
    2. Group sharing INNs by their exact co-occurring motif set
       ``frozenset(mutation_strings)`` — all INNs shared by the same pair/group
       of motifs are processed together.
    3. For each co-sharing group:

       a. Flag all participating motifs ``bispecific_pair = True``.
       b. Compute the common token intersection of all sharing motifs.
       c. Find the *best existing anchor*: the largest motif whose token set
          ⊆ ``common_tokens`` → flag ``bispecific_anchor = True``.
       d. If no such motif exists, add a **synthetic anchor row per unique
          (gene, gene_species) combination** present in the sharing motifs.
          Cross-gene sharing therefore produces one anchor row per gene group.

    Parameters
    ----------
    motif_df :
        Final motif table (post-subtraction, post-clustering).
    mutation_col :
        Column holding canonical mutation strings.
    inn_col :
        Column whose sets are inspected for lateral sharing (``"adjusted_inns"``).
    group_by :
        Optional stratification columns; defaults to ``["gene", "gene_species"]``
        when present in the DataFrame.

    Returns
    -------
    Annotated copy of *motif_df*, possibly with synthetic anchor rows appended.
    Adds ``bispecific_pair``, ``bispecific_anchor``, ``bispecific_source`` if not
    already present.
    """
    df = motif_df.copy().reset_index(drop=True)

    # ── Ensure bispecific flag columns exist ─────────────────────────────────
    for _col, _default in [
        ("bispecific_pair", False),
        ("bispecific_anchor", False),
        ("bispecific_source", None),
    ]:
        if _col not in df.columns:
            df[_col] = _default

    if inn_col not in df.columns:
        _log.warning(
            "annotate_bispecific_lateral_sharing: column '%s' not found — skipping.",
            inn_col,
        )
        return df

    # ── Build INN → row indices ──────────────────────────────────────────────
    inn_to_idxs: dict[str, list[int]] = defaultdict(list)
    for i, row in df.iterrows():
        adj = row.get(inn_col)
        if isinstance(adj, (set, frozenset)):
            for inn in adj:
                inn_to_idxs[inn].append(i)

    lateral = {inn: idxs for inn, idxs in inn_to_idxs.items() if len(idxs) >= 2}
    if not lateral:
        _log.info("annotate_bispecific_lateral_sharing: no lateral INN sharing found.")
        return df

    # ── Group sharing INNs by their co-occurring motif set ───────────────────
    # INNs shared by the exact same set of motifs are processed together so
    # we only compute one anchor per unique co-sharing group.
    group_to_inns: dict[frozenset, set[str]] = defaultdict(set)
    for inn, idxs in lateral.items():
        motif_key = frozenset(df.at[i, mutation_col] for i in idxs)
        group_to_inns[motif_key].add(inn)

    # ── Token lookups ────────────────────────────────────────────────────────
    def _tok_row(ms: str) -> frozenset:
        return frozenset(_parse_tok(t) for t in str(ms).split(" / ") if t.strip())

    ms_to_idx: dict[str, int] = {
        row[mutation_col]: i
        for i, row in df.iterrows()
        if isinstance(row.get(mutation_col), str)
    }
    all_token_sets: dict[str, frozenset] = {ms: _tok_row(ms) for ms in ms_to_idx}

    _GROUP_COLS = ["gene", "chain_type", "gene_species"]
    _grp_cols = [c for c in (group_by or []) if c in df.columns] or [
        c for c in _GROUP_COLS if c in df.columns
    ]

    new_rows: list[dict] = []
    # Tracks (anchor_mutation_string, *group_key) already added to avoid dups
    _added_anchors: set[tuple] = set()

    # ── Main annotation loop ─────────────────────────────────────────────────
    for motif_set, sharing_inns in group_to_inns.items():
        # Common token intersection of ALL sharing motifs
        toks_list = [all_token_sets.get(ms, frozenset()) for ms in motif_set]
        common_tokens: frozenset = toks_list[0]
        for ts in toks_list[1:]:
            common_tokens = common_tokens & ts

        # Canonical source string (used on both pair and anchor rows)
        source_str = " | ".join(sorted(motif_set))

        # 3a. Flag participating motifs as bispecific_pair
        for ms in motif_set:
            idx = ms_to_idx.get(ms)
            if idx is None:
                continue
            df.at[idx, "bispecific_pair"] = True
            if not df.at[idx, "bispecific_source"]:
                df.at[idx, "bispecific_source"] = source_str

        if not common_tokens:
            _log.debug(
                "annotate: motif set %s has no common tokens; pair flagged, no anchor.",
                sorted(motif_set),
            )
            continue

        # 3b. Find best existing anchor: largest motif with tokens ⊆ common_tokens
        anchor_ms: Optional[str] = find_best_subset_motif(common_tokens, all_token_sets)

        if anchor_ms is not None:
            # 3c. Flag existing row as anchor (OR into existing value)
            anchor_idx = ms_to_idx[anchor_ms]
            df.at[anchor_idx, "bispecific_anchor"] = True
            if not df.at[anchor_idx, "bispecific_source"]:
                df.at[anchor_idx, "bispecific_source"] = source_str
            _log.info(
                "annotate: existing anchor '%s' flagged for %d sharing INN(s): %s",
                anchor_ms,
                len(sharing_inns),
                sorted(sharing_inns),
            )
        elif (
            anchor_min_mutations is not None
            and len(common_tokens) < anchor_min_mutations
        ):
            # common_tokens is below the configured anchor threshold — skip.
            _log.debug(
                "annotate: common_tokens=%s has %d token(s) < anchor_min_mutations=%d; "
                "pairs flagged but no synthetic anchor added.",
                sorted(common_tokens),
                len(common_tokens),
                anchor_min_mutations,
            )
        elif anchor_min_mutations is None and len(common_tokens) < 2:
            # Default behaviour: single-token common intersection → skip.
            # Single-token motifs are intentionally excluded from
            # motif_df_clustered_post_filter (num_mutations > 1 filter); adding
            # them back here as synthetic rows would conflict with that gate and
            # produce inn_count/profile mismatches in debug validation.
            # Set MotifIndexConfig.bispecific_anchor_min_mutations=1 to allow these.
            _log.debug(
                "annotate: common_tokens=%s is single-token; "
                "pairs flagged but no synthetic anchor added "
                "(set bispecific_anchor_min_mutations=1 to enable).",
                sorted(common_tokens),
            )
        else:
            # 3d. Add synthetic anchor rows — one per unique gene/species group
            # in the co-sharing motifs.  Cross-gene pairs produce one row each.
            anchor_sorted_toks = sorted(common_tokens, key=lambda t: (str(t[0]), t[1]))
            anchor_ms_str = " / ".join(f"{t[0]}:{t[1]}" for t in anchor_sorted_toks)

            # ── Compute true inns for the anchor ─────────────────────────────
            # All INNs in any motif whose token set ⊇ common_tokens — these are
            # all INNs that carry the anchor pattern (not just the bispecific ones
            # that triggered this detection).
            anchor_inns: set[str] = set(sharing_inns)  # always include the trigger INNs
            for ms, ts in all_token_sets.items():
                if common_tokens <= ts:  # anchor tokens ⊆ this motif's tokens
                    ref_ms_idx = ms_to_idx.get(ms)
                    if ref_ms_idx is not None:
                        _ms_inns = df.at[ref_ms_idx, "inns"]
                        if isinstance(_ms_inns, (set, frozenset)):
                            anchor_inns.update(_ms_inns)

            _seen_grps: set[tuple] = set()
            for ms in sorted(motif_set):
                ref_idx = ms_to_idx.get(ms)
                if ref_idx is None:
                    continue
                ref_row = df.iloc[ref_idx]
                gk = tuple(ref_row.get(c) for c in _grp_cols)
                if gk in _seen_grps:
                    continue
                _seen_grps.add(gk)

                dedup_key = (anchor_ms_str,) + gk
                if dedup_key in _added_anchors:
                    continue
                _added_anchors.add(dedup_key)

                inn_total = ref_row.get("inn_total") or 1
                syn: dict = {c: ref_row.get(c) for c in df.columns}
                syn[mutation_col] = anchor_ms_str
                syn["mutation_string_pe"] = _derive_mutation_string_pe(anchor_ms_str)
                syn["motif"] = common_tokens
                syn["mutation_list"] = anchor_sorted_toks
                syn["num_mutations"] = len(common_tokens)
                syn["inns"] = set(anchor_inns)
                syn[inn_count_col] = len(anchor_inns)
                syn["adjusted_inns"] = set(anchor_inns)
                syn[adjusted_count_col] = len(anchor_inns)
                syn["freq_motif"] = round(len(anchor_inns) / inn_total, 5)
                syn["adjusted_freq"] = round(len(anchor_inns) / inn_total, 5)
                syn["bispecific_pair"] = False
                syn["bispecific_anchor"] = True
                syn["bispecific_source"] = source_str
                # Populate sub_motifs from existing table subset lookup
                _anchor_subs = _derive_sub_motifs(common_tokens, all_token_sets)
                syn["sub_motifs"] = _anchor_subs
                if "raw_sub_motifs" in syn:
                    syn["raw_sub_motifs"] = _anchor_subs
                for _s in _STALE_COLS:
                    if _s in syn:
                        syn[_s] = None
                new_rows.append(syn)
                _log.info(
                    "annotate: synthetic anchor '%s' ADDED for group %s, %d INN(s): %s",
                    anchor_ms_str,
                    dict(zip(_grp_cols, gk)),
                    len(anchor_inns),
                    sorted(anchor_inns),
                )

    # ── Append synthetic rows ────────────────────────────────────────────────
    n_new = len(new_rows)
    if n_new:
        new_df = pd.DataFrame(new_rows, columns=df.columns)
        df = pd.concat([df, new_df], ignore_index=True)

        # Re-derive sub_motifs for all new anchor rows using the complete
        # post-concat token set (construction-time lookup missed sibling rows).
        _final_tok: dict[str, frozenset] = {
            row[mutation_col]: row["motif"]
            for _, row in df.iterrows()
            if isinstance(row.get("motif"), frozenset)
        }
        _new_mask = df["bispecific_pair"].fillna(False).astype(bool) | df[
            "bispecific_anchor"
        ].fillna(False).astype(bool)
        for idx in df.index[_new_mask]:
            _tok = df.at[idx, "motif"]
            if not isinstance(_tok, frozenset):
                continue
            _ms = df.at[idx, mutation_col]
            _subs = _derive_sub_motifs(
                _tok,
                {ms: ts for ms, ts in _final_tok.items() if ms != _ms},
            )
            df.at[idx, "sub_motifs"] = _subs
            if "raw_sub_motifs" in df.columns:
                df.at[idx, "raw_sub_motifs"] = _subs

    n_pairs = int(df["bispecific_pair"].fillna(False).astype(bool).sum())
    n_anchors = int(df["bispecific_anchor"].fillna(False).astype(bool).sum())
    _log.info(
        "annotate_bispecific_lateral_sharing: %d lateral-sharing INN(s) detected "
        "→ %d pair row(s), %d anchor row(s) (%d new synthetic).",
        len(lateral),
        n_pairs,
        n_anchors,
        n_new,
    )
    return df
