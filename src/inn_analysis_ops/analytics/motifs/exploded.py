"""Motif exploding helpers shared by analytics and plotting."""

from __future__ import annotations

import logging
from typing import Optional

import pandas as pd

_log = logging.getLogger(__name__)


def build_exploded_motif_df(
    motif_df: pd.DataFrame,
    top_n: int,
    adjusted_count_col: str,
    inn_count_col: str,
    num_mutations_col: str,
    mutation_pe_col: str = "mutation_string_pe",
    sub_motifs_col: str = "sub_motifs",
    seed_col: Optional[str] = None,
    expand_linked_seeds: bool = False,
    row_filters: Optional[dict] = None,
    pre_selected_top_seeds: Optional[set] = None,
    pre_selected_original_top_seeds: Optional[set] = None,
    mutation_col: Optional[str] = None,
) -> pd.DataFrame:
    """Build a long-form (motif × seed_group) DataFrame with unambiguous columns.

    Each row represents one (motif, seed_group) pair. A motif that belongs
    to N seed groups produces N rows. ``mutation_pe_col`` is the single key
    column used throughout — no colon-notation column is read or required.

    Required columns in *motif_df*
    ------------------------------
    ``mutation_pe_col``, ``sub_motifs_col``, ``adjusted_count_col``,
    ``inn_count_col``, ``num_mutations_col``

    pre_selected_top_seeds
    ----------------------
    When provided, the top-N ranking and expansion steps are **skipped** —
    the given set is used directly as ``top_seeds``. Used by
    :func:`motif_cluster_viz._data_prep.prepare_motif_chart_data`
    which already ran the expansion.

    mutation_col
    ------------
    Backward-compatible alias for ``mutation_pe_col``.

    Returns
    -------
    pd.DataFrame
        Long-form DataFrame, one row per (motif_pe, seed_group).
    """
    if mutation_col is not None:
        if mutation_pe_col != "mutation_string_pe" and mutation_pe_col != mutation_col:
            raise ValueError(
                "build_exploded_motif_df: both mutation_pe_col and mutation_col "
                "were provided with different values."
            )
        mutation_pe_col = mutation_col

    required = [
        mutation_pe_col,
        sub_motifs_col,
        adjusted_count_col,
        inn_count_col,
        num_mutations_col,
    ]
    missing = [c for c in required if c not in motif_df.columns]
    if missing:
        raise ValueError(
            f"build_exploded_motif_df: required column(s) missing from motif_df: {missing}"
        )

    def _pe_toks(pe: str) -> frozenset:
        return frozenset(t.strip() for t in pe.split("/") if t.strip())

    def _is_leaf(v) -> bool:
        return isinstance(v, (set, list, frozenset)) and len(v) == 0

    seed_mask = motif_df[sub_motifs_col].apply(_is_leaf)
    seeds: list[str] = motif_df.loc[seed_mask, mutation_pe_col].tolist()

    if not seeds:
        raise ValueError(
            f"build_exploded_motif_df: no seeds found — no rows have an empty "
            f"'{sub_motifs_col}' collection."
        )

    seed_tokens: dict[str, frozenset] = {s: _pe_toks(s) for s in seeds}
    for i, sa in enumerate(seeds):
        for sb in seeds[i + 1 :]:
            ta, tb = seed_tokens[sa], seed_tokens[sb]
            if ta and tb:
                if ta < tb:
                    raise ValueError(
                        f"build_exploded_motif_df: seed {sa!r} is a strict subset "
                        f"of {sb!r} — seeds must be disjoint."
                    )
                if tb < ta:
                    raise ValueError(
                        f"build_exploded_motif_df: seed {sb!r} is a strict subset "
                        f"of {sa!r} — seeds must be disjoint."
                    )

    if pre_selected_top_seeds is not None:
        top_seeds: set[str] = set(pre_selected_top_seeds)
        if pre_selected_original_top_seeds is None:
            _original_top_seeds: set[str] = set(pre_selected_top_seeds)
        else:
            _original_top_seeds = set(pre_selected_original_top_seeds)
    else:
        inn_lookup: dict[str, int] = (
            motif_df.set_index(mutation_pe_col)[inn_count_col]
            .apply(lambda v: int(v) if pd.notna(v) else 0)
            .to_dict()
        )
        ranked_seeds = sorted(seeds, key=lambda s: inn_lookup.get(s, 0), reverse=True)
        top_seeds = set(ranked_seeds[:top_n]) if top_n else set(ranked_seeds)
        _original_top_n_count = (
            min(top_n, len(ranked_seeds)) if top_n else len(ranked_seeds)
        )
        _original_top_seeds = set(ranked_seeds[:_original_top_n_count])

    _all_toks: dict[str, frozenset] = {
        pe: _pe_toks(pe) for pe in motif_df[mutation_pe_col]
    }

    def _seed_groups(pe: str) -> list[str]:
        toks = _all_toks.get(pe, frozenset())
        return [s for s, st in seed_tokens.items() if st and st <= toks]

    motif_df = motif_df.copy()
    motif_df["_seed_groups"] = motif_df[mutation_pe_col].apply(_seed_groups)

    exploded = motif_df.explode("_seed_groups").rename(
        columns={"_seed_groups": "seed_group"}
    )
    exploded = exploded[exploded["seed_group"].notna()].copy()
    exploded = exploded.reset_index(drop=True)

    seed_set = set(seeds)
    if "is_seed" not in exploded.columns:
        exploded["is_a_seed"] = exploded[mutation_pe_col].isin(seed_set)
    else:
        exploded["is_a_seed"] = exploded["is_seed"].astype(bool)

    _seeds_with_non_self_rows: set[str] = set(
        exploded.loc[
            ~exploded["is_a_seed"] & exploded["seed_group"].isin(top_seeds),
            "seed_group",
        ].unique()
    )
    _pruned_orphans: set[str] = (
        top_seeds - _seeds_with_non_self_rows
    ) - _original_top_seeds
    if _pruned_orphans:
        _log.info(
            "build_exploded_motif_df: pruned %d orphan seed(s) with no motif rows: %s",
            len(_pruned_orphans),
            sorted(_pruned_orphans)[:5],
        )
        top_seeds = top_seeds - _pruned_orphans
        exploded = exploded[exploded["seed_group"].isin(top_seeds)].copy()
        exploded = exploded.reset_index(drop=True)

    exploded["is_primary_seed_group"] = exploded["seed_group"].isin(top_seeds)

    def _classify_row(row) -> str:
        pe = row[mutation_pe_col]
        if row["is_a_seed"]:
            return "top seed" if pe in _original_top_seeds else "expanded seed"
        if "assignment" in row.index and pd.notna(row["assignment"]):
            if row["assignment"] == "motif_associated_chain":
                return "motif row paired to chain"
        return "motif row with top seed"

    exploded["row_type"] = exploded.apply(_classify_row, axis=1)
    return exploded
