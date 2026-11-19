"""Co-mutation frequency analytics for inn_summary data.

Functions
---------
compute_co_mutation_frequencies(inn_summary, genes, mode, ...)
    Compute intra- or inter-chain pairwise co-mutation counts with
    Fisher's exact test for independence.

remove_intra_pairs(inter_df, intra_df)
    Remove from an inter-chain table any position pairs that also appear
    as intra-chain co-mutations (based on shared EU position indices).
"""

from __future__ import annotations

import re
import warnings
from dataclasses import dataclass
from itertools import combinations
from itertools import product as iproduct
from math import comb as _comb_count
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple

import networkx as nx
import numpy as np
import pandas as pd
from scipy.stats import binomtest as _binom
from scipy.stats import fisher_exact
from tqdm.auto import tqdm as _tqdm

from ._helpers import (
    _build_mutation_string_pe,
    _canonical_str,
    _parse_tokens,
    build_inn_lookup,
)

if TYPE_CHECKING:
    # Import only for static type checking — both would otherwise create a
    # circular import at runtime (analytics.constant_chain_analysis imports
    # FROM this module; data_io.merge.loader is import-safe at runtime,
    # but kept alongside for consistency). `from __future__ import
    # annotations` above means these names are never resolved at runtime.
    from inn_analysis_ops.analytics.constant_chain_analysis import ConstantChainAnalysisResult
    from inn_analysis_ops.data_io.merge.loader import INNAnalysisDataset


@dataclass(frozen=True)
class MutEvent:
    """A single somatic mutation on one chain of one INN.

    Fields map directly to inn_summary columns. ``region_classification``
    (e.g. "FC", "HINGE") is carried so paired events can be classified by
    their region context (FC-FC, HINGE-FC, etc.).
    """

    EU: str  # position key — may be non-canonical like "230.i1"
    mutation_type: str
    germline_seq: str
    mutated_seq: str
    region_classification: str

    def __repr__(self) -> str:
        return f"MutEvent(EU={self.EU}, {self.germline_seq}->{self.mutated_seq}, {self.region_classification})"


def _prep(
    inn_summary: pd.DataFrame,
    position_col: str,
    regions: Optional[List[str]],
) -> pd.DataFrame:
    """Flatten inn_summary and filter. Keeps non-canonical positions as strings.

    Excludes insertion mutations (mutation_type == 'insertion').
    Position column ``_pos`` is the raw string EU value (e.g. '234', '230.i1').
    """
    flat = inn_summary.reset_index()

    def _is_list_like(v) -> bool:
        return isinstance(v, (list, tuple, set, frozenset, np.ndarray, pd.Series))

    # Newer summary tables can store per-row carrier INNs as a list/set in "inn".
    # Expand to one row per carrier INN so downstream unique-INN counting and
    # pair generation stay correct.
    if "inn" in flat.columns and flat["inn"].map(_is_list_like).any():
        flat = flat.copy()
        flat["inn"] = flat["inn"].map(lambda v: list(v) if _is_list_like(v) else v)
        flat = flat.explode("inn")

    # Co-mutation pair classification depends on chain identity.
    assert "chain_id" in flat.columns, (
        "compute_co_mutation_frequencies requires chain-resolved input: "
        "missing 'chain_id' in inn_summary/index."
    )

    # Some pre-aggregated tables expose "residue" instead of "mutated_seq".
    if "mutated_seq" not in flat.columns and "residue" in flat.columns:
        flat = flat.rename(columns={"residue": "mutated_seq"})

    flat = flat[flat["mutation_type"] != "insertion"]
    if regions is not None:
        flat = flat[
            flat["region_classification"].str.upper().isin([r.upper() for r in regions])
        ]
    flat = flat.dropna(subset=[position_col, "inn"]).copy()
    flat["_pos"] = flat[position_col].astype(str).str.strip()
    flat["inn"] = flat["inn"].astype(str).str.strip()
    flat["gene"] = flat["gene"].astype(str).str.strip().str.lower()
    flat["chain_id"] = flat["chain_id"].astype(str).str.strip()
    flat = flat[
        flat["inn"].ne("")
        & flat["inn"].ne("nan")
        & flat["chain_id"].ne("")
        & flat["chain_id"].ne("nan")
    ]
    assert "gene_species" in flat.columns, (
        "compute_co_mutation_frequencies requires 'gene_species' in inn_summary/index."
    )
    flat["gene_species"] = flat["gene_species"].astype(str).str.strip().str.lower()
    return flat


def build_marginals_table(
    inn_summary: pd.DataFrame,
    position_col: str = "EU",
    regions: Optional[List[str]] = None,
    stratify_by: List[str] | None = None,
) -> pd.DataFrame:
    """Build a pre-computed marginals table for co-mutation analysis.

    Each row is a unique mutation event within a stratification group.
    The index of the returned table is the stratification columns +
    (region_classification, _pos, germline_seq, mutated_seq, mutation_type),
    so that ``compute_co_mutation_frequencies`` can look up ``n_p0``/``n_p1``
    for a pair using the pair's stratification values.

    Parameters
    ----------
    inn_summary:
        Full mutation profile DataFrame.
    position_col:
        ``"EU"``, ``"IMGT"``, or ``"KABAT"``.
    regions:
        Optional region_classification filter.
    stratify_by:
        List of columns that define the stratification group.
        Default ``["gene_species"]`` (original behaviour — collapsed across genes).
        Pass ``["gene", "gene_species"]`` to scope marginals per gene so that
        IGHG4 marginals only count IGHG4 INNs.

    Returns
    -------
    DataFrame indexed by (stratify_cols..., region_classification, _pos,
    germline_seq, mutated_seq, mutation_type) with columns:
        representative_gene, region, gene_counts,
        n_inn, raw_counts, inn_set, inn_set_multi_chain, n_inn_multi_chain,
        inn_set_intra_repeat, n_inn_intra_repeat.
    """
    flat = _prep(inn_summary, position_col, regions)

    # ── Resolve stratification columns ────────────────────────────────────
    _strat = stratify_by or ["gene_species"]
    # Add derived isotype column if requested
    if "isotype" in _strat:
        flat["_isotype"] = flat["gene"].str[:4].str.upper()
        _strat = [("_isotype" if c == "isotype" else c) for c in _strat]
    _strat = [c for c in _strat if c in flat.columns]
    # Ensure gene_species is always present (needed for species-level fallback)
    if "gene_species" not in _strat:
        _strat = ["gene_species"] + _strat

    # ── Core grouping key: stratification + mutation event identity ────────
    _KEY = list(
        dict.fromkeys(  # deduplicate while preserving order
            _strat
            + [
                "region",
                "region_classification",
                "_pos",
                "germline_seq",
                "mutated_seq",
                "mutation_type",
            ]
        )
    )
    # Key used for the gene-counts dict: same as _KEY but with "gene" removed so
    # that "gene" stays as a column (not a groupby key) inside each group.
    # This lets the lambda access g["gene"] even when gene is in _strat.
    _KEY_NO_GENE = [c for c in _KEY if c != "gene"]

    # Gene-granular key (always includes gene for INN-per-gene counts)
    _KEY_GENE = list(
        dict.fromkeys(
            _strat
            + [
                "gene",
                "region",
                "region_classification",
                "_pos",
                "germline_seq",
                "mutated_seq",
                "mutation_type",
            ]
        )
    )

    # ── Gene-level INN counts ─────────────────────────────────────────────
    gene_inn_counts = (
        flat.groupby(_KEY_GENE)["inn"].nunique().rename("n_inn_gene").reset_index()
    )

    # ── Collapsed grouping (stratification-scoped, gene-agnostic within strat) ─
    inn_counts = flat.groupby(_KEY)["inn"].nunique().rename("n_inn")
    raw_counts = flat.groupby(_KEY).size().rename("raw_counts")
    inn_sets = flat.groupby(_KEY)["inn"].apply(set).rename("inn_set")

    # Gene counts dict: {gene: n_unique_inns}
    # Group by _KEY_NO_GENE so "gene" remains available as a column in each group.
    gene_counts_series = (
        gene_inn_counts.groupby(_KEY_NO_GENE)
        .apply(
            lambda g: dict(
                sorted(
                    zip(g["gene"], g["n_inn_gene"]),
                    key=lambda x: (-x[1], x[0]),
                )
            ),
            include_groups=False,
        )
        .rename("gene_counts")
    )

    # Representative gene: most globally frequent, tiebreak alphabetical
    _gene_rank_order: dict[str, int] = {
        g: i for i, g in enumerate(flat["gene"].value_counts().index)
    }

    def _pick_rep_gene(genes_series):
        return min(genes_series, key=lambda g: (_gene_rank_order.get(g, 999), g))

    rep_gene = (
        gene_inn_counts.groupby(_KEY_NO_GENE)["gene"]
        .apply(_pick_rep_gene)
        .rename("representative_gene")
    )

    # ── Multi-chain detection ─────────────────────────────────────────────
    chains_per_inn_mut = (
        flat.groupby(["inn", "_pos", "germline_seq", "mutated_seq"])["chain_id"]
        .nunique()
        .rename("n_chains")
        .reset_index()
    )
    multi_chain = chains_per_inn_mut[chains_per_inn_mut["n_chains"] > 1]
    multi_chain_counts = (
        multi_chain.groupby(["_pos", "germline_seq", "mutated_seq"])["inn"]
        .apply(set)
        .rename("inn_set_multi_chain")
    )

    # ── Intra-repeat detection ────────────────────────────────────────────
    repeats_per_chain = (
        flat.groupby(["inn", "chain_id", "_pos", "germline_seq", "mutated_seq"])
        .size()
        .rename("n_repeats")
        .reset_index()
    )
    intra_repeat = repeats_per_chain[repeats_per_chain["n_repeats"] > 1]
    intra_repeat_counts = (
        intra_repeat.groupby(["_pos", "germline_seq", "mutated_seq"])["inn"]
        .apply(set)
        .rename("inn_set_intra_repeat")
    )

    # ── Assemble ──────────────────────────────────────────────────────────
    marginals = pd.concat([inn_counts, raw_counts, inn_sets], axis=1).reset_index()
    marginals = marginals.merge(
        gene_counts_series.reset_index(), on=_KEY_NO_GENE, how="left"
    )
    marginals = marginals.merge(rep_gene.reset_index(), on=_KEY_NO_GENE, how="left")
    marginals = marginals.merge(
        multi_chain_counts.reset_index(),
        on=["_pos", "germline_seq", "mutated_seq"],
        how="left",
    )
    marginals = marginals.merge(
        intra_repeat_counts.reset_index(),
        on=["_pos", "germline_seq", "mutated_seq"],
        how="left",
    )
    marginals["inn_set_multi_chain"] = marginals["inn_set_multi_chain"].apply(
        lambda x: x if isinstance(x, set) else set()
    )
    marginals["n_inn_multi_chain"] = marginals["inn_set_multi_chain"].apply(len)
    marginals["inn_set_intra_repeat"] = marginals["inn_set_intra_repeat"].apply(
        lambda x: x if isinstance(x, set) else set()
    )
    marginals["n_inn_intra_repeat"] = marginals["inn_set_intra_repeat"].apply(len)

    # ── Set stratification-aware index ───────────────────────────────────
    # Index = (stratify_cols..., gene_species, region_classification, _pos,
    #          germline_seq, mutated_seq, mutation_type)
    # This index is used by compute_co_mutation_frequencies to look up
    # n_p0/n_p1 for each pair using the pair's stratification values.
    _idx_cols = list(
        dict.fromkeys(
            _strat
            + [
                "gene_species",
                "region_classification",
                "_pos",
                "germline_seq",
                "mutated_seq",
                "mutation_type",
            ]
        )
    )
    _idx_cols = [c for c in _idx_cols if c in marginals.columns]
    marginals = marginals.set_index(_idx_cols)

    cols_front = [
        "representative_gene",
        "region",
        "gene_counts",
        "n_inn",
        "raw_counts",
    ]
    col_order = cols_front + [c for c in marginals.columns if c not in cols_front]
    return marginals[[c for c in col_order if c in marginals.columns]].sort_values(
        ["region_classification", "_pos", "n_inn"],
        ascending=[True, True, False],
    )


def _assert_marginals(
    internal: pd.DataFrame,
    external: pd.DataFrame,
) -> None:
    """Assert that an externally-built marginals table agrees with the internal one.

    Compares per-(stratification_cols, mutation_string_pe) unique INN counts
    between ``internal`` (from ``build_marginals_table``) and ``external``
    (e.g. from ``build_mutation_residues``).

    ``internal`` must have ``n_inn`` as a column and its index must include
    ``_pos``, ``germline_seq``, and ``mutated_seq``.

    ``external`` must have ``unique_inn_count`` as a column and
    ``mutation_string_pe`` in its index.  Any additional index levels are
    treated as stratification keys and are matched against the same columns in
    ``internal``.

    String join-key columns are normalised to uppercase before comparison so
    that case differences (e.g. ``ighg1`` vs ``IGHG1``) do not cause false
    mismatches.

    Raises
    ------
    AssertionError
        If any (key, mutation) pair has mismatched counts, listing every
        discrepancy.
    """
    # ── Flatten internal to (strat_cols..., mutation_string_pe) → n_inn ──────
    int_flat = internal.reset_index().copy()
    # Only compare substitutions — insertions are already excluded from _prep;
    # deletions are excluded here too so both sides of the comparison are consistent.
    if "mutation_type" in int_flat.columns:
        int_flat = int_flat[int_flat["mutation_type"] == "substitution"]
    int_flat["mutation_string_pe"] = int_flat.rename(
        columns={"_pos": "EU", "mutated_seq": "residue"}
    ).apply(_build_mutation_string_pe, axis=1)
    # Identify stratification columns: everything in the index except the fixed
    # mutation-identity columns
    _fixed = {
        "region_classification",
        "_pos",
        "germline_seq",
        "mutated_seq",
        "mutation_type",
    }
    int_strat_cols = [
        c for c in internal.index.names if c not in _fixed and c in int_flat.columns
    ]
    int_key_cols = int_strat_cols + ["mutation_string_pe"]

    # Normalise string strat columns to uppercase so case differences don't
    # produce spurious mismatches (build_mutation_residues uppercases gene,
    # raw data may be lowercase).
    # Use is_string_dtype to cover both legacy object dtype and pandas 2.x StringDtype.
    import pandas.api.types as pat

    _str_cols = [c for c in int_strat_cols if pat.is_string_dtype(int_flat[c])]
    for c in _str_cols:
        int_flat[c] = int_flat[c].str.upper()
    # Note: mutation_string_pe is NOT uppercased — it contains position labels
    # like "P230.i1C" where ".i" is part of the position notation, not a classifier.

    int_counts = (
        int_flat.groupby(int_key_cols)["n_inn"]
        .max()  # same mutation may appear in multiple region rows; take max
        .reset_index()
    )

    # ── Flatten external to the same key → unique_inn_count ──────────────────
    ext_flat = external.reset_index().copy()
    if "mutation_string_pe" not in ext_flat.columns:
        raise ValueError(
            "'assert_against_marginals' must have 'mutation_string_pe' in its index or columns."
        )
    if "unique_inn_count" not in ext_flat.columns:
        raise ValueError(
            "'assert_against_marginals' must have a 'unique_inn_count' column."
        )
    # Drop insertions and deletions — _prep excludes insertions from the
    # internal marginals, and deletions can have non-standard position strings
    # that produce false mismatches.  Only compare substitutions.
    if "mutation_type" in ext_flat.columns:
        ext_flat = ext_flat[~ext_flat["mutation_type"].isin(["insertion", "deletion"])]
    # Find any shared stratification columns with internal
    ext_strat_cols = [c for c in int_strat_cols if c in ext_flat.columns]
    ext_key_cols = ext_strat_cols + ["mutation_string_pe"]

    # Normalise classifier columns in external (gene, gene_species) — NOT mutation_string_pe
    for c in _str_cols:
        if c in ext_flat.columns:
            ext_flat[c] = ext_flat[c].str.upper()

    ext_counts = ext_flat.groupby(ext_key_cols)["unique_inn_count"].max().reset_index()

    # ── Merge and compare — join on only the columns present in *external* ──
    # external may not have all stratification columns that internal does (e.g.
    # gene_species absent when external only has gene).  Use ext_key_cols as the
    # common join key for both sides.
    merged = int_counts.merge(ext_counts, on=ext_key_cols, how="outer")
    merged["n_inn"] = merged["n_inn"].fillna(-1).astype(int)
    merged["unique_inn_count"] = merged["unique_inn_count"].fillna(-1).astype(int)
    mismatches = merged[merged["n_inn"] != merged["unique_inn_count"]]

    if not mismatches.empty:
        lines = ["Marginals cross-validation FAILED — INN counts differ:"]
        lines.append(
            f"  {'Key':<40}  {'internal n_inn':>14}  {'external unique_inn_count':>25}"
        )
        lines.append("  " + "-" * 83)
        for _, row in mismatches.head(20).iterrows():
            key_str = " × ".join(str(row[c]) for c in ext_key_cols)
            lines.append(
                f"  {key_str:<40}  {row['n_inn']:>14}  {row['unique_inn_count']:>25}"
            )
        if len(mismatches) > 20:
            lines.append(f"  ... and {len(mismatches) - 20} more rows")
        raise AssertionError("\n".join(lines))


def compute_co_mutation_frequencies(
    inn_summary: pd.DataFrame,
    position_col: str = "EU",
    min_count: int = 1,
    regions: Optional[List[str]] = None,
    stratify_n_by: List[str] | None = None,
    assert_against_marginals: pd.DataFrame | None = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Compute pairwise co-mutation frequencies with independence statistics.

    Generates ALL pair classifications in a single pass:
      - ``intra_repeat``: same chain, same position, same mutation (repeated domain)
      - ``intra_pos_repeat_diff_mut``: same chain, same position, different mutation
      - ``intra``: same chain, different positions
      - ``inter_same_gene``: different chains, same gene
      - ``inter_cross_gene``: different chains, different gene
      - ``shared_mutation_cross_chain``: different chains, same position + same mutation
      - ``unexpected``: does not match any of the above rules

    Parameters
    ----------
    inn_summary:
        Full mutation profile DataFrame (multi-indexed from Excel).
    position_col:
        Which numbering to use: ``"EU"``, ``"IMGT"``, or ``"KABAT"``.
    min_count:
        Minimum co-occurrence count to retain.
    regions:
        Optional list of ``region_classification`` values to include.
    stratify_n_by:
        List of columns to group by when computing ``n_total`` per pair.
        ``None`` (default) gives the original behaviour: N = unique INNs
        per ``gene_species``.

        Examples::

            stratify_n_by=["gene", "gene_species"]   # N per gene × species
            stratify_n_by=["isotype"]                 # N per isotype (all species)
            stratify_n_by=["gene_species"]            # original behaviour

        The lookup key for each pair is built from ``gene_0`` (the canonical
        first gene) plus any other requested columns. Missing denominator keys
        raise an ``AssertionError``.
    assert_against_marginals:
        Optional pre-built marginals table (e.g. from
        ``build_mutation_residues``) to cross-validate against the internally
        computed ``build_marginals_table`` result.

        The table must have ``mutation_string_pe``
        (``{germline}{pos}{mutated}``, e.g. ``"L235A"``) in its index and a
        ``unique_inn_count`` column.  Any stratification columns present in the
        index (e.g. ``gene``) are also used for the comparison.

        If any (mutation_string_pe, strat_key) pair has a different INN count
        between the two tables an ``AssertionError`` is raised listing the
        mismatches.

    Returns
    -------
    (co_mutations_df, marginals_table)
    """
    flat = _prep(inn_summary, position_col, regions)
    flat["_mut"] = flat["germline_seq"] + "->" + flat["mutated_seq"]
    n_total = flat["inn"].nunique()
    n_total_by_species = flat.groupby("gene_species")["inn"].nunique().to_dict()
    inn_species_map = flat.groupby("inn")["gene_species"].first().to_dict()

    # ── Build per-stratification N lookup ─────────────────────────────────
    # _n_lookup maps a tuple of stratification values (uppercased) → int
    _n_lookup: dict[tuple, int] = {}
    _stratify_cols: list[str] = stratify_n_by or ["gene_species"]

    # Add isotype column if requested
    if "isotype" in _stratify_cols:
        flat["_isotype"] = flat["gene"].str[:4].str.upper()
        # map "isotype" → "_isotype" column name
        _col_map = {"isotype": "_isotype"}
    else:
        _col_map = {}

    _real_cols = [_col_map.get(c, c) for c in _stratify_cols]
    _missing_strat = [c for c in _real_cols if c not in flat.columns]
    assert not _missing_strat, (
        "Missing stratification column(s) in input: " + ", ".join(_missing_strat)
    )
    _valid_cols = _real_cols

    if _valid_cols and _valid_cols != ["gene_species"]:
        for keys, grp in flat.groupby(_valid_cols):
            key = tuple(
                str(k).upper() for k in (keys if isinstance(keys, tuple) else (keys,))
            )
            _n_lookup[key] = grp["inn"].nunique()
    # else: use n_total_by_species (gene_species only, already computed)

    # Global gene frequency rank (for representative gene selection)
    _gene_rank: dict[str, int] = {
        g: i for i, g in enumerate(flat["gene"].value_counts().index)
    }

    # Build marginals table — stratified so n_p0/n_p1 are scoped to the same
    # group as N.  Pass stratify_n_by so the marginals index includes those cols.
    marginals_table = build_marginals_table(
        inn_summary, position_col, regions, stratify_by=stratify_n_by
    )
    # Work on a flat copy for lookup building (marginals_table keeps its index
    # for public consumption; we iterate on a reset copy internally).
    _marg_flat = marginals_table.reset_index()
    _marg_flat["_mut_sig"] = (
        _marg_flat["germline_seq"] + "->" + _marg_flat["mutated_seq"]
    )

    # ── Optional cross-validation against a pre-built marginals table ────────
    if assert_against_marginals is not None:
        _assert_marginals(marginals_table, assert_against_marginals)

    # ── Resolve which stratification columns were baked into the marginals ──
    # Mirror the logic in build_marginals_table so we use the same column names.
    _strat_marg = _stratify_cols  # already resolved above (real column names)
    # Only keep columns that are actually in the flat marginals frame
    _strat_marg = [c for c in _strat_marg if c in _marg_flat.columns]

    # ── Helper: extract strat-key from a pair's representative-gene / species ─
    # We need to map a (gene, gene_species) row back to a strat-key tuple so we
    # can look up the correct n_p0/n_p1 from the stratified marginals.
    # The strat key mirrors how build_marginals_table built its index:
    #   (_strat cols..., gene_species, region_classification, _pos,
    #    germline_seq, mutated_seq, mutation_type)
    # For the position-level marginal we need: strat_key + (species, rc, pos)
    # For the variant-level marginal:          strat_key + (species, rc, pos, msig)

    def _strat_key_from_gene(gene: str, species: str) -> tuple:
        """Build the stratification prefix for a given gene and species."""
        vals = []
        for c in _strat_marg:
            if c == "gene_species":
                vals.append(species)
            elif c == "gene":
                vals.append(gene)
            elif c == "_isotype":
                vals.append(gene[:4].upper() if gene else "")
            else:
                raise AssertionError(f"Unhandled stratification column: {c}")
        return tuple(vals)

    # Position-level marginal: unique INNs with ANY mutation at (strat_key, rc, pos)
    _pos_marg: dict[tuple, int] = {}
    _grp_pos_cols = _strat_marg + ["region_classification", "_pos"]
    _grp_pos_cols = [c for c in _grp_pos_cols if c in _marg_flat.columns]
    for gkeys, grp in _marg_flat.groupby(_grp_pos_cols):
        gkeys_t = gkeys if isinstance(gkeys, tuple) else (gkeys,)
        _pos_marg[gkeys_t] = len(set.union(*grp["inn_set"]))

    # Variant-level marginal + gene info — grouped within strat × rc × pos × mut_sig
    _var_marg: dict[tuple, int] = {}
    _gene_detail_map: dict[tuple, dict] = {}
    _rep_gene_map: dict[tuple, str] = {}

    _grp_var_cols = _strat_marg + [
        "region_classification",
        "_pos",
        "_mut_sig",
    ]
    _grp_var_cols = [c for c in _grp_var_cols if c in _marg_flat.columns]
    for gkeys, grp in _marg_flat.groupby(_grp_var_cols):
        gkeys_t = gkeys if isinstance(gkeys, tuple) else (gkeys,)
        _var_marg[gkeys_t] = len(set.union(*grp["inn_set"]))
        # Merge gene_counts across rows (different mutation_type rows)
        merged: dict[str, int] = {}
        for gc in grp["gene_counts"]:
            if isinstance(gc, dict):
                for gene, cnt in gc.items():
                    merged[gene] = merged.get(gene, 0) + cnt
        merged = dict(sorted(merged.items(), key=lambda x: (-x[1], x[0])))
        # strip last element (_mut_sig) for gene-detail key so callers use
        # strat_key + (rc, pos, msig)
        _gene_detail_map[gkeys_t] = merged if len(merged) > 1 else {}
        if merged:
            missing_gene_rank = [g for g in merged if g not in _gene_rank]
            assert not missing_gene_rank, (
                "Representative-gene ranking missing gene(s): "
                + ", ".join(missing_gene_rank)
            )
            _rep_gene_map[gkeys_t] = min(
                merged.keys(), key=lambda g: (_gene_rank[g], g)
            )
        else:
            raise AssertionError(
                f"Empty representative gene pool for marginal key: {gkeys_t}"
            )

    def _paired_species_fallback_strat_key(strat_key: tuple) -> tuple | None:
        """Return cat↔dog fallback strat key when gene_species is part of stratification."""
        if "gene_species" not in _strat_marg:
            return None
        species_idx = _strat_marg.index("gene_species")
        if species_idx >= len(strat_key):
            return None
        species_val = str(strat_key[species_idx]).strip().lower()
        species_fallback = {"cat": "dog", "dog": "cat"}.get(species_val)
        if species_fallback is None:
            return None
        alt = list(strat_key)
        alt[species_idx] = species_fallback
        return tuple(alt)

    def _resolve_lookup_key_with_species_fallback(
        mapping: dict, strat_key: tuple, suffix: tuple
    ) -> tuple:
        lookup = strat_key + suffix
        if lookup in mapping:
            return lookup
        fallback_strat = _paired_species_fallback_strat_key(strat_key)
        if fallback_strat is not None:
            fallback_lookup = fallback_strat + suffix
            if fallback_lookup in mapping:
                return fallback_lookup
        return lookup

    def _marginal(strat_key: tuple, rc, p) -> int:
        lookup = _resolve_lookup_key_with_species_fallback(
            _pos_marg, strat_key, (rc, p)
        )
        assert lookup in _pos_marg, f"Missing position marginal for key: {lookup}"
        return _pos_marg[lookup]

    def _var_marginal(strat_key: tuple, rc, p, m) -> int:
        lookup = _resolve_lookup_key_with_species_fallback(
            _var_marg, strat_key, (rc, p, m)
        )
        assert lookup in _var_marg, f"Missing variant marginal for key: {lookup}"
        return _var_marg[lookup]

    def _get_gene_detail(strat_key: tuple, rc, p, m) -> dict:
        lookup = _resolve_lookup_key_with_species_fallback(
            _gene_detail_map, strat_key, (rc, p, m)
        )
        assert lookup in _gene_detail_map, (
            f"Missing gene-detail marginal for key: {lookup}"
        )
        return _gene_detail_map[lookup]

    def _missing_rep_gene_debug_message(lookup: tuple) -> str:
        """Build a high-signal debug message for missing representative-gene keys."""
        base = list(lookup[:-1])  # strat + rc + pos
        mutation = lookup[-1]
        same_base_keys = [
            k
            for k in _rep_gene_map.keys()
            if len(k) == len(lookup) and list(k[:-1]) == base
        ]
        same_base_mutations = sorted({k[-1] for k in same_base_keys})

        # Fallback diagnostics when pos typing/normalization differs.
        same_strat_rc = [
            k
            for k in _rep_gene_map.keys()
            if len(k) == len(lookup)
            and k[: len(lookup) - 2] == lookup[: len(lookup) - 2]
        ]
        pos_examples = sorted({k[-2] for k in same_strat_rc})
        mutation_examples = sorted({k[-1] for k in same_strat_rc})[:20]

        return (
            f"Missing representative gene for key: {lookup}\n"
            f"  - variant marginal exists: {lookup in _var_marg}\n"
            f"  - gene-detail marginal exists: {lookup in _gene_detail_map}\n"
            f"  - same strat+region+pos key count: {len(same_base_keys)}\n"
            f"  - mutations at same strat+region+pos: {same_base_mutations[:20]}\n"
            f"  - requested mutation: {mutation}\n"
            f"  - same strat+region key count: {len(same_strat_rc)}\n"
            f"  - position examples under same strat+region: {pos_examples[:20]}\n"
            f"  - mutation examples under same strat+region: {mutation_examples}"
        )

    def _get_rep_gene(strat_key: tuple, rc, p, m) -> str:
        lookup = _resolve_lookup_key_with_species_fallback(
            _rep_gene_map, strat_key, (rc, p, m)
        )
        assert lookup in _rep_gene_map, _missing_rep_gene_debug_message(lookup)
        return _rep_gene_map[lookup]

    # --- Pair generation ---
    rows = []

    def _to_events(grp: pd.DataFrame) -> list[MutEvent]:
        """All unique (pos, germline, mutated) mutations within a group."""
        return [
            MutEvent(
                row["_pos"],
                row["mutation_type"],
                row["germline_seq"],
                row["mutated_seq"],
                row["region_classification"],
            )
            for _, row in grp.drop_duplicates(
                subset=["_pos", "germline_seq", "mutated_seq"]
            ).iterrows()
        ]

    def _classify_intra(e0: MutEvent, e1: MutEvent) -> str:
        if e0.EU == e1.EU:
            if e0.germline_seq == e1.germline_seq and e0.mutated_seq == e1.mutated_seq:
                return "intra_repeat"
            return "intra_pos_repeat_diff_mut"
        return "intra"

    def _classify_inter(e0: MutEvent, e1: MutEvent, g0: str, g1: str) -> str:
        if (
            e0.EU == e1.EU
            and e0.germline_seq == e1.germline_seq
            and e0.mutated_seq == e1.mutated_seq
        ):
            return "shared_mutation_cross_chain"
        if g0 == g1:
            return "inter_same_gene"
        if g0 != g1:
            return "inter_cross_gene"
        return "unexpected"

    # Intra: pairs within same (inn, chain_id, gene)
    for (inn, chain_id, gene), grp in flat.groupby(["inn", "chain_id", "gene"]):
        assert inn in inn_species_map, f"Missing species mapping for INN: {inn}"
        species = inn_species_map[inn]
        sk = _strat_key_from_gene(gene, species)
        events = _to_events(grp)
        for e0, e1 in combinations(events, 2):
            classification = _classify_intra(e0, e1)
            m0 = f"{e0.germline_seq}->{e0.mutated_seq}"
            m1 = f"{e1.germline_seq}->{e1.mutated_seq}"
            rg0 = _get_rep_gene(sk, e0.region_classification, e0.EU, m0)
            rg1 = _get_rep_gene(sk, e1.region_classification, e1.EU, m1)
            gd0 = _get_gene_detail(sk, e0.region_classification, e0.EU, m0)
            gd1 = _get_gene_detail(sk, e1.region_classification, e1.EU, m1)
            rows.append(
                (
                    classification,
                    species,
                    rg0,
                    e0.region_classification,
                    e0.EU,
                    m0,
                    gd0,
                    rg1,
                    e1.region_classification,
                    e1.EU,
                    m1,
                    gd1,
                    inn,
                )
            )

    # Inter: pairs across different (chain_id, gene) within same INN
    for inn, grp in flat.groupby("inn"):
        assert inn in inn_species_map, f"Missing species mapping for INN: {inn}"
        species = inn_species_map[inn]
        chain_gene = (
            grp[["chain_id", "gene"]]
            .drop_duplicates()
            .sort_values("chain_id")
            .values.tolist()
        )
        if len(chain_gene) < 2:
            continue
        events_by_chain = {
            (cid, g): _to_events(grp[(grp["chain_id"] == cid) & (grp["gene"] == g)])
            for cid, g in chain_gene
        }
        for (c0, g0), (c1, g1) in combinations(chain_gene, 2):
            sk0 = _strat_key_from_gene(g0, species)
            sk1 = _strat_key_from_gene(g1, species)
            for e0, e1 in iproduct(
                events_by_chain[(c0, g0)], events_by_chain[(c1, g1)]
            ):
                classification = _classify_inter(e0, e1, g0, g1)
                m0 = f"{e0.germline_seq}->{e0.mutated_seq}"
                m1 = f"{e1.germline_seq}->{e1.mutated_seq}"
                rg0 = _get_rep_gene(sk0, e0.region_classification, e0.EU, m0)
                rg1 = _get_rep_gene(sk1, e1.region_classification, e1.EU, m1)
                gd0 = _get_gene_detail(sk0, e0.region_classification, e0.EU, m0)
                gd1 = _get_gene_detail(sk1, e1.region_classification, e1.EU, m1)
                # Canonical order: sort by (rep_gene, pos)
                if (rg0, e0.EU) > (rg1, e1.EU):
                    rg0, rg1, e0, e1, m0, m1, gd0, gd1 = (
                        rg1,
                        rg0,
                        e1,
                        e0,
                        m1,
                        m0,
                        gd1,
                        gd0,
                    )
                rows.append(
                    (
                        classification,
                        species,
                        rg0,
                        e0.region_classification,
                        e0.EU,
                        m0,
                        gd0,
                        rg1,
                        e1.region_classification,
                        e1.EU,
                        m1,
                        gd1,
                        inn,
                    )
                )

    # --- Aggregation ---
    _GCOLS = [
        "classification",
        "gene_species",
        "gene_0",
        "region_0",
        "p0",
        "gene_1",
        "region_1",
        "p1",
    ]
    _VAR_STAT_COLS = [
        "n_variants",
        "var_count_max",
        "var_count_min",
        "var_lift_max",
        "var_lift_min",
        "var_lift_mean",
        "var_log_lift_max",
        "var_log_lift_min",
        "var_log_lift_mean",
        "var_npmi_max",
        "var_npmi_min",
        "var_npmi_mean",
        "var_or_max",
        "var_or_min",
        "var_or_mean",
        "var_pvalue_min",
        "var_pvalue_max",
        "var_pvalue_mean",
        "var_min_conf_max",
        "var_min_conf_min",
        "var_min_conf_mean",
        "var_cosine_max",
        "var_cosine_min",
        "var_cosine_mean",
        "var_kulczynski_max",
        "var_kulczynski_min",
        "var_kulczynski_mean",
        "top_var_sig",
        "top_var_count",
        "top_var_n_p0",
        "top_var_n_p1",
        "top_var_lift",
        "top_var_log_lift",
        "top_var_npmi",
        "top_var_or",
        "top_var_pvalue",
        "top_var_min_conf",
        "top_var_cosine",
        "top_var_kulczynski",
        "mutation_pair_stats",
    ]
    _empty_cols = (
        _GCOLS
        + [
            "gene_detail_0",
            "gene_detail_1",
            "mutations",
            "inns",
            "count",
            "n_p0",
            "n_p1",
            "n_total",
            "expected",
            "lift",
            "log_lift",
            "npmi",
            "min_conf",
            "cosine",
            "kulczynski",
            "odds_ratio",
            "pvalue",
        ]
        + _VAR_STAT_COLS
    )
    if not rows:
        return pd.DataFrame(columns=_empty_cols), marginals_table

    pair_df = pd.DataFrame(
        rows,
        columns=[
            "classification",
            "gene_species",
            "gene_0",
            "region_0",
            "p0",
            "mut_0",
            "gene_detail_0",
            "gene_1",
            "region_1",
            "p1",
            "mut_1",
            "gene_detail_1",
            "inn",
        ],
    )

    # Gene detail aggregation: collect unique gene_detail dicts per pair group
    gene_detail_agg = (
        pair_df.groupby(_GCOLS)
        .agg(
            gene_detail_0=("gene_detail_0", "first"),
            gene_detail_1=("gene_detail_1", "first"),
        )
        .reset_index()
    )

    co_counts = pair_df.groupby(_GCOLS)["inn"].nunique().reset_index(name="count")
    co_counts = co_counts[co_counts["count"] >= min_count]

    # Mutation detail: {"T->S;A->C": count}
    mut_agg = (
        pair_df.assign(mut_pair=pair_df["mut_0"] + ";" + pair_df["mut_1"])
        .groupby(_GCOLS + ["mut_pair"])["inn"]
        .nunique()
        .reset_index(name="n")
        .groupby(_GCOLS)
        .apply(
            lambda g: dict(sorted(zip(g["mut_pair"], g["n"]), key=lambda x: -x[1])),
            include_groups=False,
        )
        .rename("mutations")
    )

    # Variant-level pair counts (unique INNs per specific mutation pair)
    var_pair_counts = pair_df.groupby(
        [
            "classification",
            "gene_species",
            "gene_0",
            "region_0",
            "p0",
            "mut_0",
            "gene_1",
            "region_1",
            "p1",
            "mut_1",
        ]
    )["inn"].nunique()

    inn_sets = pair_df.groupby(_GCOLS)["inn"].apply(set).rename("inns")

    def _pmi_stats(n_ab, n_a, n_b, n_total):
        if n_a <= 0 or n_b <= 0 or n_ab <= 0 or n_total <= 0:
            return float("nan"), float("nan")
        p_ab = n_ab / n_total
        # NPMI is undefined when p_ab is 0 or 1 (denominator is 0 at p_ab == 1).
        if not (0.0 < p_ab < 1.0):
            return float("nan"), float("nan")

        with np.errstate(divide="ignore", invalid="ignore"):
            ll = np.log(p_ab / ((n_a / n_total) * (n_b / n_total)))
            denom = -np.log(p_ab)
            npmi = ll / denom if denom > 0 else float("nan")

        if not np.isfinite(ll) or not np.isfinite(npmi):
            return float("nan"), float("nan")
        return round(float(ll), 6), round(float(npmi), 6)

    results = []
    for _, r in co_counts.iterrows():
        cls = r["classification"]
        sp = r["gene_species"]
        g0, rc0, p0 = r["gene_0"], r["region_0"], r["p0"]
        g1, rc1, p1 = r["gene_1"], r["region_1"], r["p1"]
        key = (cls, sp, g0, rc0, p0, g1, rc1, p1)
        n_ab = int(r["count"])

        # ── Strat-keyed lookups (scoped per gene × species) ───────────────
        sk0 = _strat_key_from_gene(g0, sp)
        sk1 = _strat_key_from_gene(g1, sp)

        # N denominator: strict lookup in requested stratum (or species when strat is species)
        assert sp in n_total_by_species, (
            f"Missing species denominator for species: {sp}"
        )
        if _n_lookup:
            _n_key = tuple(str(v).upper() for v in sk0)
            assert _n_key in _n_lookup, (
                f"Missing stratified denominator for key: {_n_key}"
            )
            nt = _n_lookup[_n_key]
        else:
            nt = n_total_by_species[sp]
        n_a = _marginal(sk0, rc0, p0)
        n_b = _marginal(sk1, rc1, p1)
        a, b = n_ab, max(n_a - n_ab, 0)
        c, d = max(n_b - n_ab, 0), max(nt - n_a - n_b + n_ab, 0)
        oddsratio, pvalue = fisher_exact([[a, b], [c, d]], alternative="greater")
        expected = (n_a * n_b) / nt if nt > 0 else 0
        lift = (
            (n_ab / nt) / ((n_a / nt) * (n_b / nt))
            if n_a > 0 and n_b > 0
            else float("nan")
        )
        log_lift, npmi = _pmi_stats(n_ab, n_a, n_b, nt)

        # Gene detail for this pair group
        gd_row = gene_detail_agg[
            (gene_detail_agg["classification"] == cls)
            & (gene_detail_agg["gene_species"] == sp)
            & (gene_detail_agg["gene_0"] == g0)
            & (gene_detail_agg["region_0"] == rc0)
            & (gene_detail_agg["p0"] == p0)
            & (gene_detail_agg["gene_1"] == g1)
            & (gene_detail_agg["region_1"] == rc1)
            & (gene_detail_agg["p1"] == p1)
        ]
        assert len(gd_row) > 0, f"Missing gene detail row for pair key: {key}"
        gd0 = gd_row["gene_detail_0"].iloc[0]
        gd1 = gd_row["gene_detail_1"].iloc[0]

        # Per-variant statistics
        assert key in mut_agg.index, f"Missing mutation aggregation for pair key: {key}"
        _mut_map = mut_agg.loc[key]
        var_stats = []
        mutation_pair_stats: dict = {}

        def _fmt_mut_key(m: str, p) -> str:
            if "->" in m:
                wt, mt = m.split("->", 1)
                return f"{wt}{p}{mt}"
            return f"?{p}?"

        for sig in _mut_map:
            m0, m1 = sig.split(";", 1)
            _var_key = (cls, sp, g0, rc0, p0, m0, g1, rc1, p1, m1)
            assert _var_key in var_pair_counts.index, (
                f"Missing variant pair count for key: {_var_key}"
            )
            n_ab_v = int(var_pair_counts.loc[_var_key])
            n_a_v = _var_marginal(sk0, rc0, p0, m0)
            n_b_v = _var_marginal(sk1, rc1, p1, m1)
            a_v, b_v = n_ab_v, max(n_a_v - n_ab_v, 0)
            c_v = max(n_b_v - n_ab_v, 0)
            d_v = max(nt - n_a_v - n_b_v + n_ab_v, 0)
            or_v, pv_v = fisher_exact(
                [[a_v, b_v], [c_v, d_v]],
                alternative="greater",
            )
            lift_v = (
                (n_ab_v / nt) / ((n_a_v / nt) * (n_b_v / nt))
                if n_a_v > 0 and n_b_v > 0
                else float("nan")
            )
            log_lift_v, npmi_v = _pmi_stats(n_ab_v, n_a_v, n_b_v, nt)
            min_conf_v = (
                n_ab_v / min(n_a_v, n_b_v) if min(n_a_v, n_b_v) > 0 else float("nan")
            )
            cosine_v = (
                n_ab_v / (n_a_v * n_b_v) ** 0.5
                if n_a_v > 0 and n_b_v > 0
                else float("nan")
            )
            kulc_v = (
                0.5 * (n_ab_v / n_a_v + n_ab_v / n_b_v)
                if n_a_v > 0 and n_b_v > 0
                else float("nan")
            )
            var_stats.append(
                {
                    "count": n_ab_v,
                    "lift": lift_v,
                    "log_lift": log_lift_v,
                    "npmi": npmi_v,
                    "or": float(or_v),
                    "pvalue": pv_v,
                    "min_conf": min_conf_v,
                    "cosine": cosine_v,
                    "kulczynski": kulc_v,
                }
            )
            _pair_key = f"{_fmt_mut_key(m0, p0)}/{_fmt_mut_key(m1, p1)}"
            mutation_pair_stats[_pair_key] = {
                "count": n_ab_v,
                "n_p0": n_a_v,
                "n_p1": n_b_v,
                "lift": round(lift_v, 3),
                "log_lift": log_lift_v,
                "npmi": npmi_v,
                "or": round(float(or_v), 3),
                "pvalue": round(float(pv_v), 10),
                "min_conf": round(min_conf_v, 6),
                "cosine": round(cosine_v, 6),
                "kulczynski": round(kulc_v, 6),
            }

        if var_stats:
            vs = pd.DataFrame(var_stats)
            var_agg_dict = {
                "n_variants": len(var_stats),
                "var_count_max": int(vs["count"].max()),
                "var_count_min": int(vs["count"].min()),
                "var_lift_max": round(vs["lift"].max(), 3),
                "var_lift_min": round(vs["lift"].min(), 3),
                "var_lift_mean": round(vs["lift"].mean(), 3),
                "var_log_lift_max": round(vs["log_lift"].max(), 6),
                "var_log_lift_min": round(vs["log_lift"].min(), 6),
                "var_log_lift_mean": round(vs["log_lift"].mean(), 6),
                "var_npmi_max": round(vs["npmi"].max(), 6),
                "var_npmi_min": round(vs["npmi"].min(), 6),
                "var_npmi_mean": round(vs["npmi"].mean(), 6),
                "var_or_max": round(vs["or"].max(), 3),
                "var_or_min": round(vs["or"].min(), 3),
                "var_or_mean": round(vs["or"].mean(), 3),
                "var_pvalue_min": round(vs["pvalue"].min(), 10),
                "var_pvalue_max": round(vs["pvalue"].max(), 10),
                "var_pvalue_mean": round(vs["pvalue"].mean(), 10),
                "var_min_conf_max": round(vs["min_conf"].max(), 6),
                "var_min_conf_min": round(vs["min_conf"].min(), 6),
                "var_min_conf_mean": round(vs["min_conf"].mean(), 6),
                "var_cosine_max": round(vs["cosine"].max(), 6),
                "var_cosine_min": round(vs["cosine"].min(), 6),
                "var_cosine_mean": round(vs["cosine"].mean(), 6),
                "var_kulczynski_max": round(vs["kulczynski"].max(), 6),
                "var_kulczynski_min": round(vs["kulczynski"].min(), 6),
                "var_kulczynski_mean": round(vs["kulczynski"].mean(), 6),
            }
        else:
            var_agg_dict = {k: float("nan") for k in _VAR_STAT_COLS}

        top_var_dict = {
            "top_var_sig": float("nan"),
            "top_var_count": float("nan"),
            "top_var_n_p0": float("nan"),
            "top_var_n_p1": float("nan"),
            "top_var_lift": float("nan"),
            "top_var_log_lift": float("nan"),
            "top_var_npmi": float("nan"),
            "top_var_or": float("nan"),
            "top_var_pvalue": float("nan"),
            "top_var_min_conf": float("nan"),
            "top_var_cosine": float("nan"),
            "top_var_kulczynski": float("nan"),
        }
        if _mut_map:
            top_sig = max(_mut_map, key=_mut_map.get)
            tm0, tm1 = top_sig.split(";", 1)
            _top_key = (cls, sp, g0, rc0, p0, tm0, g1, rc1, p1, tm1)
            assert _top_key in var_pair_counts.index, (
                f"Missing top-variant pair count for key: {_top_key}"
            )
            tn_ab = int(var_pair_counts.loc[_top_key])
            if top_sig:
                tn_a = _var_marginal(sk0, rc0, p0, tm0)
                tn_b = _var_marginal(sk1, rc1, p1, tm1)
                ta, tb = tn_ab, max(tn_a - tn_ab, 0)
                tc = max(tn_b - tn_ab, 0)
                td = max(nt - tn_a - tn_b + tn_ab, 0)
                tor, tpv = fisher_exact(
                    [[ta, tb], [tc, td]],
                    alternative="greater",
                )
                tlift = (
                    (tn_ab / nt) / ((tn_a / nt) * (tn_b / nt))
                    if tn_a > 0 and tn_b > 0
                    else float("nan")
                )
                tlog_lift, tnpmi = _pmi_stats(tn_ab, tn_a, tn_b, nt)
                tmin_conf = (
                    tn_ab / min(tn_a, tn_b) if min(tn_a, tn_b) > 0 else float("nan")
                )
                tcos = (
                    tn_ab / (tn_a * tn_b) ** 0.5
                    if tn_a > 0 and tn_b > 0
                    else float("nan")
                )
                tkulc = (
                    0.5 * (tn_ab / tn_a + tn_ab / tn_b)
                    if tn_a > 0 and tn_b > 0
                    else float("nan")
                )
                top_var_dict = {
                    "top_var_sig": top_sig,
                    "top_var_count": int(tn_ab),
                    "top_var_n_p0": int(tn_a),
                    "top_var_n_p1": int(tn_b),
                    "top_var_lift": round(tlift, 3),
                    "top_var_log_lift": tlog_lift,
                    "top_var_npmi": tnpmi,
                    "top_var_or": round(float(tor), 3),
                    "top_var_pvalue": round(float(tpv), 10),
                    "top_var_min_conf": round(tmin_conf, 6),
                    "top_var_cosine": round(tcos, 6),
                    "top_var_kulczynski": round(tkulc, 6),
                }

        # Null-invariant association measures
        min_conf = n_ab / min(n_a, n_b) if min(n_a, n_b) > 0 else float("nan")
        cosine = n_ab / (n_a * n_b) ** 0.5 if n_a > 0 and n_b > 0 else float("nan")
        kulc = 0.5 * (n_ab / n_a + n_ab / n_b) if n_a > 0 and n_b > 0 else float("nan")

        results.append(
            {
                "classification": cls,
                "gene_species": sp,
                "gene_0": g0,
                "region_0": rc0,
                "p0": p0,
                "gene_detail_0": gd0,
                "gene_1": g1,
                "region_1": rc1,
                "p1": p1,
                "gene_detail_1": gd1,
                "mutations": _mut_map,
                "inns": inn_sets.loc[key],
                "count": n_ab,
                "n_p0": n_a,
                "n_p1": n_b,
                "n_total": nt,
                "expected": round(expected, 3),
                "lift": round(lift, 3),
                "log_lift": log_lift,
                "npmi": npmi,
                "min_conf": round(min_conf, 4),
                "cosine": round(cosine, 4),
                "kulczynski": round(kulc, 4),
                "odds_ratio": round(float(oddsratio), 3),
                "pvalue": round(pvalue, 10),
                **var_agg_dict,
                **top_var_dict,
                "mutation_pair_stats": mutation_pair_stats,
            }
        )

    co_df = (
        pd.DataFrame(results)
        .sort_values("count", ascending=False)
        .reset_index(drop=True)
    )

    # --- Post-process: fix gene assignments for cross-gene mismatches ---
    # If gene_0 != gene_1 but gene_detail shows the other gene is present,
    # reassign to make them consistent.
    mismatch = co_df["gene_0"] != co_df["gene_1"]
    for idx in co_df[mismatch].index:
        g0, g1 = co_df.at[idx, "gene_0"], co_df.at[idx, "gene_1"]
        gd0, gd1 = (
            co_df.at[idx, "gene_detail_0"],
            co_df.at[idx, "gene_detail_1"],
        )
        # Prefer making them the same: if gene_1 is in gene_detail_0, use gene_1
        if isinstance(gd0, dict) and g1 in gd0:
            co_df.at[idx, "gene_0"] = g1
        elif isinstance(gd1, dict) and g0 in gd1:
            co_df.at[idx, "gene_1"] = g0

    # Fix classification after gene reassignment
    now_same = (co_df["gene_0"] == co_df["gene_1"]) & (
        co_df["classification"] == "inter_cross_gene"
    )
    co_df.loc[now_same, "classification"] = "inter_same_gene"

    return co_df, marginals_table


def _canon_pair_key(gene_0, p0, gene_1, p1) -> tuple[tuple, bool]:
    """Canonical (position-sorted) pair key.

    Pairs are sorted numerically by position so that ``(gene, p=234)`` before
    ``(gene, p=235)`` regardless of which side of the pair they appear on.

    Returns ``(canon_key, swapped)`` where *swapped* indicates the endpoints
    were reversed relative to the input order.
    """
    try:
        swapped = float(p0) > float(p1)
    except (TypeError, ValueError):
        swapped = str(p0) > str(p1)
    if swapped:
        return (str(gene_1), str(p1), str(gene_0), str(p0)), True
    return (str(gene_0), str(p0), str(gene_1), str(p1)), False


def _canon_sig(sig: str, swapped: bool) -> str:
    """Reverse ``'mut0;mut1'`` signature when pair endpoints were swapped."""
    if not swapped:
        return sig
    parts = sig.split(";", 1)
    return f"{parts[1]};{parts[0]}" if len(parts) == 2 else sig


def remove_intra_pairs(
    co_df: pd.DataFrame,
    stratify_by: List[str] | None = None,
) -> pd.DataFrame:
    """Reclassify inter pairs whose mutation signatures also appear as intra pairs.

    Pairs are matched by *canonical* position-sorted key so that
    ``(gene, p=235, gene, p=234)`` and ``(gene, p=234, gene, p=235)`` are
    treated as the same pair regardless of storage order.

    Matching is scoped to the same stratification group (default:
    ``["gene_species"]``) so that an intra pair in ``human`` cannot filter
    an inter pair in ``dog``.  Pass the same ``stratify_by`` columns used
    when calling ``compute_co_mutation_frequencies``.

    For each inter row whose canonical position pair also appears in intra rows
    *within the same stratification group*, shared mutation signatures are
    removed. If ALL signatures are removed the row is reclassified as
    ``filtered_by_intra_match``. If some remain, the row keeps its original
    classification with updated statistics.

    Returns the FULL co_df (all classifications including intra rows) with
    affected inter rows reclassified in-place.

    Parameters
    ----------
    co_df:
        Full output of ``compute_co_mutation_frequencies`` (all classifications).
    stratify_by:
        Columns that define a stratification group.  Default ``["gene_species"]``.
        Pass the same list you used in ``stratify_n_by`` (minus any derived
        columns). ``"gene"`` is interpreted as pair-side ``"gene_0"`` and
        ``"isotype"`` is derived from ``gene_0[:4].upper()``.
    """
    internal_required_cols = {
        "classification",
        "gene_0",
        "p0",
        "gene_1",
        "p1",
        "mutations",
        "n_total",
        "n_p0",
        "n_p1",
        "count",
    }
    export_required_cols = {
        "pair_type",
        "species",
        "position_1",
        "position_2",
        "gene_position_1",
        "region_position_1",
        "gene_position_2",
        "region_position_2",
        "num_unique_inn_co_occurring",
        "num_unique_inn_with_any_mut_position_1",
        "num_unique_inn_with_any_mut_position_2",
        "num_unique_inn_in_group",
        "all_mutation_pairs",
    }

    input_is_internal = internal_required_cols.issubset(co_df.columns)
    input_is_export = export_required_cols.issubset(co_df.columns)
    assert input_is_internal or input_is_export, (
        "remove_intra_pairs received unsupported table schema. "
        "Expected either compute_co_mutation_frequencies output columns "
        f"{sorted(internal_required_cols)} or export table columns "
        f"{sorted(export_required_cols)}."
    )

    def _parse_export_mut_token(tok: str, pos: str) -> str:
        s = str(tok).strip()
        p = str(pos).strip()
        assert len(s) >= 3, f"Invalid mutation token in export table: {tok!r}"
        wt = s[0]
        tail = s[1:]
        assert tail.startswith(p), (
            f"Mutation token {tok!r} does not match position {pos!r}"
        )
        mt = tail[len(p) :]
        assert mt, f"Missing mutated residue in mutation token: {tok!r}"
        return f"{wt}->{mt}"

    def _parse_all_mutation_pairs(raw, p0, p1) -> dict[str, int]:
        if raw is None or (isinstance(raw, float) and pd.isna(raw)):
            return {}
        s = str(raw).strip()
        if not s:
            return {}
        out: dict[str, int] = {}
        for entry in [x.strip() for x in s.split("|") if str(x).strip()]:
            parts = [x.strip() for x in entry.split(":", 1)]
            assert len(parts) == 2, f"Invalid all_mutation_pairs entry: {entry!r}"
            pair_label, count_str = parts
            pair_parts = [x.strip() for x in pair_label.split("/", 1)]
            assert len(pair_parts) == 2, f"Invalid mutation pair label: {pair_label!r}"
            m0 = _parse_export_mut_token(pair_parts[0], p0)
            m1 = _parse_export_mut_token(pair_parts[1], p1)
            cnt = int(count_str)
            assert cnt >= 0, f"Negative mutation-pair count in export table: {entry!r}"
            out[f"{m0};{m1}"] = cnt
        return out

    input_schema = "internal"
    working_df = co_df.copy()
    if input_is_export and not input_is_internal:
        input_schema = "export"
        rename_map = {
            "pair_type": "classification",
            "species": "gene_species",
            "position_1": "p0",
            "position_2": "p1",
            "gene_position_1": "gene_0",
            "region_position_1": "region_0",
            "gene_position_2": "gene_1",
            "region_position_2": "region_1",
            "num_unique_inn_co_occurring": "count",
            "num_unique_inn_with_any_mut_position_1": "n_p0",
            "num_unique_inn_with_any_mut_position_2": "n_p1",
            "num_unique_inn_in_group": "n_total",
            "min_confidence": "min_conf",
            "fisher_pvalue": "pvalue",
        }
        working_df = working_df.rename(columns=rename_map)
        working_df["mutations"] = working_df.apply(
            lambda r: _parse_all_mutation_pairs(
                r["all_mutation_pairs"], r["p0"], r["p1"]
            ),
            axis=1,
        )
        working_df["top_var_sig"] = working_df["mutations"].apply(
            lambda d: max(d, key=d.get) if isinstance(d, dict) and d else float("nan")
        )
        working_df["top_var_count"] = working_df["mutations"].apply(
            lambda d: (
                int(max(d.values())) if isinstance(d, dict) and d else float("nan")
            )
        )

    _strat_cols = stratify_by or ["gene_species"]
    if input_schema == "export":
        _strat_export_to_internal = {"species": "gene_species"}
        _strat_cols = [_strat_export_to_internal.get(c, c) for c in _strat_cols]

    # remove_intra_pairs operates on pair-level rows where "gene" is represented
    # as gene_0/gene_1. For grouping symmetry with the co-mutation builder, use
    # gene_0 as the single-gene stratification proxy, and derive isotype from it.
    _strat_aliases = {"gene": "gene_0"}
    _resolved_strat_cols: list[str] = []
    for c in _strat_cols:
        resolved = _strat_aliases.get(c, c)
        if resolved == "isotype":
            if "_isotype" not in working_df.columns:
                working_df["_isotype"] = (
                    working_df["gene_0"].astype(str).str[:4].str.upper()
                )
            resolved = "_isotype"
        _resolved_strat_cols.append(resolved)
    _strat_cols = _resolved_strat_cols
    missing_strat = [c for c in _strat_cols if c not in working_df.columns]
    assert not missing_strat, (
        "remove_intra_pairs missing stratification column(s): "
        + ", ".join(missing_strat)
    )

    intra_mask = working_df["classification"].isin(
        ["intra", "intra_repeat", "intra_pos_repeat_diff_mut"]
    )
    inter_mask = working_df["classification"].isin(
        [
            "inter_same_gene",
            "inter_cross_gene",
            "shared_mutation_cross_chain",
            "unexpected",
        ]
    )

    intra_df = working_df[intra_mask]
    inter_df = working_df[inter_mask]

    # Build lookup: (strat_values..., canon_pos_key) -> set of canonical mutation sigs
    # strat_values are the values of _strat_cols for that row — scopes matching
    intra_sigs: dict[tuple, set] = {}
    for _, row in intra_df.iterrows():
        ckey, swapped = _canon_pair_key(
            row["gene_0"], row["p0"], row["gene_1"], row["p1"]
        )
        assert isinstance(row["mutations"], dict), (
            "Expected dict in 'mutations' for intra row, got "
            f"{type(row['mutations']).__name__}"
        )
        strat_vals = tuple(str(row[c]) for c in _strat_cols)
        full_key = strat_vals + ckey
        for sig in row["mutations"].keys():
            intra_sigs.setdefault(full_key, set()).add(_canon_sig(sig, swapped))

    updated = []
    for _, r in inter_df.iterrows():
        assert isinstance(r["mutations"], dict), (
            "Expected dict in 'mutations' for inter row, got "
            f"{type(r['mutations']).__name__}"
        )
        muts = dict(r["mutations"])
        ckey, swapped = _canon_pair_key(r["gene_0"], r["p0"], r["gene_1"], r["p1"])
        strat_vals = tuple(str(r[c]) for c in _strat_cols)
        full_key = strat_vals + ckey
        if full_key in intra_sigs:
            for sig in list(muts.keys()):
                if _canon_sig(sig, swapped) in intra_sigs[full_key]:
                    muts.pop(sig, None)

        if not muts:
            # All signatures matched intra — reclassify
            updated_row = r.to_dict()
            updated_row["classification"] = "filtered_by_intra_match"
            updated_row["top_var_sig"] = float("nan")
            updated_row["top_var_count"] = float("nan")
            updated.append(updated_row)
            continue

        if muts == r["mutations"]:
            # No change — keep as-is
            updated.append(r.to_dict())
            continue

        # Partial removal — recompute statistics
        n_ab = sum(muts.values())
        n_total = int(r["n_total"])
        n_a, n_b = int(r["n_p0"]), int(r["n_p1"])
        a, b = n_ab, max(n_a - n_ab, 0)
        c, d = max(n_b - n_ab, 0), max(n_total - n_a - n_b + n_ab, 0)
        oddsratio, pvalue = fisher_exact([[a, b], [c, d]], alternative="greater")
        expected = (n_a * n_b) / n_total if n_total > 0 else 0
        lift = (
            (n_ab / n_total) / ((n_a / n_total) * (n_b / n_total))
            if n_a > 0 and n_b > 0
            else float("nan")
        )
        p_ab = n_ab / n_total
        log_lift = (
            float(np.log(lift)) if np.isfinite(lift) and lift > 0 else float("nan")
        )
        npmi = (
            round(log_lift / (-np.log(p_ab)), 6)
            if np.isfinite(log_lift) and p_ab < 1
            else float("nan")
        )
        updated.append(
            {
                **r.to_dict(),
                "mutations": muts,
                "count": n_ab,
                "top_var_sig": max(muts, key=muts.get),
                "top_var_count": int(max(muts.values())),
                "expected": round(expected, 3),
                "lift": round(lift, 3),
                "log_lift": (round(log_lift, 6) if np.isfinite(log_lift) else log_lift),
                "npmi": npmi,
                "odds_ratio": round(float(oddsratio), 3),
                "pvalue": round(pvalue, 10),
            }
        )

    # Combine: intra rows (unchanged) + processed inter rows
    result = pd.concat(
        [
            intra_df,
            pd.DataFrame(updated),
        ],
        ignore_index=True,
    )

    result = result.sort_values("count", ascending=False).reset_index(drop=True)
    if input_schema == "export":
        drop_conflicting_export_cols = [
            c
            for c in [
                "top_mutation_pair",
                "top_mutation_position_1",
                "top_mutation_position_2",
            ]
            if c in result.columns
        ]
        if drop_conflicting_export_cols:
            result = result.drop(columns=drop_conflicting_export_cols)
        return co_df_to_export_table(result)
    return result


# ---------------------------------------------------------------------------
# Hierarchical co-mutation graph
# ---------------------------------------------------------------------------


def build_comutation_hierarchy(
    pair_df: pd.DataFrame,
    min_lift: float = 1.5,
    max_pvalue: float = 0.05,
    min_count: int = 2,
) -> Tuple[nx.DiGraph, nx.Graph]:
    """Build a hierarchical DAG of co-occurring specific mutation events.

    Algorithm
    ---------
    1. Filter ``pair_df`` to statistically significant pairs (position level).
    2. Explode the ``mutations`` dict so each edge is a specific event pair
       ``(gene, position, "A->B")`` — ``(gene, position, "C->D")``.
    3. Compute per-event marginals from the exploded data and re-derive PMI
       = log(lift) at the mutation-event level.
    4. Build weighted co-occurrence graph *G* and enumerate maximal cliques
       via Bron-Kerbosch; each clique is a set of mutation events that all
       mutually co-occur.
    5. Score every clique by mean pairwise PMI and construct a Hasse DAG *H*.

    Parameters
    ----------
    pair_df:
        Output of ``compute_co_mutation_frequencies()``.
    min_lift:
        Minimum position-level lift to pre-filter pairs.
    max_pvalue:
        Maximum p-value threshold (position level).
    min_count:
        Minimum co-occurrence count at the specific mutation-event level.

    Returns
    -------
    H : nx.DiGraph
        Hierarchical DAG.  Node ``type`` attribute:

        * ``"clique"`` — group node.  Attrs: ``members`` (set of event tuples),
          ``size``, ``score`` (mean pairwise PMI).
        * ``"mutation"`` — leaf ``(gene, position, "A->B")``.

        Edge ``kind`` attribute: ``"contains"`` or ``"subsumes"``.
    G : nx.Graph
        Raw event-level co-occurrence graph; edge attrs: ``pmi``, ``lift``,
        ``count``, ``joint_prob``.
    """
    sig = pair_df[
        (pair_df["lift"] >= min_lift)
        & (pair_df["pvalue"] <= max_pvalue)
        & (pair_df["count"] >= min_count)
    ].copy()

    if sig.empty:
        return nx.DiGraph(), nx.Graph()

    n_total = int(sig["n_total"].iloc[0])

    # ------------------------------------------------------------------ #
    # 1. Explode mutations dict → specific event pairs with counts        #
    # ------------------------------------------------------------------ #
    exploded: list[dict] = []
    for _, row in sig.iterrows():
        for mut_sig, k in row["mutations"].items():
            if k < min_count:
                continue
            parts = mut_sig.split(";", 1)
            m0 = parts[0] if len(parts) > 0 else ""
            m1 = parts[1] if len(parts) > 1 else ""
            exploded.append(
                {
                    "e0": (row["gene_0"], int(row["p0"]), m0),
                    "e1": (row["gene_1"], int(row["p1"]), m1),
                    "count": int(k),
                }
            )

    if not exploded:
        return nx.DiGraph(), nx.Graph()

    # ------------------------------------------------------------------ #
    # 2. Marginals per mutation event (sum of co-occurrence counts)       #
    # ------------------------------------------------------------------ #
    # This sums pair counts — a conservative approximation of the true
    # marginal frequency (slightly overcounts when an event appears in many
    # pairs, but relative PMI ordering is preserved).
    marginals: dict[tuple, int] = {}
    for r in exploded:
        marginals[r["e0"]] = marginals.get(r["e0"], 0) + r["count"]
        marginals[r["e1"]] = marginals.get(r["e1"], 0) + r["count"]

    # ------------------------------------------------------------------ #
    # 3. Co-occurrence graph at the mutation-event level                  #
    # ------------------------------------------------------------------ #
    G: nx.Graph = nx.Graph()
    for r in exploded:
        e0, e1, k = r["e0"], r["e1"], r["count"]
        m0, m1 = marginals[e0], marginals[e1]
        lift = (k / n_total) / ((m0 / n_total) * (m1 / n_total)) if m0 and m1 else 1.0
        pmi = float(np.log(max(lift, 1e-9)))
        if G.has_edge(e0, e1):
            # Multiple position pairs can produce the same event pair; sum counts
            G[e0][e1]["count"] += k
        else:
            G.add_edge(
                e0,
                e1,
                lift=round(lift, 3),
                count=k,
                pmi=pmi,
                joint_prob=round(k / n_total, 5),
            )

    if G.number_of_edges() == 0:
        return nx.DiGraph(), G

    # ------------------------------------------------------------------ #
    # 4. Maximal cliques                                                   #
    # ------------------------------------------------------------------ #
    cliques = [frozenset(c) for c in nx.find_cliques(G) if len(c) >= 2]

    def _score(clique: frozenset) -> float:
        pmis = [G[u][v]["pmi"] for u, v in combinations(clique, 2) if G.has_edge(u, v)]
        return float(np.mean(pmis)) if pmis else 0.0

    # ------------------------------------------------------------------ #
    # 5. Hasse DAG                                                         #
    # ------------------------------------------------------------------ #
    H: nx.DiGraph = nx.DiGraph()

    for clique in cliques:
        H.add_node(
            clique,
            type="clique",
            members=set(clique),
            size=len(clique),
            score=_score(clique),
        )

    for clique in cliques:
        for evt in clique:
            if evt not in H:
                H.add_node(evt, type="mutation")
            H.add_edge(clique, evt, kind="contains")

    sorted_cliques = sorted(cliques, key=len, reverse=True)
    for i, big in enumerate(sorted_cliques):
        for small in sorted_cliques[i + 1 :]:
            if not (small < big):
                continue
            if not any(
                small < mid < big for mid in sorted_cliques if mid not in (big, small)
            ):
                H.add_edge(big, small, kind="subsumes")

    return H, G


def select_top_seeds_and_filter(
    motif_df: pd.DataFrame,
    top_n: int,
    adjusted_count_col: str,
    inn_count_col: str,
    num_mutations_col: str,
    seed_col: str = "cluster_seeds",
    primary_seed_col: str = "primary_cluster_seeds",
    motif_col: str = "mutation_string",
    sub_motifs_col: str | None = "sub_motifs",
) -> tuple[pd.DataFrame, set, list]:
    """Select the top-N seed clusters and return a filtered motif_df with a
    ``primary_seed`` column.

    Delegates to ``build_exploded_motif_df`` in
    ``analytics.motifs.exploded`` so that the seed-ranking and
    primary-seed assignment are identical to ``plot_motif_cluster_bubbles``.

    Returns
    -------
    (filtered_df, top_seeds, ranked_seeds)
    """
    from inn_analysis_ops.analytics.motifs.exploded import build_exploded_motif_df

    exploded_df = build_exploded_motif_df(
        motif_df,
        top_n=top_n,
        adjusted_count_col=adjusted_count_col,
        inn_count_col=inn_count_col,
        num_mutations_col=num_mutations_col,
        mutation_pe_col=motif_col,
        sub_motifs_col=sub_motifs_col,
    )

    top_seeds: set = set(
        exploded_df.loc[exploded_df["is_primary_seed_group"], "seed_group"]
    )
    _seed_inn = (
        exploded_df[exploded_df["is_a_seed"]]
        .drop_duplicates("seed_group")
        .set_index("seed_group")[inn_count_col]
        .to_dict()
    )
    ranked_seeds: list = sorted(
        top_seeds, key=lambda s: _seed_inn.get(s, 0), reverse=True
    )

    filtered_df = (
        exploded_df[exploded_df["is_primary_seed_group"]]
        .drop_duplicates(motif_col)
        .drop(columns=["seed_group", "is_a_seed", "is_primary_seed_group"])
        .reset_index(drop=True)
    )
    return filtered_df, top_seeds, ranked_seeds


def _column_or_index_values(df: pd.DataFrame, name: str) -> pd.Series:
    """Return the values of *name* in *df*, whether it's a regular column or
    a named index level — covers both a simple named ``Index`` and any level
    of a ``MultiIndex`` (e.g. an ``inn_df`` indexed by INN name, or by
    ``(inn_name, other)``). If *name* is both a column and an index level,
    the column takes precedence.

    The returned Series always carries a fresh positional ``0..n-1`` index
    (not *df*'s original index/labels), so results from multiple calls on
    the same *df* can be combined/compared position-wise without needing to
    align on mismatched index labels first.

    Raises
    ------
    KeyError
        If *name* is neither a column nor an index level — never a silent
        empty/missing fallback.
    """
    if name in df.columns:
        return df[name].reset_index(drop=True)
    if name in list(df.index.names):
        return pd.Series(df.index.get_level_values(name)).reset_index(drop=True)
    raise KeyError(
        f"{name!r} not found as a column or index level. Available columns: "
        f"{sorted(df.columns.tolist())}; index level names: {list(df.index.names)}"
    )


def _resolve_inn_flat(
    inn_df: pd.DataFrame,
    inn_name_source: str,
    date_col: str | list[str],
) -> tuple[pd.DataFrame, dict, set]:
    """Build a resolved (inn_name, _year) table from *inn_df*.

    Supports ``date_col`` as a single column name or a preference-ordered list:
    the first non-null value found across the list is used for each row.
    ``inn_name_source`` and every ``date_col`` candidate may each be either a
    regular column or a named index level (including a level of a
    MultiIndex) — resolved via ``_column_or_index_values``.

    Returns
    -------
    (inn_flat, year_inns, no_date_inns)
        ``inn_flat``      – DataFrame with columns [inn_name_source, "_year"]
                            containing only rows where ``_year`` is not null.
        ``year_inns``     – dict mapping year (int) → set of INN name strings.
        ``no_date_inns``  – set of INN names with no date across all columns.
    """
    _date_cols: list[str] = [date_col] if isinstance(date_col, str) else list(date_col)
    _all_names = set(inn_df.columns) | set(inn_df.index.names)
    _valid = [c for c in _date_cols if c in _all_names]

    _base = pd.DataFrame(
        {inn_name_source: _column_or_index_values(inn_df, inn_name_source)}
    )
    if _valid:
        _base["_year"] = _column_or_index_values(inn_df, _valid[0])
        for _c in _valid[1:]:
            _base["_year"] = _base["_year"].combine_first(
                _column_or_index_values(inn_df, _c)
            )
    else:
        _base["_year"] = None

    inn_flat = _base[[inn_name_source, "_year"]].dropna(subset=["_year"]).copy()
    inn_flat["_year"] = inn_flat["_year"].astype(int)

    year_inns: dict = inn_flat.groupby("_year")[inn_name_source].apply(set).to_dict()
    no_date_inns: set = set(_base.loc[_base["_year"].isna(), inn_name_source].dropna())
    if no_date_inns:
        year_inns[None] = no_date_inns

    return inn_flat, year_inns, no_date_inns


def _inn_names_with_date(
    inn_df: pd.DataFrame, inn_name_source: str, date_col: str | list[str]
) -> set:
    """Cheap set of INN names in *inn_df* with a valid (non-null) date across
    any ``date_col`` candidate — the narrow piece of ``_resolve_inn_flat``
    that ``motif_yearly_counts`` actually needs (to classify rare "gap rows"),
    without paying for the full year-binning machinery that function does
    (which only ``build_motif_timeline_table`` requires for its core
    per-year counting).

    ``inn_name_source``/``date_col`` may each be a column or a named index
    level — resolved via ``_column_or_index_values``, mirroring
    ``_resolve_inn_flat``.
    """
    _date_cols = [date_col] if isinstance(date_col, str) else list(date_col)
    _all_names = set(inn_df.columns) | set(inn_df.index.names)
    _valid = [c for c in _date_cols if c in _all_names]
    if not _valid:
        return set()
    _names = _column_or_index_values(inn_df, inn_name_source)
    _date_vals = pd.concat([_column_or_index_values(inn_df, c) for c in _valid], axis=1)
    _has_date = _date_vals.notna().any(axis=1)
    return set(_names[_has_date].dropna())


def _process_constant_chain_no_mut(
    result: "ConstantChainAnalysisResult",
    ds: "INNAnalysisDataset",
    species: str,
    gene: str,
    sequence_must_cover_regions: list[str] | None = None,
) -> pd.DataFrame:
    """Filter ``result.supp_tabl_constant_mut_summary`` down to INNs whose
    V-domain (variable-region) germline allele matches ``species``/``gene``
    — the correctly-scoped population for constant-chain mutation cross-
    referencing (e.g. only human IGHG1-derived antibodies), rather than
    every INN in the dataset regardless of germline background.

    Suitable as (or as the direct basis for) the ``constant_chain_info``
    argument to ``motif_yearly_counts``.

    Parameters
    ----------
    result:
        A ``ConstantChainAnalysisResult`` — supplies
        ``supp_tabl_constant_mut_summary`` (one row per INN, with an
        ``inn`` column, ``has_constant_chain_mutations``, and per-region
        mutation-count columns).
    ds:
        The corresponding ``INNAnalysisDataset`` — supplies
        ``constant_chains`` (the raw multi-level sequence/allele pivot used
        to resolve each INN's V-gene allele and, optionally, its constant-
        region sequence coverage).
    species:
        Constant-chain species to match against ``ds.constant_chains``'s
        ``constant_chain_species`` level, e.g. ``"human"``.
    gene:
        V-gene family to match against each INN's V-domain ``c_allele``
        (case-insensitive; matched against the allele string's ``*``-
        delimited gene-family prefix), e.g. ``"IGHG1"``.
    sequence_must_cover_regions:
        If given, an INN is only kept when at least one of these regions'
        ``sequence`` columns (in ``ds.constant_chains``) is non-null for it
        — e.g. ``["CH2", "CH3"]`` to require some resolved Fc sequence
        coverage before including the INN at all.

    Returns
    -------
    DataFrame
        The row-subset of ``result.supp_tabl_constant_mut_summary`` (same
        columns/schema) restricted to INNs matching ``species``/``gene``
        (and ``sequence_must_cover_regions`` if given).
    """
    gene = gene.upper()
    species = species.upper()
    constant_chain_mut_data = result.supp_tabl_constant_mut_summary
    raw_constant_chain_info = ds.constant_chains.swaplevel(axis=1)

    if sequence_must_cover_regions:
        raw_constant_chain_info = raw_constant_chain_info[
            ~raw_constant_chain_info["sequence"][sequence_must_cover_regions]
            .isnull()
            .all(axis=1)
        ].copy()

    rough_inns_by_gene_name = (
        raw_constant_chain_info.query('sub_domain=="V"')["c_allele"]
        .stack()
        .dropna()
        .groupby(by="inn")
        .apply(lambda x: sorted(set([y.split("*")[0].upper() for y in x])))
    )
    rough_inns_by_species_name = (
        raw_constant_chain_info.query('sub_domain=="V"')["c_allele_species"]
        .stack()
        .dropna()
        .groupby(by="inn")
        .apply(lambda x: sorted(set([y.upper() for y in x])))
    )

    has_at_least_one_gene_ref = rough_inns_by_gene_name.apply(lambda x: gene in x)
    has_at_least_one_species_ref = rough_inns_by_species_name.apply(
        lambda x: species in x
    )
    inns_with_gene_species = rough_inns_by_gene_name[
        has_at_least_one_gene_ref & has_at_least_one_species_ref
    ].index
    return constant_chain_mut_data[
        constant_chain_mut_data["inn"].isin(inns_with_gene_species)
    ]


def motif_yearly_counts(
    motif_df: pd.DataFrame,
    inn_df: pd.DataFrame,
    constant_chain_info: pd.DataFrame,
    inn_mut_info: pd.DataFrame = None,
    date_col: str = "year_received",
    inn_name_source: str = "inn_name",
    motif_col: str = "mutation_string",
    mutation_col: str = "mutation_string",
    num_mutations_col: str = "num_mutations",
    count_col: str = "adjusted_count",
    rank_by_col: str | None = None,
    top_n: int | None = None,
    top_n_by_seed_only: bool = False,
    min_motif_size: int = 1,
    min_count: int = 0,
    group_other_motifs_by_panel: bool = True,
    force_other_into_single_panel: bool = False,
    seed_col: str = "cluster_seeds",
    inns_col: str = "inns",
    adjusted_inns_col: str = "adjusted_inns",
    adjusted_count_col: str = "adjusted_count",
    inn_count_col: str = "inn_count",
    constant_chain_inn_col: str = "inn",
    has_constant_chain_mutations_col: str = "has_constant_chain_mutations",
) -> pd.DataFrame:
    """Build a per-motif summary table with panel/group/classification metadata.

    One row per distinct motif in ``motif_df`` (plus any INN-level mutations
    from ``inn_mut_info`` not captured by a clustered motif — see
    ``inn_mut_info`` below), PLUS three synthesized bucket rows classifying
    every INN referenced by ``constant_chain_info`` that isn't otherwise
    covered by a real motif row (see ``constant_chain_info`` below). Every
    INN accounted for by either input is represented somewhere in the
    output: none are ever silently dropped, only reclassified into the
    appropriate ``panel_id``/``panel_classification``.

    Parameters
    ----------
    motif_df:
        Output of build_motif_index / compute_motif_ancestry (optionally via
        subtract_parent_inns) — one row per motif. Must contain
        ``motif_col``, ``num_mutations_col``, a cluster-seed column
        (``seed_col`` or literally ``"cluster_seeds"``), and all four count/
        INN-identity columns: ``inns_col``, ``adjusted_inns_col``,
        ``inn_count_col``, ``adjusted_count_col`` (both the raw/"total" and
        adjusted variants are always required and always returned,
        regardless of which one ``count_col`` selects for filtering/ranking).
    inn_df:
        One row per INN; must contain ``inn_name_source`` and ``date_col``
        (each may be a column or a named index level).
    constant_chain_info:
        One row per INN; must contain ``constant_chain_inn_col`` (default
        ``"inn"``) and ``has_constant_chain_mutations_col`` (default
        ``"has_constant_chain_mutations"``) — see
        ``_process_constant_chain_no_mut``/``analytics.constant_chain_analysis``
        for a table with this shape (``supp_tabl_constant_mut_summary``).
        Used to validate and then classify every INN referenced anywhere in
        ``motif_df``/``inn_mut_info`` (union of ``inns_col``/
        ``adjusted_inns_col``, called ``inns_in_motif_table`` below) against
        the confirmed ground truth of which INNs actually have a constant-
        chain mutation at all:

        - Hard-fail (``AssertionError``) if any motif-table INN is
          confirmed to have NO constant-chain mutation
          (``inns_in_motif_table & no_inn_mut_set`` must be empty) — a real
          motif can never legitimately reference a mutation-free INN.
        - Hard-fail (``AssertionError``) if any motif-table INN is not
          confirmed to have a mutation at all — i.e. it's either the above
          case, or simply absent from ``constant_chain_info``/of unknown
          status (``inns_in_motif_table - any_inn_mut_set`` must be empty).

        Two additional bucket rows are then synthesized from whatever
        remains uncovered by ``inns_in_motif_table``:

        - ``mutation_string_pe=""`` — ONE row combining every INN confirmed
          to have NO constant-chain mutation (``has_constant_chain_mutations_col``
          explicitly ``False``). ``panel_classification="no mutation"``,
          ``panel_id="No mutations"`` — a third, distinct generic sentinel
          from ``"Other motifs"``/``"Other mutations"`` (never folded into
          either of those two catch-all buckets; ``mutation_string_pe``
          itself stays the reserved ``""`` sentinel, unrelated to this
          ``panel_id`` string).
        - ``mutation_string_pe=f"other_mutations_{inn}"`` — one row PER
          INN confirmed to have a mutation (``has_constant_chain_mutations_col``
          explicitly ``True``) but not referenced by any real motif row —
          i.e. a real, dated mutation that isn't individually tracked by
          this motif table. ``panel_classification="untracked mutation"``,
          ``panel_id="Other mutations"`` (this INN never had a fair chance
          to be represented at all — grouped with ``min_motif_size``-
          clipped motifs under the same sentinel, distinct from
          ``"Other motifs"``; see the ``panel_classification``/``panel_id``
          section of ``Returns`` below for the full two-sentinel scheme).
    inn_mut_info:
        Optional flat (INN, mutation) table. When given, individual
        mutations not fully captured by any clustered motif are folded in
        as extra rows via ``select_mutations_outside_of_motifs`` — ensuring
        every real mutation observed in the raw INN data is represented
        somewhere in the output, not just the ones that survived motif
        clustering.
    count_col:
        Which count metric drives filtering and the ``count_col_val``
        output column: ``"adjusted_count"`` (default — the post
        ``subtract_parent_inns`` deduplicated count/INN-set) or
        ``"inn_count"`` (the raw, pre-adjustment count/INN-set — ``inns``).
        Both metrics are always returned as their own columns
        (``adjusted_count``, ``inn_count``) regardless of this choice —
        ``count_col`` controls ``min_count`` filtering and ``count_col_val``.
    rank_by_col:
        Ranking metric for direct top-N winner selection. ``None`` (default)
        means "same as ``count_col``". Must be one of
        ``{"adjusted_count", "inn_count"}``.
    top_n:
        If given, selects the top-N direct winners by ``rank_by_col`` — see
        ``top_n_by_seed_only`` for the two selection modes. ``None`` treats
        every motif surviving the zero-count drop as eligible (no ranking
        restriction — every motif resolves its own panel/group natively).
    top_n_by_seed_only:
        ``False`` (default) — direct winners are top-N motifs ranked by
        ``rank_by_col``.
        ``True`` — direct winners are top-N seed motifs (``is_seed=True``)
        ranked by ``rank_by_col``.
        In both modes, after direct winners are chosen, a seed-reference
        expansion pass can rescue additional motifs (including motifs that
        initially failed ``min_count``) when their ``cluster_seeds`` references
        a selected seed.
    min_motif_size:
        Motifs with fewer than this many mutations (``"size"`` column if
        present, else ``num_mutations_col``) are excluded from ranking (can
        never win their own top-N/seed slot, nor be reassigned via
        ``group_other_motifs_by_panel``) and reclassified into the
        ``"Other mutations"`` catch-all bucket — distinct from the
        ``"Other motifs"`` bucket used for motifs that DID compete and
        lost (see ``group_other_motifs_by_panel``/``Returns`` below) —
        tagged ``panel_classification="panel below min motif size cutoff"``
        (``panel_id="Other mutations"``), since a size-clipped motif never
        got a fair chance to compete at all. It is never removed from the
        output entirely. The sole exception is a seed (``is_seed=True``),
        which stays completely exempt from this threshold regardless of
        ranking mode (a seed is very often a single mutation, and must
        never be declassified purely for being "small"). Default ``1`` is
        a no-op (every motif has at least one mutation). Contrast with
        ``min_count`` below, which only restricts ranking eligibility and
        does not apply this distinct reclassification.
    min_count:
        Applied after direct top-N + expansion selection to demote
        below-threshold non-seed motifs out of the selected set. Seed motifs
        may remain selected below threshold when needed as panel headers. Rows
        demoted by this step remain in output and flow through normal "other"
        regrouping/classification. ``0`` disables this filter (default).
    group_other_motifs_by_panel:
        ``True`` (default) — after native panel/group assignment, make a
        second pass over every motif still bucketed under the generic
        ``"Other motifs"`` panel. Candidate panels are every distinct
        winning panel. A motif whose own tokens are a superset of a
        candidate panel's tokens is reassigned into that REAL panel —
        classified ``panel_classification="panel other"`` — but ONLY when
        exactly one *mutually-exclusive* (non-nested) candidate panel
        matches after collapsing redundant/nested matches (e.g. if a motif
        matches both ``"L234A / L235A"`` and the more specific
        ``"L234A / L235A / P329G"``, the former is dropped as redundant,
        leaving one unambiguous match). A motif matching two or more
        independent, non-nested panels (e.g. a superset of both an
        unrelated ``"LALA"`` panel and an unrelated ``"YTE"`` panel) is
        genuinely ambiguous — since it never won its own top-N ranking
        slot, it has no independently-earned claim to either panel (or to
        a new combined one — that's reserved for motifs that DID win a
        top-N slot with a multi-seed ``cluster_seeds``, see above), so it
        stays in the generic ``"Other motifs"`` bucket, classified
        ``panel_classification="motif failed: ambiguous panel match"``
        rather than being forced into one arbitrary match. A motif
        matching zero candidate panels at all is classified
        ``"motif failed: no panel match"``. ``False`` skips this whole
        pass entirely — every non-winning motif stays under the generic
        ``"Other motifs"`` panel, classified
        ``"motif failed: not ranked"`` (no matching was even attempted).
    force_other_into_single_panel:
        Only used when ``group_other_motifs_by_panel=True``. If an "other"
        motif matches multiple candidate panel headers, ``True`` forces
        assignment into exactly one panel: pick the panel with highest total
        selected-member cumulative count (sum of ``count_col`` across selected
        rows in that panel), tie-breaking by higher panel-header
        ``count_col``. ``False`` keeps current behavior: ambiguous multi-panel
        matches remain in generic ``"Other motifs"``.
    seed_col:
        Column in ``motif_df`` holding each motif's cluster seed label(s)
        (default ``"cluster_seeds"``); a comma-joined string for motifs
        belonging to more than one seed.
    constant_chain_inn_col, has_constant_chain_mutations_col:
        Column names in ``constant_chain_info`` — see above.

    Decision flow
    -------------
    ::

        START
          |
          |-- 0) Validate required columns + strict canonical motif/seed format
          |       (fail fast and list non-canonical rows)
          |-- 1) rank metric = rank_by_col or count_col
          |-- 2) Initial filtering
          |       - min_motif_size clip (non-seed; later reclassified)
          |-- 3) Ranking branch
          |       A) top_n_by_seed_only: direct winners = top-N seeds by rank metric
          |       B) general top_n: direct winners = top-N motifs by rank metric
          |-- 4) Expansion (both branches)
          |       pull motifs whose cluster_seeds references selected seed(s)
          |-- 5) Apply min_count cutoff demotion
          |       demote selected non-seeds below cutoff; keep selected seeds
          |-- 6) Multi-seed resolution
          |       same token-subset combo-panel clustering in both branches
          |-- 7) End-stage zero-count cleanup loop
          |       drop rows with count_col <= 0, then remove selected panels
          |       whose member sum(count_col) is still 0, recomputing panel
          |       membership/clustering until stable
          |-- 8) Classification
          |       selected -> panel header / panel variant
          |       non-selected -> other regrouping (header matching), optional
          |       force-to-single-panel for ambiguous multi-panel matches
          |-- 9) Final buckets
          |       min-size clipped, no-mutation, untracked-mutation unchanged
          |
        END

    Returns
    -------
    DataFrame with these columns:

    - ``motif_col`` (e.g. ``"mutation_string_pe"``) — the raw motif string,
      unchanged/untouched for real motif rows; ``""`` for the "no
      mutation" bucket; ``f"other_mutations_{inn}"`` for each "untracked
      mutation" bucket row (see ``constant_chain_info`` above).
    - ``adjusted_count`` — this motif's post-adjustment count (always
      present, regardless of ``count_col``).
    - ``inn_count`` — this motif's raw/total count (always present,
      regardless of ``count_col``).
    - ``year`` — the earliest year this motif's raw INN set was observed,
      resolved via ``inn_df``/``date_col``/``inn_name_source`` (float,
      ``NaN`` if none of its INNs have a resolvable date).
    - ``count_col_val`` — whichever of ``adjusted_count``/``inn_count`` was
      selected by ``count_col`` — the value used for ``min_count`` filtering
      and exported as the active count metric.
    - ``panel_classification`` — one of ``"panel header"`` (this motif IS
      its panel's own representative/head), ``"panel variant"`` (a genuine
      top-N motif under a panel, but not its head), ``"panel other"``
      (reclassified into a real panel via ``group_other_motifs_by_panel``),
      ``"panel below min motif size cutoff"`` (excluded from ranking
      entirely by ``min_motif_size`` — never got a fair chance to
      compete), ``"panel other mutations"`` (reserved
      fallback for any OTHER future reason a motif might land in the
      ``"Other mutations"`` bucket without a more specific classification),
      ``"motif failed: <reason>"`` (a real
      motif that DID compete — was eligible for ranking/matching — but
      lost; ``<reason>`` is ``"no panel match"``, ``"ambiguous panel
      match"``, or ``"not ranked"`` — see ``group_other_motifs_by_panel``
      above for exactly when each applies), ``"no mutation"`` (the single
      synthesized confirmed-mutation-free bucket), or ``"untracked
      mutation"`` (a synthesized per-INN bucket — real mutation, not
      tracked by any real motif row at all).
    - ``panel_id`` — this motif's assigned panel: a real winning seed/combo-
      head label; ``"Other motifs"`` for ``"motif failed: ..."`` rows (real
      motifs that competed and lost); ``"Other mutations"`` for
      ``"panel below min motif size cutoff"``/``"panel other
      mutations"``/``"untracked mutation"`` rows (never got a fair chance
      to compete at all); or ``"No mutations"`` for the "no mutation"
      bucket (confirmed mutation-free — deliberately its own third
      sentinel, distinct from both of the above). These are THREE DISTINCT
      sentinels, not one — do not conflate them.
    - ``variant_label`` — the display/grouping identity consumed downstream
      by ``build_motif_timeline_table``/``plot_motif_timeline``. Equal to
      ``motif_col`` for ``"panel header"``/``"panel variant"`` rows;
      ``"Other {panel_id}/+"`` for ``"panel other"`` rows (several distinct
      real motifs matched into the same real panel share this one label);
      otherwise equal to ``panel_id`` itself — i.e. ``"Other motifs"`` for
      ``"motif failed: ..."`` rows, ``"Other mutations"`` for
      ``"panel below min motif size cutoff"``/``"panel other
      mutations"``/``"untracked mutation"`` rows, or ``"No mutations"`` for
      the "no mutation" bucket (each collapses into its own respective
      generic catch-all bucket).
    - ``is_seed`` — whether this row's motif is a seed motif in the source
      motif table (``False`` for synthesized rows such as "no mutation",
      "untracked mutation", and any coverage/gap-fill synthetic rows).
    - ``is_top_n_selected`` — whether this motif ended up in the final top-N
      selected set used for panel assignment (``True`` for both direct top-N
      winners and motifs pulled in via seed expansion; ``False`` otherwise,
      including synthesized rows).
    - ``selection_source`` — provenance label for how/if this motif entered
      top-N selection:
        - ``"direct_top_n"``: selected directly by ranking.
        - ``"expanded_seed"``: a seed motif included only because a direct
          top-N motif referenced it via seed expansion.
        - ``"seed_member"``: a non-seed motif included because it is
          associated with a selected seed/direct-top-N motif.
        - ``"selected_other"``: not top-N-selected, but explicitly
          reassigned into a real panel as ``panel_classification="panel other"``.
        - ``"no_top_n_filter"``: ``top_n`` was not applied (all surviving
          motifs were eligible without rank restriction).
        - ``"not_selected"``: not in selected top-N set.
    - ``inns`` — this motif's raw (pre-adjustment) INN identity set.
    - ``adjusted_inns`` — this motif's adjusted (post-``subtract_parent_inns``)
      INN identity set.
    - ``group``, ``sub_classification`` — *only if present in* ``motif_df``
      *on input* — passed through verbatim (unmodified, untouched by any
      panel-assignment logic above). These are a wholly separate, pre-
      existing concept from an earlier pipeline stage (the Will-effector
      biological classification produced by
      ``build_will_effector_motif_pe_dataframe``, e.g.
      ``"Effector"``/``"Unknown"``/``"FcRn"``) — deliberately NOT reused or
      overwritten for panel-assignment (hence the distinct
      ``panel_id``/``panel_classification`` names above). Rows synthesized
      by this function itself (from ``inn_mut_info`` coverage, the
      coverage-assertion gap-fill, or the ``constant_chain_info``-driven
      "no mutation"/"untracked mutation" buckets) have no original
      classification to pass through and so get ``NaN`` here, not a
      placeholder value.
    """
    # ── resolve count/rank columns ────────────────────────────────────────
    _COUNT_COL_SOURCES = {
        "adjusted_count": (adjusted_count_col, adjusted_inns_col),
        "inn_count": (inn_count_col, inns_col),
    }
    if count_col not in _COUNT_COL_SOURCES:
        raise ValueError(
            f"motif_yearly_counts: count_col must be one of "
            f"{sorted(_COUNT_COL_SOURCES)}, got {count_col!r}."
        )
    _count_col_name, _count_inns_col_name = _COUNT_COL_SOURCES[count_col]
    _rank_col_key = count_col if rank_by_col is None else rank_by_col
    if _rank_col_key not in _COUNT_COL_SOURCES:
        raise ValueError(
            f"motif_yearly_counts: rank_by_col must be one of "
            f"{sorted(_COUNT_COL_SOURCES)} or None, got {rank_by_col!r}."
        )
    _rank_col_name, _ = _COUNT_COL_SOURCES[_rank_col_key]

    _required_cols = [
        motif_col,
        num_mutations_col,
        inns_col,
        adjusted_inns_col,
        inn_count_col,
        adjusted_count_col,
    ]
    _missing_cols = [
        c for c in dict.fromkeys(_required_cols) if c not in motif_df.columns
    ]
    if _missing_cols:
        raise KeyError(
            f"motif_yearly_counts: motif_df is missing required column(s) "
            f"{_missing_cols}. Available columns: {sorted(motif_df.columns.tolist())}"
        )

    _cluster_col = (
        seed_col
        if seed_col in motif_df.columns
        else ("cluster_seeds" if "cluster_seeds" in motif_df.columns else None)
    )
    if _cluster_col is None:
        raise KeyError(
            f"motif_yearly_counts: motif_df is missing the cluster-seed column "
            f"(looked for seed_col={seed_col!r} and the literal 'cluster_seeds'). "
            f"This column is required — group/sub_classification assignment and "
            f"top-N seed expansion all depend on it. Available columns: "
            f"{sorted(motif_df.columns.tolist())}"
        )

    if top_n_by_seed_only and "is_seed" not in motif_df.columns:
        raise KeyError(
            "motif_yearly_counts: top_n_by_seed_only=True requires an "
            "'is_seed' column in motif_df, but it is missing. Available "
            f"columns: {sorted(motif_df.columns.tolist())}"
        )

    _required_cc_cols = (constant_chain_inn_col, has_constant_chain_mutations_col)
    _missing_cc = [c for c in _required_cc_cols if c not in constant_chain_info.columns]
    if _missing_cc:
        raise KeyError(
            f"motif_yearly_counts: constant_chain_info is missing required "
            f"column(s) {_missing_cc}. Available: "
            f"{list(constant_chain_info.columns)}"
        )
    if not constant_chain_info[constant_chain_inn_col].is_unique:
        _cc_dupes = constant_chain_info.loc[
            constant_chain_info[constant_chain_inn_col].duplicated(),
            constant_chain_inn_col,
        ].tolist()
        raise AssertionError(
            f"motif_yearly_counts: '{constant_chain_inn_col}' has duplicate "
            f"value(s) in constant_chain_info — expected exactly one row "
            f"per INN. Duplicates: {_cc_dupes[:10]}"
        )

    # ── strict canonical-format assertion (no auto-rewrite/coercion) ─────
    import re as _re_motif

    _TUPLE_TOK = _re_motif.compile(
        r"""\(\s*["']?([^"'\s,)]+)["']?\s*,\s*["']([^"']*)["']\s*\)"""
    )
    _SEED_SENTINELS = {"other mutations", "other", "multiple seeds"}

    def _canonical_motif_str(v, *, allow_empty: bool = False) -> str:
        if not isinstance(v, str):
            return ""
        s = v.strip()
        if not s:
            return "" if allow_empty else ""
        if _TUPLE_TOK.search(s):
            return ""
        parts = [tok.strip() for tok in s.split("/") if tok.strip()]
        return " / ".join(parts)

    def _canonical_cluster_seeds(v) -> str:
        if v is None or (isinstance(v, float) and pd.isna(v)):
            return ""
        if isinstance(v, (list, tuple, set, frozenset)):
            parts = [str(x).strip() for x in v if x is not None and str(x).strip()]
            if not parts:
                return ""
            if isinstance(v, (set, frozenset)):
                parts = sorted(parts)
            canon_parts = []
            for part in parts:
                if part.lower() in _SEED_SENTINELS:
                    return ""
                canon = _canonical_motif_str(part, allow_empty=False)
                if not canon:
                    return ""
                canon_parts.append(canon)
            return ", ".join(canon_parts)
        if not isinstance(v, str):
            return ""
        s = v.strip()
        if not s:
            return ""
        if s.lower() in _SEED_SENTINELS:
            return s
        parts = [c.strip() for c in s.split(",") if c.strip()]
        if not parts:
            return ""
        canon_parts = []
        for part in parts:
            canon = _canonical_motif_str(part, allow_empty=False)
            if not canon:
                return ""
            canon_parts.append(canon)
        return ", ".join(canon_parts)

    def _assert_canonical_column(
        df: pd.DataFrame,
        column: str,
        *,
        canonicalizer,
        label: str,
        allow_empty: bool = False,
        allow_sequence_values: bool = False,
    ) -> None:
        failures: list[str] = []
        for idx, raw in df[column].items():
            if raw is None or (isinstance(raw, float) and pd.isna(raw)):
                if allow_empty:
                    continue
                failures.append(f"row={idx}: value={raw!r} (missing)")
                continue
            if allow_sequence_values and isinstance(raw, (list, tuple, set, frozenset)):
                expected = canonicalizer(raw)
                if not expected:
                    failures.append(
                        f"row={idx}: value={raw!r} (invalid canonical {label} format)"
                    )
                continue
            if not isinstance(raw, str):
                failures.append(
                    f"row={idx}: value={raw!r} (type={type(raw).__name__}, expected str)"
                )
                continue
            expected = canonicalizer(raw)
            if expected == "" and allow_empty and raw.strip() == "":
                continue
            if not expected:
                failures.append(
                    f"row={idx}: value={raw!r} (invalid canonical {label} format)"
                )
                continue
            if raw != expected:
                # Allow spacing-only differences (e.g. "A/B" vs "A / B")
                # while still failing true canonical-format mismatches.
                if raw.replace(" ", "") == expected.replace(" ", ""):
                    continue
                failures.append(f"row={idx}: value={raw!r}, expected={expected!r}")
        if failures:
            preview = "\n".join(f"  - {f}" for f in failures[:30])
            suffix = "\n  - ..." if len(failures) > 30 else ""
            raise AssertionError(
                f"motif_yearly_counts: non-canonical {label} values found "
                f"({len(failures)} rows). Please canonicalize upstream.\n"
                f"{preview}{suffix}"
            )

    def _as_inn_set(raw) -> set:
        if isinstance(raw, (set, frozenset)):
            return set(raw)
        if raw is None:
            return set()
        if np.isscalar(raw):
            if pd.isna(raw):
                return set()
            if isinstance(raw, str):
                _raw = raw.strip()
                return {_raw} if _raw else set()
            return {raw}
        return set(raw)

    motif_df = motif_df.copy()
    if "cluster_seeds" not in motif_df.columns:
        motif_df = motif_df.rename(columns={seed_col: "cluster_seeds"})
    _assert_canonical_column(
        motif_df,
        motif_col,
        canonicalizer=lambda x: _canonical_motif_str(x, allow_empty=False),
        label=motif_col,
    )
    _assert_canonical_column(
        motif_df,
        "cluster_seeds",
        canonicalizer=_canonical_cluster_seeds,
        label="cluster_seeds",
        allow_empty=True,
        allow_sequence_values=True,
    )
    if inn_mut_info is not None:
        inn_mut_info = inn_mut_info.copy()
        _assert_canonical_column(
            inn_mut_info,
            mutation_col,
            canonicalizer=lambda x: _canonical_motif_str(x, allow_empty=False),
            label=mutation_col,
        )
        inn_mut_info[mutation_col] = inn_mut_info[mutation_col].apply(
            lambda x: _canonical_motif_str(x, allow_empty=False)
        )

    # Normalize after validation so matching/ranking is robust to spacing-only
    # differences while still enforcing strict canonical token content.
    motif_df[motif_col] = motif_df[motif_col].apply(
        lambda x: _canonical_motif_str(x, allow_empty=False)
    )
    motif_df["cluster_seeds"] = (
        motif_df["cluster_seeds"].apply(_canonical_cluster_seeds).astype("object")
    )

    _size_col = "size" if "size" in motif_df.columns else num_mutations_col
    if _size_col not in motif_df.columns:
        raise KeyError(
            f"motif_yearly_counts: min_motif_size requires a 'size' or "
            f"{num_mutations_col!r} column in motif_df, but neither is "
            f"present. Available columns: {sorted(motif_df.columns.tolist())}"
        )

    # ── min_motif_size deferred reclassification: a non-seed motif below
    # this threshold is excluded from the active ranking/expansion universe
    # (never wins a slot, never expanded in as a seed reference) — but,
    # unlike a full drop, it is held aside in `_clipped_df` and reintegrated
    # verbatim after panel assignment below, tagged distinctly as
    # panel_classification="panel below min motif size cutoff", panel_id=
    # "Other mutations" (not simply removed, and not conflated with the
    # "motif failed: ..."/"Other motifs" catch-all reserved for motifs that
    # DID get a fair chance to compete — see STEP 2e below). Seeds
    # (is_seed=True) are the sole exception: they stay fully exempt from
    # this threshold regardless of ranking mode (a seed is very often a
    # single mutation, and must never be reclassified away purely for
    # being "small") — see the seed-ranking-pool exemption below
    # for how a small seed can still WIN a slot in top_n_by_seed_only mode.
    _clipped_df = motif_df.iloc[:0]
    if min_motif_size > 1:
        _is_seed_pre_clip = (
            motif_df["is_seed"].fillna(False).astype(bool)
            if "is_seed" in motif_df.columns
            else pd.Series(False, index=motif_df.index)
        )
        _clip_mask = (motif_df[_size_col].fillna(0) < min_motif_size) & (
            ~_is_seed_pre_clip
        )
        _clipped_df = motif_df[_clip_mask].copy()
        motif_df = motif_df[~_clip_mask].copy()

    # ── STEP 1: filter / rank / select the winning top-N motif set ─────────
    _all_motifs: set = set(motif_df[motif_col])
    _below_count = (
        motif_df[_count_col_name].fillna(0) < min_count
        if min_count > 0
        else pd.Series(False, index=motif_df.index)
    )
    _is_seed_all = (
        motif_df["is_seed"].fillna(False).astype(bool)
        if "is_seed" in motif_df.columns
        else pd.Series(False, index=motif_df.index)
    )

    def _seed_candidates(raw) -> list:
        """Parse canonical cluster_seeds cell into candidate labels."""
        if not isinstance(raw, str):
            return []
        _raw = raw.strip()
        if not _raw or _raw.lower() in _SEED_SENTINELS:
            return []
        return [c.strip() for c in _raw.split(",") if c.strip()]

    final_top_n: set = set()
    direct_top_n_labels: set = set()
    expanded_top_n_labels: set = set()
    selected_seed_labels: set = set()

    if top_n_by_seed_only:
        _seed_pool = motif_df[_is_seed_all]
        direct_top_n_labels = set(
            _seed_pool.sort_values(_rank_col_name, ascending=False)[motif_col].iloc[
                :top_n
            ]
            if top_n is not None
            else _seed_pool[motif_col]
        )
        selected_seed_labels = set(direct_top_n_labels)
    else:
        _rank_pool = motif_df
        direct_top_n_labels = set(
            _rank_pool.sort_values(_rank_col_name, ascending=False)[motif_col].iloc[
                :top_n
            ]
            if top_n is not None
            else _rank_pool[motif_col]
        )
        _topn_rows = motif_df.loc[
            motif_df[motif_col].isin(direct_top_n_labels), ["cluster_seeds", motif_col]
        ]
        for _, _row in _topn_rows.iterrows():
            selected_seed_labels.update(_seed_candidates(_row["cluster_seeds"]))
            if bool(_row.get("is_seed", False)):
                selected_seed_labels.add(_row[motif_col])
        selected_seed_labels &= _all_motifs

    _tmp = motif_df[[motif_col, "cluster_seeds"]].copy()
    _tmp["_candidates"] = _tmp["cluster_seeds"].apply(_seed_candidates)
    _exploded = _tmp.explode("_candidates")
    _pulled_in = set(
        _exploded.loc[_exploded["_candidates"].isin(selected_seed_labels), motif_col]
    )
    _pulled_in |= set(
        motif_df.loc[motif_df[motif_col].isin(selected_seed_labels), motif_col]
    )

    final_top_n = set(direct_top_n_labels) | set(_pulled_in)
    expanded_top_n_labels = set(_pulled_in) - set(direct_top_n_labels)

    # Drop zero-count motifs unless selected (direct or expansion-rescued).
    _zero_mask = (motif_df[_count_col_name].fillna(0) == 0) & ~motif_df[motif_col].isin(
        final_top_n
    )
    motif_df = motif_df[~_zero_mask].copy()
    _top_n_labels: set = final_top_n
    if min_count > 0:
        _below_count_survivors = motif_df[_count_col_name].fillna(0) < min_count
        _is_seed_survivors = (
            motif_df["is_seed"].fillna(False).astype(bool)
            if "is_seed" in motif_df.columns
            else pd.Series(False, index=motif_df.index)
        )
        _demote_mask = (
            motif_df[motif_col].isin(_top_n_labels)
            & _below_count_survivors
            & (~_is_seed_survivors)
        )
        _demoted = set(motif_df.loc[_demote_mask, motif_col])
        if _demoted:
            _top_n_labels = set(_top_n_labels) - _demoted
            direct_top_n_labels = set(direct_top_n_labels) - _demoted
            expanded_top_n_labels = set(expanded_top_n_labels) - _demoted

    # ── merge in inn_mut_info-derived "other mutations" coverage ───────────
    long_df = motif_df.copy()
    if inn_mut_info is not None:
        other_motifs_df = select_mutations_outside_of_motifs(
            inn_mut_info, long_df, yearly_mutation_col=motif_col
        )
        if mutation_col not in other_motifs_df.columns:
            raise KeyError(
                f"motif_yearly_counts: expected mutation column {mutation_col!r} "
                f"in select_mutations_outside_of_motifs output, but only found "
                f"{list(other_motifs_df.columns)}"
            )
        _assert_canonical_column(
            other_motifs_df,
            mutation_col,
            canonicalizer=lambda x: _canonical_motif_str(x, allow_empty=False),
            label=mutation_col,
        )
        other_motifs_df[motif_col] = other_motifs_df[mutation_col].apply(
            lambda x: _canonical_motif_str(x, allow_empty=False)
        )
        other_motifs_df["cluster_seeds"] = "Other Mutations"
        long_df = pd.concat([long_df, other_motifs_df], ignore_index=True, sort=False)

    # ── coverage assertion: every canonical input mutation string must be
    # accounted for (strict-format assertion above guarantees canonical form). ──
    _all_input_strings: set = set(motif_df[motif_col].dropna().tolist())
    if inn_mut_info is not None:
        _all_input_strings |= set(inn_mut_info[mutation_col].dropna().tolist())

    _present_strings: set = set(long_df[motif_col].dropna().tolist())
    _missing: set = _all_input_strings - _present_strings

    if _missing:
        # Build INN set lookup from both input dfs — keyed by canonical string.
        # Both inns_col and adjusted_inns_col are populated from the same
        # single resolved set for these synthesized rows (no separate
        # raw-vs-adjusted source is available for a motif that was entirely
        # absorbed/consumed elsewhere).
        _inns_lookup: dict = {}
        for _src_df, _col in [
            (motif_df, motif_col),
            (
                (inn_mut_info if inn_mut_info is not None else motif_df.iloc[:0]),
                mutation_col,
            ),
        ]:
            for _, _row in _src_df.iterrows():
                _ms = _row.get(_col)
                if _ms in _missing and _ms not in _inns_lookup:
                    _raw = _row.get(adjusted_inns_col)
                    _inns_lookup[_ms] = (
                        _raw
                        if isinstance(_raw, (set, frozenset))
                        else (set(_raw) if _raw else set())
                    )

        _gap_rows = []
        for _ms in _missing:
            _inns = _inns_lookup.get(_ms, set())
            _gap_rows.append(
                {
                    motif_col: _ms,
                    inns_col: _inns,
                    adjusted_inns_col: _inns,
                    inn_count_col: len(_inns),
                    adjusted_count_col: len(_inns),
                    "cluster_seeds": "Other Mutations",
                }
            )

        long_df = pd.concat(
            [long_df, pd.DataFrame(_gap_rows)], ignore_index=True, sort=False
        )

    _still_missing = _all_input_strings - set(long_df[motif_col].dropna().tolist())
    if _still_missing:
        raise AssertionError(
            f"motif_yearly_counts: {len(_still_missing)} mutation string(s) from "
            f"input dfs are unaccounted for in yearly output:\n"
            + "\n".join(f"  {s}" for s in sorted(_still_missing)[:20])
            + ("\n  ..." if len(_still_missing) > 20 else "")
        )

    # ── STEP 2a/2b: panel_id/panel_classification assignment. A motif not
    # in _top_n_labels always falls to "Other motifs" natively (2a) — it
    # may still be reclassified into a real panel afterwards by the
    # group_other_motifs_by_panel token-superset pass (2b). Among
    # _top_n_labels motifs: 0 cluster_seeds matches → "Other motifs",
    # exactly 1 match → that seed's own panel, 2+ matches → re-clustered by
    # mutation-token subset relationships (shortest first; each motif joins
    # the first previously-seen motif whose token set is a subset of its
    # own). Named panel_id/panel_classification (not group/sub_classification)
    # to avoid colliding with the pre-existing, semantically unrelated
    # group/sub_classification columns produced upstream by
    # build_will_effector_motif_pe_dataframe (Will-effector biological
    # classification, e.g. "Effector"/"Unknown"/"FcRn") — a motif_df passed
    # in here may already carry those columns with a completely different
    # meaning; this function must never overwrite or shadow them. ─────────
    #
    # Three distinct generic-bucket sentinels, not two — they mean
    # different things and must never be conflated:
    #   _OTHER_PANEL ("Other motifs") — for real motifs that DID compete
    #     (had a fair chance to win a top-N slot or match a real panel) and
    #     lost. panel_classification is "motif failed: <reason>".
    #   _OTHER_MUTATIONS_PANEL ("Other mutations") — for anything that
    #     never got a fair chance at all: a real motif excluded from
    #     ranking entirely by min_motif_size ("panel below min motif size
    #     cutoff"), or a real, confirmed mutation on an INN that isn't
    #     referenced by any row in the motif table whatsoever ("untracked
    #     mutation" — see STEP 2f below).
    #   _NO_MUTATIONS_PANEL ("No mutations") — the single synthesized
    #     bucket combining every INN CONFIRMED to have no constant-chain
    #     mutation at all ("no mutation" — see STEP 2f below). Distinct
    #     from both of the above: this INN population was never a
    #     candidate for any panel in the first place (not merely excluded
    #     or unranked — genuinely mutation-free).
    _OTHER_PANEL = "Other motifs"
    _OTHER_MUTATIONS_PANEL = "Other mutations"
    _NO_MUTATIONS_PANEL = "No mutations"

    def _panel_seed_candidates(raw) -> list:
        if not isinstance(raw, str) or not raw.strip():
            return []
        return [
            " / ".join(t.strip() for t in c.split("/") if t.strip())
            for c in raw.split(",")
            if c.strip()
        ]

    def _matching_seeds(motif: str, raw_cluster_seeds) -> list:
        """Top-N seed labels referenced by this motif's cluster_seeds cell."""
        if motif not in _top_n_labels:
            return []
        _raw = raw_cluster_seeds.strip() if isinstance(raw_cluster_seeds, str) else ""
        if not _raw or _raw.lower() in ("other mutations", "other", "multiple seeds"):
            return []
        return list(
            dict.fromkeys(c for c in _panel_seed_candidates(_raw) if c in _top_n_labels)
        )

    def _is_literal_multiple_seeds(motif: str, raw_cluster_seeds) -> bool:
        """True for the raw sentinel value itself (no discoverable candidate
        list), which must still resolve to "Multiple Seeds" — distinct from
        a motif with 0 matching top-N seeds, which falls to "Other motifs"."""
        return (
            motif in _top_n_labels
            and isinstance(raw_cluster_seeds, str)
            and raw_cluster_seeds.strip().lower() == "multiple seeds"
        )

    def _resolve_panel(motif: str, is_literal_multi: bool, matches: list) -> str:
        if is_literal_multi:
            return "Multiple Seeds"
        if not matches:
            # A top-N motif with no recognized seed match becomes its own
            # self-headed singleton panel — never the shared "Other
            # motifs" sentinel. This keeps that sentinel exclusively
            # reserved for genuinely non-top-N motifs, so panel_id alone
            # unambiguously signals "not individually plotted" without a
            # caller also having to cross-check is_top_n/panel_classification
            # — otherwise a motif that wins its own top-N ranking slot
            # purely by count (no seed affiliation at all — a real,
            # possible outcome in the non-seed-only ranking mode) would
            # share panel_id="Other motifs" with the true generic
            # catch-all bucket despite being individually plotted, not
            # aggregated with it.
            return motif if motif in _top_n_labels else _OTHER_PANEL
        if len(matches) == 1:
            return matches[0]
        return "Multiple Seeds"

    def _collapse_multiple_seed_panels(panel_map: dict[str, str]) -> dict[str, str]:
        _ms_motifs = [m for m, p in panel_map.items() if p == "Multiple Seeds"]
        if not _ms_motifs:
            return panel_map

        def _panel_tokens(motif: str) -> frozenset:
            return frozenset(t.strip() for t in motif.split("/") if t.strip())

        _ms_tokens = {m: _panel_tokens(m) for m in _ms_motifs}
        _panel_head_tokens: dict = {}
        for _m in sorted(_ms_motifs, key=lambda m: len(_ms_tokens[m])):
            _mem_tokens = _ms_tokens[_m]
            _head = next(
                (c for c, _tc in _panel_head_tokens.items() if _tc <= _mem_tokens),
                None,
            )
            if _head is None:
                _panel_head_tokens[_m] = _mem_tokens
                panel_map[_m] = _m
            else:
                panel_map[_m] = _head
        return panel_map

    # End-stage cleanup loop:
    #  1) remove rows with no count data (count_col <= 0),
    #  2) remove selected panels whose total selected-member count is still 0,
    # then recompute panel memberships/clusters until stable.
    panel_id_map: dict[str, str] = {}
    for _iter_idx in range(20):
        _changed = False

        _zero_row_mask = long_df[_count_col_name].fillna(0) <= 0
        if _zero_row_mask.any():
            _removed_motifs = set(
                long_df.loc[_zero_row_mask, motif_col].dropna().tolist()
            )
            long_df = long_df.loc[~_zero_row_mask].copy()
            if _removed_motifs:
                _top_n_labels = set(_top_n_labels) - _removed_motifs
                direct_top_n_labels = set(direct_top_n_labels) - _removed_motifs
                expanded_top_n_labels = set(expanded_top_n_labels) - _removed_motifs
            _changed = True

        _panel_src = long_df.drop_duplicates(subset=[motif_col])
        _cluster_seeds_map: dict = (
            _panel_src.set_index(motif_col)["cluster_seeds"].to_dict()
            if not _panel_src.empty
            else {}
        )
        panel_id_map = {
            m: _resolve_panel(
                m, _is_literal_multiple_seeds(m, cs), _matching_seeds(m, cs)
            )
            for m, cs in _cluster_seeds_map.items()
        }
        panel_id_map = _collapse_multiple_seed_panels(panel_id_map)
        long_df["panel_id"] = long_df[motif_col].map(panel_id_map)

        _selected_mask = long_df[motif_col].isin(_top_n_labels)
        _selected_panel_totals = (
            long_df.loc[_selected_mask].groupby("panel_id")[_count_col_name].sum()
            if _selected_mask.any()
            else pd.Series(dtype=float)
        )
        _zero_total_panels = set(
            _selected_panel_totals[_selected_panel_totals <= 0].index
        )
        if _zero_total_panels:
            _demoted_zero_total = set(
                long_df.loc[
                    _selected_mask & long_df["panel_id"].isin(_zero_total_panels),
                    motif_col,
                ].dropna()
            )
            if _demoted_zero_total:
                _top_n_labels = set(_top_n_labels) - _demoted_zero_total
                direct_top_n_labels = set(direct_top_n_labels) - _demoted_zero_total
                expanded_top_n_labels = set(expanded_top_n_labels) - _demoted_zero_total
                _changed = True

        if not _changed:
            break
    else:
        raise RuntimeError(
            "motif_yearly_counts: panel assignment cleanup exceeded 20 iterations; "
            "unable to converge."
        )

    # ── is_top_n (internal): was this motif selected by the initial top-N
    # ranking pass (directly by its own count, or expanded in because a
    # top-N motif's cluster_seeds referenced it)? ──────────────────────────
    long_df["_is_top_n"] = long_df[motif_col].isin(_top_n_labels)
    if "is_seed" in long_df.columns:
        long_df["is_seed"] = long_df["is_seed"].fillna(False).astype(bool)
    else:
        long_df["is_seed"] = False
    _is_seed_long = long_df["is_seed"].fillna(False).astype(bool)
    _is_expanded = long_df[motif_col].isin(expanded_top_n_labels)
    long_df["_selection_source"] = np.select(
        [
            long_df[motif_col].isin(direct_top_n_labels),
            _is_expanded & _is_seed_long,
            _is_expanded & (~_is_seed_long),
            (
                pd.Series(
                    bool((not top_n_by_seed_only) and (top_n is None)),
                    index=long_df.index,
                )
                & long_df["_is_top_n"]
            ),
        ],
        ["direct_top_n", "expanded_seed", "seed_member", "no_top_n_filter"],
        default="not_selected",
    )

    # ── 2b: group_other_motifs_by_panel — reassign "Other mutations" motifs
    # into a REAL panel when their own tokens are a superset of that panel's.
    # These motifs genuinely join that panel (panel_classification="panel
    # other" below) despite never having won a top-N ranking slot
    # themselves — that fact (never a top-N winner) is exactly why this
    # reassignment must be conservative: a motif here never gets to form a
    # brand-new combo panel of its own (that mechanism — "Multiple Seeds",
    # step 2a above — is reserved for motifs that DID win a top-N slot and
    # whose own cluster_seeds explicitly names 2+ seeds).
    #
    # Matching algorithm per candidate motif:
    #   1. Find every real panel whose tokens are fully contained in the
    #      motif's own tokens (panel_tokens ⊆ motif_tokens — i.e. the motif
    #      is "at least as specific as" that panel).
    #   2. Collapse redundant/nested matches: drop any matched panel whose
    #      own tokens are a STRICT subset of another matched panel's tokens
    #      (e.g. if both "L234A / L235A" (LALA) and "L234A / L235A / P329G"
    #      (LALA+P329G) match, LALA is redundant — implied by the more
    #      specific LALA+P329G match — and is dropped, keeping only the
    #      most-specific/"maximal" matches).
    #   3. Reassign ONLY when exactly one maximal match remains. Zero
    #      matches means no real panel is a subset at all (stays "Other
    #      motifs"). Two or more maximal matches means the motif spans
    #      multiple mutually-independent (non-nested) panels — e.g. a
    #      motif whose tokens are a superset of BOTH an unrelated "LALA"
    #      panel's tokens AND an unrelated "YTE" panel's tokens — which is
    #      genuinely ambiguous: since this motif was never itself a top-N
    #      winner, it has no independently-earned claim to either panel (or
    #      to a new combined one), so it is left as the generic "Other
    #      motifs" bucket rather than arbitrarily forced into one match. ──
    if group_other_motifs_by_panel:

        def _token_set(motif: str) -> frozenset:
            return frozenset(t.strip() for t in motif.split("/") if t.strip())

        _candidate_panels = sorted(
            {panel_id_map[m] for m in _top_n_labels if m in panel_id_map},
            key=lambda p: (-len(_token_set(p)), p),
        )
        _candidate_tokens = {p: _token_set(p) for p in _candidate_panels}
        _selected_rows = long_df[long_df[motif_col].isin(_top_n_labels)].copy()
        _panel_total_counts = (
            _selected_rows.groupby("panel_id")[_count_col_name].sum().to_dict()
            if not _selected_rows.empty
            else {}
        )
        _panel_header_counts = (
            _selected_rows.loc[_selected_rows[motif_col] == _selected_rows["panel_id"]]
            .groupby("panel_id")[_count_col_name]
            .max()
            .to_dict()
            if not _selected_rows.empty
            else {}
        )

        _other_mask = long_df["panel_id"] == _OTHER_PANEL
        _other_motifs = long_df.loc[_other_mask, motif_col].dropna().unique()

        _regroup_map: dict[str, str] = {}  # motif → matched (real) panel
        # motif → why it was NOT reassigned (feeds panel_classification's
        # "motif failed: <reason>" wording below) — populated for every
        # candidate motif considered here, whether or not it ends up
        # matched, so the exact reason is always traceable.
        _fail_reason_map: dict[str, str] = {}
        for _m in _other_motifs:
            _m_tokens = _token_set(_m)
            _matches = [
                _p for _p in _candidate_panels if _candidate_tokens[_p] <= _m_tokens
            ]
            _maximal = [
                _p
                for _p in _matches
                if not any(
                    _q != _p and _candidate_tokens[_p] < _candidate_tokens[_q]
                    for _q in _matches
                )
            ]
            if len(_maximal) == 1:
                _regroup_map[_m] = _maximal[0]
            elif len(_maximal) == 0:
                _fail_reason_map[_m] = "no panel match"
            else:
                if force_other_into_single_panel:
                    _regroup_map[_m] = max(
                        _maximal,
                        key=lambda p: (
                            _panel_total_counts.get(p, 0),
                            _panel_header_counts.get(p, 0),
                            p,
                        ),
                    )
                else:
                    _fail_reason_map[_m] = "ambiguous panel match"

        if _regroup_map:
            _regrouped_mask = long_df[motif_col].isin(_regroup_map)
            long_df.loc[_regrouped_mask, "panel_id"] = long_df.loc[
                _regrouped_mask, motif_col
            ].map(_regroup_map)
    else:
        _fail_reason_map = {}

    # ── 2d: panel_classification — always exactly one "panel header" row
    # per real panel_id value (none for the "Other motifs" sentinel,
    # since no real motif is ever literally named "Other motifs"). The
    # generic "other" bucket for real, competing-but-losing motifs is
    # labeled "motif failed: <reason>" (not a fixed string) — <reason> is
    # "no panel match" (this motif's tokens aren't a superset of any real
    # panel), "ambiguous panel match" (superset of 2+ independent, non-
    # nested panels — see the group_other_motifs_by_panel loop above), or
    # the fallback "not ranked" (group_other_motifs_by_panel=False, so no
    # matching was attempted at all). ───────────────────────────────────
    _is_panel_id = long_df[motif_col] == long_df["panel_id"]
    _fail_reason = long_df[motif_col].map(_fail_reason_map).fillna("not ranked")
    long_df["panel_classification"] = np.select(
        [
            _is_panel_id,
            long_df["_is_top_n"] & ~_is_panel_id,
            (~long_df["_is_top_n"]) & (long_df["panel_id"] != _OTHER_PANEL),
        ],
        [
            "panel header",
            "panel variant",
            "panel other",
        ],
        default="motif failed: " + _fail_reason,
    )
    long_df.loc[
        long_df["panel_classification"] == "panel other", "_selection_source"
    ] = "selected_other"

    # ── 2e: reintegrate min_motif_size-clipped motifs — deferred from the
    # ranking/expansion pipeline above (they never participate in
    # _resolve_panel/group_other_motifs_by_panel), reinstated here with an
    # explicit, distinct classification rather than left dropped. Assigned
    # to the _OTHER_MUTATIONS_PANEL sentinel ("Other mutations") — distinct
    # from _OTHER_PANEL ("Other motifs") — since this motif never got a
    # fair chance to compete at all (excluded before ranking even started),
    # unlike a "motif failed: ..." row which did compete and lost. Every
    # other column (motif_col, inns/adjusted_inns, counts, cluster_seeds,
    # ...) is preserved verbatim from their original motif_df row. ────────
    if not _clipped_df.empty:
        _clipped_reint = _clipped_df.copy()
        _clipped_reint["panel_id"] = _OTHER_MUTATIONS_PANEL
        _clipped_reint["panel_classification"] = "panel below min motif size cutoff"
        _clipped_reint["_is_top_n"] = False
        _clipped_reint["_selection_source"] = "not_selected"
        if "is_seed" not in _clipped_reint.columns:
            _clipped_reint["is_seed"] = False
        long_df = pd.concat([long_df, _clipped_reint], ignore_index=True, sort=False)

    # ── 2c: variant_label — the display/grouping identity consumed
    # downstream (build_motif_timeline_table/plot_motif_timeline). A static,
    # per-motif attribute fully determined by columns already computed above
    # (panel_id, panel_classification, motif_col) — no year-dependence — so
    # it's a pass-through here, same as panel_id/panel_classification,
    # rather than a derived value recomputed later. Equal to the motif's own
    # raw string for "panel header"/"panel variant"; "Other {panel_id}/+"
    # for "panel other" (several distinct real motifs share this one label,
    # all matched into the same real panel); otherwise equal to panel_id
    # itself — i.e. "Other motifs" for "motif failed: ..." rows, or "Other
    # mutations" for "panel below min motif size cutoff"/"untracked
    # mutation" rows (each collapses into its own respective generic
    # catch-all bucket — see docstring).
    long_df["variant_label"] = np.select(
        [
            long_df["panel_classification"].isin(["panel header", "panel variant"]),
            long_df["panel_classification"] == "panel other",
        ],
        [
            long_df[motif_col],
            "Other " + long_df["panel_id"].astype(str) + "/+",
        ],
        default=long_df["panel_id"],
    )

    # ── STEP 2f: constant_chain_info cross-reference — validate every INN
    # referenced anywhere in long_df (via inns_col/adjusted_inns_col) is
    # confirmed to actually have a constant-chain mutation, then synthesize
    # the two remaining bucket rows for whatever INN population is left
    # uncovered by a real motif row (see docstring for full semantics). ───
    inns_in_motif_table: set = set()
    for _col in (inns_col, adjusted_inns_col):
        for _raw in long_df[_col].dropna():
            inns_in_motif_table |= _as_inn_set(_raw)

    no_inn_mut_set: set = set(
        constant_chain_info.loc[
            constant_chain_info[has_constant_chain_mutations_col] == False,  # noqa: E712
            constant_chain_inn_col,
        ]
    )
    any_inn_mut_set: set = set(
        constant_chain_info.loc[
            constant_chain_info[has_constant_chain_mutations_col] == True,  # noqa: E712
            constant_chain_inn_col,
        ]
    )

    _bad_no_mut = inns_in_motif_table & no_inn_mut_set
    if _bad_no_mut:
        raise AssertionError(
            f"motif_yearly_counts: {len(_bad_no_mut)} INN(s) referenced by a "
            f"real motif row are confirmed in constant_chain_info to have NO "
            f"constant-chain mutation — a real motif can never reference a "
            f"mutation-free INN. Offending INNs: {sorted(_bad_no_mut)[:20]}"
        )
    _unconfirmed = inns_in_motif_table - any_inn_mut_set
    if _unconfirmed:
        raise AssertionError(
            f"motif_yearly_counts: {len(_unconfirmed)} INN(s) referenced by "
            f"a real motif row are not confirmed in constant_chain_info to "
            f"have a constant-chain mutation (either absent from the table, "
            f"or of unknown status) — every motif-table INN must be a "
            f"confirmed constant-chain-mutation carrier. Offending INNs: "
            f"{sorted(_unconfirmed)[:20]}"
        )

    _NO_MUT_LABEL = ""
    _UNTRACKED_PREFIX = "other_mutations_"
    _bucket_rows = []
    if no_inn_mut_set:
        _bucket_rows.append(
            {
                motif_col: _NO_MUT_LABEL,
                inns_col: set(no_inn_mut_set),
                adjusted_inns_col: set(no_inn_mut_set),
                inn_count_col: len(no_inn_mut_set),
                adjusted_count_col: len(no_inn_mut_set),
                "panel_id": _NO_MUTATIONS_PANEL,
                "panel_classification": "no mutation",
                "variant_label": _NO_MUTATIONS_PANEL,
                "_is_top_n": False,
                "_selection_source": "not_selected",
                "is_seed": False,
            }
        )
    _untracked_inns = any_inn_mut_set - inns_in_motif_table
    for _inn in sorted(_untracked_inns):
        _bucket_rows.append(
            {
                motif_col: f"{_UNTRACKED_PREFIX}{_inn}",
                inns_col: {_inn},
                adjusted_inns_col: {_inn},
                inn_count_col: 1,
                adjusted_count_col: 1,
                "panel_id": _OTHER_MUTATIONS_PANEL,
                "panel_classification": "untracked mutation",
                "variant_label": _OTHER_MUTATIONS_PANEL,
                "_is_top_n": False,
                "_selection_source": "not_selected",
                "is_seed": False,
            }
        )
    if _bucket_rows:
        long_df = pd.concat(
            [long_df, pd.DataFrame(_bucket_rows)], ignore_index=True, sort=False
        )

    # ── year: earliest year this motif's raw INN set was observed, resolved
    # via inn_df/date_col/inn_name_source — independent of count_col, so a
    # motif's first-appearance date doesn't shift depending on which
    # count/INN-set the adjustment-deduplication method happened to select. ─
    inn_flat, _year_inns, _no_date_inns = _resolve_inn_flat(
        inn_df, inn_name_source, date_col
    )
    _inn_year_lookup: dict = dict(zip(inn_flat[inn_name_source], inn_flat["_year"]))

    def _first_year(raw_inns) -> float:
        _inns = _as_inn_set(raw_inns)
        _years = [_inn_year_lookup[i] for i in _inns if i in _inn_year_lookup]
        return float(min(_years)) if _years else np.nan

    long_df["year"] = long_df[inns_col].apply(_first_year)

    # ── STEP 3: final minimal return schema ────────────────────────────────
    result = pd.DataFrame(
        {
            motif_col: long_df[motif_col],
            "adjusted_count": long_df[adjusted_count_col],
            "inn_count": long_df[inn_count_col],
            "year": long_df["year"],
            "count_col_val": long_df[_count_col_name],
            "panel_classification": long_df["panel_classification"],
            "panel_id": long_df["panel_id"],
            "variant_label": long_df["variant_label"],
            "is_seed": long_df["is_seed"],
            "is_top_n_selected": long_df["_is_top_n"],
            "selection_source": long_df["_selection_source"],
            "inns": long_df[inns_col],
            "adjusted_inns": long_df[adjusted_inns_col],
        }
    ).reset_index(drop=True)

    # Pass through the pre-existing Will-effector "group"/"sub_classification"
    # columns verbatim, if present on input — a wholly separate concept from
    # panel_id/panel_classification above (see docstring). Rows synthesized
    # by this function (inn_mut_info coverage / gap-fill) never had one of
    # these, so they correctly get NaN here rather than a placeholder.
    for _passthrough_col in ("group", "sub_classification"):
        if _passthrough_col in long_df.columns:
            result[_passthrough_col] = long_df[_passthrough_col].reset_index(drop=True)

    result.attrs["_motif_yearly_params"] = {
        "date_col": date_col,
        "inn_name_source": inn_name_source,
        "motif_col": motif_col,
        "count_col": count_col,
        "rank_by_col": rank_by_col,
        "min_count": min_count,
        "top_n": top_n,
        "top_n_by_seed_only": top_n_by_seed_only,
        "force_other_into_single_panel": force_other_into_single_panel,
    }
    return result


def _tl_bin_edges(years: pd.Series, date_bin) -> np.ndarray:
    vals = years.dropna().astype(int).values
    if date_bin == "auto":
        edges = np.histogram_bin_edges(vals, bins="fd")
    else:
        step = int(date_bin)
        lo = (vals.min() // step) * step
        hi = (vals.max() // step + 1) * step
        edges = np.arange(lo, hi + step, step, dtype=float)
    edges = np.unique(np.round(edges).astype(int)).astype(float)
    edges[0] = min(edges[0], float(vals.min()))
    edges[-1] = max(edges[-1], float(vals.max()) + 1)
    return edges


def build_motif_timeline_table(
    motif_counts_df: pd.DataFrame,
    inn_df: pd.DataFrame,
    date_col: str | list[str] | None = None,
    inn_name_source: str | None = None,
    date_bin="auto",
) -> pd.DataFrame:
    """Explode ``motif_yearly_counts``'s per-motif summary table into a
    per-(motif, year) timeline table — one row per real motif (or
    synthesized bucket row) per calendar year, adding a temporal layer on
    top of ``yearly`` without aggregating, regrouping, or reinterpreting
    anything ``motif_yearly_counts`` already computed. This function has NO
    knowledge of constant-chain mutation status at all — every motif-vs-
    no-mutation-vs-untracked classification decision is fully resolved
    upstream, once, in ``motif_yearly_counts`` (see its ``constant_chain_info``
    parameter); this function purely explodes whatever rows it's given
    across the calendar-year axis.

    ``motif_counts_df`` (``yearly``) must be the direct output of
    ``motif_yearly_counts`` — one row per distinct real motif (plus its
    synthesized "no mutation"/"untracked mutation" bucket rows). This
    function does not group by ``variant_label`` or combine multiple rows
    into one — every row of ``yearly`` keeps its own row here, once per
    calendar year.

    Only three things are added/redefined here, all inherently temporal:

    - ``year``, ``bin``, ``bin_group`` — the new per-row calendar-year axis.
    - ``inns`` — redefined from "this row's full raw INN set" (as in
      ``yearly``) to "this row's raw INNs approved in *this row's* year".
    - ``adjusted_inns`` — same narrowing, for the adjusted INN set.
    - ``count_differential``/``count_cumulative`` — new columns: unique
      *adjusted* INNs (``adjusted_inns``) with this row's identity approved
      in this year / up to and including this year. Always based on
      ``adjusted_inns`` — never ``inns`` — regardless of whichever
      ``count_col`` choice drove ranking/filtering *eligibility* upstream in
      ``motif_yearly_counts`` (that choice's scope is ranking/filtering
      only; it does not change which INN set is the "real" deduplicated
      count for timeline purposes).

    ``yearly``'s own ``year`` column (a single first-observed-year scalar
    per row) would otherwise collide with the new per-row calendar ``year``
    above — it is carried through unchanged, verbatim, under ``first_year``
    instead (a pure relabeling; the value/meaning from
    ``motif_yearly_counts`` is untouched by this rename).

    Every *other* column already in ``yearly`` — ``mutation_string_pe``,
    ``panel_id``, ``panel_classification``, ``variant_label``,
    ``adjusted_count``, ``inn_count``, ``count_col_val``, and the optional
    ``group``/``sub_classification`` Will-effector passthrough — is
    broadcast verbatim across every year-row for its motif. This function
    never re-derives, re-groups, or reinterprets any of them; it only reads
    them opaquely to carry them through unchanged.

    Parameters
    ----------
    inn_df, date_col, inn_name_source:
        ``inn_df``'s own schema is caller-specific (a separate table,
        unrelated to ``motif_yearly_counts``'s fixed output) — still the
        source of each INN's calendar year and the total per-year INN
        population (``inn_differential``/``inn_cumulative`` — across ALL
        INNs, motif or not; ``yearly`` only carries a single summary year
        per row, not a per-INN breakdown). ``date_col``/``inn_name_source``
        default to whatever ``motif_yearly_counts`` itself was called with,
        via attrs.

    Columns
    -------
    year : int  — calendar year for this row
    bin : int  — 0-based bin index for this year
    bin_group : str  — human-readable bin label (use for groupby)
    mutation_string_pe : str  — this row's own string, unchanged verbatim from ``yearly``
    count_differential : int  — unique adjusted INNs with this row's identity approved in this year
    count_cumulative : int  — unique adjusted INNs with this row's identity approved up to and including this year
    inn_differential : int  — distinct INNs approved in this year (across all of inn_df, independent of motif/mutation status)
    inn_cumulative : int  — distinct INNs approved up to and including this year
    inns : set  — this row's raw INNs approved in this year
    adjusted_inns : set  — this row's adjusted INNs approved in this year
    first_year : float  — yearly's own original ``year`` value, renamed (only present if ``yearly`` had it)
    panel_id, panel_classification, variant_label, is_seed,
    is_top_n_selected, selection_source, adjusted_count, inn_count,
    count_col_val, group, sub_classification : passthrough, verbatim from ``yearly`` (whichever are present)
    """
    _p = motif_counts_df.attrs.get("_motif_yearly_params")
    if _p is None:
        raise ValueError(
            "build_motif_timeline_table: motif_counts_df has no '_motif_yearly_params' attrs. "
            "Pass the direct output of motif_yearly_counts()."
        )

    date_col = date_col if date_col is not None else _p["date_col"]
    inn_name_source = (
        inn_name_source if inn_name_source is not None else _p["inn_name_source"]
    )
    motif_col = _p["motif_col"]

    _required_cols = (motif_col, "inns", "adjusted_inns")
    _missing = [c for c in _required_cols if c not in motif_counts_df.columns]
    if _missing:
        raise KeyError(
            f"build_motif_timeline_table: motif_counts_df is missing required "
            f"column(s) {_missing}. Pass the direct output of "
            f"motif_yearly_counts(). Available: {list(motif_counts_df.columns)}"
        )
    if not motif_counts_df[motif_col].is_unique:
        _dupes = motif_counts_df.loc[
            motif_counts_df[motif_col].duplicated(), motif_col
        ].tolist()
        raise AssertionError(
            f"build_motif_timeline_table: '{motif_col}' has duplicate value(s) "
            f"in motif_counts_df — this function requires exactly one row per "
            f"motif (the direct, unaggregated output of motif_yearly_counts), "
            f"not a variant_label-grouped table. Duplicates: {_dupes[:10]}"
        )

    def _as_inn_set(raw) -> set:
        if isinstance(raw, (set, frozenset)):
            return set(raw)
        if raw is None:
            return set()
        if np.isscalar(raw):
            if pd.isna(raw):
                return set()
            if isinstance(raw, str):
                _raw = raw.strip()
                return {_raw} if _raw else set()
            return {raw}
        return set(raw)

    # Per-year INN population, purely from inn_df — this function has no
    # knowledge of constant-chain mutation status at all (that
    # classification is fully resolved upstream, once, in
    # motif_yearly_counts via its own constant_chain_info parameter).
    _, year_inn_sets, _ = _resolve_inn_flat(inn_df, inn_name_source, date_col)
    all_years = sorted(y for y in year_inn_sets if y is not None)
    if not all_years:
        raise ValueError(
            "build_motif_timeline_table: no dated INNs found in inn_df via "
            f"date_col={date_col!r}/inn_name_source={inn_name_source!r} — "
            "cannot build a calendar-year timeline axis."
        )
    _all_years_vals = pd.Series(all_years, dtype=int)

    edges = _tl_bin_edges(_all_years_vals, date_bin)
    n_bins = len(edges) - 1

    def _label(k):
        lo, hi = int(edges[k]), int(edges[k + 1])
        return str(lo) if hi - lo <= 1 else f"{lo}–{hi - 1}"

    bin_labels = [_label(k) for k in range(n_bins)]

    def _assign_bin(series: pd.Series) -> pd.Series:
        return pd.cut(
            series.astype(int),
            bins=edges,
            right=False,
            labels=False,
            include_lowest=True,
        ).astype("Int64")

    year_bin: dict[int, int] = dict(
        zip(all_years, _assign_bin(_all_years_vals).astype(int).tolist())
    )

    inn_diff_year = pd.Series(
        {y: len(year_inn_sets.get(y, set())) for y in all_years},
        name="inn_differential",
    )
    inn_cum_year = inn_diff_year.cumsum().rename("inn_cumulative")

    # Every yearly column except inns/adjusted_inns (redefined to per-year
    # subsets below) and year (renamed to first_year to avoid colliding
    # with the new per-row calendar year) is carried through UNCHANGED,
    # broadcast across every year-row for its row — this function never
    # re-derives or reinterprets mutation_string_pe/panel_id/
    # panel_classification/variant_label/etc.
    _carry_cols = [
        c for c in motif_counts_df.columns if c not in ("inns", "adjusted_inns", "year")
    ]
    _has_scalar_year = "year" in motif_counts_df.columns

    records = []
    for _, row in motif_counts_df.iterrows():
        _inns_full = _as_inn_set(row["inns"])
        _adj_inns_full = _as_inn_set(row["adjusted_inns"])

        for y in all_years:
            _year_pop = year_inn_sets.get(y, set())
            rec = {c: row[c] for c in _carry_cols}
            if _has_scalar_year:
                rec["first_year"] = row["year"]
            rec["year"] = y
            rec["inns"] = _inns_full & _year_pop
            rec["adjusted_inns"] = _adj_inns_full & _year_pop
            # Always adjusted_inns (deduplicated) for the "real" count — see
            # docstring; count_col's scope upstream is ranking/filtering
            # eligibility only, not which INN set the timeline counts.
            rec["count_differential"] = len(_adj_inns_full & _year_pop)
            records.append(rec)

    agg = pd.DataFrame(records).sort_values(["year", motif_col]).reset_index(drop=True)
    agg["count_cumulative"] = agg.groupby(motif_col)["count_differential"].transform(
        "cumsum"
    )

    # Attach year-level metadata
    year_meta = pd.DataFrame(
        {
            "year": all_years,
            "bin": [year_bin[y] for y in all_years],
            "bin_group": [bin_labels[year_bin[y]] for y in all_years],
            "inn_differential": inn_diff_year.values,
            "inn_cumulative": inn_cum_year.values,
        }
    )
    result = agg.merge(year_meta, on="year")

    # Defensive invariant: every mutation_string_pe value in the output must
    # come from — and exactly match — motif_counts_df's own values. No
    # exceptions/reserved sentinels: this function no longer synthesizes ANY
    # new mutation_string_pe identity itself (the "" no-mutation and
    # "other_mutations_{inn}" untracked buckets already arrive as ordinary
    # rows of motif_counts_df, fully resolved by motif_yearly_counts). This
    # check exists purely to hard-fail immediately, rather than silently
    # drift, if that ever stops being true (e.g. a future edit reintroduces
    # aggregation).
    _real_motifs_out = set(result[motif_col].dropna())
    _real_motifs_in = set(motif_counts_df[motif_col].dropna())
    if _real_motifs_out != _real_motifs_in:
        _extra = _real_motifs_out - _real_motifs_in
        _missing = _real_motifs_in - _real_motifs_out
        raise AssertionError(
            f"build_motif_timeline_table: internal invariant violated — "
            f"output {motif_col!r} values must exactly match "
            f"motif_counts_df's own values. "
            f"Extra (not in motif_counts_df): {sorted(_extra)[:10]}. "
            f"Missing (in motif_counts_df but absent from output): "
            f"{sorted(_missing)[:10]}. If you are seeing this after editing "
            f"motif_yearly_counts/build_motif_timeline_table in a running "
            f"notebook, restart the kernel — a stale imported module/"
            f"variable is a common cause of exactly this symptom."
        )

    _other_cols = [c for c in _carry_cols if c != motif_col] + (
        ["first_year"] if _has_scalar_year else []
    )
    col_order = [
        "year",
        "bin",
        "bin_group",
        motif_col,
        "count_differential",
        "count_cumulative",
        "inn_differential",
        "inn_cumulative",
        "inns",
        "adjusted_inns",
    ] + _other_cols
    return result[col_order].sort_values(["year", motif_col]).reset_index(drop=True)


def build_motif_timeline_table_from_counts(
    motif_df: pd.DataFrame,
    motif_col: str = "mutation_string_pe",
    min_year_col: str = "min_year",
    count_col: str = "adjusted_count",
    date_bin: int | str = 1,
    extra_cols: list[str] | None = None,
    cluster_col: str | None = "cluster_seeds",
) -> pd.DataFrame:
    """Build a long (year, motif) timeline table directly from a motif DataFrame.

    Unlike :func:`build_motif_timeline_table`, this function does **not** require
    INN-name sets or a separate ``inn_df``.  Instead it uses the ``min_year_col``
    already present on ``motif_df`` as the motif's first-appearance year and treats
    ``count_col`` as the motif's size at that year (step-function semantics: once a
    motif appears it contributes its count for every subsequent year).

    Parameters
    ----------
    motif_df:
        One row per motif.  Must contain ``motif_col``, ``min_year_col``, and
        ``count_col``.
    motif_col:
        Column used as the motif label (default ``"mutation_string_pe"``).
    min_year_col:
        Column holding the first year the motif was observed (default ``"min_year"``).
    count_col:
        Column holding the motif count/size (default ``"adjusted_count"``).
    date_bin:
        Bin width in years (``1`` = per-year, ``"auto"`` = Freedman–Diaconis).
    extra_cols:
        Additional columns from ``motif_df`` to carry through into the output
        (e.g. ``["group", "gene", "inn_count"]``).
    cluster_col:
        Column in ``motif_df`` whose value is emitted as ``cluster_seed`` in the
        output (used by ``plot_motif_timeline`` for coloring/grouping).  Set/list/
        frozenset values are sorted and joined with ``", "`` so all cluster names
        are visible.  Defaults to ``"cluster_seeds"``; pass ``None`` to omit.

    Returns
    -------
    DataFrame with columns:
        ``year``, ``bin``, ``bin_group``, ``<motif_col>``,
        ``count_differential`` (count_col value in min_year, else 0),
        ``count_cumulative`` (running sum — equals count_col from min_year onward),
        ``cluster_seed`` (from ``cluster_col`` if present).
    """
    required = [motif_col, min_year_col, count_col]
    missing = [c for c in required if c not in motif_df.columns]
    if missing:
        raise KeyError(
            f"build_motif_timeline_table_from_counts: missing columns {missing}. "
            f"Available: {list(motif_df.columns)}"
        )

    df = motif_df.dropna(subset=[motif_col, min_year_col]).copy()
    df[min_year_col] = df[min_year_col].astype(int)

    year_min = int(df[min_year_col].min())
    year_max = int(df[min_year_col].max())
    all_years = list(range(year_min, year_max + 1))

    # Bin edges
    year_series = pd.Series(all_years)
    edges = _tl_bin_edges(year_series, date_bin)
    n_bins = len(edges) - 1

    def _label(k):
        lo, hi = int(edges[k]), int(edges[k + 1])
        return str(lo) if hi - lo <= 1 else f"{lo}–{hi - 1}"

    bin_labels = [_label(k) for k in range(n_bins)]
    year_bin: dict[int, int] = {}
    for yr in all_years:
        idx = int(
            pd.cut(
                [yr],
                bins=edges,
                right=False,
                labels=False,
                include_lowest=True,
            )[0]
        )
        year_bin[yr] = idx

    _extra = [c for c in (extra_cols or []) if c in motif_df.columns]
    # cluster_col is emitted as cluster_seed; don't double-carry via extra_cols
    _cluster_src = (
        cluster_col if (cluster_col and cluster_col in motif_df.columns) else None
    )
    _carry = [c for c in _extra if c != cluster_col]

    def _normalize_cluster(v) -> str:
        """Coerce a set/list/frozenset/str cluster value to a comma-joined string."""
        if isinstance(v, (set, frozenset, list, tuple)):
            parts = sorted(str(x) for x in v if x is not None and str(x).strip())
            return ", ".join(parts) if parts else ""
        if v is None or (isinstance(v, float) and pd.isna(v)):
            return ""
        return str(v).strip()

    records = []
    for _, row in df.iterrows():
        label = row[motif_col]
        first_yr = int(row[min_year_col])
        cnt = row[count_col] if pd.notna(row[count_col]) else 0
        carry_vals = {c: row[c] for c in _carry}
        cluster_val = _normalize_cluster(row[_cluster_src]) if _cluster_src else ""
        cum = 0
        for yr in all_years:
            diff = cnt if yr == first_yr else 0
            cum += diff
            rec = {
                "year": yr,
                motif_col: label,
                "count_differential": diff,
                "count_cumulative": cum,
                "cluster_seeds": cluster_val,
            }
            rec.update(carry_vals)
            records.append(rec)

    agg = pd.DataFrame(records)

    # Attach bin metadata
    year_meta = pd.DataFrame(
        {
            "year": all_years,
            "bin": [year_bin[y] for y in all_years],
            "bin_group": [bin_labels[year_bin[y]] for y in all_years],
        }
    )
    result = agg.merge(year_meta, on="year")

    col_order = [
        "year",
        "bin",
        "bin_group",
        motif_col,
        "count_differential",
        "count_cumulative",
        "cluster_seeds",
    ] + _carry
    return result[col_order].sort_values(["year", motif_col]).reset_index(drop=True)


def cluster_motifs_by_substring(df, mutation_col="mutation_string"):
    """
    Cluster motifs by mutation-set subset membership.

    Motifs are processed fewest-mutations-first. For each motif M:
      - If any existing cluster seed S has mutations that are a subset of M → add M to all such clusters.
      - If no existing seed is a subset of M → M starts its own cluster (singleton).

    A member can belong to multiple clusters.
    Representative of each cluster = member with highest count, then size.
    Primary cluster(s) = the largest cluster(s) the member belongs to.
    """
    df = df[df["num_mutations"] > 1].copy()
    rows = (
        df[[mutation_col, "inn_count", "num_mutations"]].copy().reset_index(drop=True)
    )
    count_map = dict(zip(rows[mutation_col], rows["inn_count"]))
    size_map = dict(zip(rows[mutation_col], rows["num_mutations"]))

    def _to_muts(ms):
        return frozenset(
            m.strip() for m in re.split(r"\s*/\s*", str(ms).strip()) if m.strip()
        )

    muts_map = {ms: _to_muts(ms) for ms in rows[mutation_col]}

    sorted_ms = sorted(rows[mutation_col].tolist(), key=lambda ms: len(muts_map[ms]))

    clusters: dict[str, set] = {}
    for ms in sorted_ms:
        matched = [seed for seed in clusters if muts_map[seed] <= muts_map[ms]]
        if matched:
            for seed in matched:
                clusters[seed].add(ms)
        else:
            clusters[ms] = {ms}

    cluster_info = {
        seed: {
            "members": list(members),
            "n_members": len(members),
            "representative": max(
                members,
                key=lambda m: (count_map.get(m, 0), size_map.get(m, 0)),
            ),
        }
        for seed, members in clusters.items()
    }

    member_to_seeds: dict[str, list] = {}
    for seed, info in cluster_info.items():
        for ms in info["members"]:
            member_to_seeds.setdefault(ms, []).append(seed)

    def _primary(ms):
        seeds = member_to_seeds.get(ms, [])
        if not seeds:
            return [], None
        max_sz = max(cluster_info[s]["n_members"] for s in seeds)
        primaries = [s for s in seeds if cluster_info[s]["n_members"] == max_sz]
        return primaries, cluster_info[primaries[0]]["representative"]

    df_out = df.copy()
    df_out["cluster_seeds"] = df_out[mutation_col].map(
        lambda ms: set(member_to_seeds.get(ms, []))
    )
    df_out["n_clusters"] = df_out["cluster_seeds"].map(len)
    df_out["primary_cluster_seeds"] = df_out[mutation_col].map(
        lambda ms: _primary(ms)[0]
    )
    df_out["primary_cluster_rep"] = df_out[mutation_col].map(lambda ms: _primary(ms)[1])

    return cluster_info, df_out


def build_inn_mutation_timeline(
    ds: "INNAnalysisDataset",
    date_col: str = "min_year",
    date_bin="auto",
    chain_subset: Optional[list] = None,
    region_subset: Optional[list] = None,
) -> pd.DataFrame:
    """Build a per-(year, INN) table of mutation statistics.

    One row per INN in the constant-chain universe (INNs with no mutations
    are included with zeros). Bin columns are added for downstream aggregation.

    Parameters
    ----------
    ds : :class:`~data_io.merge.loader.INNAnalysisDataset` — the loaded
        analysis dataset.  Three attributes are consumed:

        * ``ds.constant_chains`` — defines the INN universe (INNs with at
          least one constant-chain sequence).
        * ``ds.kept_inn`` — provides the date column (``date_col``) for each
          INN; indexed by ``inn_name``.
        * ``ds.constant_chain_motifs_by_inn_region`` — one row per
          INN × region; ``constant_chain_mutational_motif`` (comma-separated
          mutation tokens) is used to count unique mutations per INN.

    date_col : column in ``ds.kept_inn`` to use as the year.  Default
        ``"min_year"`` (the earliest of ``year_received``, ``year_proposed``,
        ``year_recommended``).
    date_bin : bin width in years, or ``"auto"`` for Freedman-Diaconis bins.
    chain_subset : optional list of ``chain_type`` values to include
        (e.g. ``["HC"]``).  ``None`` includes all chains.
    region_subset : optional list of ``region_classification`` values to
        include (e.g. ``["FC", "HINGE"]``).  ``None`` includes all regions.

    Returns
    -------
    DataFrame with one row per (year, inn_name):
        year, bin, bin_group, inn_name,
        num_mutations   — unique mutation tokens across included chains/regions,
        chains_mutated  — # distinct ``chain_type`` values with ≥1 mutation,
        regions_mutated — # distinct ``region_classification`` values with ≥1 mutation,
        chain_type_list — sorted list of mutated chain types,
        region_list     — sorted list of mutated region classifications,
        is_mutated      — bool
    """
    from inn_analysis_ops.data_io.merge.loader import _inn_set

    # ── INN universe: kept INNs that have at least one constant-chain sequence ──
    inn_universe = _inn_set(ds.constant_chains)

    flat = (
        ds.kept_inn.reset_index()[["inn_name", date_col]]
        .dropna(subset=[date_col])
        .query("inn_name in @inn_universe")
        .copy()
    )
    flat[date_col] = flat[date_col].astype(int)

    edges = _tl_bin_edges(flat[date_col], date_bin)

    def _label(k):
        lo, hi = int(edges[k]), int(edges[k + 1])
        return str(lo) if hi - lo <= 1 else f"{lo}–{hi - 1}"

    flat["bin"] = pd.cut(
        flat[date_col],
        bins=edges,
        right=False,
        labels=False,
        include_lowest=True,
    ).astype("Int64")
    flat["bin_group"] = flat["bin"].apply(lambda k: _label(int(k)))

    # ── Apply optional chain/region filters ────────────────────────────────
    mc = ds.constant_chain_motifs_by_inn_region
    if chain_subset is not None:
        mc = mc[mc["chain_type"].isin(chain_subset)]
    if region_subset is not None:
        mc = mc[mc["region_classification"].isin(region_subset)]

    # ── Per-(inn, region_classification): unique mutation tokens ───────────
    def _unique_tokens(series: pd.Series) -> set:
        tokens: set = set()
        for s in series:
            if isinstance(s, str):
                tokens.update(t.strip() for t in s.split(",") if t.strip())
        return tokens

    motifs_by_region = (
        mc.groupby(["inn", "region_classification"])["constant_chain_mutational_motif"]
        .apply(_unique_tokens)
        .reset_index(name="_mut_set")
    )
    motifs_by_region["_n"] = motifs_by_region["_mut_set"].apply(len)

    # Attach chain_type per (inn, region_classification) for chain stats
    chain_per_region = mc[
        ["inn", "region_classification", "chain_type"]
    ].drop_duplicates()
    motifs_by_region = motifs_by_region.merge(
        chain_per_region, on=["inn", "region_classification"], how="left"
    )

    # Keep only regions that carry ≥1 mutation for chain/region counts
    mutated = motifs_by_region[motifs_by_region["_n"] > 0]

    # ── Aggregate per INN ──────────────────────────────────────────────────
    # num_mutations: count unique tokens across all regions for each INN
    all_tokens = (
        motifs_by_region.groupby("inn")["_mut_set"]
        .apply(lambda sets: len(set().union(*sets)))
        .rename("num_mutations")
    )
    chain_stats = mutated.groupby("inn").agg(
        chains_mutated=("chain_type", "nunique"),
        chain_type_list=("chain_type", lambda x: sorted(x.unique())),
        regions_mutated=("region_classification", "nunique"),
        region_list=("region_classification", lambda x: sorted(x.unique())),
    )
    inn_stats = (
        all_tokens.to_frame()
        .join(chain_stats, how="left")
        .reset_index()
        .rename(columns={"inn": "inn_name"})
    )

    # ── Merge onto flat INN×year table; fill zeros for unmutated INNs ──────
    result = flat.merge(inn_stats, on="inn_name", how="left")
    result["num_mutations"] = result["num_mutations"].fillna(0).astype(int)
    result["chains_mutated"] = result["chains_mutated"].fillna(0).astype(int)
    result["regions_mutated"] = result["regions_mutated"].fillna(0).astype(int)
    result["chain_type_list"] = result["chain_type_list"].apply(
        lambda x: x if isinstance(x, list) else []
    )
    result["region_list"] = result["region_list"].apply(
        lambda x: x if isinstance(x, list) else []
    )
    result["is_mutated"] = result["num_mutations"] > 0

    return (
        result.rename(columns={date_col: "year"})
        .sort_values(["year", "inn_name"])
        .reset_index(drop=True)
    )


def subtract_parent_inns(
    motif_df: pd.DataFrame,
    mutation_col: str = "mutation_string",
    ancestry_col: str = "sub_motifs",
    id_col: str = "motif_id",
    filters: Optional[dict] = None,
) -> pd.DataFrame:
    """Remove INNs from sub-motifs that are already explained by a parent motif.

    For each motif, any INN that also belongs to a larger parent motif (recorded
    in ``ancestry_col``) is removed from the sub-motif's ``inns`` set so each INN
    is attributed only to its most complex motif.  Rows whose INN set becomes
    empty after subtraction are dropped.

    Each row is identified by ``id_col`` (default ``motif_id``), a unique integer
    added by ``build_motif_index``.  Ancestry references (stored as a list of
    ``mutation_string`` values in ``ancestry_col``) are resolved within the same
    group context (same gene / chain_type / gene_species) so that identically-named
    motifs from different genes do not incorrectly share INN subtraction.

    Parameters
    ----------
    motif_df : DataFrame produced by ``build_motif_index`` (must have ``id_col``).
    mutation_col : column holding the motif label string.
    ancestry_col : column holding a list of sub-motif mutation_string labels (default ``sub_motifs``).
    id_col : unique-per-row integer identifier column (added by build_motif_index).
    filters : optional dict with any of:
        - ``min_size``        : int
        - ``max_size``        : int
        - ``min_count``       : int
        - ``max_pvalue``      : float
        - ``max_pvalue_ancestry`` : float
        - ``min_lift``        : float
        - ``max_primary_cluster_seeds`` : int

    Returns
    -------
    DataFrame with added ``adjusted_inns``, ``adjusted_count``,
    ``adjusted_freq``, and ``superset_adjusted_count`` columns; the
    original ``inns``, ``count``, and ``freq`` columns are preserved
    unchanged.  Rows where ``adjusted_count == 0`` are dropped.

    ``superset_adjusted_count``: for each row, the sum of ``adjusted_count``
    values across all rows that contain **at least** this row's mutations
    (i.e., this motif's mutation set is a subset of theirs).  Unlike a unique
    INN union, this sum allows the same INN to be counted multiple times across
    different superset rows — capturing cases where the same INN appears with
    both a simpler motif (e.g. LALA) and a larger motif (e.g. LALA+PG) on the
    same chain.  Scoped by gene/chain_type/gene_species group.
    """
    filters = filters or {}
    _VALID_FILTER_KEYS = {
        "min_size",
        "max_size",
        "min_count",
        "max_pvalue",
        "max_pvalue_ancestry",
        "min_lift",
        "max_primary_cluster_seeds",
    }
    unknown_keys = set(filters.keys()) - _VALID_FILTER_KEYS
    if unknown_keys:
        raise ValueError(
            f"subtract_parent_inns: unrecognised filter key(s): {sorted(unknown_keys)}. "
            f"Valid keys are: {sorted(_VALID_FILTER_KEYS)}"
        )
    _GROUP_COLS = ["gene", "chain_type", "gene_species"]

    df = motif_df.copy()

    # Map each filter key to the dataframe column it requires
    _FILTER_COLUMN_MAP = {
        "min_size": "num_mutations",
        "max_size": "num_mutations",
        "min_count": "inn_count",
        "max_pvalue": "pvalue_vs_indep",
        "max_pvalue_ancestry": "pvalue_ancestry",
        "min_lift": "lift_vs_indep",
        "max_primary_cluster_seeds": "primary_cluster_seeds",
    }
    for fkey, col in _FILTER_COLUMN_MAP.items():
        if fkey in filters and col not in df.columns:
            raise ValueError(
                f"subtract_parent_inns: filter {fkey!r} requires column {col!r} "
                f"which is not present in the dataframe. "
                f"Available columns: {sorted(df.columns.tolist())}"
            )

    # Apply pre-filters
    if "min_size" in filters:
        df = df[df["num_mutations"] >= filters["min_size"]]
    if "max_size" in filters:
        df = df[df["num_mutations"] <= filters["max_size"]]
    if "min_count" in filters:
        df = df[df["inn_count"] >= filters["min_count"]]
    if "max_pvalue" in filters:
        df = df[df["pvalue_vs_indep"] <= filters["max_pvalue"]]
    if "max_pvalue_ancestry" in filters:
        mask = df["pvalue_ancestry"].isna() | (
            df["pvalue_ancestry"] <= filters["max_pvalue_ancestry"]
        )
        df = df[mask]
    if "min_lift" in filters:
        df = df[df["lift_vs_indep"] >= filters["min_lift"]]
    if "max_primary_cluster_seeds" in filters:
        df = df[
            df["primary_cluster_seeds"].apply(
                lambda x: len(x) if isinstance(x, (set, list)) else 0
            )
            <= filters["max_primary_cluster_seeds"]
        ]

    tmp = df.sort_values(by="num_mutations", ascending=False).reset_index(drop=True)

    if not tmp[id_col].is_unique:
        raise ValueError(
            f"subtract_parent_inns: '{id_col}' contains duplicate values. "
            "Re-run build_motif_index with group_by to get globally unique motif_ids."
        )

    # Delegate to the canonical _subtract_inns_inline from inn_analysis_ops.analytics.motifs.subtraction.
    # pre-filters have already been applied above; ancestry_col is passed through.
    from inn_analysis_ops.analytics.motifs.subtraction import (
        _subtract_inns_inline as _sub_fn,
    )

    tmp = _sub_fn(tmp, ancestry_col=ancestry_col, id_col=id_col)

    # Drop rows where adjusted_count == 0 (absorbed by larger motifs)
    result = tmp[tmp["adjusted_count"] > 0].copy().reset_index(drop=True)

    # ── superset_adjusted_count ──────────────────────────────────────────────
    # For each row A, find all rows B where A's mutations are a subset of B's
    # mutations (B "contains" A), then sum their adjusted_counts.  Unlike a
    # unique-INN union, this sum allows the same INN to be counted multiple
    # times across different superset rows — useful when the same INN appears
    # with distinct larger motifs (e.g. LALA alone vs. LALA+PG on the same
    # chain).  Lookup is scoped by gene/chain_type/gene_species group.

    # Build frozenset of mutation tokens per row for fast subset testing.
    # mutation_list is a list column if present; otherwise parse mutation_string.
    if "mutation_list" in result.columns:
        _mut_sets = (
            result["mutation_list"]
            .apply(
                lambda x: frozenset(x) if isinstance(x, (list, set)) else frozenset()
            )
            .tolist()
        )
    elif "motif" in result.columns:
        _mut_sets = (
            result["motif"]
            .apply(
                lambda x: (
                    frozenset(x) if isinstance(x, (set, frozenset)) else frozenset()
                )
            )
            .tolist()
        )
    else:
        _mut_sets = (
            result[mutation_col]
            .apply(
                lambda s: (
                    frozenset(t.strip() for t in s.split("+"))
                    if isinstance(s, str)
                    else frozenset()
                )
            )
            .tolist()
        )

    _adj_counts = result["adjusted_count"].tolist()

    _group_keys = [
        tuple(result.at[i, c] for c in group_cols if c in result.columns)
        for i in range(len(result))
    ]

    superset_sums = []
    for i in range(len(result)):
        my_muts = _mut_sets[i]
        my_gkey = _group_keys[i]
        total = 0
        for j in range(len(result)):
            if _group_keys[j] != my_gkey:
                continue
            if my_muts.issubset(_mut_sets[j]):  # row j contains AT LEAST my mutations
                total += _adj_counts[j]
        superset_sums.append(total)

    result["superset_adjusted_count"] = superset_sums
    return result


def select_mutations_outside_of_motifs(motif_df, yearly, yearly_mutation_col: str):
    """Classify mutations that fall outside of seed (yearly) motifs as "other mutations".

    Takes motif_df and yearly dataframes and identifies mutations in motif_df that are NOT
    fully present in any yearly motif. These "other mutations" are grouped by gene/chain
    and deduplicated by remaining mutation set.

    Parameters
    ----------
    motif_df : pd.DataFrame
        Motif dataframe with columns: gene, gene_species, chain_type_h_or_l, mutation_string,
        inn_count, inn_total, inns, num_mutations, etc.
    yearly : pd.DataFrame
        Yearly motif dataframe with mutation_string column (seed motifs).

    Returns
    -------
    pd.DataFrame
        Deduplicated and collapsed dataframe of "other mutations" with all
        required columns to match motif_df schema:
        - gene, gene_species, chain_type_h_or_l (grouping keys)
        - mutation_string (str representation)
        - num_mutations (count)
        - inns, inn_count, inn_total (aggregated)
        - adjusted_inns, adjusted_inn_count (after deduplication by superset)
        - motif_id (sequential index)
        - cluster_seeds: empty set (for compatibility)
        - primary_cluster_seeds: "other mutations" label
    """
    # Add pre-formatted PE label column
    try:
        from .position_labeling import format_motif_label as _fmt_motif
    except ImportError:
        from ._helpers.position_labeling import (
            format_motif_label as _fmt_motif,
        )

    def _safe_fmt(x):
        if not isinstance(x, str) or not x:
            return x
        try:
            return _fmt_motif(x)
        except (ValueError, Exception):
            return x

    def remove_all_motifs(mutation_set, motifs_to_remove):
        """Remove all yearly motif patterns from a mutation set."""
        for m in motifs_to_remove:
            mutation_set = mutation_set - m
            if not len(mutation_set):
                return set()
        return mutation_set

    def _tok_normalize(tok):
        import re as _re2

        tok = tok.strip()
        # Match tuple repr: (pos, 'mut') or ('pos', 'mut') — convert to pos:mut
        m = _re2.fullmatch(
            r"\(\s*[\"']?([^\"',)\s]+)[\"']?\s*,\s*[\"']([^\"']*)[\"']\s*\)",
            tok,
        )
        if m:
            pos, mut = m.group(1).strip(), m.group(2).strip()
            try:
                f = float(pos)
                pos = str(int(f)) if f == int(f) else pos
            except (ValueError, OverflowError):
                pass
            return f"{pos}:{mut}"
        return tok

    def _split_ms(v):
        if not isinstance(v, str) or not v:
            return set()
        parts = [r.strip() for r in v.replace(" / ", "|||").split("|||") if r.strip()]
        if len(parts) == 1 and "/" in v:
            parts = [r.strip() for r in v.split("/") if r.strip()]
        return set(_tok_normalize(t) for t in parts if t)

    # Extract yearly motif patterns
    if yearly_mutation_col not in yearly.columns:
        raise KeyError(
            f"select_mutations_outside_of_motifs: column '{yearly_mutation_col}' not found in yearly. "
            f"Available: {list(yearly.columns)}"
        )
    yearly_mutation_string_sets = yearly[yearly_mutation_col].apply(_split_ms)

    # Parse motif_df mutations
    motif_df_copy = motif_df.sort_values(by="num_mutations").copy()
    all_mutation_set = motif_df_copy["mutation_string"].apply(_split_ms)

    # Remove yearly motifs from each motif
    motif_df_copy["remaining_mutations"] = all_mutation_set.apply(
        lambda x: tuple(sorted(remove_all_motifs(x, yearly_mutation_string_sets)))
    )

    # Filter to only motifs with remaining mutations
    motif_df_copy_trimmed = motif_df_copy[
        motif_df_copy["remaining_mutations"].apply(lambda x: len(x) > 0)
    ]

    # Collapse by (gene, gene_species, chain_type_h_or_l, remaining_mutations)
    motif_df_copy_trimmed_collapsed = motif_df_copy_trimmed.groupby(
        by=[
            "gene",
            "gene_species",
            "chain_type_h_or_l",
            "remaining_mutations",
        ],
        as_index=False,
    ).agg(
        {
            "inn_count": "sum",
            "inn_total": "first",
            "inns": lambda x: set([a for y in x for a in y]),
        }
    )

    # Reconstruct mutation columns
    motif_df_copy_trimmed_collapsed["mutation_string"] = (
        motif_df_copy_trimmed_collapsed["remaining_mutations"].apply(
            lambda x: " / ".join(x)
        )
    )
    motif_df_copy_trimmed_collapsed["mutations"] = motif_df_copy_trimmed_collapsed[
        "remaining_mutations"
    ].apply(set)
    motif_df_copy_trimmed_collapsed["num_mutations"] = motif_df_copy_trimmed_collapsed[
        "remaining_mutations"
    ].apply(len)
    motif_df_copy_trimmed_collapsed["motif_id"] = np.arange(
        len(motif_df_copy_trimmed_collapsed)
    )

    # Remove INN duplicates: if a mutation set is a subset of another, remove its INNs
    new_set = []
    for idx, row in motif_df_copy_trimmed_collapsed.sort_values(
        by="num_mutations", ascending=False
    ).iterrows():
        current_inns = list(row["inns"])

        # Check against already-added motifs (larger sets first)
        for s in new_set:
            # Match by gene attributes
            all_match = all(
                row[chk] == s[chk]
                for chk in ["gene", "gene_species", "chain_type_h_or_l"]
            )
            if not all_match:
                continue

            # If row's mutations are a subset of s's mutations, remove row's INNs
            if len(row["mutations"] - s["mutations"]) == 0:
                current_inns = list(set(current_inns) - set(s["adjusted_inns"]))

        row_counts = len(current_inns)
        if row_counts == 0:
            continue
        if row_counts < 0:
            raise AssertionError(f"Negative INN count at idx {idx}: {row}")

        row["adjusted_inns"] = set(current_inns)
        row["adjusted_count"] = row_counts
        new_set.append(row)

    result = pd.DataFrame(new_set)

    # Add required columns for compatibility with motif_df schema
    result["cluster_seeds"] = [set() for _ in range(len(result))]
    result["primary_cluster_seeds"] = "other mutations"
    result["sub_motifs"] = [[] for _ in range(len(result))]

    result["mutation_string_pe"] = result["mutation_string"].map(_safe_fmt)
    result["novel_mutations"] = result["mutation_string"]
    result["most_sig_sub_motif"] = None
    result["mutation_list"] = result["mutations"].apply(lambda x: list(x))

    return result


# ── Export helper ─────────────────────────────────────────────────────────────

_CO_COLUMN_LABELS: dict[str, str] = {
    "classification": "pair_type",
    "gene_species": "species",
    "p0": "position_1",
    "gene_0": "gene_position_1",
    "region_0": "region_position_1",
    "gene_detail_0": "inn_count_by_gene_position_1",
    "p1": "position_2",
    "gene_1": "gene_position_2",
    "region_1": "region_position_2",
    "gene_detail_1": "inn_count_by_gene_position_2",
    # derived columns added by this function
    "top_mutation_A": "top_mutation_position_1",
    "top_mutation_B": "top_mutation_position_2",
    "top_variant_pair_label": "top_mutation_pair",
    "all_mutation_pairs": "all_mutation_pairs",
    # counts
    "count": "num_unique_inn_co_occurring",
    "n_p0": "num_unique_inn_with_any_mut_position_1",
    "n_p1": "num_unique_inn_with_any_mut_position_2",
    "n_total": "num_unique_inn_in_group",
    # statistics
    "lift": "lift",
    "log_lift": "log_lift",
    "npmi": "npmi",
    "min_conf": "min_confidence",
    "odds_ratio": "odds_ratio",
    "pvalue": "fisher_pvalue",
    # top variant detail
    "top_var_sig": "top_variant_pair_raw",
    "top_var_count": "top_variant_num_unique_inn_co_occurring",
    "top_var_n_p0": "top_variant_num_unique_inn_with_mut_position_1",
    "top_var_n_p1": "top_variant_num_unique_inn_with_mut_position_2",
    "top_var_lift": "top_variant_lift",
    "top_var_npmi": "top_variant_npmi",
    "top_var_or": "top_variant_odds_ratio",
    "top_var_pvalue": "top_variant_fisher_pvalue",
    "top_var_min_conf": "top_variant_min_confidence",
    "n_variants": "num_variant_pairs",
}

_CO_EXPORT_COLS: list[str] = [
    # ── Lead: pair type then position pair identity ───────────────────────────
    "pair_type",
    "species",
    "position_1",
    "position_2",
    "top_mutation_pair",  # e.g. "L235A / P238S"
    "all_mutation_pairs",  # e.g. "L235A / P238S: 5 | L235E / P238S: 2"
    # ── Per-position context ──────────────────────────────────────────────────
    "gene_position_1",
    "region_position_1",
    "inn_count_by_gene_position_1",
    "gene_position_2",
    "region_position_2",
    "inn_count_by_gene_position_2",
    # ── Core counts ───────────────────────────────────────────────────────────
    "num_unique_inn_co_occurring",
    "num_unique_inn_with_any_mut_position_1",
    "num_unique_inn_with_any_mut_position_2",
    "num_unique_inn_in_group",
    # ── Key statistics ────────────────────────────────────────────────────────
    "lift",
    "log_lift",
    "npmi",
    "min_confidence",
    "odds_ratio",
    "fisher_pvalue",
]


def co_df_to_export_table(
    co_df: pd.DataFrame,
    keep_classification_pairs: list[str] | None = None,
) -> pd.DataFrame:
    """Convert a raw ``compute_co_mutation_frequencies`` result to a clean
    export-ready DataFrame.

    Steps applied
    -------------
    1. Optionally filter to specific ``classification`` values via
       ``keep_classification_pairs``.
    2. Derive ``top_mutation_position_1``, ``top_mutation_position_2``,
       ``top_mutation_pair`` (e.g. ``"L235A / P238S"``), and
       ``all_mutation_pairs`` (all variant pairs with counts, sorted by count
       descending, e.g. ``"L235A / P238S: 5 | L235E / P238S: 2"``) from the
       raw ``top_var_sig`` and ``mutations`` columns.
    3. Rename all columns to human-readable labels.
    4. Keep only the key statistical and identity columns.
    5. Sort by ``num_unique_inn_in_group`` descending, then
       ``fisher_pvalue`` ascending.

    Parameters
    ----------
    co_df:
        DataFrame returned by ``compute_co_mutation_frequencies`` (or filtered
        by ``remove_intra_pairs``).
    keep_classification_pairs:
        Optional list of ``classification`` values to retain.  If ``None``
        all rows are kept.  Example::

            keep_classification_pairs=["intra", "inter_same_gene"]

        Available values: ``"intra"``, ``"intra_repeat"``,
        ``"intra_pos_repeat_diff_mut"``, ``"inter_same_gene"``,
        ``"inter_cross_gene"``, ``"shared_mutation_cross_chain"``,
        ``"filtered_by_intra_match"``.

    Returns
    -------
    Clean DataFrame ready for ``DataFrame.to_excel`` / ``to_csv``.
    """
    df = co_df.copy()

    # ── 0. Optional classification filter ────────────────────────────────────
    if keep_classification_pairs is not None:
        df = df[df["classification"].isin(keep_classification_pairs)]

    # ── 1. Derive mutation labels ─────────────────────────────────────────────
    def _fmt(mut_raw, pos):
        if not isinstance(mut_raw, str) or "->" not in mut_raw:
            return None
        wt, mt = mut_raw.split("->", 1)
        return f"{wt}{pos}{mt}"

    def _fmt_pair_sig(sig: str, p0, p1) -> str | None:
        """Convert raw 'L->A;P->S' + positions to 'L235A / P238S'."""
        if not isinstance(sig, str):
            return None
        parts = sig.split(";", 1)
        if len(parts) != 2:
            return None
        m0 = _fmt(parts[0], p0)
        m1 = _fmt(parts[1], p1)
        if m0 and m1:
            return f"{m0} / {m1}"
        return None

    splits = df["top_var_sig"].fillna("").str.split(";", n=1, expand=True)
    df["top_mutation_A"] = [
        _fmt(m, p) for m, p in zip(splits.get(0, pd.Series(dtype=str)), df["p0"])
    ]
    df["top_mutation_B"] = [
        _fmt(m, p) for m, p in zip(splits.get(1, pd.Series(dtype=str)), df["p1"])
    ]
    df["top_variant_pair_label"] = df.apply(
        lambda r: (
            f"{r['top_mutation_A']} / {r['top_mutation_B']}"
            if pd.notna(r["top_mutation_A"]) and pd.notna(r["top_mutation_B"])
            else None
        ),
        axis=1,
    )

    # All variant pairs with unique INN counts.
    # Use mutation_pair_stats (which has per-variant unique INN counts) rather
    # than the raw mutations dict (which counts pairings, not unique INNs — an
    # INN with the same mutation on two chains would be counted twice there).
    def _all_pairs_str(row) -> str | None:
        pair_stats = row.get("mutation_pair_stats")
        if isinstance(pair_stats, dict) and pair_stats:
            # Keys are already formatted as "L234A/L235A"; values have "count" = unique INNs
            parts = sorted(pair_stats.items(), key=lambda x: -x[1].get("count", 0))
            return " | ".join(
                f"{k.replace('/', ' / ')}: {v['count']}" for k, v in parts
            )
        # Fallback: raw mutations dict (pair occurrences, not unique INNs)
        mut_map = row.get("mutations")
        if not isinstance(mut_map, dict) or not mut_map:
            return None
        p0, p1 = row["p0"], row["p1"]
        parts = []
        for sig, cnt in sorted(mut_map.items(), key=lambda x: -x[1]):
            label = _fmt_pair_sig(sig, p0, p1)
            if label:
                parts.append(f"{label}: {cnt}")
        return " | ".join(parts) if parts else None

    df["all_mutation_pairs"] = df.apply(_all_pairs_str, axis=1)

    # ── 2. Rename ─────────────────────────────────────────────────────────────
    df = df.rename(columns=_CO_COLUMN_LABELS)

    # ── 3. Select columns ────────────────────────────────────────────────────
    keep = [c for c in _CO_EXPORT_COLS if c in df.columns]
    df = df[keep]

    # ── 4. Sort ───────────────────────────────────────────────────────────────
    sort_cols = [
        c for c in ["num_unique_inn_in_group", "fisher_pvalue"] if c in df.columns
    ]
    if sort_cols:
        df = df.sort_values(sort_cols, ascending=[False, True])

    return df.reset_index(drop=True)


def add_inn_date_info(
    motif_df: pd.DataFrame,
    inn_df: pd.DataFrame,
    date_col: str | list[str] = "year_received",
    inn_col: str = "inn",
    inn_name_source: str = "inn_name",
    unresolved_inn_policy: str = "warn",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Attach min_year, max_year, date_range to each motif row.

    Parameters
    ----------
    motif_df:
        Output of build_motif_index / compute_motif_ancestry — must have an
        ``inns`` column (set of INN identifiers).
    inn_df:
        DataFrame with one row per INN; must contain ``inn_col``, ``inn_name_source``
        and ``date_col``.
    date_col:
        Column in ``inn_df`` containing the year/date value (default ``year_received``).
    inn_col:
        Column in ``inn_df`` containing the INN identifier (default ``inn``).
    inn_name_source:
        Column in ``inn_df`` whose values match the identifiers stored in
        ``motif_df["inns"]`` (default ``inn_name``).
    unresolved_inn_policy:
        What to do when motif INNs still cannot be mapped after exact and
        first-word fallback matching. One of ``"warn"``, ``"raise"``, or
        ``"ignore"`` (default ``"warn"``).

    Returns
    -------
    (motif_df_dated, inn_info_resolved_name)
        ``motif_df_dated`` has ``min_year``, ``max_year``, and ``date_range``
        columns added.
        ``inn_info_resolved_name`` is a copy of ``inn_df`` with additional
        alias rows where unresolved motif INNs were matched by first word;
        these alias rows use motif INN naming in ``inn_name_source``.
    """
    policy = str(unresolved_inn_policy).strip().lower()
    if policy not in {"warn", "raise", "ignore"}:
        raise ValueError(
            "add_inn_date_info: unresolved_inn_policy must be one of "
            "{'warn', 'raise', 'ignore'}."
        )

    base = inn_df.reset_index().copy()
    if inn_name_source not in base.columns:
        raise KeyError(
            f"add_inn_date_info: column '{inn_name_source}' not found in inn_df."
        )

    _date_cols = [date_col] if isinstance(date_col, str) else list(date_col)
    _valid_date_cols = [c for c in _date_cols if c in base.columns]
    if not _valid_date_cols:
        raise KeyError(
            f"add_inn_date_info: none of date_col={_date_cols!r} were found in inn_df."
        )

    base["_resolved_year"] = base[_valid_date_cols[0]]
    for _c in _valid_date_cols[1:]:
        base["_resolved_year"] = base["_resolved_year"].combine_first(base[_c])
    base["_resolved_year"] = pd.to_numeric(base["_resolved_year"], errors="coerce")

    def _iter_inns(raw) -> list[str]:
        if isinstance(raw, (set, frozenset, list, tuple)):
            return [str(v).strip() for v in raw if pd.notna(v) and str(v).strip()]
        if raw is None or (isinstance(raw, float) and pd.isna(raw)):
            return []
        s = str(raw).strip()
        return [s] if s else []

    def _first_word(name: str) -> str:
        n = str(name).strip()
        return n.split()[0].lower() if n else ""

    motif_inn_universe: set[str] = set()
    for _inns in motif_df["inns"].dropna():
        motif_inn_universe.update(_iter_inns(_inns))

    _name_series = base[inn_name_source].dropna().astype(str).map(str.strip)
    exact_names = set(_name_series.tolist())
    first_word_to_names: dict[str, set[str]] = {}
    for _name in exact_names:
        _token = _first_word(_name)
        if _token:
            first_word_to_names.setdefault(_token, set()).add(_name)

    alias_to_canonical: dict[str, str] = {}
    unresolved_after_exact: set[str] = set()
    ambiguous_aliases: dict[str, list[str]] = {}

    for _motif_inn in sorted(motif_inn_universe):
        if _motif_inn in exact_names:
            continue
        unresolved_after_exact.add(_motif_inn)
        _candidates = sorted(first_word_to_names.get(_first_word(_motif_inn), set()))
        if len(_candidates) == 1:
            alias_to_canonical[_motif_inn] = _candidates[0]
        elif len(_candidates) > 1:
            ambiguous_aliases[_motif_inn] = _candidates

    unresolved_after_fallback = sorted(
        unresolved_after_exact - set(alias_to_canonical.keys())
    )
    if unresolved_after_fallback:
        _sample = unresolved_after_fallback[:10]
        _ambiguous = {
            k: ambiguous_aliases[k] for k in _sample if k in ambiguous_aliases
        }
        msg = (
            "add_inn_date_info: unable to map "
            f"{len(unresolved_after_fallback)} motif INN name(s) after exact and "
            "first-word fallback matching. "
            f"Examples: {_sample}"
        )
        if _ambiguous:
            msg += f". Ambiguous first-word matches: {_ambiguous}"
        if policy == "raise":
            raise KeyError(msg)
        if policy == "warn":
            warnings.warn(msg, UserWarning, stacklevel=2)

    year_map: dict[str, float] = (
        base[[inn_name_source, "_resolved_year"]]
        .dropna(subset=[inn_name_source])
        .drop_duplicates(subset=[inn_name_source], keep="last")
        .set_index(inn_name_source)["_resolved_year"]
        .to_dict()
    )

    def _years(inns):
        inn_items = _iter_inns(inns)
        raw = [year_map.get(alias_to_canonical.get(i, i)) for i in inn_items]
        vals = [
            int(v)
            for v in raw
            if v is not None and not (isinstance(v, float) and pd.isna(v))
        ]
        if not vals:
            return pd.NA, pd.NA, pd.NA
        mn, mx = min(vals), max(vals)
        return mn, mx, max(1, mx - mn)

    rows = motif_df["inns"].map(_years)
    motif_df = motif_df.copy()
    motif_df[["min_year", "max_year", "date_range"]] = pd.DataFrame(
        rows.tolist(), index=motif_df.index
    )

    inn_info_resolved_name = base.drop(columns=["_resolved_year"]).copy()
    if alias_to_canonical:
        canonical_rows = (
            base[[inn_name_source] + [c for c in base.columns if c != inn_name_source]]
            .drop_duplicates(subset=[inn_name_source], keep="last")
            .set_index(inn_name_source)
        )
        alias_rows = []
        for alias, canonical in sorted(alias_to_canonical.items()):
            if canonical not in canonical_rows.index:
                continue
            row = canonical_rows.loc[canonical].to_dict()
            row[inn_name_source] = alias
            row["_resolved_from_inn_name"] = canonical
            alias_rows.append(row)
        if alias_rows:
            inn_info_resolved_name = pd.concat(
                [inn_info_resolved_name, pd.DataFrame(alias_rows)],
                ignore_index=True,
                sort=False,
            )

    return motif_df, inn_info_resolved_name
