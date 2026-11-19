"""Debugging and validation utilities for motif-level pipeline outputs.

Primary entry points
--------------------
debug_motif_inn_counts
    Cross-validate ``inn_count`` / ``adjusted_count`` in a motif DataFrame
    against the raw ``individual_mutation_profile``.
investigate_inn_count_mismatches
    Drill into the top-N unexplained mismatches returned by
    ``debug_motif_inn_counts``.
debug_pairwise_inn_overlap
    Show pairwise INN overlap stats for all sub-combinations of a motif.
debug_cluster_overlaps
    Report token and INN overlaps between every pair of cluster seeds.
"""

from __future__ import annotations

import re
import warnings
from dataclasses import dataclass
from itertools import combinations
from typing import Any, Optional

import pandas as pd

from .._helpers import _canonical_str, _parse_tokens, build_inn_lookup


@dataclass
class INNDiscrepancyDetail:
    """Per-INN breakdown for one side of a ``debug_motif_inn_counts`` mismatch.

    Attributes
    ----------
    inn : str
        INN identifier.
    chains_in_motif : list[str]
        Chain IDs associated with this INN in the motif table
        (from ``unique_inn_chains``), or ``[]`` if the column is absent.
    genes_in_motif : list[str]
        Gene values attached to the motif row for this INN (typically just
        the motif-level gene, included for completeness).
    tokens_in_profile : frozenset[tuple[str, str]] | None
        All ``(eu_canon, change)`` tokens present for this INN in the
        gene-scoped profile lookup.  ``None`` when the INN was not found
        in the lookup at all.
    missing_tokens : frozenset[tuple[str, str]]
        Motif tokens that are absent from ``tokens_in_profile``
        (non-empty only for "missing" INNs).
    genes_in_profile : list[str]
        Distinct ``gene`` values seen for this INN across all stratification
        groups in the profile (useful when a gene-scope mismatch is the cause).
    """

    inn: str
    chains_in_motif: list[str]
    genes_in_motif: list[str]
    tokens_in_profile: "frozenset[tuple[str, str]] | None"
    missing_tokens: "frozenset[tuple[str, str]]"
    genes_in_profile: list[str]
    """Distinct genes seen for this INN across ALL its chains in the profile.
    May include genes from unrelated chains — use ``genes_on_motif_chains``
    for the chain-scoped view."""
    genes_on_motif_chains: "list[str]"
    """Distinct genes seen only on the chains listed in ``chains_in_motif``.
    This is the chain-scoped equivalent of ``genes_in_profile`` and is the
    correct field to compare against ``genes_in_motif``."""
    # ── Cross-scope findings ──────────────────────────────────────────────────
    found_in_global_profile: bool
    """True when ALL motif tokens are present for this INN anywhere in the
    profile (ignoring gene/chain scope).  If True but ``missing_tokens`` is
    non-empty, the tokens exist but in a different stratification group."""
    genes_with_full_motif: "list[str]"
    """Gene values under which this INN carries *all* motif tokens.
    Non-empty means the full motif is present but attributed to a different
    gene than the motif row (gene-scope mismatch)."""
    chains_with_full_motif: "list[str]"
    """Chain IDs under which this INN carries *all* motif tokens.
    Non-empty means the full motif is present on a different chain than the
    motif row expected (chain-scope mismatch)."""

    def __repr__(self) -> str:
        lines = [self.inn]
        lines.append(f"  chains in motif        : {self.chains_in_motif}")
        lines.append(f"  genes in motif         : {self.genes_in_motif}")
        lines.append(
            f"  missing tokens         : {self.missing_tokens or '(none — tokens present globally)'}"
        )
        lines.append(
            f"  genes on motif chains  : {self.genes_on_motif_chains}  (chain-scoped)"
        )
        lines.append(
            f"  genes in profile (all) : {self.genes_in_profile}  (all chains)"
        )
        lines.append(
            f"  found in global profile: {self.found_in_global_profile}"
        )
        if self.genes_with_full_motif:
            gene_note = (
                "  ← gene-scope mismatch"
                if set(self.genes_with_full_motif)
                != set(self.genes_in_motif)
                else "  ← same gene, confirmed"
            )
            lines.append(
                f"  genes with full motif  : {self.genes_with_full_motif}{gene_note}"
            )
        else:
            lines.append(f"  genes with full motif  : (none)")
        if self.chains_with_full_motif:
            chain_note = (
                "  ← chain-scope mismatch"
                if set(self.chains_with_full_motif)
                != set(self.chains_in_motif)
                else "  ← same chain, confirmed"
            )
            lines.append(
                f"  chains with full motif : {self.chains_with_full_motif}{chain_note}"
            )
        else:
            lines.append(f"  chains with full motif : (none)")
        return "\n".join(lines)


@dataclass
class MotifMismatchDetail:
    """Full diagnostic detail for one discrepant motif row.

    Attributes
    ----------
    mutation_string : str
        Human-readable motif string (PE or colon notation).
    gene : str
        Gene group this row belongs to.
    inn_count : int
        ``inn_count`` as stored in the motif table.
    recomputed_inn_count : int
        INN count recomputed from the mutation profile.
    motif_tokens : frozenset[tuple[str, str]]
        Canonical ``(eu_str, change)`` token set for this motif.
    stored_inns : frozenset[str]
        INN set as stored in the motif table.
    recomputed_inns : frozenset[str]
        INN set found by subset-matching against the profile.
    missing_from_profile : list[INNDiscrepancyDetail]
        Detail for each INN that is stored in the motif but absent from the
        recomputed set (profile couldn't confirm all tokens).
    extra_in_motif : list[INNDiscrepancyDetail]
        Detail for each INN found in the profile but not stored in the motif
        (pipeline skipped them).
    """

    mutation_string: str
    gene: str
    inn_count: int
    recomputed_inn_count: int
    motif_tokens: "frozenset[tuple[str, str]]"
    stored_inns: "frozenset[str]"
    recomputed_inns: "frozenset[str]"
    missing_from_profile: "list[INNDiscrepancyDetail]"
    extra_in_motif: "list[INNDiscrepancyDetail]"

    def __repr__(self) -> str:
        sep = "─" * 60
        lines = [
            sep,
            f"  {self.mutation_string}  |  gene={self.gene}",
            f"  stored={self.inn_count}  recomputed={self.recomputed_inn_count}"
            f"  |  missing={len(self.missing_from_profile)}"
            f"  extra={len(self.extra_in_motif)}",
            sep,
        ]
        if self.missing_from_profile:
            lines.append(
                f"\n── Missing from profile ({len(self.missing_from_profile)}) ──"
            )
            for d in self.missing_from_profile:
                lines.append(repr(d))
        if self.extra_in_motif:
            lines.append(
                f"\n── Extra in profile ({len(self.extra_in_motif)}) ──"
            )
            for d in self.extra_in_motif:
                lines.append(repr(d))
        return "\n".join(lines)


@dataclass
class DebugMotifResult:
    """Return value of :func:`debug_motif_inn_counts`.

    Attributes
    ----------
    summary : pd.DataFrame
        One row per motif with scalar validation columns — same layout as the
        old plain-DataFrame return so existing code continues to work
        (``result["inn_count_ok"]``, etc.).
    mismatch_detail : dict[str, MotifMismatchDetail]
        Keyed by ``mutation_string`` (the display string used in the motif
        row).  Contains only rows that are unexplained mismatches (i.e. rows
        that appear in the ``AssertionError`` / unexplained-inn printout).
        Each value holds the full per-INN breakdown:

        * which chains/genes the INN has in the motif table
        * which tokens it has in the profile lookup (and which are missing)
        * which profile genes the INN appears under

    Examples
    --------
    ::

        result = debug_motif_inn_counts(..., raise_on_error=False)
        # summary DataFrame — same as before
        result.summary[result.summary["inn_count_ok"]]

        # drill into a specific mismatch
        detail = result.mismatch_detail["M234A/L235A"]
        for inn_detail in detail.missing_from_profile:
            print(inn_detail.inn, inn_detail.missing_tokens, inn_detail.genes_in_profile)
    """

    summary: pd.DataFrame
    mismatch_detail: "dict[str, MotifMismatchDetail]"

    # ── Convenience accessors ────────────────────────────────────────────────

    def __getitem__(self, key: str) -> "MotifMismatchDetail":
        """Allow ``result["M234A/L235A"]`` as shorthand for ``result.mismatch_detail[...]``."""
        return self.mismatch_detail[key]

    def unexplained(self) -> pd.DataFrame:
        """Rows from ``summary`` that are unexplained mismatches (same filter as the error report)."""
        df = self.summary
        mask = (
            ~df["inn_count_ok"]
            & ~df.get(
                "inn_count_ok_cross_region",
                pd.Series(False, index=df.index),
            )
            & ~df.get(
                "inn_count_bispecific_ok", pd.Series(False, index=df.index)
            )
            & ~df.get(
                "has_insertion_token", pd.Series(False, index=df.index)
            )
            & ~df.get(
                "is_bispecific_row", pd.Series(False, index=df.index)
            )
        )
        return df[mask]

    def __repr__(self) -> str:
        n_total = len(self.summary)
        n_ok = (
            int(self.summary["inn_count_ok"].sum())
            if "inn_count_ok" in self.summary
            else 0
        )
        n_detail = len(self.mismatch_detail)
        return (
            f"DebugMotifResult("
            f"motifs={n_total}, inn_count_ok={n_ok}, "
            f"mismatch_detail={n_detail} unexplained motifs)"
        )


# ── Debug helper ──────────────────────────────────────────────────────────────


def debug_motif_inn_counts(
    motif_df: pd.DataFrame,
    mutation_profile: pd.DataFrame,
    position_col: str = "EU",
    inn_col: str = "inn",
    motifs_by_region: pd.DataFrame | None = None,
    stratify_by: list[str] | None = None,
    head: int = 20,
    raise_on_error: bool = True,
) -> DebugMotifResult:
    """Cross-validate ``inn_count`` and ``adjusted_count`` in a motif DataFrame
    against the raw mutation profile.

    For each motif, queries ``mutation_profile`` to find all INNs that carry
    *every* mutation in the motif's token set, then compares that to the stored
    ``inn_count``.  Also validates ``adjusted_count`` by re-deriving the
    "leftover" INNs after subtracting those explained by larger seed motifs.

    Parameters
    ----------
    motif_df:
        Output of ``build_motif_index`` / ``motif_df_clustered_post_filter``.
        Supports two token formats:

        * **New format** (``motif_df_clustered_post_filter``): must have
          ``motif_token_set`` (``frozenset`` of colon-notation strings like
          ``"366:T->S"``), which is present on all rows from the chain motif
          pipeline.  The ``motif`` column (``frozenset`` of tuples) is optional
          — ``motif_token_set`` is used as fallback, then ``mutation_string``
          (colon-notation string).  ``mutation_string_pe`` is display-only.
        * **Old format**: must have ``motif`` (``frozenset`` of ``(pos, change)``
          2-tuples) and ``mutation_string``.

        Both formats require ``inn_count``, ``adjusted_count``, ``inns`` (set of
        INN names), and ``adjusted_inns`` (set of INN names after seed
        subtraction).
    mutation_profile:
        Long-form mutation table — one row per INN × mutation.  Must have
        columns ``inn``, ``EU`` (or ``position_col``), ``germline_seq``,
        ``mutated_seq``.
    position_col:
        Position column to use (default ``"EU"``).
    inn_col:
        INN column name in ``mutation_profile`` (default ``"inn"``).
    motifs_by_region:
        Optional ``constant_chain_motifs_by_inn_region`` DataFrame (one row
        per INN × chain × region, with list-valued ``EU`` and ``mutated_seq``
        columns).  When provided, cross-region cases are **confirmed** by
        checking that each extra INN has all the motif's tokens present in
        that table but split across different region rows — rather than just
        inferred from the pattern.
    stratify_by:
        Columns to use for stratification — should match the ``group_by``
        argument used when calling ``build_motif_index`` / ``collapse_motifs_chains``.
        Default ``["gene", "gene_species"]``.  Pass your
        ``MUTATION_FREQ_RESOLUTION`` variable directly::

            debug_motif_inn_counts(
                motif_df, profile,
                stratify_by=MUTATION_FREQ_RESOLUTION,
                motifs_by_region=ds.constant_chain_motifs_by_inn_region,
            )
    head:
        Number of mismatching rows to print (default 20).
    raise_on_error:
        If ``True`` (default), raise ``AssertionError`` when unexplained
        discrepancies are found.  Set to ``False`` to always return the result
        even when there are errors — useful for subsequent inspection via
        :func:`investigate_inn_count_mismatches` or
        ``result.mismatch_detail``.

    Returns
    -------
    :class:`DebugMotifResult`
        ``.summary`` — DataFrame with one row per motif (same columns as the
        old plain-DataFrame return, so ``result.summary["inn_count_ok"]``
        still works).

        ``.mismatch_detail`` — ``{mutation_string: MotifMismatchDetail}``
        for every unexplained mismatch.  Each value contains the full per-INN
        breakdown: chains and genes in the motif table, tokens present/absent
        in the profile lookup, and all gene groups the INN appears under in
        the profile.
    """
    flat = mutation_profile.reset_index().copy()

    # Normalise synthetic gene aliases — ighgsyn is a synthetic IgG1 backbone
    # and should be treated identically to ighg1 in all debug lookups.
    _GENE_ALIASES: dict[str, str] = {"ighgsyn": "ighg1"}
    if "gene" in flat.columns:
        flat["gene"] = flat["gene"].apply(
            lambda g: _GENE_ALIASES.get(str(g).lower(), g)
        )
    # Build _change for all mutation types:
    # substitution/deletion: "{germline_seq}->{mutated_seq}"
    # insertion: "{germline_seq}{mutated_seq}"  (e.g. "T^SS")
    if "mutation_type" in flat.columns:
        flat["_change"] = flat.apply(
            lambda r: (
                f"{r['germline_seq']}{r['mutated_seq']}"
                if r.get("mutation_type") == "insertion"
                else f"{r['germline_seq']}->{r['mutated_seq']}"
            ),
            axis=1,
        )
    else:
        flat["_change"] = flat["germline_seq"] + "->" + flat["mutated_seq"]

    # Canonicalise EU positions to string so profile and motif tokens always use
    # the same type. Rules:
    #   - float/int 217.0 or 217  → "217"   (whole-number positions)
    #   - string "214.i1"         → "214.i1" (insertion-slot: base.ix → kept as-is)
    # This avoids int vs float mismatches from Excel loading EU as float64.
    def _canon_eu(x) -> str:
        if isinstance(x, str):
            return x  # already canonical (e.g. "214.i1", "217")
        try:
            f = float(x)
            i = int(f)
            return str(i) if f == i else str(x)
        except (ValueError, TypeError):
            return str(x)

    flat["_eu_canon"] = flat[position_col].apply(_canon_eu)
    flat["_token"] = list(zip(flat["_eu_canon"], flat["_change"]))

    def _parse_motif_string(ms: str) -> frozenset:
        """Parse a ``mutation_string_pe`` / ``mutation_string`` in colon notation
        (e.g. ``"234:L->A / 235:L->A"``) into a frozenset of ``(pos, change)``
        tuples compatible with profile ``_token`` values."""
        out = []
        for tok in re.split(r"\s*/\s*", ms.strip()):
            tok = tok.strip()
            if not tok:
                continue
            if ":" in tok:
                pos, _, change = tok.partition(":")
                out.append((pos.strip(), change.strip()))
        return frozenset(out)

    def _norm_motif(motif: frozenset) -> frozenset:
        """Normalise motif tokens to canonical ``(eu_str, change)`` tuples.

        Handles two token formats:
        - Old: ``(eu_pos, change)`` 2-tuples (from the ``motif`` frozenset column).
        - New: ``"pos:change"`` strings (from ``mutation_string_pe`` colon notation).
        """
        out = []
        for t in motif:
            if isinstance(t, tuple) and len(t) >= 2:
                out.append((_canon_eu(t[0]), t[1]))
            elif isinstance(t, str) and ":" in t:
                pos, _, change = t.partition(":")
                out.append((_canon_eu(pos.strip()), change.strip()))
        return frozenset(out)

    def _is_insertion_token(t) -> bool:
        """Return True if token represents an insertion mutation.

        Handles both bare-caret (``^GGGGS``) and germline-prefixed (``T^GGGGS``)
        change strings, in both tuple and colon-notation string formats.
        """
        if isinstance(t, tuple) and len(t) >= 2:
            change = str(t[1])
            return "^" in change or change == "+"
        if isinstance(t, str) and ":" in t:
            change = t.partition(":")[2]
            return "^" in change or change == "+"
        return False

    # ── Build motifs_by_region lookup for cross-region confirmation ───────────
    _mbr_rows: dict[tuple[str, str, str], list[frozenset]] = {}
    if motifs_by_region is not None:
        mbr = motifs_by_region.reset_index().copy()
        for _, mr in mbr.iterrows():
            inn_v = mr.get(inn_col, mr.get("inn", None))
            gene_v = str(mr.get("gene", "")).upper()
            gsp_v = str(mr.get("gene_species", "")).upper()
            if inn_v is None:
                continue
            eu_list = mr.get("EU", [])
            ms_list = mr.get("mutated_seq", [])
            gl_list = mr.get("germline_seq", [])
            if not isinstance(eu_list, list):
                continue
            mt_list = mr.get("mutation_type", [])
            if not isinstance(mt_list, list):
                mt_list = ["substitution"] * len(eu_list)
            # Use canonical EU strings — same format as _token in profile
            row_tokens = frozenset(
                (
                    _canon_eu(eu),
                    (
                        f"{gl}{ms}"
                        if str(mt).lower() == "insertion"
                        else f"{gl}->{ms}"
                    ),
                )
                for eu, gl, ms, mt in zip(
                    eu_list, gl_list, ms_list, mt_list
                )
            )
            key = (str(inn_v), gene_v, gsp_v)
            _mbr_rows.setdefault(key, []).append(row_tokens)

    def _confirm_cross_region(
        extra_inn: str, motif: frozenset, gene_upper: str, gsp_upper: str
    ) -> str:
        """Return 'confirmed', 'refuted', or 'unknown' for a cross-region hypothesis."""
        if not _mbr_rows:
            return "unknown (motifs_by_region not provided)"
        key = (extra_inn, gene_upper, gsp_upper)
        rows = _mbr_rows.get(key)
        if not rows:
            # Try without gene_species constraint as fallback
            rows_any = [
                v
                for k, v in _mbr_rows.items()
                if k[0] == extra_inn and k[1] == gene_upper
            ]
            if not rows_any:
                return "unknown (INN/gene not in motifs_by_region)"
            # INN found but under a different gene_species — not same group
            gsp_found = [
                k[2]
                for k in _mbr_rows
                if k[0] == extra_inn and k[1] == gene_upper
            ]
            return (
                f"unknown (INN in motifs_by_region but gene_species={gsp_found} "
                f"differs from motif group={gsp_upper!r} — different stratification group, not a bug)"
            )
        # rows is now a list of frozensets (canonical token sets per region row)
        motif_canon = _norm_motif(motif)
        all_tokens_union = frozenset().union(*rows)
        if not motif_canon.issubset(all_tokens_union):
            return "refuted (tokens not present even across all rows)"
        if any(motif_canon.issubset(row) for row in rows):
            return "refuted (motif fits within a single row — pipeline should have counted it)"
        return "confirmed (tokens present but split across region rows)"

    # ── Build gene-scoped token lookup ───────────────────────────────────────
    # For each (inn, chain_id), resolve ONE effective gene using the same
    # priority rule as enumerate_chain_motifs, applied to the motif's tokens:
    #
    #   1. If any token on this chain is FC → use the FC gene
    #   2. Elif any token is non-secondary (CH1/CH2/CH3/CL/CK/…) → use that gene
    #   3. Else all tokens are hinge/secondary → use the hinge gene
    #
    # All tokens for that (inn, chain) are then filed under the resolved gene.
    # This means:
    #  - hinge-only chains (efparepoetin alfa) → all tokens go to hinge gene (ighg1)
    #  - mixed chains with FC (nemolizumab) → all tokens including hinge go to FC gene (ighg4)
    _strat = (
        stratify_by
        if stratify_by is not None
        else ["gene", "gene_species"]
    )
    _grp_cols_avail = [c for c in _strat if c in flat.columns]
    _SECONDARY_REGIONS: frozenset[str] = frozenset(
        {"HINGE", "CHS", "HINGE-REGION"}
    )

    _global_tokens: dict[str, frozenset] = {
        str(inn): frozenset(grp["_token"])
        for inn, grp in flat.groupby(inn_col)
    }

    # Build per-token region lookup: used in genes_on_motif_chains computation
    _token_region: dict[tuple, str] = {}
    if "region_classification" in flat.columns:
        for _, trow in flat.iterrows():
            _token_region[trow["_token"]] = str(
                trow["region_classification"]
            )

    def _effective_gene_key(chain_rows: pd.DataFrame) -> tuple | None:
        """Resolve stratification key for a set of chain rows using FC > non-secondary > hinge."""
        if "region_classification" not in chain_rows.columns:
            return None
        for region_filter in (
            lambda r: r == "FC",
            lambda r: r not in _SECONDARY_REGIONS,
            lambda r: True,
        ):
            candidates = chain_rows[
                chain_rows["region_classification"].apply(region_filter)
            ]
            if candidates.empty:
                continue
            key_vals = []
            for col in _grp_cols_avail:
                unique_vals = candidates[col].dropna().unique()
                if len(unique_vals) != 1:
                    key_vals = []
                    break
                key_vals.append(str(unique_vals[0]).upper())
            if key_vals:
                return tuple(key_vals)
        return None

    # Build group cache: {gene_key: {inn: frozenset(tokens)}}
    _group_cache: dict[tuple, dict[str, frozenset]] = {}
    _chain_col = next(
        (
            c
            for c in ["chain_id", "chain_type_h_or_l"]
            if c in flat.columns
        ),
        None,
    )
    if (
        _grp_cols_avail
        and _chain_col
        and "region_classification" in flat.columns
    ):
        for (inn_v, ch_v), chain_grp in flat.groupby(
            [inn_col, _chain_col], sort=False
        ):
            gk = _effective_gene_key(chain_grp)
            if gk is None:
                continue
            tokens = frozenset(chain_grp["_token"])
            inn_str = str(inn_v)
            if gk not in _group_cache:
                _group_cache[gk] = {}
            _group_cache[gk][inn_str] = (
                _group_cache[gk].get(inn_str, frozenset()) | tokens
            )
    elif _grp_cols_avail:
        # No chain_id or region column — raw per-row gene grouping
        for gkeys, subset in flat.groupby(_grp_cols_avail, sort=False):
            gk = gkeys if isinstance(gkeys, tuple) else (gkeys,)
            gk_upper = tuple(str(v).upper() for v in gk)
            _group_cache[gk_upper] = {
                inn: frozenset(grp["_token"])
                for inn, grp in subset.groupby(inn_col)
            }

    def _pick_token_dict(group_key: tuple | None) -> dict[str, frozenset]:
        if group_key and _group_cache:
            gk_upper = tuple(str(v).upper() for v in group_key)
            if gk_upper in _group_cache:
                return _group_cache[gk_upper]
        return _global_tokens

    # ── Build global bispecific INN set ──────────────────────────────────────
    # Collect all INNs removed from branch rows by bispecific pruning (present
    # in any stem row's bispecific_inns). Excluded from the "unexplained extra"
    # check — their absence from branch inn_count is intentional.
    _all_bispecific_inns: set[str] = set()
    if "bispecific_inns" in motif_df.columns:
        for _bi in motif_df["bispecific_inns"].dropna():
            if isinstance(_bi, (set, frozenset)):
                _all_bispecific_inns.update(_bi)

    # ── Build global inn → genes lookup (across all stratification groups) ──────
    # Intentionally unscoped — covers all chains of each INN.
    # For chain-scoped gene lookup use _inn_chain_genes below.
    _inn_to_genes: dict[str, list[str]] = {}
    if "gene" in flat.columns:
        for inn_v, grp in flat.groupby(inn_col):
            _inn_to_genes[str(inn_v)] = sorted(
                grp["gene"].dropna().unique()
            )

    # ── Build (inn, gene) → tokens and (inn, token) → full strat key indexes ──
    _inn_gene_tokens: dict[tuple[str, str], frozenset] = {}
    # (inn, token) → full uppercase stratification tuple — used for gene matching
    # in _motif_gene_matches so ALL stratification columns are checked, not just gene.
    _inn_token_strat_key: dict[tuple[str, tuple], tuple] = {}
    if "gene" in flat.columns:
        for (inn_v, gene_v), grp in flat.groupby([inn_col, "gene"]):
            _inn_gene_tokens[(str(inn_v), str(gene_v))] = frozenset(
                grp["_token"]
            )
    if _grp_cols_avail:
        for _, trow in flat.iterrows():
            strat_vals = tuple(
                str(trow[c]).upper()
                for c in _grp_cols_avail
                if c in trow.index
            )
            if strat_vals:
                _inn_token_strat_key[
                    (str(trow[inn_col]), trow["_token"])
                ] = strat_vals

    # Pre-build chain-level indexes for fast per-motif gene resolution.
    # _inn_chain_tokens:      (inn, chain_id) → frozenset of (eu, change) tokens
    # _inn_chain_token_region:(inn, chain_id, token) → region_classification string
    # _inn_to_chains:         inn → set of chain_ids
    _inn_chain_tokens: dict[tuple[str, str], frozenset] = {}
    _inn_chain_token_region: dict[tuple[str, str, tuple], str] = {}
    _inn_to_chains: dict[str, set[str]] = {}
    chain_col = next(
        (
            c
            for c in ["chain_id", "chain_type_h_or_l"]
            if c in flat.columns
        ),
        None,
    )
    has_region_col = "region_classification" in flat.columns
    if chain_col:
        for (inn_v, ch_v), grp in flat.groupby([inn_col, chain_col]):
            inn_str, ch_str = str(inn_v), str(ch_v)
            key = (inn_str, ch_str)
            _inn_chain_tokens[key] = frozenset(grp["_token"])
            _inn_to_chains.setdefault(inn_str, set()).add(ch_str)
            if has_region_col:
                for _, trow in grp.iterrows():
                    _inn_chain_token_region[
                        (inn_str, ch_str, trow["_token"])
                    ] = str(trow["region_classification"])

    mismatch_detail: dict[str, MotifMismatchDetail] = {}
    records = []
    for _, row in motif_df.iterrows():
        motif: frozenset = row.get("motif", frozenset())

        # New format: the frozenset of (pos, change) tuples may be absent.
        # Resolution order:
        #   1. motif_token_set — frozenset of colon-notation strings ("366:T->S")
        #      present on all rows from enumerate_chain_motifs / build_chain_motif_index.
        #   2. mutation_string — " / "-joined colon-notation string; parsed into tuples.
        # mutation_string_pe is compact PE notation ("T366S") and is not parseable
        # into profile tokens, so it is never used for motif derivation.
        if not isinstance(motif, frozenset) or not motif:
            mts = row.get("motif_token_set")
            if isinstance(mts, frozenset) and mts:
                motif = mts
        if not isinstance(motif, frozenset) or not motif:
            ms = row.get("mutation_string", "")
            if ms:
                motif = _parse_motif_string(str(ms))
        if not motif:
            continue

        mut_str = (
            row.get("mutation_string_pe")
            or row.get("mutation_string")
            or str(motif)
        )

        # ── Detect insertion tokens: change starts with "^" or is bare "+" ──────
        has_insertion = any(_is_insertion_token(t) for t in motif)

        # Build group key from this motif row matching the stratification columns
        gene_val = row.get("gene", None)
        gsp_val = row.get("gene_species", None)
        gene_upper = str(gene_val).upper() if gene_val is not None else ""
        gsp_upper = str(gsp_val).upper() if gsp_val is not None else ""
        group_key = (
            tuple(row[c] for c in _grp_cols_avail if c in row.index)
            or None
        )

        # ── Recompute inn_count from profile ─────────────────────────────────
        # For each INN, find chains carrying all motif tokens. Apply the gene
        # priority rule to only those tokens' regions on that chain, get the
        # gene from the winning tier, check it matches the motif's group.
        # Uses pre-built indexes — no DataFrame scanning in the hot loop.
        motif_canon = _norm_motif(motif)
        gk_upper = (
            tuple(str(v).upper() for v in group_key) if group_key else None
        )

        def _motif_gene_matches(inn_name: str) -> bool:
            if not chain_col or not has_region_col:
                return motif_canon.issubset(
                    _global_tokens.get(inn_name, frozenset())
                )
            for ch_str in _inn_to_chains.get(inn_name, set()):
                if not motif_canon.issubset(
                    _inn_chain_tokens.get((inn_name, ch_str), frozenset())
                ):
                    continue
                tok_regions = [
                    (
                        _inn_chain_token_region.get(
                            (inn_name, ch_str, t), "UNKNOWN"
                        ),
                        t,
                    )
                    for t in motif_canon
                ]
                for region_filter in (
                    lambda r: r == "FC",
                    lambda r: r not in _SECONDARY_REGIONS,
                    lambda r: True,
                ):
                    tier_toks = [
                        t for r, t in tok_regions if region_filter(r)
                    ]
                    if not tier_toks:
                        continue
                    tier_strat_keys = {
                        _inn_token_strat_key.get((inn_name, t))
                        for t in tier_toks
                        if (inn_name, t) in _inn_token_strat_key
                    } - {None}
                    if not tier_strat_keys:
                        continue
                    if gk_upper:
                        if gk_upper in tier_strat_keys:
                            return True
                    else:
                        return True
                    break
            return False

        recomputed_inns = frozenset(
            inn for inn in _global_tokens if _motif_gene_matches(inn)
        )
        recomputed_count = len(recomputed_inns)
        stored_inns = row.get("inns", set())
        if not isinstance(stored_inns, (set, frozenset)):
            stored_inns = set()
        stored_count = row.get("inn_count", -1)

        missing_from_profile = stored_inns - recomputed_inns
        extra_in_motif = recomputed_inns - stored_inns

        # ── Exclude bispecific-pruned INNs from the extra set ─────────────────
        # When bispecific pruning is active, some INNs are intentionally removed
        # from a branch row and attributed to a common stem.  Those INNs still
        # carry the branch motif's tokens (so the profile-based recompute finds
        # them), but their absence from stored_inns is expected — not a bug.
        # Any INN that appears in any stem row's bispecific_inns set is excluded
        # from the unexplained-extra check for all branch rows.
        extra_in_motif_excl_bispecific = (
            extra_in_motif - _all_bispecific_inns
        )
        cross_region_verdicts: dict[str, str] = {}
        for extra_inn in extra_in_motif_excl_bispecific:
            cross_region_verdicts[extra_inn] = _confirm_cross_region(
                extra_inn, motif, gene_upper, gsp_upper
            )

        n_confirmed = sum(
            1
            for v in cross_region_verdicts.values()
            if v.startswith("confirmed")
        )
        n_refuted = sum(
            1
            for v in cross_region_verdicts.values()
            if v.startswith("refuted")
        )
        n_unknown = sum(
            1
            for v in cross_region_verdicts.values()
            if v.startswith("unknown")
        )
        extra_cross_region = n_confirmed
        n_bispecific_explained = len(extra_in_motif) - len(
            extra_in_motif_excl_bispecific
        )

        # inn_count is correct, OR all discrepancy is explained by cross-region
        # or bispecific pruning (intentional removal to the stem).
        inn_count_ok = stored_count == recomputed_count
        inn_count_cross_region_ok = inn_count_ok or (
            len(missing_from_profile) == 0
            and n_refuted == 0
            and len(extra_in_motif_excl_bispecific) > 0
        )
        inn_count_bispecific_ok = inn_count_ok or (
            len(missing_from_profile) == 0
            and n_refuted == 0
            and n_bispecific_explained == len(extra_in_motif)
        )
        # Any refuted extra that is NOT explained by bispecific pruning = genuine bug
        has_refuted_extra = n_refuted > 0

        # ── Validate adjusted_count using stored adjusted_inns directly ───────
        stored_adj_count = row.get("adjusted_count", None)
        adj_inns_stored = row.get("adjusted_inns", None)
        if not isinstance(adj_inns_stored, (set, frozenset)):
            adj_inns_stored = None

        if adj_inns_stored is not None:
            recomputed_adj_count = len(adj_inns_stored)
            adj_not_in_profile = adj_inns_stored - recomputed_inns
            # Stem rows receive bispecific INNs into adjusted_inns; those INNs
            # carry the stem tokens (not the branch tokens) — exclude them.
            adj_not_in_profile_excl_bispecific = (
                adj_not_in_profile - _all_bispecific_inns
            )
        else:
            recomputed_adj_count = None
            adj_not_in_profile = set()
            adj_not_in_profile_excl_bispecific = set()

        records.append(
            {
                "mutation_string": mut_str,
                "num_mutations": row.get("num_mutations", len(motif)),
                "gene": row.get("gene", ""),
                "gene_species": row.get("gene_species", ""),
                "has_insertion_token": has_insertion,
                # inn_count validation
                "inn_count": stored_count,
                "recomputed_inn_count": recomputed_count,
                "inn_count_ok": inn_count_ok,
                "inn_count_ok_cross_region": inn_count_cross_region_ok,
                "inn_count_bispecific_ok": inn_count_bispecific_ok,
                "missing_from_profile": len(missing_from_profile),
                "extra_in_motif": len(extra_in_motif),
                "extra_bispecific_explained": n_bispecific_explained,
                "extra_cross_region_confirmed": n_confirmed,
                "extra_cross_region_refuted": n_refuted,
                "extra_cross_region_unknown": n_unknown,
                "has_refuted_extra": has_refuted_extra,
                "cross_region_verdicts": cross_region_verdicts,
                "missing_inns": sorted(missing_from_profile)[:10],
                "extra_inns": sorted(extra_in_motif)[:10],
                # adjusted_count validation
                "adjusted_count": stored_adj_count,
                "recomputed_adjusted_count": recomputed_adj_count,
                "adjusted_count_ok": stored_adj_count
                == recomputed_adj_count,
                "adj_inns_not_in_profile": len(
                    adj_not_in_profile_excl_bispecific
                ),
                "adj_inn_examples": sorted(
                    adj_not_in_profile_excl_bispecific
                )[:5],
                # Bispecific arm/anchor rows: their inns come from a cross-gene
                # chain search (constant_chains), not gene-scoped profile.
                # inn_count validation does not apply to these synthetic rows.
                "is_bispecific_row": bool(
                    row.get("bispecific_pair")
                    or row.get("bispecific_anchor")
                ),
            }
        )

        # ── Build MotifMismatchDetail for unexplained mismatches ───────────────
        # We collect detail for any row where inn_count differs and it is not
        # yet explained (cross-region or bispecific).  The final "unexplained"
        # filter (which mirrors the AssertionError mask) is applied after all
        # records are built, so we build detail eagerly for any mismatch here.
        is_bispecific_row = bool(
            row.get("bispecific_pair") or row.get("bispecific_anchor")
        )
        if (
            not inn_count_ok
            and not is_bispecific_row
            and not has_insertion
        ):
            # Chains in the motif table for each INN: from unique_inn_chains
            # column (dict {inn: [chain_ids]}) or unique_inn_chain_counts.
            _uic = row.get("unique_inn_chains")
            _chains_by_inn: dict[str, list[str]] = {}
            if isinstance(_uic, dict):
                _chains_by_inn = {
                    str(k): (
                        sorted(v)
                        if isinstance(v, (list, set))
                        else [str(v)]
                    )
                    for k, v in _uic.items()
                }
            elif isinstance(_uic, (set, frozenset, list)):
                for item in _uic:
                    if isinstance(item, tuple) and len(item) == 2:
                        inn_v, ch_v = str(item[0]), str(item[1])
                        _chains_by_inn.setdefault(inn_v, []).append(ch_v)
                    elif isinstance(item, str):
                        # flat chain_id string — resolve inn_name via profile lookup
                        for inn_v, chains_set in _inn_to_chains.items():
                            if item in chains_set:
                                _chains_by_inn.setdefault(
                                    inn_v, []
                                ).append(item)

            motif_gene_val = str(row.get("gene", ""))

            def _build_inn_detail(
                inn_name: str, side: str
            ) -> INNDiscrepancyDetail:
                """Build INNDiscrepancyDetail for one INN on the given side."""
                chains_in_motif = sorted(_chains_by_inn.get(inn_name, []))
                # Resolve gene for this INN's motif tokens using the priority rule:
                # look at the regions of the motif tokens specifically, then pick
                # the gene from FC > non-secondary > hinge tiers.
                genes_on_chains_set: set[str] = set()
                for ch in chains_in_motif:
                    ch_toks_in_profile = _inn_chain_tokens.get(
                        (inn_name, ch), frozenset()
                    )
                    motif_toks_on_chain = motif_canon & ch_toks_in_profile
                    if not motif_toks_on_chain:
                        continue
                    tok_regions = [
                        _token_region.get(t, "UNKNOWN")
                        for t in motif_toks_on_chain
                    ]
                    for tier_filter in (
                        lambda r: r == "FC",
                        lambda r: r not in _SECONDARY_REGIONS,
                        lambda r: True,
                    ):
                        tier_toks = {
                            t
                            for t, r in zip(
                                motif_toks_on_chain, tok_regions
                            )
                            if tier_filter(r)
                        }
                        if not tier_toks:
                            continue
                        tier_genes = {
                            gene_v
                            for (
                                inn_v2,
                                gene_v,
                            ), toks in _inn_gene_tokens.items()
                            if inn_v2 == inn_name and tier_toks & toks
                        }
                        if tier_genes:
                            genes_on_chains_set |= tier_genes
                            break
                genes_on_chains = sorted(genes_on_chains_set)
                # tokens this INN has in the scoped profile lookup
                scoped_dict = _pick_token_dict(group_key)
                inn_toks = scoped_dict.get(inn_name)
                # also check global for cross-scope cases
                if inn_toks is None:
                    inn_toks = _global_tokens.get(inn_name)
                missing_toks: frozenset = (
                    motif_canon - inn_toks
                    if inn_toks is not None
                    else motif_canon
                )
                # ── Cross-scope: check if full motif is found under a different gene/chain
                global_toks = _global_tokens.get(inn_name, frozenset())
                found_globally = motif_canon.issubset(global_toks)
                genes_with_full = sorted(
                    gene_v
                    for (inn_v, gene_v), toks in _inn_gene_tokens.items()
                    if inn_v == inn_name and motif_canon.issubset(toks)
                )
                chains_with_full = sorted(
                    ch_v
                    for (inn_v, ch_v), toks in _inn_chain_tokens.items()
                    if inn_v == inn_name and motif_canon.issubset(toks)
                )
                return INNDiscrepancyDetail(
                    inn=inn_name,
                    chains_in_motif=chains_in_motif,
                    genes_in_motif=[motif_gene_val],
                    tokens_in_profile=inn_toks,
                    missing_tokens=missing_toks,
                    genes_in_profile=_inn_to_genes.get(inn_name, []),
                    genes_on_motif_chains=genes_on_chains,
                    found_in_global_profile=found_globally,
                    genes_with_full_motif=genes_with_full,
                    chains_with_full_motif=chains_with_full,
                )

            missing_detail = [
                _build_inn_detail(inn_name, "missing")
                for inn_name in sorted(missing_from_profile)
            ]
            extra_detail = [
                _build_inn_detail(inn_name, "extra")
                for inn_name in sorted(extra_in_motif_excl_bispecific)
            ]
            mismatch_detail[mut_str] = MotifMismatchDetail(
                mutation_string=mut_str,
                gene=motif_gene_val,
                inn_count=int(stored_count) if stored_count != -1 else -1,
                recomputed_inn_count=recomputed_count,
                motif_tokens=motif_canon,
                stored_inns=frozenset(stored_inns),
                recomputed_inns=recomputed_inns,
                missing_from_profile=missing_detail,
                extra_in_motif=extra_detail,
            )

    result_df = pd.DataFrame(records)
    if result_df.empty:
        print("No motif rows processed.")
        return DebugMotifResult(summary=result_df, mismatch_detail={})

    n_inn_ok = result_df["inn_count_ok"].sum()
    n_inn_ok_cr = result_df["inn_count_ok_cross_region"].sum()
    n_inn_skip = result_df["has_insertion_token"].sum()
    n_adj_ok = result_df["adjusted_count_ok"].sum()
    total = len(result_df)
    print(f"inn_count (exact)        : {n_inn_ok}/{total} correct")
    print(
        f"inn_count (cross-region) : {n_inn_ok_cr}/{total} correct "
        f"(extra INNs whose mutations span region rows are expected)"
    )
    print(
        f"inn_count (skipped)      : {n_inn_skip} motifs skipped "
        f"(contain insertion tokens absent from profile)"
    )
    print(
        f"adjusted_count           : {n_adj_ok}/{total} correct  "
        f"(validates len(adjusted_inns)==adjusted_count + adjusted_inns⊆profile)"
    )

    # Only show motifs that are neither exactly correct, cross-region explained, nor insertions
    true_mismatches = result_df[
        ~result_df["inn_count_ok"]
        & ~result_df["inn_count_ok_cross_region"]
        & ~result_df["has_insertion_token"]
    ]
    if not true_mismatches.empty:
        print(
            f"\n── True inn_count mismatches ({len(true_mismatches)}) ──"
        )
        print(
            true_mismatches[
                [
                    "mutation_string",
                    "gene",
                    "inn_count",
                    "recomputed_inn_count",
                    "missing_from_profile",
                    "extra_in_motif",
                ]
            ]
            .head(head)
            .to_string(index=False)
        )

    cross_region_cases = result_df[
        ~result_df["inn_count_ok"] & result_df["inn_count_ok_cross_region"]
    ]
    if not cross_region_cases.empty:
        has_mbr = motifs_by_region is not None
        label = (
            "confirmed"
            if has_mbr
            else "inferred (pass motifs_by_region to confirm)"
        )
        print(
            f"\n── Cross-region cases ({len(cross_region_cases)}) [{label}] ──"
        )
        print(
            cross_region_cases[
                [
                    "mutation_string",
                    "gene",
                    "inn_count",
                    "recomputed_inn_count",
                    "extra_cross_region_confirmed",
                    "extra_cross_region_refuted",
                    "extra_cross_region_unknown",
                ]
            ]
            .head(head)
            .to_string(index=False)
        )

    insertion_cases = result_df[result_df["has_insertion_token"]]
    if not insertion_cases.empty:
        print(
            f"\n── Skipped motifs with insertion tokens ({len(insertion_cases)}) ──"
        )
        print(
            insertion_cases[
                [
                    "mutation_string",
                    "gene",
                    "inn_count",
                    "recomputed_inn_count",
                ]
            ]
            .head(head)
            .to_string(index=False)
        )

    adj_mismatches = result_df[~result_df["adjusted_count_ok"]]
    if not adj_mismatches.empty:
        print(f"\n── adjusted_count mismatches ({len(adj_mismatches)}) ──")
        print(
            adj_mismatches[
                [
                    "mutation_string",
                    "gene",
                    "adjusted_count",
                    "recomputed_adjusted_count",
                    "adj_inns_not_in_profile",
                ]
            ]
            .head(head)
            .to_string(index=False)
        )

    # ── Raise on unexplained issues ───────────────────────────────────────────
    # Bispecific arm/anchor rows are exempt: their inns come from a cross-gene
    # constant_chains search rather than the gene-scoped individual_mutation_profile,
    # so inn_count validation and refuted-cross-region checks don't apply to them.
    _bispecific_mask = (
        result_df.get(
            "is_bispecific_row", pd.Series(False, index=result_df.index)
        )
        .fillna(False)
        .astype(bool)
    )

    unexplained_inn = result_df[
        ~result_df["inn_count_ok"]
        & ~result_df["inn_count_ok_cross_region"]
        & ~result_df["inn_count_bispecific_ok"]
        & ~result_df["has_insertion_token"]
        & ~_bispecific_mask
    ]
    # Refuted cross-region: extra INN fits in a single motifs_by_region row
    # but wasn't counted — genuine pipeline bug (unless explained by bispecific).
    refuted_cross_region = (
        result_df[result_df["has_refuted_extra"] & ~_bispecific_mask]
        if "has_refuted_extra" in result_df.columns
        else pd.DataFrame()
    )

    unexplained_adj = result_df[
        ~result_df["adjusted_count_ok"]
        & (result_df["adj_inns_not_in_profile"] > 0)
    ]

    errors = []
    if not unexplained_inn.empty:
        errors.append(
            f"{len(unexplained_inn)} unexplained inn_count mismatch(es):\n"
            + unexplained_inn[
                [
                    "mutation_string",
                    "gene",
                    "inn_count",
                    "recomputed_inn_count",
                    "missing_from_profile",
                    "extra_in_motif",
                ]
            ].to_string(index=False)
        )
    if not refuted_cross_region.empty:
        errors.append(
            f"{len(refuted_cross_region)} REFUTED cross-region case(s) — "
            f"extra INN fits in a single region row; pipeline should have counted it:\n"
            + refuted_cross_region[
                [
                    "mutation_string",
                    "gene",
                    "inn_count",
                    "recomputed_inn_count",
                    "extra_cross_region_refuted",
                    "cross_region_verdicts",
                ]
            ].to_string(index=False)
        )
    if not unexplained_adj.empty:
        errors.append(
            f"{len(unexplained_adj)} unexplained adjusted_count mismatch(es) "
            f"(adjusted INNs don't carry the motif mutations):\n"
            + unexplained_adj[
                [
                    "mutation_string",
                    "gene",
                    "adjusted_count",
                    "recomputed_adjusted_count",
                    "adj_inns_not_in_profile",
                    "adj_inn_examples",
                ]
            ].to_string(index=False)
        )

    debug_result = DebugMotifResult(
        summary=result_df, mismatch_detail=mismatch_detail
    )

    if errors:
        if raise_on_error:
            raise AssertionError(
                "debug_motif_inn_counts: unexplained discrepancies found\n\n"
                + "\n\n".join(errors)
            )
        else:
            print(
                "\n⚠ Unexplained discrepancies found (raise_on_error=False):\n"
                + "\n\n".join(errors)
            )
            return debug_result

    print("\n✓ No unexplained debug issues found.")
    return debug_result


def investigate_inn_count_mismatches(
    debug_result: pd.DataFrame,
    motif_df: pd.DataFrame,
    mutation_profile: pd.DataFrame,
    constant_chains: pd.DataFrame,
    top_n: int = 5,
    position_col: str = "EU",
    inn_col: str = "inn",
) -> None:
    """Explain why ``inn_count`` mismatches exist for the top-N discrepant motifs.

    For each mismatch calls out:
    - **missing INNs** (stored but not found in profile): checks whether their
      mutations appear in ``constant_chains`` (full-chain EU) vs
      ``mutation_profile`` (per-domain/region), revealing cross-chain or
      missing-domain issues.
    - **extra INNs** (found in profile but not stored): shows the relevant
      profile rows to explain why the pipeline skipped them.

    Parameters
    ----------
    debug_result:
        The DataFrame returned by :func:`debug_motif_inn_counts`.
    motif_df:
        The same ``motif_df_clustered_post_filter`` passed to
        ``debug_motif_inn_counts``.
    mutation_profile:
        ``ds.individual_mutation_profile.reset_index()`` — per-domain profile.
    constant_chains:
        ``ds.constant_chains.reset_index()`` — full-chain EU mutation strings.
    top_n:
        Number of largest mismatches to investigate (by ``inn_count``).
    position_col:
        EU position column name in ``mutation_profile`` (default ``"EU"``).
    inn_col:
        INN column name (default ``"inn"``).

    Examples
    --------
    ::

        investigate_inn_count_mismatches(
            debug_result,
            result.motif_df_clustered_post_filter,
            ds.individual_mutation_profile.reset_index(),
            ds.constant_chains.reset_index(),
        )
    """
    if debug_result is None:
        raise ValueError(
            "debug_result is None. "
            "Call debug_motif_inn_counts(..., raise_on_error=False) first "
            "to capture the result even when there are discrepancies."
        )
    # Accept both the new DebugMotifResult and the legacy plain DataFrame
    _summary: pd.DataFrame
    if isinstance(debug_result, DebugMotifResult):
        _summary = debug_result.summary
    elif isinstance(debug_result, pd.DataFrame):
        _summary = debug_result
    else:
        raise ValueError(
            f"debug_result must be a DebugMotifResult or DataFrame, got {type(debug_result)}"
        )
    if "inn_count_ok" not in _summary.columns:
        raise ValueError(
            "debug_result is missing 'inn_count_ok' column — it appears "
            "debug_motif_inn_counts did not complete successfully. "
            "Call it with raise_on_error=False to get the full result."
        )

    mismatches = (
        _summary[
            ~_summary["inn_count_ok"]
            & ~_summary["inn_count_ok_cross_region"]
            & ~_summary.get(
                "has_insertion_token",
                pd.Series(False, index=_summary.index),
            )
        ]
        .sort_values("inn_count", ascending=False)
        .head(top_n)
    )

    if mismatches.empty:
        print("No unexplained mismatches to investigate.")
        return

    flat_profile = mutation_profile.copy()
    flat_cc = (
        constant_chains.reset_index()
        if hasattr(constant_chains, "index")
        and not isinstance(constant_chains.index, pd.RangeIndex)
        else constant_chains.copy()
    )

    def _canon_eu(x) -> str:
        if isinstance(x, str):
            return x
        try:
            f = float(x)
            return str(int(f)) if f == int(f) else str(x)
        except (ValueError, TypeError):
            return str(x)

    if "mutation_type" in flat_profile.columns:
        flat_profile["_change"] = flat_profile.apply(
            lambda r: (
                f"{r['germline_seq']}{r['mutated_seq']}"
                if r.get("mutation_type") == "insertion"
                else f"{r['germline_seq']}->{r['mutated_seq']}"
            ),
            axis=1,
        )
    else:
        flat_profile["_change"] = (
            flat_profile["germline_seq"]
            + "->"
            + flat_profile["mutated_seq"]
        )
    flat_profile["_eu_canon"] = flat_profile[position_col].apply(_canon_eu)
    flat_profile["_token"] = list(
        zip(flat_profile["_eu_canon"], flat_profile["_change"])
    )

    _eu_col_cc = next(
        (
            c
            for c in [
                "constant_chain_mutations_EU",
                "mutations_EU",
                "EU_mutations",
            ]
            if c in flat_cc.columns
        ),
        None,
    )

    sep = "=" * 72

    for rank, (_, row) in enumerate(mismatches.iterrows(), 1):
        ms = row["mutation_string"]
        gene = row["gene"]
        stored = int(row["inn_count"])
        recomp = int(row["recomputed_inn_count"])
        missing_inns: list = list(row.get("missing_inns") or [])
        extra_inns: list = list(row.get("extra_inns") or [])

        print(f"\n{sep}")
        print(
            f"  #{rank}  {ms}  |  gene={gene}"
            f"  |  stored={stored}  recomputed={recomp}"
            f"  |  missing={len(missing_inns)}  extra={len(extra_inns)}"
        )
        print(sep)

        # ── Reconstruct motif token set from motif_df ─────────────────────────
        motif_row_mask = motif_df.apply(
            lambda r: (
                r.get("mutation_string", "") == ms
                or r.get("mutation_string_pe", "") == ms
            )
            and str(r.get("gene", "")).lower() == str(gene).lower(),
            axis=1,
        )
        motif_row = motif_df[motif_row_mask]
        mts: frozenset = frozenset()
        if not motif_row.empty:
            mts = motif_row.iloc[0].get("motif_token_set", frozenset())
            if not isinstance(mts, frozenset):
                mts = motif_row.iloc[0].get("motif", frozenset())

        motif_tokens_canon: frozenset = frozenset(
            (
                (_canon_eu(t[0]), t[1])
                if isinstance(t, tuple) and len(t) >= 2
                else (
                    t.partition(":")[0].strip(),
                    t.partition(":")[2].strip(),
                )
            )
            for t in mts
            if (isinstance(t, tuple) and len(t) >= 2)
            or (isinstance(t, str) and ":" in t)
        )
        print(
            f"  Motif tokens : {sorted(str(t) for t in motif_tokens_canon)}"
        )

        # ── Missing INNs: stored in motif but absent from profile ─────────────
        if missing_inns:
            print(f"\n  ── MISSING from profile ({len(missing_inns)}) ──")
            for inn in missing_inns:
                print(f"\n    INN: {inn!r}")

                prof_rows = flat_profile[
                    (flat_profile[inn_col] == inn)
                    & (flat_profile["gene"].str.lower() == gene.lower())
                ]
                if prof_rows.empty:
                    # Retry without gene filter — maybe gene mismatch
                    prof_rows = flat_profile[flat_profile[inn_col] == inn]

                if prof_rows.empty:
                    print(
                        "      → NOT in individual_mutation_profile at all"
                    )
                else:
                    inn_tokens = frozenset(prof_rows["_token"])
                    missing_tok = motif_tokens_canon - inn_tokens
                    print(
                        f"      → {len(prof_rows)} profile rows  |  "
                        f"has all motif tokens: {not missing_tok}"
                    )
                    if missing_tok:
                        print(
                            f"      → Tokens ABSENT from profile: "
                            f"{sorted(str(t) for t in missing_tok)}"
                        )
                    genes_p = (
                        sorted(prof_rows["gene"].dropna().unique())
                        if "gene" in prof_rows
                        else []
                    )
                    chains_p = []
                    for col in ["chain_id", "chain_type_h_or_l"]:
                        if col in prof_rows.columns:
                            chains_p = sorted(
                                prof_rows[col].dropna().unique()
                            )
                            break
                    regions_p = (
                        sorted(
                            prof_rows["region_classification"]
                            .dropna()
                            .unique()
                        )
                        if "region_classification" in prof_rows.columns
                        else []
                    )
                    print(
                        f"      → genes={genes_p}  chains={chains_p}"
                        f"  regions={regions_p}"
                    )

                if inn_col in flat_cc.columns:
                    cc_rows = flat_cc[flat_cc[inn_col] == inn]
                    if cc_rows.empty:
                        print("      → NOT in constant_chains")
                    else:
                        genes_cc = (
                            sorted(cc_rows["gene"].dropna().unique())
                            if "gene" in cc_rows.columns
                            else []
                        )
                        if _eu_col_cc:
                            eu_vals = list(
                                cc_rows[_eu_col_cc].dropna().unique()
                            )[:3]
                            print(
                                f"      → constant_chains EU mutations: {eu_vals}"
                            )
                        print(
                            f"      → genes in constant_chains: {genes_cc}"
                        )

        # ── Extra INNs: found in profile but not stored ───────────────────────
        if extra_inns:
            print(
                f"\n  ── EXTRA in profile ({len(extra_inns)}) — pipeline skipped these ──"
            )
            for inn in extra_inns:
                print(f"\n    INN: {inn!r}")
                prof_rows = flat_profile[
                    (flat_profile[inn_col] == inn)
                    & (flat_profile["gene"].str.lower() == gene.lower())
                ]
                if prof_rows.empty:
                    prof_rows = flat_profile[flat_profile[inn_col] == inn]

                if not prof_rows.empty:
                    inn_tokens = frozenset(prof_rows["_token"])
                    present_tok = motif_tokens_canon & inn_tokens
                    genes_p = (
                        sorted(prof_rows["gene"].dropna().unique())
                        if "gene" in prof_rows
                        else []
                    )
                    chains_p = []
                    for col in ["chain_id", "chain_type_h_or_l"]:
                        if col in prof_rows.columns:
                            chains_p = sorted(
                                prof_rows[col].dropna().unique()
                            )
                            break
                    regions_p = (
                        sorted(
                            prof_rows["region_classification"]
                            .dropna()
                            .unique()
                        )
                        if "region_classification" in prof_rows.columns
                        else []
                    )
                    print(
                        f"      → {len(prof_rows)} profile rows  |  "
                        f"motif tokens present: {sorted(str(t) for t in present_tok)}"
                    )
                    print(
                        f"      → genes={genes_p}  chains={chains_p}"
                        f"  regions={regions_p}"
                    )
                    # Show per-region token coverage (cross-region diagnosis)
                    if (
                        "region_classification" in prof_rows.columns
                        and len(regions_p) > 1
                    ):
                        by_region = {
                            reg: frozenset(grp["_token"])
                            for reg, grp in prof_rows.groupby(
                                "region_classification"
                            )
                        }
                        for reg, toks in by_region.items():
                            covered = motif_tokens_canon & toks
                            print(
                                f"        region {reg!r}: covers "
                                f"{sorted(str(t) for t in covered)}"
                            )

                if inn_col in flat_cc.columns:
                    cc_rows = flat_cc[flat_cc[inn_col] == inn]
                    if not cc_rows.empty:
                        genes_cc = (
                            sorted(cc_rows["gene"].dropna().unique())
                            if "gene" in cc_rows.columns
                            else []
                        )
                        if _eu_col_cc:
                            eu_vals = list(
                                cc_rows[_eu_col_cc].dropna().unique()
                            )[:3]
                            print(
                                f"      → constant_chains EU mutations: {eu_vals}"
                            )
                        cc_gene_rows = (
                            cc_rows[
                                cc_rows["gene"].str.lower() == gene.lower()
                            ]
                            if "gene" in cc_rows.columns
                            else pd.DataFrame()
                        )
                        status = "IS" if not cc_gene_rows.empty else "NOT"
                        print(
                            f"      → {status} in constant_chains for "
                            f"gene={gene!r} — pipeline would "
                            + (
                                "find it"
                                if not cc_gene_rows.empty
                                else "miss it"
                            )
                        )

    print(f"\n{sep}")
    print("Investigation complete.")


def trace_discrepancy_pipeline(
    debug_result: "DebugMotifResult",
    motif_result: "Any",
    mutation_string: str | None = None,
) -> None:
    """Walk every pipeline stage to find where a motif's ``inn_count`` first diverges.

    For each stage (enumeration → collapse → pre/post-resolve → filter →
    restore → pre/post-cluster → final) this function looks up the motif row
    and reports the ``inn_count`` it carries at that point.  The first stage
    where the count differs from the profile-recomputed value is highlighted
    as the origin of the discrepancy.

    Parameters
    ----------
    debug_result:
        :class:`DebugMotifResult` returned by :func:`debug_motif_inn_counts`
        (call with ``raise_on_error=False``).
    motif_result:
        ``result.motif_result`` — a ``ChainMotifIndexResult`` instance.
    mutation_string:
        The motif to trace, matching a key in ``debug_result.mismatch_detail``.
        If ``None`` and there is exactly one mismatch, that one is used.

    Examples
    --------
    ::

        trace_discrepancy_pipeline(debug_result, result.motif_result)
        trace_discrepancy_pipeline(debug_result, result.motif_result, "M234A/L235A")
    """
    if not debug_result.mismatch_detail:
        print("No mismatches in debug_result.")
        return

    if mutation_string is None:
        if len(debug_result.mismatch_detail) == 1:
            mutation_string = next(iter(debug_result.mismatch_detail))
        else:
            keys = list(debug_result.mismatch_detail)
            print(
                f"Multiple mismatches — pass mutation_string explicitly. "
                f"Available:\n  " + "\n  ".join(keys)
            )
            return

    if mutation_string not in debug_result.mismatch_detail:
        print(
            f"mutation_string {mutation_string!r} not in mismatch_detail."
        )
        return

    detail = debug_result.mismatch_detail[mutation_string]
    expected_count = detail.recomputed_inn_count
    stored_count = detail.inn_count
    gene = detail.gene

    # ── Build ordered list of (label, DataFrame) ─────────────────────────────
    # Mirrors build_chain_motif_index pipeline order exactly.
    stages: list[tuple[str, pd.DataFrame | None]] = [
        ("enumeration_df", getattr(motif_result, "enumeration_df", None)),
        (
            "enumeration_collapsed_df",
            getattr(motif_result, "enumeration_collapsed_df", None),
        ),
    ]
    other_dfs_order = [
        "motif_df_pre_resolve_super_motifs",
        "motif_df_post_resolve_super_motifs",
        "motif_df_post_filter_remove_motif",
        "motif_df_post_filter_restore_other_gene_hits",
        "motif_df_before_clustering",
        "motif_df_after_clustering",
    ]
    other = getattr(motif_result, "other_dfs", {}) or {}
    for key in other_dfs_order:
        stages.append((key, other.get(key)))
    stages.append(
        ("motif_df [final]", getattr(motif_result, "motif_df", None))
    )

    def _lookup(df: pd.DataFrame, ms: str, g: str) -> pd.Series | None:
        """Find the motif row by mutation_string_pe or mutation_string, scoped to gene."""
        for col in ("mutation_string_pe", "mutation_string"):
            if col not in df.columns:
                continue
            mask = df[col] == ms
            if "gene" in df.columns:
                mask &= df["gene"].str.lower() == g.lower()
            hits = df[mask]
            if not hits.empty:
                return hits.iloc[0]
        return None

    sep = "─" * 68
    print(f"\n{sep}")
    print(f"  Pipeline trace: {mutation_string!r}  |  gene={gene}")
    print(f"  stored={stored_count}  profile-recomputed={expected_count}")
    print(sep)

    first_bad_stage: str | None = None

    for label, df in stages:
        if df is None or (isinstance(df, pd.DataFrame) and df.empty):
            print(f"  [{label}]  (not available)")
            continue

        row = _lookup(df, mutation_string, gene)
        if row is None:
            # Try without gene filter in case gene column doesn't exist yet
            for col in ("mutation_string_pe", "mutation_string"):
                if col in df.columns:
                    hits = df[df[col] == mutation_string]
                    if not hits.empty:
                        row = hits.iloc[0]
                        break

        if row is None:
            status = "ABSENT"
            count_str = "—"
            inns_str = ""
            marker = "  "
        else:
            count = row.get("inn_count", None)
            count_str = str(int(count)) if count is not None else "?"
            inns = row.get("inns", None)
            inns_str = (
                f"  inns={sorted(inns)[:5]}"
                if isinstance(inns, (set, frozenset))
                else ""
            )
            if count is not None and int(count) != expected_count:
                status = f"inn_count={count_str}  ✗ differs from recomputed={expected_count}"
                if first_bad_stage is None:
                    first_bad_stage = label
                    marker = "▶▶"
                else:
                    marker = "  "
            else:
                status = f"inn_count={count_str}  ✓"
                marker = "  "

        print(f"  {marker} [{label}]  {status}{inns_str}")

    # ── Walk cluster iterations ───────────────────────────────────────────────
    iterations = (
        getattr(motif_result, "motif_df_per_cluster_iteration", []) or []
    )
    if iterations:
        print(f"\n  ── Cluster iterations ({len(iterations)}) ──")
        for i, entry in enumerate(iterations):
            # entry is (df_before, df_after, ChainIterativeClusters)
            df_before, df_after, cluster_info = entry
            iter_label = (
                f"{cluster_info.name} / iter={cluster_info.iteration}"
                if cluster_info is not None
                else f"iteration {i}"
            )
            print(f"\n    [{i}] {iter_label}")

            sub_stages: list[tuple[str, Any]] = []
            if df_before is not None:
                sub_stages.append(("df_before", df_before))
            if cluster_info is not None:
                for attr in (
                    "init_df",
                    "cluster_df",
                    "stats_df",
                    "output_df",
                ):
                    sub_stages.append(
                        (attr, getattr(cluster_info, attr, None))
                    )
            if df_after is not None:
                sub_stages.append(("df_after", df_after))

            for sub_label, df in sub_stages:
                if df is None or (
                    isinstance(df, pd.DataFrame) and df.empty
                ):
                    print(f"        [{sub_label}]  (not available)")
                    continue

                row = _lookup(df, mutation_string, gene)
                if row is None:
                    for col in ("mutation_string_pe", "mutation_string"):
                        if col in df.columns:
                            hits = df[df[col] == mutation_string]
                            if not hits.empty:
                                row = hits.iloc[0]
                                break

                if row is None:
                    print(f"        [{sub_label}]  ABSENT")
                else:
                    count = row.get("inn_count", None)
                    count_str = (
                        str(int(count)) if count is not None else "?"
                    )
                    inns = row.get("inns", None)
                    inns_str = (
                        f"  inns={sorted(inns)[:5]}"
                        if isinstance(inns, (set, frozenset))
                        else ""
                    )
                    if count is not None and int(count) != expected_count:
                        status = f"inn_count={count_str}  ✗ differs from recomputed={expected_count}"
                        if first_bad_stage is None:
                            first_bad_stage = (
                                f"iteration {i} / {sub_label}"
                            )
                            marker = "▶▶"
                        else:
                            marker = "  "
                    else:
                        marker = "  "
                        status = f"inn_count={count_str}  ✓"
                    print(
                        f"      {marker}  [{sub_label}]  {status}{inns_str}"
                    )

    print(f"\n{sep}")
    if first_bad_stage:
        print(f"  ▶ First discrepancy at stage: {first_bad_stage!r}")
    else:
        print(
            f"  ✓ inn_count matches recomputed={expected_count} at every stage (motif absent in early stages is normal)"
        )


def debug_pairwise_inn_overlap(
    mutation_string: str,
    scored_df: pd.DataFrame,
    inn_total: Optional[int] = None,
    mutation_col: str = "mutation_string",
    inns_col: str = "inns",
    inn_lookup: Optional[dict] = None,
) -> pd.DataFrame:
    """Show how every pairwise sub-combination of *mutation_string* relates to
    the full motif's INN set.

    For each pair of tokens drawn from the query motif this function looks up
    the INN set that carries that pair (sourced from *scored_df*, which contains
    every enumerated sub-combination) and reports overlap statistics against the
    INNs carrying the full motif.

    Parameters
    ----------
    mutation_string:
        The full motif to inspect, e.g.
        ``"233:E->- / 234:L->P / 235:L->V / 267:S->K"``.
        Format must match ``scored_df[mutation_col]``.
    scored_df:
        ``MotifIndexResult.scored_df``.  Used only to infer *inn_total* when
        not provided and to build *inn_lookup* when not pre-built.
    inn_total:
        Total INN population used as frequency denominator.  If *None* the
        function reads ``scored_df["inn_total"].iloc[0]``.
    mutation_col:
        Column in *scored_df* holding mutation strings (default
        ``"mutation_string"``).
    inns_col:
        Column in *scored_df* holding the INN set (default ``"inns"``).
    inn_lookup:
        Pre-built dict from :func:`build_inn_lookup`.  Pass this when calling
        the function repeatedly to avoid rebuilding the lookup each time.

    Returns
    -------
    pd.DataFrame
        One row per ordered pair (alphabetical on token strings), sorted by
        ``pair_inn_count`` descending.  Columns:

        * ``token_a``, ``token_b``          – the two mutation tokens
        * ``pair_mutation_string``          – canonical " / "-joined pair string
        * ``pair_inn_count``                – |INNs carrying this pair|
        * ``pair_freq``                     – pair_inn_count / inn_total
        * ``full_motif_inn_count``          – |INNs carrying the full motif|
        * ``full_motif_freq``               – full_motif_inn_count / inn_total
        * ``overlap_count``                 – |pair_inns ∩ full_motif_inns|
        * ``inns_pair_not_full``            – INNs with pair but *not* full motif
        * ``n_inns_pair_not_full``          – count of above
        * ``frac_pair_in_full``             – overlap_count / pair_inn_count
        * ``pair_inns``                     – full INN set for the pair
        * ``full_motif_inns``               – full INN set for the motif (same
                                              for every row; shown for easy
                                              downstream joining)
    """
    if not isinstance(mutation_string, str) or not mutation_string.strip():
        raise ValueError("mutation_string must be a non-empty string.")

    tokens = _parse_tokens(mutation_string)
    if len(tokens) < 2:
        raise ValueError(
            f"mutation_string has fewer than 2 tokens; "
            f"pairwise analysis requires at least 2. Got: {mutation_string!r}"
        )

    if inn_total is None:
        if "inn_total" in scored_df.columns and not scored_df.empty:
            inn_total = int(scored_df["inn_total"].dropna().iloc[0])
        else:
            raise ValueError(
                "inn_total could not be inferred from scored_df; pass it explicitly."
            )

    if inn_lookup is None:
        inn_lookup = build_inn_lookup(scored_df, mutation_col, inns_col)

    full_key = _canonical_str(tokens)
    full_inns: set = (
        inn_lookup.get(full_key)
        or inn_lookup.get(mutation_string.strip())
        or set()
    )
    full_count = len(full_inns)
    full_freq = full_count / inn_total

    if not full_inns:
        import warnings

        warnings.warn(
            f"No INN set found for the full motif {mutation_string!r}. "
            f"The full-motif columns will be empty. "
            f"Check that scored_df contains this exact mutation_string.",
            stacklevel=2,
        )

    records = []
    for tok_a, tok_b in combinations(tokens, 2):
        pair_key = _canonical_str([tok_a, tok_b])
        pair_inns: set = inn_lookup.get(pair_key, set())
        pair_count = len(pair_inns)
        overlap = pair_inns & full_inns
        only_in_pair = pair_inns - full_inns
        records.append(
            {
                "token_a": tok_a,
                "token_b": tok_b,
                "pair_mutation_string": pair_key,
                "pair_inn_count": pair_count,
                "pair_freq": round(pair_count / inn_total, 5),
                "full_motif_inn_count": full_count,
                "full_motif_freq": round(full_freq, 5),
                "overlap_count": len(overlap),
                "n_inns_pair_not_full": len(only_in_pair),
                "inns_pair_not_full": only_in_pair,
                "frac_pair_in_full": (
                    round(len(overlap) / pair_count, 4)
                    if pair_count
                    else float("nan")
                ),
                "pair_inns": pair_inns,
                "full_motif_inns": full_inns,
            }
        )

    return (
        pd.DataFrame(records)
        .sort_values("pair_inn_count", ascending=False)
        .reset_index(drop=True)
    )


def debug_cluster_overlaps(
    cluster_info,
    inn_lookup: dict[str, set],
    inn_total: int,
    frac_threshold: float = 1.0,
    top_n: Optional[int] = None,
    mutation_col: str = "mutation_string",
) -> pd.DataFrame:
    """Report token overlaps and INN overlaps between every pair of cluster seeds.

    Accepts either:

    * The ``cluster_info`` dict returned by :func:`cluster_motifs_by_substring`
      (keys = seed strings, values = ``{"n_members", "members", "representative"}``).
    * A **DataFrame** of seed rows — any DataFrame whose rows represent seeds,
      e.g. ``motif_df[motif_df["sub_motifs"].apply(len) == 0]``.  The
      *mutation_col* column is used as the seed string; ``inn_count`` and a
      cluster-size column are read if present.

    For each pair of seeds this function reports:

    * Which tokens (mutations) they share — if any, the seeds are not fully
      independent and may need further decomposition.
    * How much their INN sets overlap — high INN overlap means many antibodies
      carry mutations from both seeds simultaneously.
    * The worst cross-seed ``frac_pair_in_full`` — a low value confirms the
      seeds are genuinely independent.

    Parameters
    ----------
    cluster_info : dict or DataFrame
        Seed source — see above.
    inn_lookup : dict
        Pre-built from :func:`build_inn_lookup`.
    inn_total : int
        Total INN population for frequency computation.
    frac_threshold : float
        Cross-seed pairs below this threshold are flagged as "leaky".
    top_n : int or None
        Restrict to the top *top_n* seeds by member/inn count.  ``None`` = all.
    mutation_col : str
        Column name to use when *cluster_info* is a DataFrame (default
        ``"mutation_string"``).

    Returns
    -------
    pd.DataFrame
        One row per (seed_a, seed_b) pair, sorted by ``shared_token_count``
        descending then ``inn_overlap_count`` descending.  Columns:

        ``seed_a``, ``seed_b``
            The two seed mutation_strings.
        ``seed_a_inn_count``, ``seed_b_inn_count``
            INN counts for each seed.
        ``seed_a_n_members``, ``seed_b_n_members``
            Cluster sizes (1 if not available from a DataFrame source).
        ``shared_tokens``
            Frozenset of mutation tokens appearing in both seeds.
        ``shared_token_count``
            \\|shared_tokens\\|.  > 0 means the seeds are NOT disjoint.
        ``inn_overlap_count``
            \\|INNs(seed_a) ∩ INNs(seed_b)\\|.
        ``inn_overlap_frac``
            inn_overlap_count / min(seed_a_inn_count, seed_b_inn_count).
        ``n_cross_pairs``
            Number of cross-seed token pairs evaluated.
        ``n_leaky_cross_pairs``
            Cross-seed pairs with frac < frac_threshold.
        ``min_cross_pair_frac``
            Worst frac across all cross-seed pairs.
        ``leakiest_cross_pair``
            Canonical string of the pair with the lowest frac.
    """
    # ── Normalise input to (seeds list, n_members dict) ───────────────────────
    n_members_map: dict[str, int] = {}

    if isinstance(cluster_info, pd.DataFrame):
        df_seeds = cluster_info
        if mutation_col not in df_seeds.columns:
            raise ValueError(
                f"debug_cluster_overlaps: column '{mutation_col}' not found in DataFrame. "
                f"Available: {list(df_seeds.columns)}"
            )
        seeds = df_seeds[mutation_col].dropna().tolist()
        # Prefer n_members / cluster size column; fall back to inn_count; default 1
        size_col = next(
            (
                c
                for c in ("n_members", "n_clusters", "inn_count")
                if c in df_seeds.columns
            ),
            None,
        )
        for _, row in df_seeds.iterrows():
            ms = row[mutation_col]
            if not isinstance(ms, str):
                continue
            n_members_map[ms] = int(row[size_col]) if size_col else 1
    else:
        # dict from cluster_motifs_by_substring
        seeds = list(cluster_info.keys())
        for s, info in cluster_info.items():
            n_members_map[s] = info.get("n_members", 1)

    if top_n is not None:
        seeds = sorted(
            seeds, key=lambda s: n_members_map.get(s, 1), reverse=True
        )[:top_n]

    # ── Resolve INN sets for each seed ────────────────────────────────────────
    seed_inns: dict[str, set] = {}
    seed_tokens: dict[str, frozenset] = {}
    for s in seeds:
        toks = _parse_tokens(s)
        seed_tokens[s] = frozenset(toks)
        seed_inns[s] = inn_lookup.get(
            _canonical_str(toks)
        ) or inn_lookup.get(s, set())

    records = []
    for i, sa in enumerate(seeds):
        for sb in seeds[i + 1 :]:
            toks_a = seed_tokens[sa]
            toks_b = seed_tokens[sb]
            inns_a = seed_inns[sa]
            inns_b = seed_inns[sb]

            shared = toks_a & toks_b
            inn_overlap = inns_a & inns_b

            # Cross-seed pair fracs: for each (tok_a from sa, tok_b from sb)
            # check how often that pair occurs outside sa's context
            n_cross = 0
            n_leaky = 0
            min_frac = 1.0
            worst_pair = ""

            for tok_a in toks_a:
                for tok_b in toks_b:
                    if tok_a == tok_b:
                        continue
                    pair_key = _canonical_str([tok_a, tok_b])
                    pair_inns = inn_lookup.get(pair_key, set())
                    if not pair_inns:
                        continue
                    # frac relative to seed_a's inns
                    frac_a = len(pair_inns & inns_a) / len(pair_inns)
                    frac_b = len(pair_inns & inns_b) / len(pair_inns)
                    frac = min(
                        frac_a, frac_b
                    )  # worst case across both seeds
                    n_cross += 1
                    if frac < frac_threshold:
                        n_leaky += 1
                    if frac < min_frac:
                        min_frac = frac
                        worst_pair = pair_key

            inn_a_count = len(inns_a)
            inn_b_count = len(inns_b)
            records.append(
                {
                    "seed_a": sa,
                    "seed_b": sb,
                    "seed_a_inn_count": inn_a_count,
                    "seed_b_inn_count": inn_b_count,
                    "seed_a_n_members": n_members_map.get(sa, 1),
                    "seed_b_n_members": n_members_map.get(sb, 1),
                    "shared_tokens": shared,
                    "shared_token_count": len(shared),
                    "inn_overlap_count": len(inn_overlap),
                    "inn_overlap_frac": (
                        round(
                            len(inn_overlap)
                            / min(inn_a_count, inn_b_count),
                            4,
                        )
                        if min(inn_a_count, inn_b_count) > 0
                        else float("nan")
                    ),
                    "n_cross_pairs": n_cross,
                    "n_leaky_cross_pairs": n_leaky,
                    "min_cross_pair_frac": (
                        round(min_frac, 4) if n_cross else float("nan")
                    ),
                    "leakiest_cross_pair": worst_pair,
                }
            )

    return (
        pd.DataFrame(records)
        .sort_values(
            ["shared_token_count", "inn_overlap_count"],
            ascending=[False, False],
        )
        .reset_index(drop=True)
    )


def audit_motif_cohesion(
    motif_df: pd.DataFrame,
    scored_df: pd.DataFrame,
    inn_total: Optional[int] = None,
    mutation_col: str = "mutation_string",
    inns_col: str = "inns",
    inn_lookup: Optional[dict] = None,
) -> pd.DataFrame:
    """Evaluate internal pairwise cohesion for every motif in *motif_df*.

    For each motif, computes pairwise INN overlap statistics and summarises how
    "captive" its token pairs are to the full motif.  A motif with uniformly
    high ``frac_pair_in_full`` is genuinely cohesive — every INN that carries
    any two of its tokens also carries the whole motif.  A motif with low
    ``min_frac_pair_in_full`` contains at least one pair that frequently
    co-occurs *outside* the full motif, indicating an over-greedy seed that
    absorbed what should have been an independent sub-pattern.

    Pass a pre-built *inn_lookup* (from :func:`build_inn_lookup`) to avoid
    rebuilding the dict when calling this alongside
    :func:`debug_pairwise_inn_overlap` or multiple times on different subsets.

    Parameters
    ----------
    motif_df:
        Motifs to audit — typically ``MotifIndexResult.motif_df`` or
        ``removed_df``.  Must contain *mutation_col*.
    scored_df:
        ``MotifIndexResult.scored_df`` — used to build the INN lookup and
        infer *inn_total* if not provided.
    inn_total:
        Total INN population.  Inferred from *scored_df* if not provided.
    mutation_col:
        Column holding mutation strings (default ``"mutation_string"``).
    inns_col:
        Column in *scored_df* holding the INN set (default ``"inns"``).
    inn_lookup:
        Pre-built dict from :func:`build_inn_lookup`.

    Returns
    -------
    pd.DataFrame
        One row per motif, sorted by ``min_frac_pair_in_full`` ascending
        (most suspicious / leaky motifs first).  Columns:

        * ``mutation_string``           – the motif
        * ``num_mutations``             – token count
        * ``inn_count``                 – INN count
        * ``n_pairs``                   – C(num_mutations, 2)
        * ``min_frac_pair_in_full``     – worst pair: fraction of its INNs
                                          captured by the full motif.
                                          < 1.0 means the pair leaks outside.
        * ``mean_frac_pair_in_full``    – average over all pairs
        * ``max_inns_pair_not_full``    – INNs carrying the leakiest pair but
                                          not the full motif
        * ``leakiest_pair``             – mutation_string of that pair
        * ``leakiest_pair_inn_count``   – inn_count of that pair
        * ``cohesion_flag``             – ``"cohesive"`` / ``"leaky"``
    """
    if inn_total is None:
        if "inn_total" in scored_df.columns and not scored_df.empty:
            inn_total = int(scored_df["inn_total"].dropna().iloc[0])
        else:
            raise ValueError(
                "inn_total could not be inferred from scored_df; pass it explicitly."
            )

    # Build lookup once from column arrays — no iterrows
    if inn_lookup is None:
        inn_lookup = build_inn_lookup(scored_df, mutation_col, inns_col)

    # Extract motif_df columns as plain Python lists — one pass, no per-row DF access
    ms_list: list[str] = motif_df[mutation_col].tolist()
    nm_list: list[int] = (
        motif_df["num_mutations"].tolist()
        if "num_mutations" in motif_df.columns
        else [None] * len(ms_list)
    )
    ic_list: list[int] = (
        motif_df["inn_count"].tolist()
        if "inn_count" in motif_df.columns
        else [None] * len(ms_list)
    )

    _nan = float("nan")
    records = []

    for ms, nm, ic in zip(ms_list, nm_list, ic_list):
        if not isinstance(ms, str) or not ms.strip():
            continue

        tokens = _parse_tokens(ms)
        if len(tokens) < 2:
            continue

        full_inns: set = (
            inn_lookup.get(_canonical_str(tokens))
            or inn_lookup.get(ms.strip())
            or set()
        )
        full_count = len(full_inns)

        # Compute all C(n,2) pairs in one pass
        min_frac = 1.0
        mean_acc = 0.0
        n_valid = 0
        max_outside = 0
        worst_key = ""
        worst_count = 0
        worst_frac = 1.0
        worst_outside = 0  # INNs in leakiest pair but NOT in seed

        for tok_a, tok_b in combinations(tokens, 2):
            pair_key = _canonical_str([tok_a, tok_b])
            pair_inns: set = inn_lookup.get(pair_key, set())
            pair_count = len(pair_inns)
            if pair_count == 0:
                continue
            overlap = len(pair_inns & full_inns)
            frac = overlap / pair_count
            outside = pair_count - overlap

            mean_acc += frac
            n_valid += 1
            if outside > max_outside:
                max_outside = outside
            if frac < worst_frac:
                worst_frac = frac
                worst_key = pair_key
                worst_count = pair_count
                worst_outside = (
                    outside  # capture outside for THIS worst pair
                )
            if frac < min_frac:
                min_frac = frac

        if n_valid == 0:
            continue

        mean_frac = mean_acc / n_valid
        n_pairs = n_valid

        records.append(
            {
                mutation_col: ms,
                "num_mutations": len(tokens) if nm is None else int(nm),
                "inn_count": full_count if ic is None else int(ic),
                "n_pairs": n_pairs,
                "min_frac_pair_in_full": round(min_frac, 4),
                "mean_frac_pair_in_full": round(mean_frac, 4),
                "max_inns_pair_not_full": max_outside,
                "leakiest_pair": worst_key,
                "leakiest_pair_inn_count": worst_count,
                "leakiest_pair_inns_outside_seed": worst_outside,
                "cohesion_flag": (
                    "cohesive" if min_frac == 1.0 else "leaky"
                ),
            }
        )

    if not records:
        return pd.DataFrame(
            columns=[
                mutation_col,
                "num_mutations",
                "inn_count",
                "n_pairs",
                "min_frac_pair_in_full",
                "mean_frac_pair_in_full",
                "max_inns_pair_not_full",
                "leakiest_pair",
                "leakiest_pair_inn_count",
                "leakiest_pair_inns_outside_seed",
                "cohesion_flag",
            ]
        )
    return (
        pd.DataFrame(records)
        .sort_values("min_frac_pair_in_full", ascending=True)
        .reset_index(drop=True)
    )


def decompose_leaky_motif(
    mutation_string: str,
    inn_lookup: dict[str, set],
    inn_total: int,
    frac_threshold: float = 1.0,
) -> dict:
    """Decompose a motif into maximally cohesive sub-motifs via tight-pair cliques.

    Algorithm
    ---------
    1. Compute ``frac_pair_in_full`` for every C(n,2) pair in *mutation_string*.
    2. Build a graph: nodes = tokens, edges = pairs whose
       ``frac_pair_in_full >= frac_threshold`` ("tight" pairs).
    3. Find all maximal cliques of that tight graph (via
       ``nx.find_cliques``).  Each clique is the largest set of tokens
       that are mutually tight — i.e. every internal pair stays cohesive.
    4. Look each clique up in *inn_lookup* to see whether it is already a
       known motif in ``scored_df``.

    If the tight graph is a single clique covering **all** tokens the motif
    is genuinely cohesive and should remain a seed.  If it fragments into
    multiple cliques the motif is over-greedy and should be split.

    Parameters
    ----------
    mutation_string:
        The full motif to decompose.
    inn_lookup:
        Pre-built dict from :func:`build_inn_lookup`.
    inn_total:
        Total INN population (denominator for frequency).
    frac_threshold:
        Minimum ``frac_pair_in_full`` for a pair to be considered "tight"
        (default ``1.0`` — only perfectly captive pairs are tight).
        Lower values (e.g. ``0.9``) allow slight leakage.

    Returns
    -------
    dict with keys:

    ``is_seed`` : bool
        True when every pair is tight (no leaky pairs at this threshold).
    ``tokens`` : list[str]
        All tokens in the query motif (sorted).
    ``leaky_pairs`` : list[dict]
        Each dict has ``pair``, ``pair_inn_count``, ``frac_pair_in_full``,
        ``n_inns_outside`` for pairs below *frac_threshold*.
    ``tight_pairs`` : list[dict]
        Same structure for pairs at or above *frac_threshold*.
    ``candidate_sub_motifs`` : list[dict]
        One entry per maximal clique (size ≥ 2), sorted by clique size
        descending then inn_count descending.  Each dict has:

        * ``tokens``         – sorted list of token strings in the clique
        * ``mutation_string`` – canonical " / "-joined string
        * ``size``           – number of tokens
        * ``inn_count``      – INN count if found in inn_lookup, else None
        * ``inn_freq``       – inn_count / inn_total if found, else None
        * ``in_scored_df``   – True if the clique string is in inn_lookup
        * ``inns``           – INN set if found, else empty set
    ``singleton_tokens`` : list[str]
        Tokens that appear in zero tight pairs (isolated nodes in the
        tight graph) — these contribute no clique and cannot be merged.
    """
    if not isinstance(mutation_string, str) or not mutation_string.strip():
        raise ValueError("mutation_string must be a non-empty string.")

    tokens = _parse_tokens(mutation_string)
    if len(tokens) < 2:
        raise ValueError(
            f"mutation_string has fewer than 2 tokens. Got: {mutation_string!r}"
        )

    full_inns: set = (
        inn_lookup.get(_canonical_str(tokens))
        or inn_lookup.get(mutation_string.strip())
        or set()
    )

    # ── 1. Score every pair ───────────────────────────────────────────────────
    leaky_pairs: list[dict] = []
    tight_pairs: list[dict] = []
    tight_edges: list[tuple[str, str]] = []

    for tok_a, tok_b in combinations(tokens, 2):
        pair_key = _canonical_str([tok_a, tok_b])
        pair_inns: set = inn_lookup.get(pair_key, set())
        pair_count = len(pair_inns)
        if pair_count == 0:
            frac = float("nan")
        else:
            frac = len(pair_inns & full_inns) / pair_count
        outside = (
            pair_count - len(pair_inns & full_inns) if pair_count else 0
        )

        entry = {
            "pair": pair_key,
            "pair_inn_count": pair_count,
            "frac_pair_in_full": (
                round(frac, 4) if not np.isnan(frac) else float("nan")
            ),
            "n_inns_outside": outside,
        }
        if np.isnan(frac) or frac < frac_threshold:
            leaky_pairs.append(entry)
        else:
            tight_pairs.append(entry)
            tight_edges.append((tok_a, tok_b))

    is_seed = len(leaky_pairs) == 0

    # ── 2. Build tight-pair graph and find maximal cliques ────────────────────
    G = nx.Graph()
    G.add_nodes_from(tokens)
    G.add_edges_from(tight_edges)

    # Isolated nodes = tokens with no tight partner
    singleton_tokens: list[str] = sorted(
        t for t in tokens if G.degree(t) == 0
    )

    # nx.find_cliques requires at least one edge; guard the empty case
    if G.number_of_edges() == 0:
        raw_cliques: list[list[str]] = []
    else:
        raw_cliques = list(nx.find_cliques(G))

    # Keep only cliques of size ≥ 2
    cliques = [sorted(c) for c in raw_cliques if len(c) >= 2]

    # ── 3. Look each clique up in inn_lookup ──────────────────────────────────
    candidate_sub_motifs: list[dict] = []
    for clique_toks in cliques:
        clique_str = _canonical_str(clique_toks)
        clique_inns: set = inn_lookup.get(clique_str, set())
        clique_count = len(clique_inns) if clique_inns else None
        candidate_sub_motifs.append(
            {
                "tokens": clique_toks,
                "mutation_string": clique_str,
                "size": len(clique_toks),
                "inn_count": clique_count,
                "inn_freq": (
                    round(clique_count / inn_total, 5)
                    if clique_count
                    else None
                ),
                "in_scored_df": bool(clique_inns),
                "inns": clique_inns,
            }
        )

    candidate_sub_motifs.sort(
        key=lambda d: (d["size"], d["inn_count"] or 0), reverse=True
    )

    return {
        "is_seed": is_seed,
        "tokens": tokens,
        "leaky_pairs": sorted(
            leaky_pairs, key=lambda d: d["frac_pair_in_full"]
        ),
        "tight_pairs": sorted(
            tight_pairs, key=lambda d: d["frac_pair_in_full"]
        ),
        "candidate_sub_motifs": candidate_sub_motifs,
        "singleton_tokens": singleton_tokens,
    }


def compute_motif_ancestry(
    motif_df: pd.DataFrame,
    min_lift: float = 1.5,
    max_pvalue: float = 0.05,
) -> pd.DataFrame:
    """Add MLE ancestry columns to a motif_df produced by build_motif_index.

    For each motif, finds the exact optimal partition of its tokens into known
    atomic sub-motifs + singletons that maximises ``∏ freq(component)`` via
    bitmask DP.

    Speedup: motifs are processed size-ascending.  A motif that ``pass_ancestry``
    is added to the *atomic candidate pool*; one that doesn't is replaced by its
    own partition's components in that pool.  This means each parent only sees
    truly irreducible building blocks, keeping the candidate set small.

    New columns
    -----------
    ancestry_components : list of dicts – each with mutation_string, size, freq,
        expected, lift, lift_conditional.
    expected_ancestry   : product of atomic component frequencies
    lift_ancestry       : freq_motif / expected_ancestry
    pvalue_ancestry     : binomtest p-value vs expected_ancestry
    pass_ancestry       : bool – lift and pvalue both pass thresholds
    """
    import math

    if motif_df.empty:
        for col in (
            "ancestry_components",
            "expected_ancestry",
            "lift_ancestry",
            "pvalue_ancestry",
            "pass_ancestry",
        ):
            motif_df[col] = None
        return motif_df

    # ── Lookups ───────────────────────────────────────────────────────────────
    indiv_map: dict[tuple, float] = {}
    for _, row in motif_df.iterrows():
        for t, f in row["marginal_freq"].items():
            indiv_map.setdefault(t, f)

    _row0 = motif_df[motif_df["inn_count"] > 0].iloc[0]
    N_total = (
        int(_row0["inn_total"])
        if "inn_total" in motif_df.columns
        else round(_row0["inn_count"] / _row0["freq_motif"])
    )

    # Eligible building blocks: passes_submotif_filter AND not consumed by a larger motif
    # (consumed motifs never appear independently — they're shadows of their parent)
    _anc_mask = motif_df["passes_submotif_filter"]
    if "is_fully_described_by_larger_motif" in motif_df.columns:
        _anc_mask = (
            _anc_mask & ~motif_df["is_fully_described_by_larger_motif"]
        )
    motif_stats: dict[frozenset, dict] = {
        r["motif"]: r
        for r in motif_df[_anc_mask][
            [
                "motif",
                "freq_motif",
                "lift_vs_indep",
                "lift_vs_submotifs",
                "expected_freq_indep",
                "mutation_string",
                "num_mutations",
            ]
        ].to_dict("records")
    }

    def _sorted_str(sub: frozenset) -> str:
        toks = sorted(sub, key=lambda t: (str(t[0]), t[1]))
        return " / ".join(f"{eu}:{ch}" for eu, ch in toks)

    NEG_INF = float("-inf")

    # atomic_pool: frozenset -> (log_freq, component_info_dict)
    # Populated as we process motifs ascending by size.
    # If a motif passes ancestry, it enters as a single atomic block.
    # If it doesn't pass, its OWN optimal partition's components are added instead
    # so parents can reuse the flattened decomposition.
    atomic_pool: dict[frozenset, tuple[float, dict]] = {}

    # Pre-seed singletons into the pool
    for t, f in indiv_map.items():
        lf = math.log(f) if f > 0 else NEG_INF
        info = {
            "mutation_string": _sorted_str(frozenset([t])),
            "size": 1,
            "freq": f,
            "expected": f,
            "lift": 1.0,
            "lift_vs_submotifs": 1.0,
        }
        atomic_pool[frozenset([t])] = (lf, info)

    def _optimal_decomp(m: frozenset):
        """Bitmask DP over atomic_pool candidates that are strict subsets of m."""
        tokens = sorted(m, key=lambda t: (str(t[0]), t[1]))
        n = len(tokens)
        idx = {t: i for i, t in enumerate(tokens)}
        full = (1 << n) - 1

        candidates: list[tuple[int, float, frozenset]] = []
        for sub, (lf, _) in atomic_pool.items():
            if sub < m and lf != NEG_INF:
                mask = 0
                valid = True
                for t in sub:
                    if t not in idx:
                        valid = False
                        break
                    mask |= 1 << idx[t]
                if valid:
                    candidates.append((mask, lf, sub))

        dp_val = [NEG_INF] * (1 << n)
        dp_parts: list[list | None] = [None] * (1 << n)
        dp_val[0] = 0.0
        dp_parts[0] = []

        for mask in range(1, full + 1):
            lowest = mask & (-mask)
            for cmask, lf, csub in candidates:
                if (cmask & lowest) and (cmask & mask) == cmask:
                    prev = mask ^ cmask
                    new_val = dp_val[prev] + lf
                    if new_val > dp_val[mask]:
                        dp_val[mask] = new_val
                        dp_parts[mask] = (dp_parts[prev] or []) + [csub]

        if dp_val[full] == NEG_INF:
            return None, None
        return math.exp(dp_val[full]), dp_parts[full]

    # ── Process size-ascending (skip consumed shadows) ───────────────────────
    _null_rec = {
        "ancestry_components": None,
        "expected_ancestry": None,
        "lift_ancestry": None,
        "pvalue_ancestry": None,
        "pass_ancestry": False,
    }
    _consumed_mask = motif_df.get(
        "is_fully_described_by_larger_motif",
        pd.Series(False, index=motif_df.index),
    )
    _work_df = motif_df[~_consumed_mask]

    result_map: dict[int, dict] = {
        i: _null_rec for i in motif_df[_consumed_mask].index
    }

    for _, row in _tqdm(
        _work_df.sort_values("num_mutations").iterrows(),
        total=len(_work_df),
        desc="Ancestry DP",
    ):
        m = row["motif"]
        freq = row["freq_motif"]
        count = row["inn_count"]
        orig_idx = row.name
        row_n = (
            int(row["inn_total"]) if "inn_total" in row.index else N_total
        )

        exp_anc, parts = _optimal_decomp(m)

        if exp_anc is None or not parts:
            rec = {
                "ancestry_components": [],
                "expected_ancestry": None,
                "lift_ancestry": None,
                "pvalue_ancestry": None,
                "pass_ancestry": False,
            }
            result_map[orig_idx] = rec
            # Add to pool as singletons only (already seeded)
            continue

        lift_anc = freq / exp_anc if exp_anc > 0 else float("inf")
        pval_anc = float(
            _binom(
                count, row_n, min(exp_anc, 1.0), alternative="greater"
            ).pvalue
        )
        _lp = len(m) == 1 or lift_anc >= min_lift
        _pp = pval_anc <= max_pvalue
        passes = _lp and _pp

        comp_info = [atomic_pool[csub][1] for csub in parts]

        rec = {
            "ancestry_components": comp_info,
            "expected_ancestry": round(exp_anc, 6),
            "lift_ancestry": round(lift_anc, 3),
            "pvalue_ancestry": round(pval_anc, 10),
            "pass_ancestry": passes,
        }
        result_map[orig_idx] = rec

        stats = motif_stats.get(m)
        if passes and stats:
            # Atomic: add this motif itself as a single block for parents
            lf = math.log(freq) if freq > 0 else NEG_INF
            info = {
                "mutation_string": stats["mutation_string"],
                "size": stats["num_mutations"],
                "freq": freq,
                "expected": stats["expected_freq_indep"],
                "lift": stats["lift_vs_indep"],
                "lift_vs_submotifs": stats["lift_vs_submotifs"],
            }
            atomic_pool[m] = (lf, info)
        elif not passes and parts:
            # Composite: propagate its flattened components into the pool
            # so parents inherit the fully-reduced building blocks
            flat_lf = math.log(exp_anc) if exp_anc > 0 else NEG_INF
            flat_info = {
                "mutation_string": (
                    _sorted_str(m)
                    if not stats
                    else stats["mutation_string"]
                ),
                "size": len(m),
                "freq": exp_anc,  # effective freq = product of atomic children
                "expected": exp_anc,
                "lift": 1.0,
                "lift_interaction": 1.0,
            }
            atomic_pool[m] = (flat_lf, flat_info)

    records = [result_map[i] for i in motif_df.index]
    anc_df = pd.DataFrame(records, index=motif_df.index)

    def _first_pos(comp):
        # extract the leading position number from the mutation_string for sorting
        first = comp["mutation_string"].split("/")[0].strip()
        try:
            return int(first.split(":")[0])
        except ValueError:
            return 0

    def _ancestry_str(comps):
        if not comps or all(c["size"] == 1 for c in comps):
            return None
        multi = [c for c in comps if c["size"] > 1]
        sorted_comps = sorted(
            multi, key=lambda c: (-c["size"], _first_pos(c))
        )
        return " | ".join(c["mutation_string"] for c in sorted_comps)

    anc_df["ancestry_string"] = anc_df["ancestry_components"].apply(
        _ancestry_str
    )

    def _novel_str(row):
        comps = row["ancestry_components"]
        if not comps or all(c["size"] == 1 for c in comps):
            return None
        # tokens covered by any multi-token block
        covered = frozenset(
            t
            for c in comps
            if c["size"] > 1
            for t in c.get("motif", frozenset())
        )
        # fall back to parsing mutation_string if motif key absent
        if not covered:
            covered_strs = {
                tok.strip()
                for c in comps
                if c["size"] > 1
                for tok in c["mutation_string"].split("/")
            }
            novel_toks = [
                tok
                for tok in row["mutation_string"].split("/")
                if tok.strip() not in covered_strs
            ]
        else:
            all_toks = [
                tok.strip() for tok in row["mutation_string"].split("/")
            ]
            covered_strs = {
                tok.strip()
                for c in comps
                if c["size"] > 1
                for tok in c["mutation_string"].split("/")
            }
            novel_toks = [t for t in all_toks if t not in covered_strs]
        return " / ".join(novel_toks) if novel_toks else None

    combined = pd.concat([motif_df, anc_df], axis=1)
    combined["novel_string"] = combined.apply(_novel_str, axis=1)
    return combined
