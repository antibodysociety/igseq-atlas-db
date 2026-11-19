"""End-to-end constant-chain mutation analysis pipeline.

Transforms an :class:`INNAnalysisDataset` into a
:class:`ConstantChainAnalysisResult` containing all derived analysis tables,
then exports the result to a multi-sheet Excel workbook.

Usage
-----
::

    from inn_analysis_ops.analytics.constant_chain_analysis import (
        ConstantChainAnalysisConfig,
        build_constant_chain_analysis,
        export_constant_chain_analysis,
    )

    cfg = ConstantChainAnalysisConfig()          # tweak defaults as needed
    result = build_constant_chain_analysis(ds, cfg)
    export_constant_chain_analysis(result, "ConstantChainAnalysis.xlsx")

    # Or access tables directly:
    result.unique_single_mutation_events
    result.co_df_export
    result.motif_df_clustered_post_filter
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

import pandas as pd
from pydantic import BaseModel, ConfigDict
from tqdm.auto import tqdm

_log = logging.getLogger(__name__)

from inn_analysis_ops.analytics._helpers import (
    find_multi_chain_inns as _find_multi_chain_inns,
)
from inn_analysis_ops.analytics.motifs.allele_mapping import (
    find_multi_chain_inns_by_subdomain as _find_multi_chain_inns_by_subdomain,
)
from inn_analysis_ops.analytics._helpers import motif_signature_without_germline
from inn_analysis_ops.analytics.co_mutation_analytics import (
    add_inn_date_info,
    cluster_motifs_by_substring,
    co_df_to_export_table,
    compute_co_mutation_frequencies,
    remove_intra_pairs,
)
from inn_analysis_ops.analytics.motifs.models import ChainMotifIndexResult
from inn_analysis_ops.analytics.motifs.pipelines.chain import (
    ChainMotifStep,
    step_add_effector_annotation,
)
from inn_analysis_ops.analytics.motifs.pipelines.chain import (
    build_chain_motif_index as _build_chain_motif_index,
)
from inn_analysis_ops.analytics.mutation_stats import (
    build_motif_stats_per_inn,
    build_motif_stats_per_motif,
    build_mutation_residues,
    mutation_repetitions_per_inn,
)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Sub-configurations
# ---------------------------------------------------------------------------


@dataclass
class CoMutationConfig:
    """Parameters for the co-mutation frequency step."""

    min_count: int = 2
    position_col: str = "EU"
    keep_classifications: List[str] = field(
        default_factory=lambda: [
            "intra",
            "inter_same_gene",
            "inter_cross_gene",
        ]
    )


@dataclass
class MotifIndexConfig:
    """Parameters for motif index building and clustering."""

    debug_step: int | None = None
    min_count: int = 2
    min_lift: float = 1.0
    max_pvalue: float = 0.05
    min_size: int = 1
    max_size: int = 20
    max_consecutive_deletions: int | None = 5
    cohesion_frac_threshold: float | None = 1.0
    max_cohesion_iterations: int | None = None
    hard_stat_filters: (
        list[Literal["pvalue", "lift", "cohesion"]] | None
    ) = field(default_factory=lambda: ["pvalue", "lift", "cohesion"])
    additional_user_set_filter_terms: (
        dict[str, tuple[str, float]] | None
    ) = None
    debug_step: int | None = None

    # No longer used — ``build_chain_motif_index`` handles bispecific cross-chain
    # pairing natively.  Kept for backwards-compatibility with existing configs;
    # the value is ignored by the new pipeline.
    #  bispecific_branching: Optional[str] = None => deprecated
    # Minimum number of mutations for a bispecific anchor row to appear in
    # motif_df_clustered_post_filter.  ``None`` (default) means bispecific
    # anchor rows follow the same ``num_mutations > 1`` gate as all other rows.
    # Set to ``1`` to allow single-token anchors (e.g. ``297:N->G``) to appear
    # as explicit anchor rows in the clustered table.
    # bispecific_anchor_min_mutations: Optional[int] = None => deprecated
    mutations_to_ignore_for_motif_ops: Dict[str, Dict[str, List[str]]] = (
        field(
            default_factory=lambda: {
                "insertion": {
                    "HINGE": ["230", "230.i", "219", "220"],
                    "FC": ["447"],
                },
                "deletion": {
                    "FC": ["447"],
                    "HINGE": [
                        "230",
                        "230.i",
                        "217",
                        "216",
                        "218",
                        "215",
                        "219",
                        "220",
                    ],
                },
            }
        )
    )
    # Motifs to force-keep regardless of statistical thresholds.
    # Passed directly to ``build_chain_motif_index`` as ``allowed_motifs_by_select``.
    allowed_motifs_by_select: Optional[List[str]] = None
    # How to handle adjusted_count > inn_count invariant violations.
    # ``"warn"`` (default) logs a warning; ``"assert"`` raises ValueError.
    inn_count_invariant_mode: str = "warn"
    # add in motifs that should always be present
    manual_motif_definitions: list[str] | None = None
    # Germline-reversion correction for cross-chain pairing.
    # Required when the dataset contains bispecific antibodies whose two arms
    # carry different alleles (e.g. IGHG1*01 vs IGHG4*01).  When alleles
    # differ for a chain pair, tokens whose mutated residue matches the other
    # allele's germline are rescued from the arm-exclusive set into the shared
    # bispecific anchor.  A ValueError is raised if alleles differ and these
    # are not provided.
    cdb: Optional[Any] = field(default=None, repr=False)
    """ConstantDB instance (from build_constant_db). Required for cross-allele bispecifics."""
    cn: Optional[Any] = field(default=None, repr=False)
    """ConstantChainNumbering instance. Required for cross-allele bispecifics."""


@dataclass
class EffectorAnnotationConfig:
    """Parameters for joining effector annotation onto analysis tables."""

    effector_df: Optional[Any] = field(default=None, repr=False)
    """DataFrame to left-join onto single-mutation and motif tables.
    Join key is always ``mutation_freq_resolution + ["mutation_string_pe"]``."""
    annotation_cols: Optional[List[str]] = None
    """Columns to keep from effector_df after join. ``None`` keeps all."""


# ---------------------------------------------------------------------------
# Top-level configuration
# ---------------------------------------------------------------------------


@dataclass
class ConstantChainAnalysisConfig:
    """Tunable parameters for the analysis pipeline.

    Sub-configs for the heavier steps can be passed independently::

        cfg = ConstantChainAnalysisConfig(
            mutation_freq_resolution=["gene_species", "gene"],
            co_mutation=CoMutationConfig(min_count=3),
            motif_index=MotifIndexConfig(max_pvalue=0.01, cohesion_frac_threshold=0.9),
            effector=EffectorAnnotationConfig(effector_df=will_effector_motif_df),
        )
    """

    # Stratification columns — must match group_by used in the motif pipeline
    mutation_freq_resolution: List[str] = field(
        default_factory=lambda: ["gene_species", "gene"]
    )

    # Region classification groups for positional mutation frequency
    region_classification_groups: Dict[str, List[str]] = field(
        default_factory=lambda: {
            "heavy": ["FC", "HINGE", "CH1"],
            "light": ["CL"],
        }
    )

    # Mutation types to include in single-event and per-INN tables
    mutation_types: List[str] = field(
        default_factory=lambda: ["substitution", "deletion", "insertion"]
    )

    # Mol-info columns to include from kept/removed_inn
    mol_info_cols: List[str] = field(
        default_factory=lambda: [
            "inn_name",
            "inn_number",
            "imgt_id",
            "format",
            "receptor_type",
            "isotype",
            "estimated_status",
            "companies",
            "year_proposed",
            "year_recommended",
            "year_received",
            "specificity",
            "species",
            "conjugated",
            "origin_species",
            "origin_clone",
            "expression_system",
            "mechanism_of_action",
            "clinical_indication",
            "clinical_domain",
            "highest_clinical_trial",
            "development_status",
            "biosimilars",
            "effector_function",
            "half_life",
            "heterodimerization",
            "site_specific_conj",
            "stabilized",
            "c_term_truncation",
            "other_fc_mod",
            "sources",
        ]
    )

    # Sequence list columns to include
    sequence_cols: List[str] = field(
        default_factory=lambda: [
            "inn",
            "inn_id",
            "chain_id",
            "chain_type",
            "sequence",
            "variable_chain_species",
            "constant_chain_species",
            "chain_description",
            "germline_sequence",
            "v_allele",
            "j_allele",
            "c_allele",
            "total_mutations",
            "total_constant_mutations",
            "total_mutations_unclassified_features",
            "variable_mutations",
            "constant_chain_mutations_IMGT",
            "constant_chain_mutations_EU",
            "constant_chain_mutations_KABAT",
        ]
    )

    # Date column in ds.kept_inn for motif timeline
    inn_date_source: str = "min_year"

    # Sub-configurations
    co_mutation: CoMutationConfig = field(default_factory=CoMutationConfig)
    motif_index: MotifIndexConfig = field(default_factory=MotifIndexConfig)
    motif_classification: EffectorAnnotationConfig = field(
        default_factory=EffectorAnnotationConfig
    )


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------


@dataclass
class ConstantChainAnalysisResult:
    """All derived analysis tables from the constant-chain pipeline."""

    # ── INN overview ─────────────────────────────────────────────────────────
    mol_info: pd.DataFrame
    """Kept INNs with molecule-level annotation."""
    skipped_info: pd.DataFrame
    """Removed INNs with removal reason."""

    # ── Sequence info ─────────────────────────────────────────────────────────
    full_sequence_list: pd.DataFrame
    """Per-chain sequence and annotation data."""

    # ── Constant chain mutation summaries ─────────────────────────────────────
    mut_info_per_constant: pd.DataFrame
    """Mutation count matrix per INN × constant chain region (wide format)."""
    supp_tabl_constant_mut_summary: pd.DataFrame
    """Summary table of constant chain mutations per INN with species info."""
    constant_chain_mutation_per_region_count: pd.DataFrame
    """Unique INN counts per (gene_species, region_classification)."""

    # ── Single-mutation events ────────────────────────────────────────────────
    insertion_mutations_for_table: pd.DataFrame
    """Insertion-only mutations indexed by EU position."""
    unique_single_mutation_events: pd.DataFrame
    """Unique (gene_species×gene, EU, residue) mutation events with INN counts."""
    pos_mutation_freq: pd.DataFrame
    """Per-position mutation frequency grouped by region group (heavy/light)."""
    unique_single_mutations_per_inn: pd.DataFrame
    """Per-INN mutation repetition counts."""

    # ── Motif-level ───────────────────────────────────────────────────────────
    constant_chain_info_per_inn: pd.DataFrame
    """Per-INN motif stats (from build_motif_stats_per_inn)."""
    constant_chains_mutations_per_molecule: pd.DataFrame
    """Per-motif stats (from build_motif_stats_per_motif)."""

    # ── Co-mutation ───────────────────────────────────────────────────────────
    co_df_export: pd.DataFrame
    """Export-ready co-mutation pair table."""

    # ── Motif index ───────────────────────────────────────────────────────────
    motif_df_clustered_post_filter: pd.DataFrame
    """Final clustered motif DataFrame after all filtering and recomputation."""

    # ── Raw intermediate objects (not exported to Excel) ─────────────────────
    marginals: pd.DataFrame = field(repr=False)
    motif_result: Any = field(default=None, repr=False)
    """``ChainMotifIndexResult`` from the chain motif pipeline (or ``None`` if not yet run)."""
    co_df_raw: pd.DataFrame = field(default=None, repr=False)
    inn_info: pd.DataFrame = field(default=None, repr=False)
    """INN info resolved with date/motif annotations (from ``add_inn_date_info``)."""
    all_unique_mutation_strings: pd.DataFrame = field(
        default=None, repr=False
    )
    """Enumeration DataFrame from ``build_chain_motif_index`` — equivalent to
    ``ChainMotifIndexResult.enumeration_df``.  Pass as
    ``cached_chain_motif_result`` (via ``result.motif_result``) to skip
    rebuilding the motif index on subsequent runs."""
    cross_chain_pairing_motifs: pd.DataFrame = field(default=None, repr=False)
    """Long-format view of ``bispecific_ind_branch`` rows from
    ``motif_result.cross_chain_df``: one row per (inn, mutation_string_pe,
    chain_id) — pivoted from the wide ``chain`` column and stacked.
    Columns: ``inn``, ``mutation_string_pe``, ``chain_id``."""
    paired_chain_motifs: pd.DataFrame = field(default=None, repr=False)
    """Grouped paired-chain motif export table indexed by ``inn``."""

    def to_excel(self, path: str | Path) -> None:
        """Export all tables to a multi-sheet Excel workbook."""
        export_constant_chain_analysis(self, path)

    def __repr__(self) -> str:  # noqa: D105
        _tables = [
            ("mol_info", self.mol_info),
            ("skipped_info", self.skipped_info),
            ("full_sequence_list", self.full_sequence_list),
            ("mut_info_per_constant", self.mut_info_per_constant),
            (
                "supp_tabl_constant_mut_summary",
                self.supp_tabl_constant_mut_summary,
            ),
            (
                "constant_chain_mutation_per_region_count",
                self.constant_chain_mutation_per_region_count,
            ),
            (
                "insertion_mutations_for_table",
                self.insertion_mutations_for_table,
            ),
            (
                "unique_single_mutation_events",
                self.unique_single_mutation_events,
            ),
            ("pos_mutation_freq", self.pos_mutation_freq),
            (
                "unique_single_mutations_per_inn",
                self.unique_single_mutations_per_inn,
            ),
            (
                "constant_chain_info_per_inn",
                self.constant_chain_info_per_inn,
            ),
            (
                "constant_chains_mutations_per_molecule",
                self.constant_chains_mutations_per_molecule,
            ),
            ("co_df_export", self.co_df_export),
            (
                "motif_df_clustered_post_filter",
                self.motif_df_clustered_post_filter,
            ),
        ]
        col_w = max(len(name) for name, _ in _tables)
        table_lines = []
        for name, df in _tables:
            if isinstance(df, pd.DataFrame):
                table_lines.append(
                    f"    {name:<{col_w}}  {df.shape[0]:>5} rows × {df.shape[1]:>3} cols"
                )
            else:
                table_lines.append(f"    {name:<{col_w}}  (not built)")

        # Summary stats derived from the tables
        n_inns = (
            len(self.mol_info)
            if isinstance(self.mol_info, pd.DataFrame)
            else "?"
        )
        n_skip = (
            len(self.skipped_info)
            if isinstance(self.skipped_info, pd.DataFrame)
            else "?"
        )
        n_pairs = (
            len(self.co_df_export)
            if isinstance(self.co_df_export, pd.DataFrame)
            else "?"
        )
        n_motifs = (
            len(self.motif_df_clustered_post_filter)
            if isinstance(
                self.motif_df_clustered_post_filter, pd.DataFrame
            )
            else "?"
        )

        return (
            "ConstantChainAnalysisResult\n"
            "  Summary\n"
            f"    INNs analysed                     : {n_inns}\n"
            f"    INNs skipped                      : {n_skip}\n"
            f"    Co-mutation pairs                 : {n_pairs}\n"
            f"    Motifs (post-filter)              : {n_motifs}\n"
            "  Tables\n" + "\n".join(table_lines)
        )


# ---------------------------------------------------------------------------
# Serialisation helper
# ---------------------------------------------------------------------------


def _serialise_df(df: pd.DataFrame) -> pd.DataFrame:
    """Convert set/frozenset columns to comma-joined strings for Excel."""
    df = df.copy()
    for col in df.columns:
        try:
            sample = (
                df[col].dropna().iloc[0]
                if not df[col].dropna().empty
                else None
            )
        except Exception:
            sample = None
        if isinstance(sample, (set, frozenset)):
            df[col] = df[col].apply(
                lambda x: (
                    ", ".join(sorted(str(v) for v in x))
                    if isinstance(x, (set, frozenset))
                    else x
                )
            )
        elif isinstance(sample, dict):
            df[col] = df[col].apply(
                lambda x: str(x) if isinstance(x, dict) else x
            )
        elif isinstance(sample, list):
            df[col] = df[col].apply(
                lambda x: (
                    ", ".join(str(v) for v in x)
                    if isinstance(x, list)
                    else x
                )
            )
    return df


def _safe_reset(df: "pd.DataFrame") -> "pd.DataFrame":
    """Reset index levels that are not already column names, drop the rest."""
    if isinstance(df.index, pd.RangeIndex):
        return df
    idx_to_reset = [
        n for n in df.index.names if n is not None and n not in df.columns
    ]
    if not idx_to_reset:
        return df.reset_index(drop=True)
    return df.reset_index(level=idx_to_reset).reset_index(drop=True)


def _drop_index_col(df: pd.DataFrame) -> pd.DataFrame:
    """Drop any column literally named 'index' — artifact of repeated reset_index calls."""
    if "index" in df.columns:
        return df.drop(columns=["index"])
    return df


def _build_cross_chain_pairing_motifs(cross_chain_df: pd.DataFrame) -> pd.DataFrame:
    """Long-format view of ``bispecific_ind_branch`` rows: one row per (inn, mutation_string_pe, chain_id).

    Pivots ``chain`` (wide, one column per ``mutation_string_pe``) then
    stacks it back down to long form, dropping empty cells.
    """
    if not isinstance(cross_chain_df, pd.DataFrame) or cross_chain_df.empty:
        return pd.DataFrame(columns=["inn", "mutation_string_pe", "chain_id"])

    # Already long-format output from this helper: pass through.
    if {"inn", "mutation_string_pe", "chain_id"}.issubset(
        cross_chain_df.columns
    ):
        return cross_chain_df[
            ["inn", "mutation_string_pe", "chain_id"]
        ].copy()

    # Legacy long-format shape that still uses "chain" as the value column.
    if {"inn", "mutation_string_pe", "chain"}.issubset(cross_chain_df.columns):
        return cross_chain_df[
            ["inn", "mutation_string_pe", "chain"]
        ].rename(columns={"chain": "chain_id"})

    ind_branch = cross_chain_df.query('type == "bispecific_ind_branch"')
    if ind_branch.empty:
        return pd.DataFrame(columns=["inn", "mutation_string_pe", "chain_id"])

    return (
        ind_branch.pivot(index="inn", values="chain", columns="mutation_string_pe")
        .stack()
        .dropna()
        .reset_index()
        .rename(columns={0: "chain_id"})
    )


def _build_paired_chain_motifs_export(
    cross_chain_pairing_motifs: pd.DataFrame,
    supp_tabl_constant_mut_summary: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """Build paired-chain motif export table grouped per INN.

    Output contains:
    - ``inn_name``: INN name looked up from ``supp_tabl_constant_mut_summary`` (when available).
    - ``inn``: INN identifier from cross-chain pairing output.
    - ``chain_id``: comma-joined sorted chain IDs participating in pairing.
    - ``Chain Motif N`` columns: one column per motif position in sorted motif list.
    """
    if (
        not isinstance(cross_chain_pairing_motifs, pd.DataFrame)
        or cross_chain_pairing_motifs.empty
    ):
        return pd.DataFrame(columns=["chain_id"])

    grouped = (
        cross_chain_pairing_motifs.groupby(by="inn", dropna=False)
        .agg(
            {
                "mutation_string_pe": lambda x: sorted(set(list(x))),
                "chain_id": lambda x: sorted(set(list(x))),
            }
        )
        .reset_index()
    )
    grouped = grouped[
        grouped["mutation_string_pe"].apply(
            lambda x: isinstance(x, list) and len(x) > 1
        )
    ]
    if grouped.empty:
        return pd.DataFrame(columns=["chain_id"])
    grouped = grouped.reset_index(drop=True)

    motif_cols = grouped["mutation_string_pe"].apply(pd.Series).rename(
        columns=lambda col: f"Chain Motif {int(col) + 1}"
    )
    paired_chain_motifs = pd.concat(
        [grouped[["inn", "chain_id"]].reset_index(drop=True), motif_cols],
        axis=1,
    )
    paired_chain_motifs["chain_id"] = paired_chain_motifs["chain_id"].apply(
        lambda x: ",".join(x) if isinstance(x, list) else x
    )

    if (
        isinstance(supp_tabl_constant_mut_summary, pd.DataFrame)
        and not supp_tabl_constant_mut_summary.empty
    ):
        supp = supp_tabl_constant_mut_summary.copy()
        if not isinstance(supp.index, pd.RangeIndex):
            supp = supp.reset_index()
        name_col = (
            "inn_name"
            if "inn_name" in supp.columns
            else ("inn" if "inn" in supp.columns else None)
        )
        if name_col and "inn_id" in supp.columns:

            def _norm_inn_id(v: Any) -> str | None:
                if pd.isna(v):
                    return None
                s = str(v).strip()
                if not s:
                    return None
                s_upper = s.upper()
                if s_upper.startswith("INN"):
                    s = s[3:].strip()
                try:
                    f = float(s)
                    if f.is_integer():
                        s = str(int(f))
                except (TypeError, ValueError):
                    pass
                return s.upper()

            ref = supp[["inn_id", name_col]].copy()
            ref["_inn_key"] = ref["inn_id"].apply(_norm_inn_id)
            ref = ref.dropna(subset=["_inn_key"]).drop_duplicates(
                subset=["_inn_key"], keep="first"
            )
            inn_name_by_key = ref.set_index("_inn_key")[name_col].to_dict()
            paired_chain_motifs["inn_name"] = (
                paired_chain_motifs["inn"]
                .astype(str)
                .apply(_norm_inn_id)
                .map(inn_name_by_key)
                .values
            )
    if "inn_name" not in paired_chain_motifs.columns:
        paired_chain_motifs["inn_name"] = None

    motif_cols = [
        c for c in paired_chain_motifs.columns if c.startswith("Chain Motif ")
    ]
    paired_chain_motifs = paired_chain_motifs[
        ["inn_name", "inn", "chain_id"] + motif_cols
    ]
    paired_chain_motifs = paired_chain_motifs.sort_values(
        by=["inn_name", "inn"], na_position="last"
    ).reset_index(drop=True)
    return paired_chain_motifs


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------


def motif_index_default_transform(df: pd.DataFrame) -> pd.DataFrame:
    """Default post-processing applied to the MotifIndex sheet during export.

    Mirrors the notebook steps:
    - Drop rows with NaN in ``chain_type_h_or_l`` or ``num_mutations``.
    - Serialise ``inns`` / ``adjusted_inns`` as comma-joined, ``sub_motifs`` as semicolon-joined.
    - Sort by ``inn_count`` desc → ``num_mutations`` asc → ``sub_motifs`` asc.
    - Keep only rows where ``num_mutations > 1``.
    """
    df = df.copy()
    _drop_cols = [
        c
        for c in ["chain_type_h_or_l", "num_mutations"]
        if c in df.columns
    ]
    if _drop_cols:
        df = df.dropna(subset=_drop_cols)

    for col, delim in [
        ("inns", ","),
        ("adjusted_inns", ","),
        ("sub_motifs", ";"),
    ]:
        if col in df.columns:
            df[col] = df[col].apply(
                lambda x: (
                    delim.join(str(v) for v in x)
                    if isinstance(x, (list, set, frozenset))
                    else x
                )
            )

    sort_cols = [
        c
        for c in ["inn_count", "num_mutations", "sub_motifs"]
        if c in df.columns
    ]
    asc = [False, True, True][: len(sort_cols)]
    if sort_cols:
        df = df.sort_values(by=sort_cols, ascending=asc)

    if "num_mutations" in df.columns:
        df = df[df["num_mutations"] > 1]

    return df


#: Default per-sheet post-processors applied during export.
#: Override or extend via the ``sheet_transforms`` parameter of
#: :func:`export_constant_chain_analysis`.
DEFAULT_SHEET_TRANSFORMS: Dict[str, Any] = {
    "MotifIndex": motif_index_default_transform,
}

#: Ordered mapping of result attribute → Excel sheet name
_SHEET_MAP: List[tuple[str, str]] = [
    ("mol_info", "MoleculeInfo"),
    ("skipped_info", "SkippedINNs"),
    ("full_sequence_list", "FullSequenceList"),
    ("supp_tabl_constant_mut_summary", "ConstantMutSummary"),
    ("constant_chain_mutation_per_region_count", "MutCountByRegion"),
    ("insertion_mutations_for_table", "InsertionMutations"),
    ("unique_single_mutation_events", "UniqueSingleMutations"),
    ("pos_mutation_freq", "PositionalMutFreq"),
    ("unique_single_mutations_per_inn", "MutRepetitionsPerINN"),
    ("constant_chain_info_per_inn", "ConstantChainInfoPerINN"),
    ("constant_chains_mutations_per_molecule", "MutationsPerMotif"),
    ("co_df_export", "CoMutationPairs"),
    ("motif_df_clustered_post_filter", "MotifIndex"),
    ("cross_chain_pairing_motifs", "CrossChainPairingMotifs"),
    ("paired_chain_motifs", "PairedChainMotifs"),
]

#: Default legend text for supplemental export worksheets.
DEFAULT_SUPPLEMENTAL_TABLE_LEGENDS: Dict[str, str] = {
    "MoleculeInfo": (
        "Molecule-level metadata for INNs retained in the analysis, including "
        "source provenance, chain/gene context, and high-level inclusion flags."
    ),
    "SkippedINNs": (
        "INNs excluded from constant-chain analyses, with explicit skip reasons "
        "to document filtering and quality-control decisions."
    ),
    "FullSequenceList": (
        "Complete constant-chain sequence records used downstream, preserving "
        "raw sequence context for traceability and manual audit."
    ),
    "ConstantMutSummary": (
        "Per-INN summary of constant-region mutational content, including motif "
        "signatures and region-specific mutation burden across Fc/constant domains."
    ),
    "MutCountByRegion": (
        "Aggregated mutation counts by region classification (e.g., CH1, HINGE, "
        "CH2/CH3/CL), stratified by configured species/gene grouping columns."
    ),
    "InsertionMutations": (
        "Catalog of insertion events observed in constant domains, with position "
        "and residue-level context for each insertion mutation token."
    ),
    "UniqueSingleMutations": (
        "Unique single-residue mutation events pooled across molecules, with "
        "occurrence counts and annotation fields for ranking recurrent variants."
    ),
    "PositionalMutFreq": (
        "Position-wise mutation frequency table under the configured numbering "
        "scheme, summarizing recurrent hotspots across genes and species."
    ),
    "MutRepetitionsPerINN": (
        "Per-INN recurrence summary showing how often individual mutation events "
        "repeat within each molecule's constant-chain mutation profile."
    ),
    "ConstantChainInfoPerINN": (
        "Domain-resolved constant-chain summary per INN, including chain identity, "
        "region assignment, and associated mutation lists."
    ),
    "MutationsPerMotif": (
        "Motif-centric aggregation linking each motif to constituent mutations and "
        "supporting INN-level occurrence statistics."
    ),
    "CoMutationPairs": (
        "Pairwise co-mutation table describing mutation pairs that co-occur across "
        "molecules, with frequency and association statistics."
    ),
    "MotifIndex": (
        "Post-filtered motif index generated by the motif pipeline, including motif "
        "counts, cluster assignments, significance metrics, and supporting INN sets."
    ),
    "CrossChainPairingMotifs": (
        "Long-format cross-chain pairing table (one motif token per row) used to "
        "reconstruct paired motif events across antibody chains within each INN."
    ),
    "PairedChainMotifs": (
        "Wide-format paired motif table (one paired event per row) summarizing "
        "chain-to-chain motif combinations observed within individual INNs."
    ),
}

#: Additional publication-style details for complex worksheets.
DEFAULT_SUPPLEMENTAL_TABLE_DETAILS: Dict[str, str] = {
    "MutCountByRegion": (
        "Useful for comparing mutation burden distribution across constant domains "
        "and for validating region-specific enrichment trends."
    ),
    "ConstantMutSummary": (
        "Includes motif signatures derived from final_mutations and region-level "
        "counts that can be used for supplementary burden summaries."
    ),
    "UniqueSingleMutations": (
        "Single-token event table suitable for ranking high-frequency substitutions, "
        "deletions, and insertions before motif-level clustering."
    ),
    "PositionalMutFreq": (
        "Supports hotspot analyses by providing per-position denominators/counts "
        "within the selected mutation frequency resolution."
    ),
    "MutationsPerMotif": (
        "Connects motif-level summaries back to constituent mutation tokens, "
        "supporting interpretation of what each motif biologically represents."
    ),
    "CoMutationPairs": (
        "Contains pairwise counts and enrichment-like statistics for assessing "
        "non-random co-occurrence relationships between mutation positions."
    ),
    "MotifIndex": (
        "Primary motif catalog after clustering/statistical filtering, including "
        "sub-motif relationships, significance columns, and cluster metadata."
    ),
    "CrossChainPairingMotifs": (
        "Each row captures one chain-local motif token for an INN; rows sharing "
        "the same INN can be grouped to reconstruct paired-chain events."
    ),
    "PairedChainMotifs": (
        "Condenses cross-chain long-format rows into explicit paired combinations "
        "(e.g., Chain Motif 1/2), enabling direct paired-event tabulation."
    ),
}


def export_constant_chain_analysis(
    result: ConstantChainAnalysisResult,
    path: str | Path,
    *,
    include_sheets: Optional[List[str]] = None,
    exclude_sheets: Optional[List[str]] = None,
    column_serializers: Optional[Dict] = None,
    sheet_transforms: Optional[Dict[str, Any]] = None,
    column_renames: Optional[Dict] = None,
    drop_columns: Optional[Any] = None,
    include_columns: Optional[Dict[str, List[str]]] = None,
    supp_table_numbering: bool = False,
    fail_on_missing_included_sheets: bool = False,
    include_table_legends: bool = True,
    table_legends: Optional[Dict[str, str]] = None,
    table_legend_details: Optional[Dict[str, str]] = None,
) -> None:
    """Write tables in ``result`` to a multi-sheet Excel workbook.

    Parameters
    ----------
    result:
        Analysis result object.
    path:
        Output path (created or overwritten).
    include_sheets:
        If provided, only export sheets whose names are in this list.
    exclude_sheets:
        Sheet names to skip. Ignored when ``include_sheets`` is set.
    column_serializers:
        Per-value serializers. Key is ``str`` (all sheets) or
        ``(sheet_name, column_name)`` (one sheet). Value is callable.
    sheet_transforms:
        Per-sheet DataFrame transforms ``(df) -> df``.  Merged with
        :data:`DEFAULT_SHEET_TRANSFORMS`; caller values win on conflict.
    column_renames:
        Column renames applied **after** transforms. Same key format as
        ``column_serializers``:

        - ``str`` key  → rename that column in **all** sheets.
        - ``(sheet_name, old_col)`` tuple → rename only in that sheet.

        Value is the new column name (str). Example::

            column_renames={"mutation_string_pe": "motif",
                            ("MotifIndex", "inn_count"): "n_inns"}

    drop_columns:
        Columns to drop before writing. Either:

        - A ``list[str]`` — dropped from **all** sheets where present.
        - A ``dict`` — keys are sheet names or the special key ``"all"``
          (which drops those columns from every sheet). Values are
          ``list[str]``.

        Example::

            drop_columns=["mutation_string"]                       # all sheets
            drop_columns={"all": ["mutation_string"], "MotifIndex": ["inns_raw"]}
            drop_columns={"MotifIndex": ["mutation_string"]}       # one sheet only
    include_columns:
        ``dict[sheet_name, list[str]]`` — for each named sheet, keep only
        the listed columns **in that order**.  Columns not in the list are
        discarded.  Applied after all other transforms, renames and drops.

        Example::

            include_columns={
                "MotifIndex": ["Motif", "gene", "gene_species",
                               "Total Unique INNs", "adjusted_count"],
                "CoMutationPairs": ["position_1", "position_2",
                                    "top_mutation_pair"],
            }
    fail_on_missing_included_sheets:
        If ``True`` and ``include_sheets`` is provided, raise ``ValueError``
        when any requested sheet is unknown, missing on ``result``, or empty
        after transforms.
    include_table_legends:
        If ``True`` (default), append a worksheet containing a legend with one
        row per exported table and a human-readable description.
    table_legends:
        Optional per-sheet legend overrides. Merged onto
        :data:`DEFAULT_SUPPLEMENTAL_TABLE_LEGENDS`.
    table_legend_details:
        Optional per-sheet detailed notes for complex worksheets. Merged onto
        :data:`DEFAULT_SUPPLEMENTAL_TABLE_DETAILS`.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    # Merge default + caller transforms; caller wins on conflict
    _transforms: Dict[str, Any] = {**DEFAULT_SHEET_TRANSFORMS}
    if sheet_transforms is not None:
        _transforms.update(sheet_transforms)

    # Resolve which sheets to write — include_sheets preserves caller order
    _attr_by_sheet = {s: a for a, s in _SHEET_MAP}
    _unknown_includes: list[str] = []
    if include_sheets is not None:
        _unknown_includes = [s for s in include_sheets if s not in _attr_by_sheet]
        sheet_iter = [
            (a, s)
            for s in include_sheets
            if (a := _attr_by_sheet.get(s)) is not None
        ]
    elif exclude_sheets is not None:
        exclude_set = set(exclude_sheets)
        sheet_iter = [
            (a, s) for a, s in _SHEET_MAP if s not in exclude_set
        ]
    else:
        sheet_iter = list(_SHEET_MAP)

    # Partition serializers into global (str key) and per-sheet ((sheet, col) key)
    _global_serializers: dict[str, Any] = {}
    _sheet_serializers: dict[tuple, Any] = {}
    if column_serializers:
        for key, fn in column_serializers.items():
            if isinstance(key, str):
                _global_serializers[key] = fn
            elif isinstance(key, tuple) and len(key) == 2:
                _sheet_serializers[key] = fn

    # Partition renames: str → all sheets, (sheet, col) → one sheet
    _global_renames: dict[str, str] = {}
    _sheet_renames: dict[tuple, str] = {}
    if column_renames:
        for key, new_name in column_renames.items():
            if isinstance(key, str):
                _global_renames[key] = new_name
            elif isinstance(key, tuple) and len(key) == 2:
                _sheet_renames[key] = new_name

    # Normalise drop_columns
    _global_drops: list[str] = []
    _sheet_drops: dict[str, list[str]] = {}
    if drop_columns is not None:
        if isinstance(drop_columns, list):
            _global_drops = list(drop_columns)
        elif isinstance(drop_columns, dict):
            # "all" key means drop from every sheet
            _global_drops = list(drop_columns.pop("all", []))
            _sheet_drops = {k: list(v) for k, v in drop_columns.items()}

    written = 0
    _written_sheet_names: list[str] = []
    _skipped_included_reasons: dict[str, str] = {}
    _table_legends = {**DEFAULT_SUPPLEMENTAL_TABLE_LEGENDS}
    if table_legends is not None:
        _table_legends.update(table_legends)
    _table_legend_details = {**DEFAULT_SUPPLEMENTAL_TABLE_DETAILS}
    if table_legend_details is not None:
        _table_legend_details.update(table_legend_details)
    _prepared_exports: list[tuple[str, str, pd.DataFrame]] = []
    for attr, sheet_name in sheet_iter:
        df: pd.DataFrame | None = getattr(result, attr, None)
        if attr == "paired_chain_motifs" and (
            not isinstance(df, pd.DataFrame)
            or df.empty
            or "inn_name" not in df.columns
        ):
            # Backward/stale-result compatibility:
            # derive paired export table from cross-chain long table when
            # paired_chain_motifs is absent/empty and ensure inn_name is present.
            df = _build_paired_chain_motifs_export(
                getattr(result, "cross_chain_pairing_motifs", None),
                getattr(result, "supp_tabl_constant_mut_summary", None),
            )
        if df is None or (isinstance(df, pd.DataFrame) and df.empty):
            if include_sheets is not None:
                _skipped_included_reasons[sheet_name] = (
                    "missing or empty before export transforms"
                )
            continue
        df = df.copy()
        # Reset non-RangeIndex levels that don't duplicate column names
        if not isinstance(df.index, pd.RangeIndex):
            idx_names = [
                n
                for n in df.index.names
                if n is not None and n not in df.columns
            ]
            if idx_names:
                df = df.reset_index(level=idx_names)
        # Apply per-sheet serializers BEFORE auto-serialise so custom lambdas
        # see the raw Python objects (dict/set/list) rather than str(x)
        for (sn, col), fn in _sheet_serializers.items():
            if sn == sheet_name and col in df.columns:
                df[col] = df[col].apply(fn)
        # Auto-serialise remaining set/list/dict columns and drop 'index' col
        df = _drop_index_col(_serialise_df(df))
        # Apply global column serializers (post auto-serialise is fine for globals)
        for col, fn in _global_serializers.items():
            if col in df.columns:
                df[col] = df[col].apply(fn)
        # Apply sheet-level transform (e.g. filter rows, sort)
        if sheet_name in _transforms:
            df = _transforms[sheet_name](df)
        if df is None or (isinstance(df, pd.DataFrame) and df.empty):
            if include_sheets is not None:
                _skipped_included_reasons[sheet_name] = (
                    "empty after export transforms"
                )
            continue
        # Restrict and reorder columns per-sheet (uses original names, before rename)
        if include_columns and sheet_name in include_columns:
            _keep = [
                c
                for c in include_columns[sheet_name]
                if c in df.columns
            ]
            df = df[_keep]
        # Rename columns (global then per-sheet; applied after include_columns)
        _rename_map = {
            k: v for k, v in _global_renames.items() if k in df.columns
        }
        _rename_map.update(
            {
                col: new
                for (sn, col), new in _sheet_renames.items()
                if sn == sheet_name and col in df.columns
            }
        )
        if _rename_map:
            df = df.rename(columns=_rename_map)
        # Drop columns (global then per-sheet)
        _to_drop = [c for c in _global_drops if c in df.columns]
        _to_drop += [
            c
            for c in _sheet_drops.get(sheet_name, [])
            if c in df.columns
        ]
        if _to_drop:
            df = df.drop(columns=_to_drop)
        written += 1
        _tab_name = (
            f"SuppT{written}-{sheet_name}"
            if supp_table_numbering
            else sheet_name
        )
        # Excel tab names max 31 chars — truncate if needed
        _prepared_exports.append((sheet_name, _tab_name[:31], df))
        _written_sheet_names.append(sheet_name)

    _legend_rows = [
        {
            "supplemental_table_title": f"Supplemental Table {idx}",
            "worksheet_name": tab_name,
            "table_name": sheet_name,
            "description": _table_legends.get(
                sheet_name,
                "No legend available.",
            ),
            "details": _table_legend_details.get(sheet_name, ""),
            "row_count": int(len(df)),
            "column_count": int(len(df.columns)),
        }
        for idx, (sheet_name, tab_name, df) in enumerate(
            _prepared_exports, start=1
        )
    ]

    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        if include_table_legends and _legend_rows:
            pd.DataFrame(_legend_rows).to_excel(
                writer,
                sheet_name="SupplementalTableLegends",
                index=False,
            )
        for _, tab_name, df in _prepared_exports:
            df.to_excel(writer, sheet_name=tab_name, index=False)

    if include_sheets is not None and fail_on_missing_included_sheets:
        _missing_requested = [
            s for s in include_sheets if s not in _written_sheet_names
        ]
        if _missing_requested:
            _detail = []
            if _unknown_includes:
                _detail.append(
                    "Unknown include_sheets: " + ", ".join(_unknown_includes)
                )
            for s in _missing_requested:
                if s in _skipped_included_reasons:
                    _detail.append(f"{s}: {_skipped_included_reasons[s]}")
            raise ValueError(
                "Requested include_sheets were not exported: "
                + ", ".join(_missing_requested)
                + (f". Details: {'; '.join(_detail)}" if _detail else "")
            )

    print(f"✓ Exported {written} sheet(s) to {path}")


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------


def _filter_mutations_from_motifs_chains(
    mut_row: pd.Series,
    del_pattern: dict[str, dict[str, list[str]]],
) -> pd.Series:
    """Remove noise mutations (e.g. HINGE insertions) from a motif row."""
    region = mut_row["region_classification"]
    pos_arr = mut_row["EU"]
    mutation_types = mut_row["mutation_type"]
    germline_seq = mut_row["germline_seq"]
    mutations = mut_row["mutated_seq"]

    results = {
        "EU": [],
        "mutation_type": [],
        "germline_seq": [],
        "mutated_seq": [],
        "constant_chain_mutational_motif": [],
    }
    for pos, mut_type, germ, mut in zip(
        pos_arr, mutation_types, germline_seq, mutations
    ):
        if mut_type in del_pattern and region in del_pattern[mut_type]:
            chk_pos = del_pattern[mut_type][region]
            found = any(
                (
                    (".i" in str(ckp) and str(pos).startswith(str(ckp)))
                    or str(pos) == str(ckp)
                )
                for ckp in chk_pos
            )
            if found:
                continue
        results["EU"].append(pos)
        results["mutation_type"].append(mut_type)
        results["germline_seq"].append(germ)
        results["mutated_seq"].append(mut)
        results["constant_chain_mutational_motif"].append(
            f"{germ}{pos}{mut}"
        )

    row_out = mut_row.copy()
    for k, v in results.items():
        row_out[k] = v
    row_out["num_mutations"] = len(results["EU"])
    return row_out


def _prepare_effector_df(
    effector_df: pd.DataFrame,
    join_cols: List[str],
    annotation_cols: Optional[List[str]],
) -> pd.DataFrame:
    """Collapse ``effector_df`` to one row per join key.

    For each duplicate join key, asserts that all annotation values are
    identical across duplicates.  If any annotation column has conflicting
    values for the same key (e.g. two different ``sub_classification`` entries
    for the same ``mutation_string_pe × gene × gene_species``), raises
    ``AssertionError`` listing the conflicts so the caller can fix the source data.

    Parameters
    ----------
    effector_df:
        Raw effector annotation DataFrame (may have duplicate join keys due to
        allele-level rows that collapsed from the same variant).
    join_cols:
        Columns that form the join key (e.g. ``["gene_species", "gene", "mutation_string_pe"]``).
    annotation_cols:
        Annotation columns to validate and keep.  ``None`` uses all non-join columns.

    Returns
    -------
    Deduplicated DataFrame with one row per join key.
    """
    eff = (
        effector_df.reset_index()
        if any(c in effector_df.index.names for c in join_cols)
        else effector_df.copy()
    )

    avail_join = [c for c in join_cols if c in eff.columns]
    if not avail_join:
        return eff

    # Uppercase only classifier string join columns so case differences don't
    # create spurious duplicates (e.g. "human-ighg1" vs "HUMAN-IGHG1").
    # mutation_string_pe and other mutation-identity columns are left as-is.
    import pandas.api.types as _pat

    _CLASSIFIER_COLS = frozenset(
        {"gene", "species", "allele", "gene_species", "isotype"}
    )
    for c in avail_join:
        if (
            c in _CLASSIFIER_COLS
            and c in eff.columns
            and _pat.is_string_dtype(eff[c])
        ):
            eff[c] = eff[c].str.upper()

    if annotation_cols is not None:
        keep_cols = avail_join + [
            c for c in annotation_cols if c in eff.columns
        ]
        eff = eff[keep_cols]

    ann_cols = [c for c in eff.columns if c not in avail_join]

    # Find keys with duplicates
    dup_mask = eff.duplicated(subset=avail_join, keep=False)
    if not dup_mask.any():
        return eff  # already unique — nothing to do

    dups = eff[dup_mask]
    conflicts = []
    for key_vals, grp in dups.groupby(avail_join, dropna=False):
        for col in ann_cols:
            unique_vals = grp[col].dropna().unique()
            if len(unique_vals) > 1:
                key_str = " × ".join(
                    str(v)
                    for v in (
                        key_vals
                        if isinstance(key_vals, tuple)
                        else (key_vals,)
                    )
                )
                conflicts.append((key_str, key_vals, grp))

    if conflicts:
        raise AssertionError(
            f"effector_df has {len(conflicts)} conflicting annotation(s) for "
            f"the same join key — cannot safely collapse duplicates:\n"
            + "\n".join(conflicts[:20])
            + ("\n  ..." if len(conflicts) > 20 else "")
        )

    # Safe to deduplicate — all duplicates have identical annotation values
    return eff.drop_duplicates(
        subset=avail_join, keep="first"
    ).reset_index(drop=True)


def classify_motif_annotation_by_sub_motifs(
    motif_df: pd.DataFrame,
    effector_df: pd.DataFrame,
    main_classifer_col: str,
    group_classifier_col: str = "group",
) -> pd.DataFrame:
    def _find_any_mutation(
        mutation_set: set[str],
        gene: str,
        map_with_gene: dict[tuple[str, str], dict[str, str]],
        map_just_mut: dict[str, dict[str, str]],
    ) -> [str, str, list[str]]:

        mutation_to_effect = []
        not_found = []

        for mut in mutation_set:
            if (mut, gene) in map_with_gene:
                mutation_to_effect.extend(
                    map_with_gene[(mut, gene)][main_classifer_col].split(
                        ","
                    )
                )

            elif mut in map_just_mut:
                mutation_to_effect.extend(
                    map_just_mut[mut][main_classifer_col].split(",")
                )

            else:
                not_found.append(mut)
                mutation_to_effect.append("Unknown")

        unique_combo = sorted(set(mutation_to_effect))

        return (
            ",".join(unique_combo),
            "Multiple" if len(unique_combo) > 1 else unique_combo[0],
            not_found,
        )

    map_with_gene = (
        effector_df.groupby(by=["mutation_string_pe", "gene"])[
            [main_classifer_col]
        ]
        .first()
        .to_dict("index")
    )
    map_just_mut = (
        effector_df.groupby(by=["mutation_string_pe"])[
            [main_classifer_col]
        ]
        .first()
        .to_dict("index")
    )
    sub = motif_df.copy()
    all_data = []

    for idx, tp in sub.iterrows():

        mapped_mutations = []
        not_annotated = []
        all_sub_classifications = []
        tokens = set(tp["mutation_string_pe"].split("/"))
        if tp["novel_motif_mutations"]:
            sub_classification, group, not_found_novel = (
                _find_any_mutation(
                    tp["novel_motif_mutations"],
                    tp["gene"],
                    map_with_gene,
                    map_just_mut,
                )
            )
            all_sub_classifications.append(sub_classification)
            del sub_classification
            del group
        else:
            not_found_novel = []
        mapped_novel = tokens - set(not_found_novel)

        for sub_motif in tp["sub_motifs"]:
            token_sub = set(sub_motif.split("/"))
            sub_motif_classification, _, not_found_group = (
                _find_any_mutation(
                    {sub_motif}, tp["gene"], map_with_gene, map_just_mut
                )
            )
            not_annotated.extend(not_found_group)
            mapped_mutations.extend(token_sub - set(not_found_group))
            all_sub_classifications.extend(
                sub_motif_classification.split(",")
            )

        unmapped_from_sub_motifs = tokens - set(mapped_mutations).union(
            mapped_novel
        )
        total_unmapped = set(not_found_novel).union(
            unmapped_from_sub_motifs
        )
        final_sub_class = sorted(set(all_sub_classifications))
        final_sub_group = (
            "Multiple" if len(final_sub_class) > 1 else final_sub_class[0]
        )
        if len(total_unmapped) == 0:
            map_just_mut[tp["mutation_string_pe"]] = {
                main_classifer_col: ",".join(final_sub_class)
            }
        all_data.append(
            {
                "mutation_string_pe": tp["mutation_string_pe"],
                "gene": tp["gene"],
                "unmapped": total_unmapped,
                main_classifer_col: ",".join(final_sub_class),
                "orig_unmapped": not_found_novel,
                "sub_motif_unmapped": not_annotated,
                group_classifier_col: final_sub_group,
                "novel_mut": tp["novel_motif_mutations"],
                "sub_motifs": tp["sub_motifs"],
            }
        )
    return pd.DataFrame(all_data, index=sub.index)


def _left_join_effector(
    df: pd.DataFrame,
    effector_df: pd.DataFrame,
    join_cols: List[str],
    annotation_cols: Optional[List[str]] = None,
) -> pd.DataFrame:
    """Left-join ``effector_df`` onto ``df`` on ``join_cols``, never dropping rows.

    Before joining, ``effector_df`` is collapsed to one row per join key via
    :func:`_prepare_effector_df` which asserts that duplicate keys have
    identical annotation values (raises if they conflict).

    ``df`` may have ``join_cols`` as index levels or regular columns — both are
    handled.  ``annotation_cols`` restricts which effector columns are kept;
    ``None`` keeps all.
    """

    df = df.copy()
    had_index = any(c in df.index.names for c in join_cols)
    if had_index:
        df = df.reset_index()

    avail_join = [c for c in join_cols if c in df.columns]

    if not avail_join:
        raise ValueError(
            f"Cannot join motif with effector df using {join_cols}"
            "Columns not found in motif"
        )

    # Only uppercase classifier columns (gene, species, allele, etc.) — NOT
    # mutation-identity columns like mutation_string_pe whose internal notation
    # (e.g. ".i1" insertion-slot labels) must not be uppercased.
    import pandas.api.types as _pat

    _CLASSIFIER_COLS = frozenset(
        {"gene", "species", "allele", "gene_species", "isotype"}
    )
    _str_join = [
        c
        for c in avail_join
        if _pat.is_string_dtype(df[c]) and c in _CLASSIFIER_COLS
    ]
    for c in _str_join:
        df[c] = df[c].str.upper()

    # Collapse effector to one row per join key — raises on conflicting annotations
    eff = _prepare_effector_df(effector_df, avail_join, annotation_cols)
    avail_join = [c for c in avail_join if c in eff.columns]

    if not avail_join:
        raise ValueError(
            f"Cannot join motif with effector df using {join_cols}."
            "Columns not found in effector"
        )

    for c in _str_join:
        if c in eff.columns and _pat.is_string_dtype(eff[c]):
            eff[c] = eff[c].str.upper()

    # Drop effector columns that duplicate existing df columns
    eff_new_cols = [
        c
        for c in eff.columns
        if c not in avail_join and c not in df.columns
    ]
    eff = eff[avail_join + eff_new_cols]

    merged = df.merge(eff, on=avail_join, how="left", indicator=True)
    assert len(merged) == len(
        df
    ), f"Left join changed row count after dedup: {len(df)} → {len(merged)}"

    # update any missing mutation strings due to gene/ mismatches
    update_missing_eff_mask = (merged["_merge"] == "left_only") & (
        merged["mutation_string_pe"].isin(eff["mutation_string_pe"])
    )
    mutation_strings_to_grab = set(
        merged.loc[update_missing_eff_mask, "mutation_string_pe"]
    )

    if mutation_strings_to_grab:
        eff_non_null = (
            eff.dropna(subset=annotation_cols)
            .groupby("mutation_string_pe")
            .first()
        )
        missing_non_null = sorted(
            mutation_strings_to_grab - set(eff_non_null.index.tolist())
        )
        if missing_non_null:
            raise AssertionError(
                "Cannot backfill left-only rows by mutation_string_pe: "
                "mutation exists in effector join domain but has no non-null values "
                f"for annotation_cols={annotation_cols}. "
                f"missing_count={len(missing_non_null)} "
                f"missing_sample={missing_non_null[:10]}"
            )

        fill_missing_mutation_strings_map = (
            eff_non_null.reindex(list(mutation_strings_to_grab))[annotation_cols]
            .to_dict("index")
        )

        mapped_backfill = merged.loc[
            update_missing_eff_mask, "mutation_string_pe"
        ].map(fill_missing_mutation_strings_map)
        backfill_df = mapped_backfill.apply(pd.Series)
        missing_backfill_cols = [
            c for c in annotation_cols if c not in backfill_df.columns
        ]
        if missing_backfill_cols:
            _left_only = merged.loc[
                update_missing_eff_mask, ["mutation_string_pe", "gene"]
            ]
            _left_only_pairs = (
                _left_only.drop_duplicates().head(10).to_dict("records")
            )
            raise AssertionError(
                "Backfill expansion did not produce required annotation columns. "
                f"required={annotation_cols} "
                f"produced={list(backfill_df.columns)} "
                f"missing={missing_backfill_cols} "
                f"left_only_count={int(update_missing_eff_mask.sum())} "
                f"left_only_pairs_sample={_left_only_pairs}"
            )

        merged.loc[update_missing_eff_mask, annotation_cols] = (
            backfill_df[annotation_cols].values
        )  # strip index so pandas aligns by position, not label

    if (
        merged.loc[update_missing_eff_mask, annotation_cols]
        .isna()
        .any()
        .any()
    ):
        raise AssertionError(
            "Missing annotation columns after attempting to fill from effector"
            f": {merged.loc[update_missing_eff_mask, annotation_cols].isna().any()}"
        )

    # ── Insertion fallback join ───────────────────────────────────────────────
    # Insertions use mutation_string_pe = "{germline}{eu}^{seq}" but effector
    # annotations may still carry the old "{germline}{eu}+" format.  For any
    # unmatched insertion row, retry with "^<seq>" → "+" substitution.
    _ann_cols = eff_new_cols
    _pe_col = "mutation_string_pe"
    _mt_col = "mutation_type"
    if _pe_col in avail_join and _mt_col in merged.columns and _ann_cols:
        _first_ann = _ann_cols[0]
        _unmatched_ins = (merged[_mt_col] == "insertion") & merged[
            _first_ann
        ].isna()
        if _unmatched_ins.any():
            import re as _re

            # Build fallback pe: strip "^<seq>" → "+"
            _fb_col = "_pe_fallback"
            merged.loc[_unmatched_ins, _fb_col] = merged.loc[
                _unmatched_ins, _pe_col
            ].apply(lambda s: _re.sub(r"\^[A-Za-z+]+$", "+", str(s)))
            # Build fallback join using _fb_col in place of _pe_col
            _fb_join = [_fb_col if c == _pe_col else c for c in avail_join]
            _eff_fb = eff.rename(columns={_pe_col: _fb_col})
            _fb_cols_needed = [c for c in _fb_join if c in _eff_fb.columns]
            _fb_result = merged.loc[
                _unmatched_ins,
                _fb_join
                + [
                    c
                    for c in merged.columns
                    if c not in _fb_join and c not in _ann_cols
                ],
            ].merge(
                _eff_fb[
                    [
                        c
                        for c in _eff_fb.columns
                        if c in _fb_join or c in _ann_cols
                    ]
                ],
                on=_fb_cols_needed,
                how="left",
            )
            for col in _ann_cols:
                if col in _fb_result.columns:
                    merged.loc[_unmatched_ins, col] = _fb_result[
                        col
                    ].values
            merged = merged.drop(columns=[_fb_col], errors="ignore")

    return merged


class _MotifFrameStep(BaseModel):
    """Schema for motif-table step IO."""

    motif_df: pd.DataFrame
    model_config = ConfigDict(arbitrary_types_allowed=True)


class _ExpandBispecificStepIn(_MotifFrameStep):
    """.. deprecated:: Input schema for the now-unused bispecific expansion step.

    Use ``result.motif_result.cross_chain_df`` instead.
    """

    constant_chains: pd.DataFrame
    group_by: Optional[List[str]] = None
    min_count: int = 1
    mutations_to_ignore: Optional[Dict[str, Dict[str, List[str]]]] = None


class _StepWithStats(_MotifFrameStep):
    """Base output schema that records row-level deltas."""

    rows_added: int = 0
    rows_dropped: int = 0


class _OtherGeneHitStepIn(_MotifFrameStep):
    """Input schema for recovering missing same-signature motifs."""

    original_motif_df: pd.DataFrame
    min_count: int = 1


class _OtherGeneHitStepOut(_StepWithStats):
    """Output schema for same-signature recovery."""

    recovered_rows: int = 0


def _step_row_key(df: pd.DataFrame) -> pd.Series:
    """Stable per-row key used for step-level add/remove tracking."""
    _key_cols = [
        c
        for c in ("mutation_string", "gene", "gene_species")
        if c in df.columns
    ]
    if not _key_cols:
        return pd.Series([str(i) for i in range(len(df))], index=df.index)
    return df[_key_cols].astype(str).agg("|".join, axis=1)


def _compute_step_delta(
    before_df: pd.DataFrame,
    after_df: pd.DataFrame,
    *,
    add_reason: str,
    remove_reason: str,
) -> Dict[str, pd.DataFrame]:
    """Return {'added','removed'} dataframes with a reason column."""
    before = before_df.copy()
    after = after_df.copy()
    before["_row_key"] = _step_row_key(before)
    after["_row_key"] = _step_row_key(after)
    common_cols = [
        c
        for c in (
            "mutation_string",
            "gene",
            "gene_species",
            "inn_count",
            "adjusted_count",
            "inns",
        )
        if c in set(before.columns) | set(after.columns)
    ]

    added = after[~after["_row_key"].isin(set(before["_row_key"]))][
        common_cols
    ].copy()
    removed = before[~before["_row_key"].isin(set(after["_row_key"]))][
        common_cols
    ].copy()
    added["reason"] = add_reason
    removed["reason"] = remove_reason
    return {
        "added": added.reset_index(drop=True),
        "removed": removed.reset_index(drop=True),
    }


def _step_expand_bispecific_rows(
    step_in: _ExpandBispecificStepIn,
    expand_bispecific_branches_fn,
) -> _StepWithStats:
    """.. deprecated:: Bispecific expansion via post-hoc ``expand_bispecific_branches``.

    This step is no longer called by the pipeline. Bispecific cross-chain
    pairing is now handled natively by ``expand_motifs_by_cross_chain_pairing``
    (step 2/15 of ``build_chain_motif_index``) before any filtering occurs.
    This function is preserved only for reference and will be removed in a
    future release.
    """
    import warnings

    warnings.warn(
        "_step_expand_bispecific_rows is deprecated and unused. "
        "Bispecific detection now runs inside build_chain_motif_index "
        "via expand_motifs_by_cross_chain_pairing.",
        DeprecationWarning,
        stacklevel=2,
    )
    before = len(step_in.motif_df)
    expanded_df = expand_bispecific_branches_fn(
        step_in.motif_df,
        constant_chains=step_in.constant_chains,
        group_by=step_in.group_by,
        min_count=step_in.min_count,
        mutations_to_ignore=step_in.mutations_to_ignore,
    ).reset_index(drop=True)
    return _StepWithStats(
        motif_df=expanded_df,
        rows_added=max(0, len(expanded_df) - before),
        rows_dropped=max(0, before - len(expanded_df)),
    )


def _step_assign_ids_and_raw_submotifs(
    step_in: _MotifFrameStep,
    add_raw_sub_motifs_fn,
) -> _MotifFrameStep:
    """Assign motif_id and rebuild raw_sub_motifs."""
    out = step_in.motif_df.reset_index(drop=True).copy()
    out["motif_id"] = out.index
    out = add_raw_sub_motifs_fn(out)
    return _MotifFrameStep(motif_df=out)


def _step_subtract_inns(
    step_in: _MotifFrameStep,
    subtract_inns_inline_fn,
) -> _MotifFrameStep:
    """Re-run INN subtraction in-place using raw_sub_motifs ancestry."""
    out = subtract_inns_inline_fn(
        step_in.motif_df, ancestry_col="raw_sub_motifs"
    )
    return _MotifFrameStep(motif_df=out)


def _step_filter_adjusted_or_anchor(
    step_in: _MotifFrameStep,
) -> _StepWithStats:
    """Keep rows with adjusted_count>0 or bispecific_anchor=True."""
    _df = step_in.motif_df.copy()
    _anchor_mask = (
        _df.get("bispecific_anchor", pd.Series(False, index=_df.index))
        .fillna(False)
        .astype(bool)
    )
    _keep_mask = (_df["adjusted_count"] > 0) | _anchor_mask
    kept = _df[_keep_mask].copy().reset_index(drop=True)
    if "motif_retention_reason" not in kept.columns:
        kept["motif_retention_reason"] = None
    if "recovered_other_gene_hit" not in kept.columns:
        kept["recovered_other_gene_hit"] = False
    return _StepWithStats(
        motif_df=kept,
        rows_added=0,
        rows_dropped=int((~_keep_mask).sum()),
    )


def _step_restore_other_gene_hits(
    step_in: _OtherGeneHitStepIn,
) -> _OtherGeneHitStepOut:
    """Re-insert motifs from original table sharing same pos->mut signature.

    This step matches motifs by signature that ignores germline residue identity
    (e.g., compares by ``pos->mut``), and restores missing rows from the original
    motif table with a retention reason ``other_gene_hit``.
    """
    current_df = step_in.motif_df.copy().reset_index(drop=True)
    original_df = step_in.original_motif_df.copy().reset_index(drop=True)
    if (
        current_df.empty
        or original_df.empty
        or "mutation_string" not in current_df.columns
    ):
        return _OtherGeneHitStepOut(
            motif_df=current_df,
            recovered_rows=0,
            rows_added=0,
            rows_dropped=0,
        )

    for _df in (current_df, original_df):
        _df["_sig_posmut"] = _df["mutation_string"].map(
            motif_signature_without_germline
        )
        _key_cols = [
            c
            for c in ("mutation_string", "gene", "gene_species")
            if c in _df.columns
        ]
        _df["_row_key"] = _df[_key_cols].astype(str).agg("|".join, axis=1)

    current_sigs = set(current_df["_sig_posmut"])
    current_keys = set(current_df["_row_key"])
    candidate = original_df[
        original_df["_sig_posmut"].isin(current_sigs)
    ].copy()
    missing = candidate[~candidate["_row_key"].isin(current_keys)].copy()
    if "inn_count" in missing.columns:
        missing = missing[
            missing["inn_count"] >= int(step_in.min_count)
        ].copy()

    if missing.empty:
        out_df = current_df
        out_df.drop(
            columns=["_sig_posmut", "_row_key"],
            inplace=True,
            errors="ignore",
        )
        return _OtherGeneHitStepOut(
            motif_df=out_df.reset_index(drop=True),
            recovered_rows=0,
            rows_added=0,
            rows_dropped=0,
        )

    missing["motif_retention_reason"] = "other_gene_hit"
    missing["recovered_other_gene_hit"] = True
    if "motif_retention_reason" not in current_df.columns:
        current_df["motif_retention_reason"] = None
    if "recovered_other_gene_hit" not in current_df.columns:
        current_df["recovered_other_gene_hit"] = False

    union_cols = list(
        dict.fromkeys(list(current_df.columns) + list(missing.columns))
    )
    current_df = current_df.reindex(columns=union_cols)
    missing = missing.reindex(columns=union_cols)
    out_df = pd.concat([current_df, missing], ignore_index=True)
    out_df = out_df.drop_duplicates(
        subset=["_row_key"], keep="first"
    ).reset_index(drop=True)
    out_df.drop(
        columns=["_sig_posmut", "_row_key"], inplace=True, errors="ignore"
    )
    return _OtherGeneHitStepOut(
        motif_df=out_df,
        recovered_rows=len(missing),
        rows_added=len(missing),
        rows_dropped=0,
    )


def build_constant_chain_analysis(
    ds: "INNAnalysisDataset",
    cfg: ConstantChainAnalysisConfig | None = None,
    *,
    cached_chain_motif_result: ChainMotifIndexResult | None = None,
    cached_motif_result: Any = None,
    cached_all_unique_mutation_strings: pd.DataFrame | None = None,
) -> ConstantChainAnalysisResult:
    """Run the full constant-chain analysis pipeline on ``ds``.

    Parameters
    ----------
    ds:
        Loaded :class:`INNAnalysisDataset` from :func:`load_inn_analysis`.
    cfg:
        Optional configuration.  Defaults to :class:`ConstantChainAnalysisConfig`.
    cached_chain_motif_result:
        Pre-computed :class:`~analytics.motifs.models.ChainMotifIndexResult`
        from a previous run.  When provided the ``build_chain_motif_index``
        step is skipped entirely (saves several minutes).  Pass
        ``result.motif_result`` to reuse from a prior call::

            result = build_constant_chain_analysis(ds, cfg,
                cached_chain_motif_result=prev_result.motif_result)

    cached_motif_result:
        *Deprecated* — pass ``cached_chain_motif_result`` instead.
    cached_all_unique_mutation_strings:
        *Deprecated* — pass ``cached_chain_motif_result`` instead.

    Returns
    -------
    :class:`ConstantChainAnalysisResult` with all tables populated.
    """
    if cfg is None:
        cfg = ConstantChainAnalysisConfig()

    # Deprecation shim: if old-style cache params are supplied, warn and ignore.
    if (
        cached_motif_result is not None
        or cached_all_unique_mutation_strings is not None
    ):
        import warnings

        warnings.warn(
            "cached_motif_result and cached_all_unique_mutation_strings are "
            "deprecated.  Store result.motif_result (a ChainMotifIndexResult) "
            "and pass it as cached_chain_motif_result= instead.",
            DeprecationWarning,
            stacklevel=2,
        )

    def _clean(df):
        """Apply _safe_reset + drop spurious 'index' column."""
        if not isinstance(df, pd.DataFrame):
            return df
        return _drop_index_col(_safe_reset(df))

    def _motif_col_order(df: pd.DataFrame) -> pd.DataFrame:
        """Promote key motif columns to the front."""
        if not isinstance(df, pd.DataFrame):
            return df
        priority = [
            "mutation_string_pe",
            "gene",
            "gene_species",
            "inn_count",
            "adjusted_count",
        ]
        front = [c for c in priority if c in df.columns]
        rest = [c for c in df.columns if c not in front]
        return df[front + rest]

    mfr = cfg.mutation_freq_resolution

    _has_cached_chain_motifs = cached_chain_motif_result is not None

    _steps = [
        "Molecule info",
        "Full sequence list",
        "Constant chain mutation summary",
        "Single mutation event tables",
        "Motif stats per INN / per motif",
        "Co-mutation frequencies",
        (
            "(cached) Build chain motif index"
            if _has_cached_chain_motifs
            else "Build chain motif index"
        ),
        "Cluster motifs",
    ]
    pbar = tqdm(_steps, desc="ConstantChainAnalysis", unit="step")

    def _step(label: str) -> None:
        pbar.set_description(f"ConstantChainAnalysis › {label}")
        pbar.update(1)

    # Effector join key: stratification cols + mutation_string_pe
    _effector_join_cols = list(dict.fromkeys(mfr + ["mutation_string_pe"]))

    # ── 1. Molecule info ──────────────────────────────────────────────────────
    _step("Molecule info")
    _kept_cols = [
        c
        for c in cfg.mol_info_cols
        if c in ds.kept_inn.reset_index().columns
    ]
    _skip_cols = _kept_cols + [
        c
        for c in ["removal_reason", "vh", "vl"]
        if c in ds.removed_inn.reset_index().columns
    ]
    mol_info = ds.kept_inn.reset_index()[_kept_cols]
    skipped_info = ds.removed_inn.reset_index()[_skip_cols]

    # ── 2. Full sequence list ─────────────────────────────────────────────────
    _step("Full sequence list")
    try:
        _seq_df = (
            ds.full_length_chain_collection.to_dataframe().reset_index()
        )
        _seq_cols = [c for c in cfg.sequence_cols if c in _seq_df.columns]
        _sort_cols = [
            c
            for c in ["inn", "chain_id", "chain_type"]
            if c in _seq_df.columns
        ]
        full_sequence_list = (
            _seq_df[_seq_cols]
            .sort_values(_sort_cols)
            .reset_index(drop=True)
        )
    except Exception:
        full_sequence_list = pd.DataFrame()

    # ── 3. Constant chain mutation summary ───────────────────────────────────
    _step("Constant chain mutation summary")
    mut_info_per_constant = ds.constant_chains.swaplevel(axis=1)[
        "mutations"
    ].fillna("")
    mut_group = mut_info_per_constant.groupby(
        level=["inn", "inn_id"], dropna=False
    )
    mut_summary_parts = []
    for col in mut_info_per_constant.columns:
        mut_summary_parts.append(
            mut_group[col].apply(
                lambda x: len(
                    sorted(
                        set(
                            [
                                s
                                for y in x
                                for s in y.split(",")
                                if s.strip()
                            ]
                        )
                    )
                )
            )
        )
    constant_mut_count = pd.concat(mut_summary_parts, axis=1)
    constant_mut_count["has_constant_chain_mutations"] = (
        constant_mut_count.index.get_level_values("inn").isin(
            ds.inn_sets.cc_with_mutation
        )
    )

    sp_info_parts = []
    for sp_col in ["variable_chain_species", "constant_chain_species"]:
        if sp_col in ds.constant_chains.reset_index().columns:
            sp_info_parts.append(
                ds.constant_chains.reset_index()
                .groupby(["inn", "inn_id"], dropna=False)[sp_col]
                .apply(
                    lambda x: ",".join(
                        sorted(
                            set(
                                y
                                for z in x
                                if isinstance(z, str)
                                for y in z.split(",")
                                if y.strip()
                            )
                        )
                    )
                )
            )
    supp_tabl_constant_mut_summary = pd.concat(
        [constant_mut_count]
        + ([pd.concat(sp_info_parts, axis=1)] if sp_info_parts else []),
        axis=1,
    )

    count_statistics = ["gene", "gene_species", "region_classification"]
    constant_chain_mutation_per_region_count = (
        ds.individual_mutation_profile.reset_index()
        .groupby(count_statistics + ["inn"], dropna=False)
        .first()
        .reset_index()
        .groupby(count_statistics, dropna=False)
        .apply(len)
        .unstack(level=[2])
        .sort_index(axis=1)
        .sort_index(level=["gene_species", "gene"])
    )

    # ── 4. Single mutation event tables ──────────────────────────────────────
    _step("Single mutation event tables")
    flat_profile = ds.individual_mutation_profile.reset_index()

    unique_single_mutation_events = build_mutation_residues(
        flat_profile, mfr, cfg.mutation_types
    )
    unique_single_mutations_per_inn = mutation_repetitions_per_inn(
        flat_profile, mfr, cfg.mutation_types
    )
    insertion_mutations_for_table = (
        build_mutation_residues(flat_profile, mfr, ["insertion"])
        .reset_index(drop=True)
        .set_index("EU")
    )

    # ── 4b. Effector annotation on single-mutation events ──────────────────
    if cfg.motif_classification.effector_df is not None:
        unique_single_mutation_events = _left_join_effector(
            unique_single_mutation_events.reset_index(),
            cfg.motif_classification.effector_df,
            _effector_join_cols,
            cfg.motif_classification.annotation_cols,
        )

    # ── 4c. Positional mutation frequency from annotated single mutations ────
    from inn_analysis_ops.analytics._helpers import _split_eu_pos_col

    _region_to_group = {
        member: group
        for group, members in cfg.region_classification_groups.items()
        for member in members
    }
    _ann_cols_pos = cfg.motif_classification.annotation_cols or []

    _ume = unique_single_mutation_events.reset_index()
    # "region" in _ume is the raw annotation; map it through region_classification
    # (from flat_profile which has both) before building position_numbering_group
    _region_col = (
        "region" if "region" in _ume.columns else "region_classification"
    )
    _rc_lookup = (
        flat_profile.groupby(_region_col)["region_classification"]
        .first()
        .to_dict()
        if "region_classification" in flat_profile.columns
        else {}
    )
    _ume["region_classification"] = (
        _ume[_region_col].map(_rc_lookup)
        if _rc_lookup
        else _ume[_region_col]
    )
    _ume["position_numbering_group"] = _ume["region_classification"].map(
        _region_to_group
    )
    _ume, (pos_col, ins_pos_col) = _split_eu_pos_col(_ume.copy(), "EU")

    _grp_by = list(
        dict.fromkeys(
            mfr
            + [
                pos_col,
                ins_pos_col,
                "region_classification",
                "position_numbering_group",
            ]
        )
    )

    _agg_spec: dict = {
        "IMGT": "first",
        "KABAT": "first",
        "EXON": "first",
        # inn values are already lists from build_mutation_residues — flatten then deduplicate
        "inn": lambda x: sorted(
            set(
                [z for y in x for z in (y if isinstance(y, list) else [y])]
            )
        ),
    }
    for _ac in _ann_cols_pos:
        if _ac == "group":
            _agg_spec[_ac] = lambda x: (
                "Multiple"
                if len(
                    {
                        t.strip()
                        for v in x
                        if pd.notna(v)
                        for t in str(v).split(",")
                        if t.strip()
                    }
                )
                > 1
                else next(
                    iter(
                        {
                            t.strip()
                            for v in x
                            if pd.notna(v)
                            for t in str(v).split(",")
                            if t.strip()
                        }
                    ),
                    None,
                )
            )
        elif _ac == "sub_classification":
            _agg_spec[_ac] = lambda x: ",".join(
                sorted(
                    set(
                        [
                            z
                            for y in x
                            if not pd.isnull(y)
                            for z in y.split(",")
                        ]
                    )
                )
            )
        else:
            _agg_spec[_ac] = lambda x: ",".join(
                sorted(
                    set(
                        [
                            z
                            for y in x
                            if not pd.isnull(y)
                            for z in y.split(",")
                        ]
                    )
                )
            )

    pos_mutation_freq = (
        _ume.groupby(_grp_by, dropna=False)
        .agg({k: v for k, v in _agg_spec.items() if k in _ume.columns})
        .reset_index()
    )
    pos_mutation_freq["inn_count"] = pos_mutation_freq["inn"].apply(len)

    for _c in ["position_numbering_group"] + mfr:
        pos_mutation_freq[_c] = pos_mutation_freq[_c].str.upper()

    pos_mutation_freq["eu_position_label"] = pos_mutation_freq.apply(
        lambda row: (
            f"{int(row[pos_col])}.{row[ins_pos_col]}"
            if pd.notna(row[ins_pos_col]) and str(row[ins_pos_col]).strip()
            else str(int(row[pos_col]))
        ),
        axis=1,
    )
    _pmf_cols = ["eu_position_label"] + [
        c for c in pos_mutation_freq.columns if c != "eu_position_label"
    ]
    pos_mutation_freq = pos_mutation_freq[_pmf_cols].sort_values(
        "inn_count", ascending=False
    )

    # ── 5. Motif stats per INN / per motif ───────────────────────────────────
    _step("Motif stats per INN / per motif")
    constant_chain_info_per_inn = build_motif_stats_per_inn(
        ds.constant_chain_motifs_by_inn_region,
        extra_groupby_keys=["region_classification"],
    )
    constant_chains_mutations_per_molecule = build_motif_stats_per_motif(
        ds.constant_chain_motifs_by_inn_region,
        extra_groupby_keys=["region_classification"],
    )

    # ── 6. Co-mutation frequencies ───────────────────────────────────────────
    _step("Co-mutation frequencies")
    co_df_raw, marginals = compute_co_mutation_frequencies(
        ds.individual_mutation_profile,
        position_col=cfg.co_mutation.position_col,
        min_count=cfg.co_mutation.min_count,
        stratify_n_by=mfr,
        assert_against_marginals=unique_single_mutation_events,
    )
    co_df_filtered = remove_intra_pairs(co_df_raw, stratify_by=mfr)
    co_df_export = co_df_to_export_table(
        co_df_filtered,
        keep_classification_pairs=cfg.co_mutation.keep_classifications,
    )

    # ── 7. Build chain motif index (skip if cached) ───────────────────────────
    # Grouping is done by sub_domain (V vs light CK/CL), not by whole-chain
    # chain_type — a fusion chain (e.g. C-KAPPA/Fc fusion, chain_type "l")
    # must still be pairwise-compared against true heavy (V) chains for its
    # Fc/CH2-CH3 mutations, not only against other "l"-labeled chains. See
    # analytics.motifs.allele_mapping.find_multi_chain_inns_by_subdomain.
    _multi_chain_inns, _same_chain_type_groups = (
        _find_multi_chain_inns_by_subdomain(ds)
    )
    if _has_cached_chain_motifs:
        _step("(cached) Build chain motif index")
    else:
        _step("Build chain motif index")

    chain_motif_result = _build_chain_motif_index(
        inn_dataset=ds,
        multi_chain_inns=_multi_chain_inns,
        same_chain_type_groups=_same_chain_type_groups,
        group_by=mfr,
        min_chain_count=cfg.motif_index.min_count,
        min_motif_size=cfg.motif_index.min_size,
        max_motif_size=cfg.motif_index.max_size,
        max_consecutive_deletions=cfg.motif_index.max_consecutive_deletions,
        mutations_to_ignore=cfg.motif_index.mutations_to_ignore_for_motif_ops,
        min_lift=cfg.motif_index.min_lift,
        max_pvalue=cfg.motif_index.max_pvalue,
        cohesion_frac_threshold=cfg.motif_index.cohesion_frac_threshold,
        marginals_table_user=marginals,
        inn_count_invariant_mode=cfg.motif_index.inn_count_invariant_mode,
        allowed_motifs_by_select=cfg.motif_index.allowed_motifs_by_select,
        max_cluster_filter_iteration=(
            cfg.motif_index.max_cohesion_iterations
            if cfg.motif_index.max_cohesion_iterations is not None
            else 0
        ),
        hard_stat_filters=cfg.motif_index.hard_stat_filters,
        additional_user_set_filter_terms=cfg.motif_index.additional_user_set_filter_terms,
        cache_object=cached_chain_motif_result,
        forced_manual_motif_strings=cfg.motif_index.manual_motif_definitions,
        debug_stop_at_step=cfg.motif_index.debug_step,
        cdb=cfg.motif_index.cdb,
        cn=cfg.motif_index.cn,
    )

    motif_result = chain_motif_result
    # Expose enumeration_df under the legacy name for caching / auditing.
    all_unique_mutation_strings = chain_motif_result.enumeration_df

    # ── 8. Date annotation and pass-through ───────────────────────────────────
    _step("Date annotation")
    motif_df = chain_motif_result.motif_df.copy()
    # Alias motif_length → num_mutations for export compatibility.
    if (
        "num_mutations" not in motif_df.columns
        and "motif_length" in motif_df.columns
    ):
        motif_df["num_mutations"] = motif_df["motif_length"]

    s1 = motif_df.shape[0]
    motif_df_clustered_post_filter, inn_info = _add_dates(
        motif_df, ds, date_col=cfg.inn_date_source
    )
    s2 = motif_df_clustered_post_filter.shape[0]
    assert s1 == s2, motif_df_clustered_post_filter["inn"]
    motif_df_clustered_post_filter = (
        motif_df_clustered_post_filter.reset_index(drop=True)
    )
    motif_df_clustered_post_filter["motif_id"] = (
        motif_df_clustered_post_filter.index
    )

    # ── 8b. Propagate effector annotation onto motif_df ──────────────────────
    _eff_cfg = cfg.motif_classification
    if (
        _eff_cfg.effector_df is not None
        and not _eff_cfg.effector_df.empty
        and _eff_cfg.annotation_cols
    ):
        effector_result = step_add_effector_annotation(
            ChainMotifStep(
                motif_df=motif_df_clustered_post_filter,
                step_name="add_effector_annotation",
                group_cols=["mutation_string_pe", "gene"],
            ),
            effector_df=_eff_cfg.effector_df,
            join_cols=["mutation_string_pe", "gene"],
            annotation_cols=_eff_cfg.annotation_cols,
        )
        motif_df_clustered_post_filter = effector_result.motif_df
    else:
        _log.info(
            "add_effector_annotation skipped "
            "(no effector_df or annotation_cols configured)"
        )

    cross_chain_pairing_motifs = _build_cross_chain_pairing_motifs(
        motif_result.cross_chain_df
        if motif_result is not None
        else pd.DataFrame()
    )
    supp_tabl_constant_mut_summary_clean = _clean(
        supp_tabl_constant_mut_summary
    )

    pbar.set_description("ConstantChainAnalysis › complete")
    pbar.close()
    return ConstantChainAnalysisResult(
        mol_info=_clean(mol_info),
        skipped_info=_clean(skipped_info),
        full_sequence_list=_clean(full_sequence_list),
        mut_info_per_constant=_clean(constant_mut_count),
        supp_tabl_constant_mut_summary=supp_tabl_constant_mut_summary_clean,
        constant_chain_mutation_per_region_count=_clean(
            constant_chain_mutation_per_region_count
        ),
        insertion_mutations_for_table=_clean(
            insertion_mutations_for_table
        ),
        unique_single_mutation_events=_drop_index_col(
            unique_single_mutation_events
            if isinstance(unique_single_mutation_events, pd.DataFrame)
            else unique_single_mutation_events.reset_index()
        ),
        pos_mutation_freq=_clean(pos_mutation_freq),
        unique_single_mutations_per_inn=_clean(
            unique_single_mutations_per_inn
        ),
        constant_chain_info_per_inn=_clean(constant_chain_info_per_inn),
        constant_chains_mutations_per_molecule=_clean(
            constant_chains_mutations_per_molecule
        ),
        co_df_export=(
            _drop_index_col(co_df_export)
            if isinstance(co_df_export, pd.DataFrame)
            else co_df_export
        ),
        motif_df_clustered_post_filter=(
            _motif_col_order(
                _drop_index_col(motif_df_clustered_post_filter)
            )
            if isinstance(motif_df_clustered_post_filter, pd.DataFrame)
            else motif_df_clustered_post_filter
        ),
        marginals=marginals,
        motif_result=motif_result,
        co_df_raw=co_df_raw,
        inn_info=inn_info,
        all_unique_mutation_strings=(
            _motif_col_order(all_unique_mutation_strings)
            if isinstance(all_unique_mutation_strings, pd.DataFrame)
            else all_unique_mutation_strings
        ),
        cross_chain_pairing_motifs=cross_chain_pairing_motifs,
        paired_chain_motifs=_build_paired_chain_motifs_export(
            cross_chain_pairing_motifs, supp_tabl_constant_mut_summary_clean
        ),
    )


def _add_dates(
    motif_df: pd.DataFrame,
    ds: "INNAnalysisDataset",
    date_col: str = "min_year",
):
    """Add date info if available;"""
    return add_inn_date_info(motif_df, ds.kept_inn, date_col=date_col)
