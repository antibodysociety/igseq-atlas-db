"""Pydantic model and Pandera schema for the ``build_motif_index`` output DataFrame.

``MotifStatsCols`` is a namespace of column-name constants for the statistics block
produced by ``step_add_statistics``.  Use ``MotifStatsCols.lift_vs_indep`` etc. instead
of bare string literals so that renames propagate automatically.

``MotifRecord`` documents every column produced by ``build_motif_index``.
``MOTIF_INDEX_SCHEMA`` is a ``pandera.DataFrameSchema`` that validates the DataFrame.
``MOTIF_STATS_SCHEMA`` validates just the statistics block in isolation.
"""

from __future__ import annotations

from typing import Any, Dict, FrozenSet, List, Optional, Set, Tuple

import pandera.pandas as pa
from pandera.pandas import Column, DataFrameSchema
from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Column-name constants — import and use these instead of string literals
# ---------------------------------------------------------------------------

class MotifStatsCols:
    """String constants for every column produced by ``step_add_statistics``.

    Group 1 — Independence baseline
    --------------------------------
    Null: product of individual marginal frequencies (mutations are independent).

    Group 2 — Möbius sub-motif baseline
    -------------------------------------
    Null: inclusion-exclusion over observed sub-motif frequencies.
    Equals Group 1 for 2-token motifs; diverges for k≥3 where intermediate
    sub-motifs are looked up from the data.

    Group 3 — Pairwise co-mutation geometry
    -----------------------------------------
    Not a significance test — measures how tightly the tokens travel together
    at the INN level (overlap / lift between token pairs).
    """

    # ── Shared ────────────────────────────────────────────────────────────────
    inn_total: str = "inn_total"
    freq_motif: str = "freq_motif"
    marginal_freq: str = "marginal_freq"
    marginal_count: str = "marginal_count"

    # ── Group 1: independence baseline ────────────────────────────────────────
    expected_freq_indep: str = "expected_freq_indep"
    lift_vs_indep: str = "lift_vs_indep"
    log_lift_vs_indep: str = "log_lift_vs_indep"
    pvalue_vs_indep: str = "pvalue_vs_indep"
    neg_log10_pvalue_vs_indep: str = "neg_log10_pvalue_vs_indep"
    lift_passes_vs_indep: str = "lift_passes_vs_indep"
    pvalue_passes_vs_indep: str = "pvalue_passes_vs_indep"
    passes_indep_filter: str = "passes_indep_filter"

    # ── Group 2: Möbius sub-motif baseline ────────────────────────────────────
    expected_freq_mobius: str = "expected_freq_mobius"
    lift_vs_submotifs: str = "lift_vs_submotifs"
    log_lift_vs_submotifs: str = "log_lift_vs_submotifs"
    pvalue_vs_submotifs: str = "pvalue_vs_submotifs"
    neg_log10_pvalue_vs_submotifs: str = "neg_log10_pvalue_vs_submotifs"
    lift_passes_vs_submotifs: str = "lift_passes_vs_submotifs"
    pvalue_passes_vs_submotifs: str = "pvalue_passes_vs_submotifs"
    passes_submotif_filter: str = "passes_submotif_filter"
    explained_by_submotifs: str = "explained_by_submotifs"

    # ── Group 3: pairwise co-mutation geometry ────────────────────────────────
    pairwise_cohesion: str = "pairwise_cohesion"
    pairwise_lift: str = "pairwise_lift"

    # ── Group 4: child lift ratio ─────────────────────────────────────────────
    # For motif m, "children" are all rows whose super_motifs list contains m
    # (i.e. smaller motifs that m is a direct ancestor of).
    # The ratio lift_vs_submotifs(m) / lift_vs_submotifs(child) tells you
    # whether m is more enriched than its sub-patterns after the Möbius correction.
    # Ratio > 1 means m adds genuine enrichment beyond what the child already captures.
    # NaN when m has no children (it is a leaf/seed — nothing smaller lists it as ancestor).
    n_children: str = "n_children"
    lift_ratio_vs_child_min: str = "lift_ratio_vs_child_min"  # lift(m) / max(lift(children)) — vs most enriched child
    lift_ratio_vs_child_max: str = "lift_ratio_vs_child_max"  # lift(m) / min(lift(children)) — vs least enriched child

    # ── Composite scores ──────────────────────────────────────────────────────
    motif_score: str = "motif_score"
    artifact_fraction: str = "artifact_fraction"
    motif_score_artifact_adjusted: str = "motif_score_artifact_adjusted"


C = MotifStatsCols  # short alias for use inside this module


# ---------------------------------------------------------------------------
# Pandera schema — validates just the statistics block
# ---------------------------------------------------------------------------

MOTIF_STATS_SCHEMA = DataFrameSchema(
    columns={
        C.inn_total:                        Column(int,   nullable=False, checks=pa.Check.ge(1)),
        C.freq_motif:                       Column(float, nullable=False, checks=pa.Check.in_range(0.0, 1.0)),
        # Group 1
        C.expected_freq_indep:              Column(float, nullable=False, checks=pa.Check.ge(0.0)),
        C.lift_vs_indep:                    Column(float, nullable=False, checks=pa.Check.ge(0.0)),
        C.log_lift_vs_indep:                Column(float, nullable=False),
        C.pvalue_vs_indep:                  Column(float, nullable=False, checks=pa.Check.in_range(0.0, 1.0)),
        C.neg_log10_pvalue_vs_indep:        Column(float, nullable=False, checks=pa.Check.ge(0.0)),
        C.lift_passes_vs_indep:             Column(bool,  nullable=False),
        C.pvalue_passes_vs_indep:           Column(bool,  nullable=False),
        C.passes_indep_filter:              Column(bool,  nullable=False),
        # Group 2
        C.expected_freq_mobius:             Column(float, nullable=False, checks=pa.Check.ge(0.0)),
        C.lift_vs_submotifs:                Column(float, nullable=False, checks=pa.Check.ge(0.0)),
        C.log_lift_vs_submotifs:            Column(float, nullable=False),
        C.pvalue_vs_submotifs:              Column(float, nullable=False, checks=pa.Check.in_range(0.0, 1.0)),
        C.neg_log10_pvalue_vs_submotifs:    Column(float, nullable=False, checks=pa.Check.ge(0.0)),
        C.lift_passes_vs_submotifs:         Column(bool,  nullable=False),
        C.pvalue_passes_vs_submotifs:       Column(bool,  nullable=False),
        C.passes_submotif_filter:           Column(bool,  nullable=False),
        C.explained_by_submotifs:           Column(bool,  nullable=False),
        # Group 3
        C.pairwise_cohesion:                Column(float, nullable=True),
        C.pairwise_lift:                    Column(float, nullable=True),
        # Group 4: child lift ratio
        C.n_children:                       Column(int,   nullable=False),
        C.lift_ratio_vs_child_min:          Column(float, nullable=True),
        C.lift_ratio_vs_child_max:          Column(float, nullable=True),
        # Scores
        C.motif_score:                      Column(float, nullable=False),
        C.artifact_fraction:                Column(float, nullable=True),
        C.motif_score_artifact_adjusted:    Column(float, nullable=False),
    },
    strict=False,
    coerce=False,
    name="MotifStats",
)

# ---------------------------------------------------------------------------
# Pydantic model — one row of the motif index
# ---------------------------------------------------------------------------

class MotifRecord(BaseModel):
    """A single scored co-mutation motif entry produced by ``build_motif_index``.

    Column meanings are documented on each field.  Complex Python types
    (frozenset, set, dict) are preserved as-is; pandera validates the scalar
    columns in ``MOTIF_INDEX_SCHEMA``.
    """

    # ── Identity ──────────────────────────────────────────────────────────────
    motif: FrozenSet[Tuple[str, str]] = Field(
        description=(
            "Canonical frozenset of (EU_position, change_string) token tuples. "
            "change_string is '+' for insertions, 'X->Y' for substitutions."
        )
    )
    mutation_list: List[Tuple[str, str]] = Field(
        description="Sorted list of (EU_position, change_string) tokens (human-readable ordering)."
    )
    num_mutations: int = Field(
        description="Number of distinct mutation tokens in this motif (== size)."
    )
    mutation_string: str = Field(
        description=(
            "Semicolon-delimited human-readable form, e.g. '234:A->G; 235:T->C'. "
            "Used as the stable join key throughout the pipeline."
        )
    )
    motif_id: int = Field(
        description=(
            "Globally unique sequential integer index assigned after the final sort "
            "(descending num_mutations then inn_count). Used as a join key in subtract_parent_inns "
            "and other downstream steps."
        )
    )

    # ── INN sets ──────────────────────────────────────────────────────────────
    inn_count: int = Field(
        description=(
            "Number of distinct INNs carrying *at least* this motif. "
            "Includes INNs that also carry additional mutations (superstring INNs). "
            "Denominator: inn_total (all INNs in the group)."
        )
    )
    inns: Set[str] = Field(
        description="Full set of INN identifiers contributing to inn_count."
    )
    inn_total: int = Field(
        description=(
            "Total unique INNs in this group (chain_type × gene_species × gene). "
            "Denominator for all frequency and lift statistics."
        )
    )

    # ── Independence baseline ─────────────────────────────────────────────────
    freq_motif: float = Field(
        description="Observed joint frequency of the full motif: inn_count / inn_total."
    )
    marginal_freq: Dict[Tuple[str, str], float] = Field(
        description=(
            "Marginal frequency of each individual token: "
            "count_carrying_token / inn_total. "
            "Computed independently of other tokens."
        )
    )
    marginal_count: Dict[Tuple[str, str], int] = Field(
        description="Raw INN count for each individual token (before dividing by inn_total)."
    )
    expected_freq_indep: float = Field(
        description=(
            "Expected joint frequency under full mutual independence: "
            "∏ marginal_freq[t] for all tokens t. "
            "Serves as the null hypothesis for the independence test."
        )
    )
    lift_vs_indep: float = Field(
        description=(
            "Lift relative to independence: freq_motif / expected_freq_indep. "
            "Interpretation: 1 = independent, >1 = positively co-occurring, <1 = mutually exclusive."
        )
    )
    log_lift_vs_indep: float = Field(description="Natural log of lift_vs_indep.")
    pvalue_vs_indep: float = Field(
        description=(
            "One-sided binomial p-value: P(X ≥ inn_count | inn_total, p=expected_freq_indep). "
            "Tests whether the observed co-occurrence significantly exceeds the independence null."
        )
    )
    neg_log10_pvalue_vs_indep: float = Field(
        description="-log10(pvalue_vs_indep). Higher = more significant."
    )
    lift_passes_vs_indep: bool = Field(description="True if lift_vs_indep ≥ min_lift threshold.")
    pvalue_passes_vs_indep: bool = Field(description="True if pvalue_vs_indep ≤ max_pvalue threshold.")
    passes_indep_filter: bool = Field(
        description="True if both lift_passes_vs_indep and pvalue_passes_vs_indep."
    )

    # ── Sub-motif conditional statistics (Möbius baseline) ───────────────────
    expected_freq_mobius: float = Field(
        description=(
            "Möbius null frequency: f_null(m) = exp(Σ_{S⊊m, S≠∅} (-1)^(|m|-|S|-1) log f(S)). "
            "Uses observed sub-motif frequencies to isolate genuine k-way synergy "
            "beyond all lower-order effects. Equals expected_freq_indep for 2-token motifs."
        )
    )
    lift_vs_submotifs: float = Field(
        description=(
            "Conditional lift: freq_motif / expected_freq_mobius. "
            "Measures independent signal beyond the best known sub-pattern baseline."
        )
    )
    log_lift_vs_submotifs: float = Field(description="Natural log of lift_vs_submotifs.")
    pvalue_vs_submotifs: float = Field(
        description="Binomial p-value using expected_freq_mobius as the null probability."
    )
    neg_log10_pvalue_vs_submotifs: float = Field(description="-log10(pvalue_vs_submotifs).")
    lift_passes_vs_submotifs: bool = Field(description="True if lift_vs_submotifs ≥ min_lift.")
    pvalue_passes_vs_submotifs: bool = Field(description="True if pvalue_vs_submotifs ≤ max_pvalue.")
    passes_submotif_filter: bool = Field(
        description=(
            "True if both sub-motif thresholds are met. "
            "This motif has statistically independent signal beyond its best sub-patterns."
        )
    )
    explained_by_submotifs: bool = Field(
        description=(
            "True if passes_indep_filter=True but passes_submotif_filter=False. "
            "The motif's frequency is fully consistent with what you'd expect from known sub-motifs; "
            "the extra tokens add no independent signal."
        )
    )
    most_sig_sub_motif: Optional[str] = Field(
        description=(
            "mutation_string of the passing sub-motif with the highest lift_vs_submotifs "
            "that explains this motif (if explained_by_submotifs=True)."
        )
    )

    # ── Sub-motif ancestry tracking ────────────────────────────────────────
    sub_motifs: List[str] = Field(
        description=(
            "List of mutation_strings for all observed strict sub-motifs (num_mutations≥2) of this motif. "
            "These sub-patterns contributed to expected_freq_mobius via inclusion-exclusion. "
            "Empty for size-1 motifs."
        )
    )
    is_seed: bool = Field(
        description=(
            "True when no strict sub-motifs are present for this motif "
            "(i.e., this motif is a seed/leaf in the sub-motif graph)."
        )
    )
    novel_mutations: str = Field(
        description=(
            "'/'-delimited string of tokens not covered by any observed size≥2 sub-motif. "
            "These tokens contribute only their marginal_freq term to the Möbius null."
        )
    )

    # ── Maximality ────────────────────────────────────────────────────────────
    longest_motif_in_group: bool = Field(
        description=(
            "No strict superset of this motif exists anywhere in the scored dataset. "
            "This is the largest motif containing all these tokens that was observed."
        )
    )
    no_larger_passing_motif_in_group: bool = Field(
        description=(
            "No strict superset exists that also passes passes_submotif_filter. "
            "The most useful maximality flag: largest motif with independent signal."
        )
    )
    is_fully_described_by_larger_motif: bool = Field(
        description=(
            "True if a larger passing motif shares an identical INN set. "
            "Computed by iterating motifs descending by num_mutations; the first motif to register "
            "a given frozenset(inns) is canonical — all smaller motifs with the same set "
            "are fully described by it."
        )
    )
    larger_motif_with_same_inns: Optional[str] = Field(
        description="mutation_string of the canonical larger motif that fully describes this one."
    )

    # ── Composite scores ──────────────────────────────────────────────────────
    motif_score: float = Field(
        description=(
            "Composite ranking score: log_lift_vs_submotifs × (−log10(pvalue_vs_submotifs)) × log1p(inn_count). "
            "Rewards simultaneously: large effect size, high statistical significance, and wide prevalence."
        )
    )
    artifact_fraction: Optional[float] = Field(
        description=(
            "Mean fraction of INN appearances for each token originating from multi-chain "
            "or intra-repeat INNs (sourced from marginals_table). "
            "Range [0, 1]; higher values indicate technical artefacts."
        )
    )
    motif_score_artifact_adjusted: float = Field(
        description=(
            "motif_score × (1 − artifact_fraction). "
            "Penalises scores for motifs enriched in artefact-prone INNs."
        )
    )
    pairwise_cohesion: Optional[float] = Field(
        description=(
            "Mean over all token pairs in the motif of max(P(B|A), P(A|B)). "
            "P(B|A) = |INN(A) ∩ INN(B)| / |INN(A)|. "
            "Measures how tightly the tokens travel together at the INN level."
        )
    )
    pairwise_lift: Optional[float] = Field(
        description="Mean pairwise lift across all (A, B) token pairs in the motif."
    )

    # ── Child lift ratio ──────────────────────────────────────────────────────
    n_children: int = Field(
        description=(
            "Number of rows in the stratum whose super_motifs list contains this motif. "
            "These are the direct sub-motifs (smaller motifs) that this motif is an ancestor of."
        )
    )
    lift_ratio_vs_child_min: Optional[float] = Field(
        description=(
            "lift_vs_submotifs(m) / max(lift_vs_submotifs(children)). "
            "Ratio vs the most enriched child — the smallest lift ratio. "
            "Ratio > 1 means m is more enriched than even its best sub-pattern. "
            "NaN when n_children == 0 (m is a leaf with no sub-motifs in the data)."
        )
    )
    lift_ratio_vs_child_max: Optional[float] = Field(
        description=(
            "lift_vs_submotifs(m) / min(lift_vs_submotifs(children)). "
            "Ratio vs the least enriched child — the largest lift ratio. "
            "NaN when n_children == 0."
        )
    )

    class Config:
        populate_by_name = True


# ---------------------------------------------------------------------------
# Pandera schema — validates the DataFrame produced by build_motif_index
# ---------------------------------------------------------------------------

MOTIF_INDEX_SCHEMA = DataFrameSchema(
    columns={
        "mutation_string":                              Column(str,   nullable=False),
        "num_mutations":                                Column(int,   nullable=False, checks=pa.Check.ge(1)),
        "motif_id":                                     Column(int,   nullable=False, checks=pa.Check.ge(0)),
        "inn_count":                                    Column(int,   nullable=False, checks=pa.Check.ge(1)),
        C.inn_total:                                    Column(int,   nullable=False, checks=pa.Check.ge(1)),
        C.freq_motif:                                   Column(float, nullable=False, checks=pa.Check.in_range(0.0, 1.0)),
        C.expected_freq_indep:                          Column(float, nullable=False, checks=pa.Check.ge(0.0)),
        C.lift_vs_indep:                                Column(float, nullable=False, checks=pa.Check.ge(0.0)),
        C.log_lift_vs_indep:                            Column(float, nullable=False),
        C.pvalue_vs_indep:                              Column(float, nullable=False, checks=pa.Check.in_range(0.0, 1.0)),
        C.neg_log10_pvalue_vs_indep:                    Column(float, nullable=False, checks=pa.Check.ge(0.0)),
        C.lift_passes_vs_indep:                         Column(bool,  nullable=False),
        C.pvalue_passes_vs_indep:                       Column(bool,  nullable=False),
        C.passes_indep_filter:                          Column(bool,  nullable=False),
        C.expected_freq_mobius:                         Column(float, nullable=False, checks=pa.Check.ge(0.0)),
        C.lift_vs_submotifs:                            Column(float, nullable=False, checks=pa.Check.ge(0.0)),
        C.log_lift_vs_submotifs:                        Column(float, nullable=False),
        C.pvalue_vs_submotifs:                          Column(float, nullable=False, checks=pa.Check.in_range(0.0, 1.0)),
        C.neg_log10_pvalue_vs_submotifs:                Column(float, nullable=False, checks=pa.Check.ge(0.0)),
        C.lift_passes_vs_submotifs:                     Column(bool,  nullable=False),
        C.pvalue_passes_vs_submotifs:                   Column(bool,  nullable=False),
        C.passes_submotif_filter:                       Column(bool,  nullable=False),
        C.explained_by_submotifs:                       Column(bool,  nullable=False),
        "most_sig_sub_motif":                           Column(object, nullable=True),
        "longest_motif_in_group":                       Column(bool,  nullable=False),
        "no_larger_passing_motif_in_group":             Column(bool,  nullable=False),
        "is_fully_described_by_larger_motif":           Column(bool,  nullable=False),
        "larger_motif_with_same_inns":                  Column(object, nullable=True),
        "sub_motifs":                                   Column(object, nullable=False),
        "is_seed":                                      Column(bool, nullable=False),
        "novel_mutations":                              Column(str,   nullable=False),
        C.motif_score:                                  Column(float, nullable=False),
        C.artifact_fraction:                            Column(float, nullable=True),
        C.motif_score_artifact_adjusted:                Column(float, nullable=False),
        C.pairwise_cohesion:                            Column(float, nullable=True),
        C.pairwise_lift:                                Column(float, nullable=True),
        C.n_children:                                   Column(int,   nullable=False),
        C.lift_ratio_vs_child_min:                      Column(float, nullable=True),
        C.lift_ratio_vs_child_max:                      Column(float, nullable=True),
        # object columns (frozenset / set / dict — not scalar-validated)
        "motif":                                        Column(object, nullable=False),
        "inns":                                         Column(object, nullable=False),
        "mutation_list":                                Column(object, nullable=False),
        "marginal_freq":                                Column(object, nullable=False),
        "marginal_count":                               Column(object, nullable=False),
    },
    strict=False,   # allow extra columns added by downstream steps
    coerce=False,
    name="MotifIndex",
)
