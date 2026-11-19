"""Per-row statistic functions for ``step_add_statistics``.

Each function takes a typed Pydantic input model and a ``StratumContext``
(stratum-level pre-computed lookups), performs its calculation, and returns a
``StatResult`` — a thin wrapper around a plain dict whose keys are
``MotifStatsCols`` constants.

The DataFrame is never touched inside these functions; ``clustering.py``
assembles the results into columns after collecting a list of ``StatResult``
objects.

Functions
---------
compute_marginals
    ``inn_total``, ``freq_motif``, ``marginal_freq``, ``marginal_count``.
compute_indep_baseline
    Independence null: ``expected_freq_indep``, ``lift_vs_indep``,
    ``pvalue_vs_indep``, log/neg-log transforms, and threshold booleans.
compute_mobius_baseline
    Möbius inclusion-exclusion null: ``expected_freq_mobius``,
    ``lift_vs_submotifs``, ``pvalue_vs_submotifs``, transforms, booleans,
    and ``explained_by_submotifs``.
compute_pairwise
    ``pairwise_cohesion``, ``pairwise_lift``.
compute_child_lift_ratio
    ``n_children``, ``lift_ratio_vs_child_min``, ``lift_ratio_vs_child_max``.
compute_composite_score
    ``motif_score``, ``artifact_fraction``, ``motif_score_artifact_adjusted``.
"""

from __future__ import annotations

import math
from itertools import combinations
from typing import Any, Optional

import numpy as np
from pydantic import BaseModel, ConfigDict
from scipy.stats import binomtest as _binom

from inn_analysis_ops.analytics.motif_schema import MotifStatsCols as C


# ---------------------------------------------------------------------------
# Sub-motif generation
# ---------------------------------------------------------------------------


def generate_submotifs(
    tokens: frozenset[str],
) -> list[tuple[int, frozenset[str]]]:
    """Return all proper non-empty subsets of *tokens* with their Möbius signs.

    For a motif of size k, yields every r-element subset (1 ≤ r < k) together
    with the inclusion-exclusion sign ``(-1)^{k-r-1}``.  The result is ordered
    by ascending size, which matches the iteration order inside
    :func:`compute_mobius_baseline`.

    Parameters
    ----------
    tokens:
        Frozenset of PE token strings for a single motif (e.g.
        ``frozenset({"L234A", "L235A", "T366W"})``).

    Returns
    -------
    list of ``(sign, subset)`` tuples where *sign* is +1 or -1 and *subset*
    is a frozenset of token strings.

    Examples
    --------
    >>> generate_submotifs(frozenset({"A", "B", "C"}))
    [(1, frozenset({'A'})), (1, frozenset({'B'})), (1, frozenset({'C'})),
     (-1, frozenset({'A', 'B'})), (-1, frozenset({'A', 'C'})),
     (-1, frozenset({'B', 'C'}))]
    """
    k = len(tokens)
    tok_list = sorted(tokens)  # stable order for deterministic output
    result: list[tuple[int, frozenset[str]]] = []
    for r in range(1, k):
        sign = (-1) ** (k - r - 1)
        for subset in combinations(tok_list, r):
            result.append((sign, frozenset(subset)))
    return result

class StratumContext(BaseModel):
    """Pre-computed stratum-level lookups passed into every row function."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    N_total: int
    token_to_inns: dict[str, set]          # token → all INNs in stratum carrying it
    token_marginal_freq: dict[str, float]  # token → len(inns) / N_total
    pe_to_freq: dict[str, float]           # mutation_string_pe → freq_motif (populated after marginals pass)
    pe_to_tokens: dict[str, frozenset]     # mutation_string_pe → frozenset of tokens


# ---------------------------------------------------------------------------
# Output wrapper
# ---------------------------------------------------------------------------

class StatResult(BaseModel):
    """Thin wrapper around a column-name → value dict.

    Keys must be ``MotifStatsCols`` string constants.  The dict is assembled
    into DataFrame columns by the caller; Pandera validates the full DataFrame.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    values: dict[str, Any]


# ---------------------------------------------------------------------------
# Input models — one per function, only the fields it actually reads
# ---------------------------------------------------------------------------

class MarginalInput(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    mutation_string_pe: str
    inn_count: int
    inns: Any  # set | frozenset


class IndepInput(BaseModel):
    mutation_string_pe: str
    inn_count: int
    freq_motif: float
    passes_indep_filter: bool  # from prior result, needed for explained_by check


class MobiusInput(BaseModel):
    mutation_string_pe: str
    inn_count: int
    freq_motif: float
    passes_indep_filter: bool


class PairwiseInput(BaseModel):
    mutation_string_pe: str


class ChildLiftInput(BaseModel):
    mutation_string_pe: str
    lift_vs_submotifs: float
    super_motifs: list[str]


class CompositeInput(BaseModel):
    mutation_string_pe: str
    log_lift_vs_submotifs: float
    neg_log10_pvalue_vs_submotifs: float
    inn_count: int


# ---------------------------------------------------------------------------
# Stat functions
# ---------------------------------------------------------------------------

def compute_marginals(
    row: MarginalInput,
    ctx: StratumContext,
    min_lift: float,
    max_pvalue: float,
) -> StatResult:
    """Compute freq_motif, marginal_freq, marginal_count, and independence null."""
    freq = row.inn_count / ctx.N_total
    tokens = [t.strip() for t in row.mutation_string_pe.split("/") if t.strip()]
    indiv_freq = {t: ctx.token_marginal_freq.get(t, 1e-10) for t in tokens}
    indiv_count = {t: len(ctx.token_to_inns.get(t, set())) for t in tokens}

    expected = max(
        float(np.prod(list(indiv_freq.values()))) if indiv_freq else 1e-10,
        1e-10,
    )
    lift = round(freq / expected, 4)
    pval = round(
        float(
            _binom(
                row.inn_count,
                ctx.N_total,
                min(expected, 1.0),
                alternative="greater",
            ).pvalue
        ),
        10,
    )
    log_lift = round(math.log(max(lift, 1e-10)), 4)
    neg_log10_pval = round(-math.log10(max(pval, 1e-300)), 4)
    lift_passes = lift >= min_lift
    pvalue_passes = pval <= max_pvalue

    return StatResult(values={
        C.freq_motif:               round(freq, 6),
        C.marginal_freq:            indiv_freq,
        C.marginal_count:           indiv_count,
        C.expected_freq_indep:      round(expected, 8),
        C.lift_vs_indep:            lift,
        C.pvalue_vs_indep:          pval,
        C.log_lift_vs_indep:        log_lift,
        C.neg_log10_pvalue_vs_indep: neg_log10_pval,
        C.lift_passes_vs_indep:     lift_passes,
        C.pvalue_passes_vs_indep:   pvalue_passes,
        C.passes_indep_filter:      lift_passes and pvalue_passes,
    })


def compute_mobius_baseline(
    row: MobiusInput,
    ctx: StratumContext,
    min_lift: float,
    max_pvalue: float,
) -> StatResult:
    """Compute Möbius inclusion-exclusion null and sub-motif statistics."""
    pe = row.mutation_string_pe
    m = ctx.pe_to_tokens[pe]
    k = len(m)

    if k <= 1:
        f_null = ctx.pe_to_freq.get(pe, 1e-10)
    else:
        log_null = 0.0
        for sign, sub_ts in generate_submotifs(m):
            if len(sub_ts) == 1:
                f = ctx.token_marginal_freq.get(next(iter(sub_ts)), 1e-10)
            else:
                sub_pe = next(
                    (p for p, ts in ctx.pe_to_tokens.items() if ts == sub_ts),
                    None,
                )
                f = ctx.pe_to_freq.get(sub_pe, 1e-10) if sub_pe else 1e-10
            log_null += sign * math.log(max(f, 1e-10))
        f_null = min(math.exp(log_null), 1.0)

    f_null = max(f_null, 1e-10)
    lift = round(row.freq_motif / f_null, 4)
    pval = round(
        float(
            _binom(
                row.inn_count,
                ctx.N_total,
                min(f_null, 1.0),
                alternative="greater",
            ).pvalue
        ),
        10,
    )
    log_lift = round(math.log(max(lift, 1e-10)), 4)
    neg_log10_pval = round(-math.log10(max(pval, 1e-300)), 4)
    lift_passes = lift >= min_lift
    pvalue_passes = pval <= max_pvalue
    passes = lift_passes and pvalue_passes

    return StatResult(values={
        C.expected_freq_mobius:             round(f_null, 8),
        C.lift_vs_submotifs:                lift,
        C.pvalue_vs_submotifs:              pval,
        C.log_lift_vs_submotifs:            log_lift,
        C.neg_log10_pvalue_vs_submotifs:    neg_log10_pval,
        C.lift_passes_vs_submotifs:         lift_passes,
        C.pvalue_passes_vs_submotifs:       pvalue_passes,
        C.passes_submotif_filter:           passes,
        C.explained_by_submotifs:           row.passes_indep_filter and not passes,
    })


def compute_pairwise(
    row: PairwiseInput,
    ctx: StratumContext,
) -> StatResult:
    """Compute pairwise_cohesion and pairwise_lift across token pairs."""
    tokens = [t.strip() for t in row.mutation_string_pe.split("/") if t.strip()]
    pairs = list(combinations(tokens, 2))
    if not pairs:
        return StatResult(values={
            C.pairwise_cohesion: float("nan"),
            C.pairwise_lift:     float("nan"),
        })

    confs, lifts = [], []
    for a, b in pairs:
        inns_a = ctx.token_to_inns.get(a, set())
        inns_b = ctx.token_to_inns.get(b, set())
        if not inns_a or not inns_b:
            continue
        n_ab = len(inns_a & inns_b)
        confs.append(max(n_ab / len(inns_a), n_ab / len(inns_b)))
        denom = (len(inns_a) / ctx.N_total) * (len(inns_b) / ctx.N_total)
        lifts.append((n_ab / ctx.N_total) / denom if denom > 0 else float("nan"))

    return StatResult(values={
        C.pairwise_cohesion: round(float(np.nanmean(confs)), 4) if confs else float("nan"),
        C.pairwise_lift:     round(float(np.nanmean(lifts)), 4) if lifts else float("nan"),
    })


def compute_child_lift_ratio(
    row: ChildLiftInput,
    children_map: dict[str, list[float]],
) -> StatResult:
    """Compute lift ratio of this motif vs its children (sub-motifs it is an ancestor of).

    ``children_map`` maps each ``mutation_string_pe`` to the list of
    ``lift_vs_submotifs`` values of all rows that list it in their
    ``super_motifs``.  Build this once per stratum before calling this function.
    """
    child_lifts = [
        v for v in children_map.get(row.mutation_string_pe, [])
        if not math.isnan(v) and v > 0
    ]
    n = len(child_lifts)
    if n == 0 or math.isnan(row.lift_vs_submotifs):
        return StatResult(values={
            C.n_children:              0,
            C.lift_ratio_vs_child_min: float("nan"),
            C.lift_ratio_vs_child_max: float("nan"),
        })

    m_lift = row.lift_vs_submotifs
    return StatResult(values={
        C.n_children:              n,
        C.lift_ratio_vs_child_min: round(m_lift / max(child_lifts), 4),
        C.lift_ratio_vs_child_max: round(m_lift / min(child_lifts), 4),
    })


def compute_composite_score(
    row: CompositeInput,
    inn_count: int,
    marginals_lookup: Optional[dict[str, float]] = None,
) -> StatResult:
    """Compute motif_score, artifact_fraction, and motif_score_artifact_adjusted."""
    score = round(
        max(row.log_lift_vs_submotifs, 0.0)
        * max(row.neg_log10_pvalue_vs_submotifs, 0.0)
        * math.log1p(inn_count),
        4,
    )

    if marginals_lookup is not None:
        tokens = [t.strip() for t in row.mutation_string_pe.split("/") if t.strip()]
        artifact_frac = round(
            float(np.mean([marginals_lookup.get(t, 0.0) for t in tokens])) if tokens else 0.0,
            4,
        )
        adjusted = round(score * (1.0 - artifact_frac), 4)
    else:
        artifact_frac = float("nan")
        adjusted = score

    return StatResult(values={
        C.motif_score:                  score,
        C.artifact_fraction:            artifact_frac,
        C.motif_score_artifact_adjusted: adjusted,
    })
