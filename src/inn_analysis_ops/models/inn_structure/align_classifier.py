"""Classify AlignTarget entries by comparing query_sequence to DB reference sequences.

Categories
----------
1  Length matches at least one DB sequence (with gaps preserved)
   1a  profiles_identical — all per-DB mutation profiles are the same
   1b  profiles           — MutationProfile per matching DB sequence
   1c  top_profile        — profile vs db_sequences[0]

2  Query is a substring of at least one DB sequence (gapless on both sides)
   2a  profiles_identical — all per-DB substring mutation profiles are the same
   2b  profiles           — MutationProfile per DB sequence containing query
   2c  top_profile        — profile vs first containing DB sequence

3  No length match and not a substring — full external alignment needed
"""

from dataclasses import dataclass, field
from typing import Optional

from ._utils import _GAP_CHARS, _gapless
from .constant_validator import AlignTarget


@dataclass
class MutationProfile:
    db_index: int  # index into AlignTarget.db_sequences
    mutations: list[tuple[int, str, str]]  # (1-based pos, query_aa, db_aa)

    @property
    def count(self) -> int:
        return len(self.mutations)


@dataclass
class ClassifiedAlignResult:
    target: AlignTarget
    category: int  # 1, 2, or 3
    profiles_identical: Optional[bool] = None  # 1a / 2a
    profiles: list[MutationProfile] = field(default_factory=list)  # 1b / 2b
    top_profile: Optional[MutationProfile] = None  # 1c / 2c


def _make_profile(query: str, ref: str, db_index: int) -> MutationProfile:
    """Positional comparison. Includes deletions (query gap, ref AA); skips ref-gap positions."""
    muts = []
    for i, (q, r) in enumerate(zip(query, ref)):
        if r in _GAP_CHARS:  # IMGT gap in ref — not a real position
            continue
        if q == r:
            continue
        muts.append((i + 1, q, r))  # q may be a gap char (deletion) or AA (substitution)
    return MutationProfile(db_index=db_index, mutations=muts)


def _identical(profiles: list[MutationProfile]) -> bool:
    if len(profiles) < 2:
        return True
    first = profiles[0].mutations
    return all(p.mutations == first for p in profiles[1:])


def classify_align_target(target: AlignTarget) -> ClassifiedAlignResult:
    if not target.query_sequence or not target.db_sequences:
        return ClassifiedAlignResult(target=target, category=3)

    q = target.query_sequence  # keep gaps for cat 1
    db_seqs = target.db_sequences  # keep gaps for cat 1

    # --- Category 1: length match WITH gaps ---
    length_matched = [(i, s) for i, s in enumerate(db_seqs) if len(s) == len(q)]
    if length_matched:
        profiles = [_make_profile(q, s, i) for i, s in length_matched]

        # Also check non-length-matched alleles via gapless — an allele with different
        # IMGT padding may still be a perfect gapless match (0 mutations). If so, add a
        # synthetic zero-mutation profile so it can be selected as top.
        q_gl = _gapless(q)
        length_matched_idx = {i for i, _ in length_matched}
        for i, s in enumerate(db_seqs):
            if i not in length_matched_idx and _gapless(s) == q_gl:
                profiles.append(MutationProfile(db_index=i, mutations=[]))

        top_i = min(range(len(profiles)), key=lambda idx: (len(profiles[idx].mutations), profiles[idx].db_index))
        return ClassifiedAlignResult(
            target=target,
            category=1,
            profiles_identical=_identical(profiles),
            profiles=profiles,
            top_profile=profiles[top_i],
        )

    # --- Category 2: substring — gapless on both ---
    q_gl = _gapless(q)
    substring_matched = [(i, _gapless(s)) for i, s in enumerate(db_seqs) if q_gl in _gapless(s)]
    if substring_matched:
        profiles = [_make_profile(q_gl, s[s.index(q_gl) : s.index(q_gl) + len(q_gl)], i) for i, s in substring_matched]
        top_i = min(range(len(profiles)), key=lambda idx: (len(profiles[idx].mutations), profiles[idx].db_index))
        return ClassifiedAlignResult(
            target=target,
            category=2,
            profiles_identical=_identical(profiles),
            profiles=profiles,
            top_profile=profiles[top_i],
        )

    # --- Category 3: needs external alignment ---
    return ClassifiedAlignResult(target=target, category=3)


def classify_align_results(targets: list[AlignTarget]) -> list[ClassifiedAlignResult]:
    return [classify_align_target(t) for t in targets]
