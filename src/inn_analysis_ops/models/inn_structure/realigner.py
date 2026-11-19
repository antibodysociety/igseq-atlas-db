"""Realign classified constant-domain entries and extract mutation profiles.

Usage
-----
from inn_analysis_ops.models.inn_structure.realigner import realign_constant_domains

mutations = realign_constant_domains(res)
bundle = ConstantRealignedCollection(collection=collection, mutations=mutations)
"""

import logging
from dataclasses import dataclass, field
from typing import Optional

from Bio.Align import PairwiseAligner

from ._utils import _GAP_CHARS, _SPECIES_MAP, _gapless, _gapless_to_imgt_pos
from .align_classifier import classify_align_results
from .constant_validator import (
    AlignTarget,
    ValidationResults,
    _inn_known_families,
    resolve_no_gene,
    resolve_unparseable,
    validate_constant_domains,
)

logger = logging.getLogger(__name__)


_HINGE_ANNOTATIONS = {"HINGE-REGION", "H1", "H2", "H3", "H4", "H-CH2"}


def _make_sw_aligner(gap_open: float = -10) -> PairwiseAligner:
    a = PairwiseAligner()
    a.mode = "global"
    a.match_score = 1
    a.mismatch_score = -1
    a.open_gap_score = gap_open
    a.extend_gap_score = -1
    a.target_end_open_gap_score = 0
    a.target_end_extend_gap_score = 0
    a.query_end_open_gap_score = 0
    a.query_end_extend_gap_score = 0
    return a


_ALIGNER = _make_sw_aligner(gap_open=-2)
_HINGE_ALIGNER = _make_sw_aligner(
    gap_open=-2
)  # gap_open < mismatch(-1) so subs win over insertions
_CONSEC_SUBS_THRESHOLD = (
    3  # consecutive substitutions that suggest a masked indel → fall back to SW
)


@dataclass
class Mutation:
    """Single typed mutation relative to germline reference."""

    position: int  # 1-based ref position (insertions: position after which they occur)
    query_aa: str  # query residue(s); "-" for deletions; "^<seq>" for insertions
    ref_aa: str  # ref residue at position; "-" only for leading/trailing insertions with no preceding ref pos
    mutation_type: str  # "substitution" | "insertion" | "deletion"
    insertion_str: Optional[str] = (
        None  # raw inserted sequence without "^" prefix (insertions only)
    )


@dataclass
class RegionMutations:
    inn_id: str
    chain_id: str
    region_annotation: str
    category: int
    germline_gene: Optional[str]  # matched_gene; None for skip_gene regions
    mutations: list  # list[Mutation]
    method: str  # "positional" or "smith-waterman"
    no_ref_found: bool = False  # cat-3 hinge with no suitable ref
    query_sequence: Optional[str] = (
        None  # original query seq for duplicate-region disambiguation
    )
    region_annotation_description: Optional[str] = None  # e.g. "CH2 (520-629) [D5]"
    ref_sequence: Optional[str] = None  # gapped germline ref used for this alignment
    consensus_allele: Optional[str] = (
        None  # plurality allele across all domains for this INN
    )
    allele_note: Optional[str] = (
        None  # set when this domain's allele differs from consensus
    )
    germline_species: Optional[str] = (
        None  # species of the DB allele used for alignment
    )

    def visual_alignment(self, chunk: int = 50, tick: int = 5) -> str:
        """Return a multi-line alignment string: IMGT ruler / Ref / Query / Mut marker.

        Dots in the Query row mean identical to Ref.  Mut codes: S=sub D=del I=ins.
        Trailing insertions (after the last ref position) are shown as a separate
        appended block.
        tick: ruler label/tick interval (default 5).
        """
        if not self.ref_sequence:
            return (
                f"[no ref_sequence stored for {self.inn_id} {self.region_annotation}]"
            )
        ref = self.ref_sequence
        mut_map = {m.position: m for m in (self.mutations or [])}
        imgt_pos_list, ref_row, qry_row, mut_row = [], [], [], []
        seen_positions: set = set()
        for imgt_pos, r_aa in enumerate(ref, 1):
            if r_aa in _GAP_CHARS:
                imgt_pos_list.append(imgt_pos)
                ref_row.append(".")
                qry_row.append(".")
                mut_row.append(" ")
                continue
            imgt_pos_list.append(imgt_pos)
            ref_row.append(r_aa)
            seen_positions.add(imgt_pos)
            m = mut_map.get(imgt_pos)
            if m is None:
                qry_row.append(".")
                mut_row.append(" ")
            elif m.mutation_type == "deletion":
                qry_row.append("-")
                mut_row.append("D")
            elif m.mutation_type == "substitution":
                qry_row.append(m.query_aa)
                mut_row.append("S")
            else:
                qry_row.append(m.query_aa[:2] if m.query_aa else "+")
                mut_row.append("I")
        lines = [
            f"INN {self.inn_id}  {self.region_annotation}  "
            f"({self.method}, {self.germline_gene or 'no-gene'})"
        ]
        n = len(imgt_pos_list)
        for start in range(0, n, chunk):
            end = min(start + chunk, n)
            positions = imgt_pos_list[start:end]
            width = end - start
            # Ruler: number right-anchored at each tick position
            ruler = [" "] * width
            ticks = [" "] * width
            for i, p in enumerate(positions):
                if p % tick == 0:
                    ticks[i] = "|"
                    label = str(p)
                    for j, ch in enumerate(label):
                        ri = i - len(label) + 1 + j
                        if 0 <= ri < width:
                            ruler[ri] = ch
            lines += [
                f"IMGT : {''.join(ruler)}",
                f"     : {''.join(ticks)}",
                f"Ref  : {''.join(ref_row[start:end])}",
                f"Query: {''.join(qry_row[start:end])}",
                f"Mut  : {''.join(mut_row[start:end])}   S=sub D=del I=ins",
                "",
            ]

        # Trailing insertions: position > max ref position (after the ref)
        # Leading insertions: position == 0 (before the ref starts)
        trailing = [
            m
            for m in (self.mutations or [])
            if m.mutation_type == "insertion"
            and m.position not in seen_positions
            and m.position > 0
        ]
        leading = [
            m
            for m in (self.mutations or [])
            if m.mutation_type == "insertion" and m.position == 0
        ]
        if leading:
            lines.append("Leading insertions (before first ref position):")
            for m in leading:
                ins_seq = m.insertion_str or (
                    m.query_aa.lstrip("^") if m.query_aa else "?"
                )
                lines.append(f"  before pos 1 (germline={m.ref_aa}): ^{ins_seq}  [I]")
            lines.append("")
        if trailing:
            lines.append("Trailing insertions (after last ref position):")
            for m in sorted(trailing, key=lambda x: x.position):
                ins_seq = m.insertion_str or (
                    m.query_aa.lstrip("^") if m.query_aa else "?"
                )
                lines.append(
                    f"  after pos {m.position} (germline={m.ref_aa}): ^{ins_seq}  [I]"
                )
            lines.append("")

        return "\n".join(lines)


@dataclass
class ConstantRealignedCollection:
    """INNStructureCollection bundled with per-region mutation results."""

    collection: object
    mutations: list = field(default_factory=list)

    def by_inn(self, inn_id: str) -> list:
        return [m for m in self.mutations if m.inn_id == inn_id]

    def by_chain(self, chain_id: str) -> list:
        return [m for m in self.mutations if m.chain_id == chain_id]


@dataclass(frozen=True)
class ConstantContextRealignRequest:
    """One constant-domain request selected for context-aware re-alignment."""

    inn_id: str
    chain_id: str
    idx_in_result_chains: int
    detail_order_idx: str
    region_annotation: str
    region_annotation_description: Optional[str]
    flank_context: str
    prev_region_annotation: Optional[str]
    next_region_annotation: Optional[str]
    sequence_len_gapless: int
    germline_len_gapless: int
    n_mut: int
    trigger_nmut_gt_0: bool
    trigger_seq_shorter_than_germline: bool


@dataclass
class ConstantContextRealignResult:
    """Outputs from context-aware constant-domain re-alignment."""

    updated_collection: object
    requests: list[ConstantContextRealignRequest] = field(default_factory=list)
    context_realigned: list[RegionMutations] = field(default_factory=list)
    context_for_update: list[RegionMutations] = field(default_factory=list)
    context_bundle_for_update: Optional[ConstantRealignedCollection] = None
    context_rows: list[dict] = field(default_factory=list)


def _aln_to_strings(r: str, q: str, aligned) -> tuple[str, str]:
    """Reconstruct gapped alignment strings from aln.aligned coordinate blocks."""
    r_blks, q_blks = aligned
    r_out, q_out = [], []
    r_pos, q_pos = 0, 0
    for (rs, re), (qs, qe) in zip(r_blks, q_blks):
        if rs > r_pos:  # ref residues before block → deletion in query
            r_out.append(r[r_pos:rs])
            q_out.append("-" * (rs - r_pos))
        if qs > q_pos:  # query residues before block → insertion in query
            r_out.append("-" * (qs - q_pos))
            q_out.append(q[q_pos:qs])
        r_out.append(r[rs:re])
        q_out.append(q[qs:qe])
        r_pos, q_pos = re, qe
    if r_pos < len(r):  # trailing ref
        r_out.append(r[r_pos:])
        q_out.append("-" * (len(r) - r_pos))
    if q_pos < len(q):  # trailing query
        r_out.append("-" * (len(q) - q_pos))
        q_out.append(q[q_pos:])
    return "".join(r_out), "".join(q_out)


def _minimize_subs(r_aln: str, q_aln: str) -> tuple[str, str]:
    """Reposition each query gap run to minimise substitutions against the ref.

    For each contiguous gap run of length k in q (deletion from ref), try all
    placements of that gap within the non-IMGT-gap portion of the alignment and
    keep the placement that has the fewest query/ref mismatches (substitutions).
    The ref string is never modified.
    """
    q = list(q_aln)
    r = r_aln  # immutable reference

    def _count_subs(q_lst):
        return sum(
            1 for rc, qc in zip(r, q_lst) if rc != "-" and qc != "-" and rc != qc
        )

    n = len(r)
    i = 0
    while i < n:
        if q[i] != "-":
            i += 1
            continue
        # gap run [i, j) — all query gaps over real ref positions (deletions)
        j = i
        while j < n and q[j] == "-":
            j += 1
        gap_len = j - i

        # Only try to reposition if there are substitutions in the alignment
        # Try each possible start position for this gap (within range [0, n - gap_len])
        best_q = list(q)
        best_subs = _count_subs(q)

        # build gapless query (excluding current gap region)
        # extract non-gap query chars from outside the gap run, in order
        q_outside = [q[k] for k in range(n) if not (i <= k < j)]

        # iterate candidate gap placements (only over real ref positions)
        for start in range(n - gap_len + 1):
            # all positions [start, start+gap_len) must be real ref positions
            if any(r[start + d] == "-" for d in range(gap_len)):
                continue
            candidate = list(q_outside)
            # insert gap_len gaps at position `start` in the full aligned vector
            new_q = []
            outside_iter = iter(q_outside)
            for k in range(n):
                if start <= k < start + gap_len:
                    new_q.append("-")
                else:
                    new_q.append(next(outside_iter))
            subs = _count_subs(new_q)
            if subs < best_subs:
                best_subs = subs
                best_q = new_q

        q = best_q
        i = j
    return r, "".join(q)


def _reframe_leading_subs_as_deletion(r_gl: str, q_gl: str, muts: list) -> list:
    """Rewrite leading consecutive subs as a deletion when the query prefix matches ref prefix.

    Pattern: SW places a gap at the start (---APGGP vs APELLGGP) but since
    gap_open > mismatch it instead emits subs (A→L, P→L). We detect this when:
      1. There are N leading consecutive subs at positions 1..N
      2. The query chars at those positions form a prefix P that also starts the ref
      3. P appears in ref right after a deletion of k chars (i.e. ref[k:k+|P|] == P)
      4. q[|P|:] matches r[k+|P|:] exactly at the start

    In that case, rewrite as k deletions at positions 1..k, keeping remaining muts.
    """
    if not muts:
        return muts
    # Collect leading consecutive subs at gapless positions 1,2,...
    sub_run = []
    for m in muts:
        if m.mutation_type == "substitution" and m.position == (
            sub_run[-1].position + 1 if sub_run else 1
        ):
            sub_run.append(m)
        else:
            break
    if len(sub_run) < 2:
        return muts

    n = len(sub_run)
    q_prefix = q_gl[:n]  # the query residues at sub positions

    # 1: q_prefix must match the start of r_gl (same chars)
    if r_gl[:n] != q_prefix:
        return muts

    # 2: find where q_gl[n:] first appears in r_gl after position n
    q_rest = q_gl[n:]
    if not q_rest:
        return muts
    idx = r_gl.find(q_rest, n)
    if idx < 0 or idx <= n:
        return muts

    # deletion count = idx - n (ref chars between end of prefix and start of q_rest)
    del_count = idx - n
    new_dels = [
        Mutation(n + k + 1, "-", r_gl[n + k], "deletion") for k in range(del_count)
    ]
    return new_dels + muts[n:]


def _best_positional(query_gl: str, refs: list) -> tuple[int, list]:
    """Char-by-char positional alignment for short fixed-length regions.

    Safer than SW for 2–3 aa CHS where end-gap-free SW exploits free trailing
    gaps over mismatches. Picks the ref with fewest mutations.
    ``query_gl`` must already be gapless and trimmed to max-ref gapless length.
    """
    best_idx, best_muts = 0, None
    for i, ref in enumerate(refs):
        ref_gl = _gapless(ref)
        muts = []
        for j, (q_aa, r_aa) in enumerate(zip(query_gl, ref_gl)):
            imgt_pos = _gapless_to_imgt_pos(j + 1, ref)
            if q_aa != r_aa:
                muts.append(Mutation(imgt_pos, q_aa, r_aa, "substitution"))
        for j in range(len(query_gl), len(ref_gl)):
            imgt_pos = _gapless_to_imgt_pos(j + 1, ref)
            muts.append(Mutation(imgt_pos, "-", ref_gl[j], "deletion"))
        if best_muts is None or len(muts) < len(best_muts):
            best_idx, best_muts = i, muts
    return best_idx, best_muts or []


def _best_sw(query, db_sequences, aligner=None) -> tuple[int, list]:
    """Return (best_index, mutations) by highest percent identity SW alignment."""
    if aligner is None:
        aligner = _ALIGNER
    best_idx, best_pct, best_muts = 0, -1.0, []
    for i, ref in enumerate(db_sequences):
        q, r = _gapless(query).upper(), _gapless(ref).upper()
        r = r.split("*")[0]  # truncate at stop codon; remainder is not coding sequence
        alns = aligner.align(r, q)
        if not alns:
            continue
        aln = alns[0]
        r_aln, q_aln = _aln_to_strings(r, q, aln.aligned)
        r_aln, q_aln = _minimize_subs(r_aln, q_aln)
        muts, ref_pos, matches, aln_len = [], 0, 0, 0
        ins_buf: list[str] = []  # accumulate consecutive query insertions
        ins_pos = 0
        ins_ref_aa = "-"  # ref AA at the position after which the insertion occurs
        for r_aa, q_aa in zip(r_aln, q_aln):
            if r_aa == "-":  # insertion in query relative to ref
                if not ins_buf:
                    ins_pos = ref_pos
                    ins_ref_aa = r[ref_pos - 1] if ref_pos > 0 else "-"
                ins_buf.append(q_aa)
                aln_len += 1
                continue
            if ins_buf:  # flush buffered insertion
                ins_str = "".join(ins_buf)
                muts.append(
                    Mutation(ins_pos, f"^{ins_str}", ins_ref_aa, "insertion", ins_str)
                )
                ins_buf = []
                ins_ref_aa = "-"
            ref_pos += 1
            aln_len += 1
            if q_aa == "-":
                muts.append(Mutation(ref_pos, "-", r_aa, "deletion"))
            elif r_aa == q_aa:
                matches += 1
            else:
                muts.append(Mutation(ref_pos, q_aa, r_aa, "substitution"))
        if ins_buf:  # flush trailing insertion
            ins_str = "".join(ins_buf)
            muts.append(
                Mutation(ins_pos, f"^{ins_str}", ins_ref_aa, "insertion", ins_str)
            )
        # reframe before IMGT remapping (positions are still gapless here)
        muts = _reframe_leading_subs_as_deletion(r, q, muts)
        # remap gapless positions to IMGT positions using the original gapped ref.
        # Insertions at gapless position 0 (before any ref residue) stay at 0
        # to distinguish them from trailing insertions — do not let them fall
        # through to len(ref)+1 which would misreport them as trailing.
        for m in muts:
            if m.mutation_type == "insertion" and m.position == 0:
                pass  # keep position=0 to signal "leading insertion (before ref)"
            else:
                m.position = _gapless_to_imgt_pos(m.position, ref)
        pct = matches / aln_len if aln_len else 0.0
        if pct > best_pct:
            best_pct = pct
            best_idx = i
            best_muts = muts
    return best_idx, best_muts


def _realign_hinge(t, known_families: set) -> tuple[int, list, str]:
    """Hinge-specific realignment with allele family prioritization.

    If N.D. flanks are present (t.nd_prefix / t.nd_suffix), uses a two-phase approach:
      1. Align the expanded query (nd_prefix + hinge + nd_suffix) against all alleles to
         select the best-matching gene — N.D. context improves family assignment.
      2. Re-align only the original hinge sequence against the selected allele to derive
         the actual mutations (so N.D. residues never appear in the mutation list).

    Without N.D. flanks, falls back to single-pass SW as before.

    Returns (db_idx, mutations, method).
    """
    q_gl = _gapless(t.query_sequence)
    labels = t.db_gene_labels or [None] * len(t.db_sequences)

    priority_idx = [
        i for i, lbl in enumerate(labels) if lbl and lbl.split("*")[0] in known_families
    ]
    other_idx = [i for i in range(len(t.db_sequences)) if i not in set(priority_idx)]
    ordered_idx = priority_idx + other_idx

    # Exact match on hinge-only query (priority-first ordering)
    for i in ordered_idx:
        if _gapless(t.db_sequences[i]) == q_gl:
            return i, [], "exact"

    if not ordered_idx:
        return 0, [], "smith-waterman"

    seqs = [t.db_sequences[i] for i in ordered_idx]

    nd_prefix_gl = _gapless(t.nd_prefix) if t.nd_prefix else ""
    nd_suffix_gl = _gapless(t.nd_suffix) if t.nd_suffix else ""

    if nd_prefix_gl or nd_suffix_gl:
        # Phase 1: use flanked query to select best allele
        expanded = nd_prefix_gl + q_gl + nd_suffix_gl
        best_rel, _ = _best_sw(expanded, seqs, aligner=_HINGE_ALIGNER)
        # Phase 2: compute mutations using expanded query so N.D. residues that fill
        # germline positions are reported as substitutions rather than deletions
        _, muts = _best_sw(expanded, [seqs[best_rel]], aligner=_HINGE_ALIGNER)
        return ordered_idx[best_rel], muts, "smith-waterman"

    rel_idx, muts = _best_sw(t.query_sequence, seqs, aligner=_HINGE_ALIGNER)
    return ordered_idx[rel_idx], muts, "smith-waterman"


def _resolve_germline(t, db_idx: int) -> Optional[str]:
    """Pick germline gene: from db_gene_labels if available, else matched_gene."""
    if t.db_gene_labels and db_idx < len(t.db_gene_labels) and t.db_gene_labels[db_idx]:
        return t.db_gene_labels[db_idx]
    return t.matched_gene


def _resolve_species(t, db_idx: int) -> Optional[str]:
    """Pick germline species from db_species_labels at db_idx."""
    if t.db_species_labels and db_idx < len(t.db_species_labels):
        return t.db_species_labels[db_idx] or None
    return None


def _has_perfect_match(r) -> bool:
    """True if the classified result has a 0-mutation top profile."""
    return r.top_profile is not None and len(r.top_profile.mutations) == 0


def _expand_target(target: AlignTarget, cdb) -> AlignTarget:
    """Rebuild target.db_sequences with all alleles for the region+species.

    Matched gene's alleles are placed first so they win ties in _best_sw.
    """
    from ._region_config import REGION_SPECS

    spec = REGION_SPECS.get(target.region_annotation)
    if spec is None:
        return target

    species = None
    if target.all_entries:
        idx = target.matched_index if target.matched_index is not None else 0
        if idx < len(target.all_entries):
            species = target.all_entries[idx][2]

    matched_gene = target.matched_gene.lower() if target.matched_gene else None
    matched_prefix = matched_gene.split("*")[0] if matched_gene else None
    priority_triples: list[tuple[str, str, str]] = []  # (label, species, seq)
    other_triples: list[tuple[str, str, str]] = []

    sp_prefix = _SPECIES_MAP.get(species.lower(), "") if species else ""
    for (gene, region_lbl), entries in cdb.gene_idx.items():
        if region_lbl not in spec.db_labels:
            continue
        for db_sp, seq in entries:
            if sp_prefix and not db_sp.startswith(sp_prefix):
                continue
            if matched_prefix and gene.split("*")[0] == matched_prefix:
                priority_triples.append((gene, db_sp, seq))
            else:
                other_triples.append((gene, db_sp, seq))

    # Original called allele first, then rest of the same gene family alphabetically
    priority_triples.sort(key=lambda x: (x[0] != matched_gene, x[0]))
    priority_labels = [g for g, _, _ in priority_triples]
    priority_species = [sp for _, sp, _ in priority_triples]
    priority_seqs = [s for _, _, s in priority_triples]
    other_labels = [g for g, _, _ in other_triples]
    other_species = [sp for _, sp, _ in other_triples]
    other_seqs = [s for _, _, s in other_triples]

    return AlignTarget(
        inn_id=target.inn_id,
        chain_id=target.chain_id,
        region_annotation=target.region_annotation,
        query_sequence=target.query_sequence,
        db_sequences=priority_seqs + other_seqs,
        db_gene_labels=priority_labels + other_labels,
        db_species_labels=priority_species + other_species,
        gene_name=target.gene_name,
        all_entries=target.all_entries,
        matched_gene=target.matched_gene,
        matched_index=target.matched_index,
        region_annotation_description=target.region_annotation_description,
    )


def _has_consecutive_subs(raw_mutations: list, n: int = 3) -> bool:
    """Return True if raw_mutations contains n or more consecutive 1-based positions."""
    if len(raw_mutations) < n:
        return False
    positions = sorted(pos for pos, _, _ in raw_mutations)
    run = 1
    for i in range(1, len(positions)):
        if positions[i] == positions[i - 1] + 1:
            run += 1
            if run >= n:
                return True
        else:
            run = 1
    return False


def _realign_one(r, known_families: dict = None) -> RegionMutations:
    if known_families is None:
        known_families = {}
    t = r.target
    base = dict(
        inn_id=t.inn_id,
        chain_id=t.chain_id,
        region_annotation=t.region_annotation,
        category=r.category,
        germline_gene=t.matched_gene,
        region_annotation_description=getattr(t, "region_annotation_description", None),
    )

    # Hinges: exact gapless match first, then best-pct-identity SW
    is_hinge = t.region_annotation in _HINGE_ANNOTATIONS
    if is_hinge:
        if not t.query_sequence or not t.db_sequences:
            return RegionMutations(
                **base,
                mutations=[],
                method="smith-waterman",
                no_ref_found=True,
                query_sequence=t.query_sequence,
            )
        known_fams = known_families.get(t.inn_id, set())
        best_idx, muts, method = _realign_hinge(t, known_fams)
        return RegionMutations(
            **{
                **base,
                "germline_gene": _resolve_germline(t, best_idx),
                "germline_species": _resolve_species(t, best_idx),
            },
            mutations=muts,
            method=method,
            query_sequence=t.query_sequence,
            ref_sequence=t.db_sequences[best_idx] if t.db_sequences else None,
        )

    if not is_hinge and r.category in (1, 2):
        # Only trust positional if we have a known parsed gene
        if t.matched_gene:
            top_idx = r.top_profile.db_index if r.top_profile else 0
            raw = r.top_profile.mutations if r.top_profile else []
            # If consecutive substitutions exceed threshold, prefer SW (likely masks an indel)
            if not _has_consecutive_subs(raw, n=_CONSEC_SUBS_THRESHOLD):
                muts = [
                    Mutation(
                        pos,
                        "-" if q in _GAP_CHARS else q,
                        ref,
                        "deletion" if q in _GAP_CHARS else "substitution",
                    )
                    for pos, q, ref in raw
                ]
                return RegionMutations(
                    **{
                        **base,
                        "germline_gene": _resolve_germline(t, top_idx),
                        "germline_species": _resolve_species(t, top_idx),
                    },
                    mutations=muts,
                    method="positional",
                    query_sequence=t.query_sequence,
                    ref_sequence=t.db_sequences[top_idx]
                    if t.db_sequences and top_idx < len(t.db_sequences)
                    else None,
                )
        # No parsed gene, or consecutive substitutions detected — fall through to SW

    # Cat 3 non-hinge — SW against ALL alleles; known family alleles tried first so
    # they win ties (equal % identity), but are never the only candidates considered.
    if not t.query_sequence or not t.db_sequences:
        return RegionMutations(
            **base,
            mutations=[],
            method="smith-waterman",
            no_ref_found=True,
            query_sequence=t.query_sequence,
        )

    labels = t.db_gene_labels or [None] * len(t.db_sequences)
    inn_fams = known_families.get(t.inn_id, set())
    priority_idx = [
        i for i, lbl in enumerate(labels) if lbl and lbl.split("*")[0] in inn_fams
    ]
    other_idx = [i for i in range(len(t.db_sequences)) if i not in set(priority_idx)]
    ordered_idx = (
        priority_idx + other_idx
    )  # priority alleles first → win ties in _best_sw
    seqs = [t.db_sequences[i] for i in ordered_idx]
    rel_idx, muts = _best_sw(t.query_sequence, seqs, aligner=_ALIGNER)
    best_idx = ordered_idx[rel_idx]
    return RegionMutations(
        **{
            **base,
            "germline_gene": _resolve_germline(t, best_idx),
            "germline_species": _resolve_species(t, best_idx),
        },
        mutations=muts,
        method="smith-waterman",
        query_sequence=t.query_sequence,
        ref_sequence=t.db_sequences[best_idx],
    )


def _split_compound(rm: RegionMutations, cdb) -> list:
    """Split a CH3-CHS (or CH4-CHS) RegionMutations back into two separate entries.

    Uses precomputed split_points from ConstantDB: primary_ref ends with any trailing
    CH3 dots, extra_ref (CHS) starts cleanly at GK with no leading dots.
    """
    from ._region_config import REGION_SPECS

    spec = REGION_SPECS.get(rm.region_annotation)
    if spec is None or not spec.extra_annotation or cdb is None:
        return [rm]

    compound_label = f"{rm.region_annotation}-{spec.extra_annotation}"

    # Use precomputed split refs: primary_ref ends with trailing CH3 dots, extra_ref starts at GK
    gene_key = (rm.germline_gene.lower() if rm.germline_gene else None, compound_label)
    sp_entry = cdb.split_points.get(gene_key)
    if sp_entry is None:
        # fallback: any split point for this compound_label
        gene_key = next((k for k in cdb.split_points if k[1] == compound_label), None)
        sp_entry = cdb.split_points.get(gene_key) if gene_key else None
    if sp_entry is None:
        return [rm]

    primary_ref_seq = sp_entry["primary_ref"]
    chs_seq_matched = sp_entry["extra_ref"]
    chs_gl_len = len(_gapless(chs_seq_matched))
    # primary_ref includes trailing CH3 dots → split boundary is its gapped length
    primary_imgt_end = len(primary_ref_seq)
    # extra_ref has no leading dots → first CHS residue is at primary_imgt_end + 1
    chs_offset = primary_imgt_end

    primary_muts = [m for m in rm.mutations if m.position <= primary_imgt_end]
    extra_muts = [
        Mutation(
            m.position - chs_offset,
            m.query_aa,
            m.ref_aa,
            m.mutation_type,
            m.insertion_str,
        )
        for m in rm.mutations
        if m.position > primary_imgt_end
    ]

    base = dict(
        inn_id=rm.inn_id,
        chain_id=rm.chain_id,
        category=rm.category,
        germline_gene=rm.germline_gene,
        germline_species=rm.germline_species,
        method=rm.method,
    )
    # CHS query: last chs_gl_len gapless chars of the stitched compound query
    chs_query_seq = (
        _gapless(rm.query_sequence)[-chs_gl_len:] if rm.query_sequence else None
    )

    return [
        RegionMutations(
            **base,
            region_annotation=rm.region_annotation,
            mutations=primary_muts,
            query_sequence=rm.query_sequence,
            ref_sequence=primary_ref_seq,
        ),
        RegionMutations(
            **base,
            region_annotation=spec.extra_annotation,
            mutations=extra_muts,
            query_sequence=chs_query_seq,
            ref_sequence=chs_seq_matched,
        ),
    ]


def realign_constant_domains(
    validation_res: ValidationResults,
    collection=None,
    cdb=None,
    expand_on_mismatch: bool = True,
    allow_leading_insertion: bool = False,
    allow_trailing_insertion: bool = True,
):
    """Classify and realign all to_align entries, plus resolved unparseable entries.

    If expand_on_mismatch=True (default): any entry whose matched gene has no
    0-mutation allele is re-classified against all alleles for that region+species,
    allowing SW to pick the best gene family rather than staying within the IMGT call.

    Parameters
    ----------
    allow_leading_insertion : bool, default False
        If False, leading insertions (position=0) on HINGE regions are moved to
        the upstream constant domain (if one exists) or discarded.
    allow_trailing_insertion : bool, default True
        If True, trailing insertions (position>0) on HINGE regions are kept even
        when there is no downstream constant domain — they trail the germline
        anchor position and are a valid annotation.

    Returns ConstantRealignedCollection if collection is given, else list[RegionMutations].
    """
    known_families = _inn_known_families(validation_res)
    classified = classify_align_results(validation_res.to_align)

    if expand_on_mismatch and cdb is not None:
        keep, to_expand = [], []
        for r in classified:
            if r.target.matched_gene and not _has_perfect_match(r):
                to_expand.append(r.target)
            else:
                keep.append(r)
        if to_expand:
            keep += classify_align_results([_expand_target(t, cdb) for t in to_expand])
        classified = keep

    mutations = [_realign_one(r, known_families) for r in classified]

    if cdb is not None and collection is not None:
        resolved = resolve_unparseable(
            validation_res, cdb, collection, known_families=known_families
        )
        resolved += resolve_no_gene(
            validation_res, cdb, collection, known_families=known_families
        )
        classified_resolved = classify_align_results(resolved)
        mutations += [_realign_one(r, known_families) for r in classified_resolved]

    # Split compound CH3-CHS / CH4-CHS back into separate region entries
    split = []
    for rm in mutations:
        split.extend(_split_compound(rm, cdb))
    mutations = split

    # For standalone CHS chains (separate scraped entities) that were NOT already
    # covered by a compound CH3-CHS alignment, run a direct SW alignment.
    # If _split_compound already produced a CHS entry (germline_gene set), trust it —
    # those positions were assigned by aligning the full CH3+CHS query to the compound
    # reference, which is the correct source of truth.
    if cdb is not None and collection is not None:
        chs_refs = cdb.region_idx.get("CHS", [])
        if chs_refs:
            # Last-resort fallback for standalone CHS chains that were NOT handled
            # by the compound CH3+CHS alignment path. This should only occur when
            # the germline has no compound CH3-CHS reference in the DB (very rare).
            # Normally validate_constant_domains now falls through to compound alignment
            # for all CH3 chains that have a sibling CHS, even when CH3 is 100% match.
            compound_chs = {
                (m.inn_id, m.chain_id)
                for m in mutations
                if m.region_annotation == "CHS" and m.germline_gene is not None
            }
            for result in collection.results:
                for chain in result.chains:
                    if chain.region_annotation != "CHS" or not chain.sequence:
                        continue
                    inn_id, chain_id = result.inn_id, chain.chain_id
                    if (inn_id, chain_id) in compound_chs:
                        continue  # compound split already assigned CHS mutations correctly
                    # Inherit gene/species from the sibling CH3/CH4 entry (e.g. IgM whose
                    # CH4-CHS compound suffix doesn't match a standalone CHS in region_idx).
                    ch_sibling = next(
                        (
                            m
                            for m in mutations
                            if m.inn_id == inn_id
                            and m.chain_id == chain_id
                            and m.region_annotation in ("CH3", "CH4")
                            and m.germline_gene
                        ),
                        None,
                    )
                    # Trim CH3 boundary bleed: scraped CHS may include trailing CH3
                    # residues prepended (e.g. "PGK" when ref is "GK"). Trim the query
                    # from the left to the longest CHS ref gapless length so only real
                    # CHS residues are aligned.
                    max_ref_gl = max(len(_gapless(r)) for r in chs_refs)
                    chs_query = _gapless(chain.sequence)
                    if len(chs_query) > max_ref_gl:
                        chs_query = chs_query[-max_ref_gl:]
                    _, muts = _best_positional(chs_query, chs_refs)
                    mutations = [
                        m
                        for m in mutations
                        if not (
                            m.inn_id == inn_id
                            and m.chain_id == chain_id
                            and m.region_annotation == "CHS"
                        )
                    ]
                    mutations.append(
                        RegionMutations(
                            inn_id=inn_id,
                            chain_id=chain_id,
                            region_annotation="CHS",
                            category=3,
                            germline_gene=ch_sibling.germline_gene
                            if ch_sibling
                            else None,
                            germline_species=ch_sibling.germline_species
                            if ch_sibling
                            else None,
                            mutations=muts,
                            method="positional",
                            query_sequence=chain.sequence,
                        )
                    )

    if cdb is not None and collection is not None:
        mutations = _absorb_nd_domains(mutations, collection, cdb, known_families)

    mutations = _filter_hinge_boundary_insertions(
        mutations,
        allow_leading=allow_leading_insertion,
        allow_trailing=allow_trailing_insertion,
    )

    _resolve_consensus_alleles(mutations)

    if collection is not None:
        return ConstantRealignedCollection(collection=collection, mutations=mutations)
    return mutations


_CH1_ANNOTATIONS = {"CH1", "C-KAPPA", "C-LAMBDA", "CL", "CH1/C-KAPPA/C-LAMBDA"}
_DOWNSTREAM_ANNOTATIONS = {"CH2", "CH3", "CH4", "CHS", "H-CH2"}


def _filter_hinge_boundary_insertions(
    mutations: list,
    allow_leading: bool = False,
    allow_trailing: bool = True,
) -> list:
    """Remove or reassign leading/trailing insertions on HINGE regions.

    Parameters
    ----------
    allow_leading : bool, default False
        If False, leading insertions (position=0) are handled as follows:
        - If an upstream constant domain (any CH/CL annotation) exists on the
          same chain, the insertion is **moved** to that domain as a trailing
          insertion at its last IMGT position.
        - If no upstream constant domain exists, the insertion is **discarded**
          (it cannot be assigned anywhere).
    allow_trailing : bool, default True
        If False, trailing insertions (position>0) with no downstream constant
        domain (CH2/CH3/CH4/CHS) are discarded.  When True (default), trailing
        insertions are always kept — they trail the germline anchor position
        and are a valid annotation regardless of what follows.
    """
    _ALL_UPSTREAM = _CH1_ANNOTATIONS | {"CH2", "CH3", "CH4", "CHS", "HINGE-REGION"}

    # Index existing regions per (inn_id, chain_id): annotation → RegionMutations
    _rm_by_key: dict = {}
    for rm in mutations:
        key = (rm.inn_id, rm.chain_id)
        _rm_by_key.setdefault(key, {})[rm.region_annotation] = rm

    _extra_muts: dict = {}

    result_rms = {id(rm): rm for rm in mutations}

    for rm in mutations:
        if rm.region_annotation not in _HINGE_ANNOTATIONS:
            continue

        key = (rm.inn_id, rm.chain_id)
        sibling_regions = _rm_by_key[key]

        # Best upstream constant domain to absorb a leading insertion
        upstream_rm = next(
            (sibling_regions[a] for a in _CH1_ANNOTATIONS if a in sibling_regions),
            next(
                (sibling_regions[a] for a in _ALL_UPSTREAM if a in sibling_regions),
                None,
            ),
        )
        has_downstream = bool(set(sibling_regions) & _DOWNSTREAM_ANNOTATIONS)

        filtered_muts = list(rm.mutations or [])
        to_remove = []
        for m in filtered_muts:
            if m.mutation_type != "insertion":
                continue
            if m.position == 0 and not allow_leading:
                to_remove.append(m)
                if upstream_rm is not None and upstream_rm.ref_sequence:
                    # Count gapless ref positions to get the last IMGT position.
                    ref_seq = upstream_rm.ref_sequence
                    gapless = [c for c in ref_seq if c not in _GAP_CHARS]
                    last_pos = len(gapless)
                    # Correct germline_seq: use the actual last ref residue,
                    # not the "-" placeholder from _best_sw position=0 fallback.
                    correct_ref_aa = gapless[-1] if gapless else "-"

                    # Assert no existing trailing insertion at last_pos on upstream.
                    existing_ins_at_last = [
                        e
                        for e in (upstream_rm.mutations or [])
                        if e.mutation_type == "insertion" and e.position == last_pos
                    ]
                    assert not existing_ins_at_last, (
                        f"_filter_hinge_boundary_insertions: cannot move leading hinge "
                        f"insertion to {upstream_rm.region_annotation} position {last_pos} "
                        f"for INN {rm.inn_id} — an insertion already exists there: "
                        f"{existing_ins_at_last}. Merge not yet supported."
                    )

                    _extra_muts.setdefault(id(upstream_rm), []).append(
                        Mutation(
                            last_pos,
                            m.query_aa,
                            correct_ref_aa,
                            "insertion",
                            m.insertion_str,
                        )
                    )
                # else: no upstream → discard
            elif m.position > 0 and not allow_trailing and not has_downstream:
                to_remove.append(m)

        if to_remove:
            keep = [m for m in filtered_muts if m not in to_remove]
            from dataclasses import replace as _dc_replace

            result_rms[id(rm)] = _dc_replace(rm, mutations=keep)

    for rm in list(result_rms.values()):
        extra = _extra_muts.get(id(rm))
        if extra:
            from dataclasses import replace as _dc_replace

            result_rms[id(rm)] = _dc_replace(
                result_rms[id(rm)],
                mutations=list(result_rms[id(rm)].mutations or []) + extra,
            )

    return list(result_rms.values())


def _absorb_nd_domains(
    mutations: list,
    collection,
    cdb,
    known_families: dict,
) -> list:
    """Stitch N.D. sequences onto preceding constant domains and re-align if it scores better.

    For each RegionMutations entry, if the collection has an N.D. chain immediately
    following that region on the same chain, a stitched query (constant_seq + nd_seq) is
    tried against all DB sequences for that region.  If the stitched alignment produces
    fewer mutations, it replaces the original result and the N.D. absorption is logged.

    The N.D. chain is never added to the output — it is only used to improve the
    preceding constant domain alignment.  Logged at INFO when absorbed.
    """
    from ._region_config import REGION_SPECS

    # Build nd_map: (inn_id, chain_id, region_annotation) -> nd_sequence
    nd_map: dict[tuple, str] = {}
    for result in collection.results:
        by_id: dict[str, list] = {}
        for chain in result.chains:
            by_id.setdefault(chain.chain_id, []).append(chain)
        for cid, chains in by_id.items():
            for i in range(len(chains) - 1):
                nxt = chains[i + 1]
                if nxt.region_annotation == "N.D" and nxt.sequence:
                    nd_map[(result.inn_id, cid, chains[i].region_annotation)] = (
                        nxt.sequence
                    )

    if not nd_map:
        return mutations

    updated = []
    total_absorbed = 0

    for rm in mutations:
        nd_seq = nd_map.get((rm.inn_id, rm.chain_id, rm.region_annotation))
        if not nd_seq or not rm.query_sequence or rm.method == "exact":
            updated.append(rm)
            continue

        spec = REGION_SPECS.get(rm.region_annotation)
        if spec is None:
            updated.append(rm)
            continue

        # Collect DB sequences for this region (with family preference)
        db_seqs: list[str] = []
        db_labels: list[str] = []
        db_species: list[str] = []
        inn_fams = known_families.get(rm.inn_id, set())
        prio_seqs: list[str] = []
        prio_labels: list[str] = []
        prio_species: list[str] = []
        for (gene, lbl), entries in cdb.gene_idx.items():
            if lbl not in spec.db_labels:
                continue
            for _sp, seq in entries:
                db_seqs.append(seq)
                db_labels.append(gene)
                db_species.append(_sp)
                if gene.split("*")[0] in inn_fams:
                    prio_seqs.append(seq)
                    prio_labels.append(gene)
                    prio_species.append(_sp)

        if not db_seqs:
            updated.append(rm)
            continue

        stitched_query = rm.query_sequence.rstrip(".-") + nd_seq

        # Try priority alleles first, fall back to all
        absorbed = False
        for search_seqs, search_labels, search_species in (
            (prio_seqs, prio_labels, prio_species),
            (db_seqs, db_labels, db_species),
        ):
            if not search_seqs or absorbed:
                continue
            rel_idx, stitched_muts = _best_sw(stitched_query, search_seqs)
            if len(stitched_muts) < len(rm.mutations):
                germline = (
                    search_labels[rel_idx]
                    if rel_idx < len(search_labels)
                    else rm.germline_gene
                )
                sp = (
                    search_species[rel_idx]
                    if rel_idx < len(search_species)
                    else rm.germline_species
                )
                logger.info(
                    "N.D absorbed into %s INN=%s chain=%s: %d→%d mutations "
                    "(nd=%r germline=%s)",
                    rm.region_annotation,
                    rm.inn_id,
                    rm.chain_id,
                    len(rm.mutations),
                    len(stitched_muts),
                    nd_seq[:30],
                    germline,
                )
                total_absorbed += 1
                absorbed = True
                updated.append(
                    RegionMutations(
                        inn_id=rm.inn_id,
                        chain_id=rm.chain_id,
                        region_annotation=rm.region_annotation,
                        category=rm.category,
                        germline_gene=germline,
                        germline_species=sp,
                        mutations=stitched_muts,
                        method="smith-waterman",
                        query_sequence=stitched_query,
                        region_annotation_description=rm.region_annotation_description,
                        ref_sequence=search_seqs[rel_idx],
                    )
                )

        if not absorbed:
            updated.append(rm)

    if total_absorbed:
        logger.info(
            "_absorb_nd_domains: absorbed %d N.D. sequences into constant domains",
            total_absorbed,
        )

    return updated


def _resolve_consensus_alleles(mutations: list) -> None:
    """Vote on consensus allele per INN across all constant domains (in-place).

    For each INN, the plurality germline allele across non-hinge domains becomes
    `consensus_allele` on every RegionMutations.  Domains whose best allele differs
    from the consensus get `allele_note` set to describe the change.
    """
    from collections import Counter, defaultdict

    by_inn: dict = defaultdict(list)
    for m in mutations:
        by_inn[m.inn_id].append(m)

    for inn_muts in by_inn.values():
        voters = [
            m
            for m in inn_muts
            if m.germline_gene
            and not m.no_ref_found
            and m.region_annotation not in _HINGE_ANNOTATIONS
        ]
        if not voters:
            continue
        counts = Counter(m.germline_gene.upper() for m in voters)
        consensus = counts.most_common(1)[0][0]
        for m in inn_muts:
            m.consensus_allele = consensus
            if m.germline_gene and m.germline_gene.upper() != consensus:
                m.allele_note = f"allelic change from {consensus}"


def _terminal_deletion_counts(
    realigned_mutations: list, ref_sequence: str
) -> tuple[int, int]:
    """Return leading and trailing deletion counts against non-gap ref positions."""
    if not ref_sequence:
        raise RuntimeError("Missing ref_sequence for terminal deletion analysis")

    ref_positions = [i for i, aa in enumerate(ref_sequence, 1) if aa not in _GAP_CHARS]
    if not ref_positions:
        raise RuntimeError("Reference has no non-gap residues")

    del_pos = {
        int(m.position)
        for m in (realigned_mutations or [])
        if getattr(m, "mutation_type", None) == "deletion"
    }

    lead = 0
    for p in ref_positions:
        if p in del_pos:
            lead += 1
        else:
            break

    trail = 0
    for p in reversed(ref_positions):
        if p in del_pos:
            trail += 1
        else:
            break

    return lead, trail


def _skip_boundary_leading_insertions(
    mutations: list[Mutation],
    prev_region_annotation: Optional[str],
    *,
    inn_id: str,
    chain_id: str,
    region_annotation: str,
    detail_order_idx: str,
) -> list[Mutation]:
    """Skip leading insertions for context-expanded domains preceded by ND/HINGE.

    These are boundary artifacts from left-flank expansion, not stable positions in
    the target region numbering space. We keep a loud logger.info trail for audits.
    """
    if (
        prev_region_annotation not in {"N.D"}
        and prev_region_annotation not in _HINGE_ANNOTATIONS
    ):
        return mutations

    kept: list[Mutation] = []
    for m in mutations:
        if m.mutation_type == "insertion" and int(m.position) == 0:
            logger.info(
                "Context realign skip: dropping leading insertion at position 0 "
                "for inn=%s chain=%s detail=%s region=%s prev_region=%s "
                "query_aa=%s ref_aa=%s insertion_str=%s",
                inn_id,
                chain_id,
                detail_order_idx,
                region_annotation,
                prev_region_annotation,
                m.query_aa,
                m.ref_aa,
                m.insertion_str,
            )
            continue
        kept.append(m)
    return kept


def _boundary_indel_count(mutations: list[Mutation], imgt_window: int) -> int:
    """Count insertions/deletions in the early reference window near the stitch."""
    boundary = 0
    for m in mutations:
        if m.mutation_type == "insertion" and m.position == 0:
            boundary += 1
        elif (
            m.mutation_type in {"insertion", "deletion"}
            and int(m.position) <= imgt_window
        ):
            boundary += 1
    return boundary


def _species_matches(
    preferred_species: Optional[str], candidate_species: Optional[str]
) -> bool:
    if not preferred_species or not candidate_species:
        return False
    pref = preferred_species.strip().lower()
    cand = candidate_species.strip().lower()
    if pref == cand:
        return True
    pref_prefix = _SPECIES_MAP.get(pref, "")
    cand_prefix = _SPECIES_MAP.get(cand, "")
    if pref_prefix and cand.startswith(pref_prefix):
        return True
    if cand_prefix and pref.startswith(cand_prefix):
        return True
    return False


def _candidate_prefix_score(
    query_prefix: str,
    ref_sequence: str,
    preferred_gene: Optional[str],
    preferred_species: Optional[str],
    candidate_gene: Optional[str],
    candidate_species: Optional[str],
) -> float:
    score = _gapped_prefix_alignment_score(query_prefix, ref_sequence)
    if score < 0:
        return score
    score_norm = score / max(1, len(query_prefix))
    return score_norm + _candidate_preference_bonus(
        preferred_gene=preferred_gene,
        preferred_species=preferred_species,
        candidate_gene=candidate_gene,
        candidate_species=candidate_species,
    )


def _gapped_prefix_alignment_score(query_prefix: str, ref_sequence: str) -> float:
    """Align ND-prefix to the germline *beginning* with gapped reference retained."""
    qry = query_prefix.upper()
    if not qry:
        return -1.0
    ref_prefix = ref_sequence.upper()[: max(len(qry) + 10, len(qry))]
    if not ref_prefix:
        return -1.0
    # Keep IMGT gap chars in the reference prefix and align against the ND prefix.
    # End gaps are free in _make_sw_aligner so this remains start-anchored while
    # permitting small boundary shifts.
    return float(_ALIGNER.score(ref_prefix, qry))


def _rewrite_boundary_block_on_current_ref(
    *,
    query_prefix: str,
    current_ref_sequence: str,
    anchor_gapless_pos: int,
    drop_first_query_char: bool,
) -> list[Mutation]:
    """Build boundary mutations by aligning ND prefix to the start of current ref.

    The rewritten block is computed only on the prefix window up to the anchor
    residue (inclusive) and emitted in IMGT coordinates of ``current_ref_sequence``.
    """
    ref_gl = _gapless(current_ref_sequence).upper()
    if anchor_gapless_pos < 1 or anchor_gapless_pos > len(ref_gl):
        return []

    qry = (
        query_prefix[1:]
        if drop_first_query_char and len(query_prefix) > 1
        else query_prefix
    )
    qry = qry.upper()
    if not qry:
        return []

    ref_seg = ref_gl[:anchor_gapless_pos]

    prefix_muts: list[Mutation] = []
    overlap = min(len(ref_seg), len(qry))
    for i in range(overlap):
        r_aa = ref_seg[i]
        q_aa = qry[i]
        if q_aa != r_aa:
            prefix_muts.append(
                Mutation(
                    position=_gapless_to_imgt_pos(i + 1, current_ref_sequence),
                    query_aa=q_aa,
                    ref_aa=r_aa,
                    mutation_type="substitution",
                )
            )

    # If ref prefix is longer than query prefix, treat the unmatched ref suffix
    # as deletions anchored before the CH2 center motif.
    for i in range(overlap, len(ref_seg)):
        prefix_muts.append(
            Mutation(
                position=_gapless_to_imgt_pos(i + 1, current_ref_sequence),
                query_aa="-",
                ref_aa=ref_seg[i],
                mutation_type="deletion",
            )
        )

    # If query is longer than the ref prefix, emit a single insertion anchored to
    # the last ref prefix position.
    if len(qry) > len(ref_seg) and len(ref_seg) > 0:
        ins_seq = qry[len(ref_seg) :]
        if ins_seq:
            prefix_muts.append(
                Mutation(
                    position=_gapless_to_imgt_pos(len(ref_seg), current_ref_sequence),
                    query_aa=f"^{ins_seq}",
                    ref_aa=ref_seg[-1],
                    mutation_type="insertion",
                    insertion_str=ins_seq,
                )
            )
    return prefix_muts


def _candidate_preference_bonus(
    *,
    preferred_gene: Optional[str],
    preferred_species: Optional[str],
    candidate_gene: Optional[str],
    candidate_species: Optional[str],
) -> float:
    pref_gene = preferred_gene.lower() if preferred_gene else None
    cand_gene = candidate_gene.lower() if candidate_gene else None
    bonus = 0.0
    if pref_gene and cand_gene:
        if pref_gene == cand_gene:
            bonus += 0.20
        elif pref_gene.split("*")[0] == cand_gene.split("*")[0]:
            bonus += 0.10
    if _species_matches(preferred_species, candidate_species):
        bonus += 0.05
    if cand_gene and "orf" in cand_gene:
        bonus -= 0.25
    return bonus


def _collect_region_reference_candidates(
    cdb,
    region_annotation: str,
    preferred_gene: Optional[str],
    preferred_species: Optional[str],
) -> list[tuple[str, Optional[str], Optional[str]]]:
    """Return unique per-sequence candidates as (sequence, gene, species)."""
    from ._region_config import REGION_SPECS

    spec = REGION_SPECS.get(region_annotation)
    if spec is None:
        return []

    grouped: dict[str, list[tuple[str, str]]] = {}
    for (gene, lbl), entries in cdb.gene_idx.items():
        if lbl not in spec.db_labels:
            continue
        for sp, seq in entries:
            grouped.setdefault(seq, []).append((gene, sp))

    out: list[tuple[str, Optional[str], Optional[str]]] = []
    for seq, meta in grouped.items():
        ranked = sorted(
            meta,
            key=lambda x: _candidate_preference_bonus(
                preferred_gene=preferred_gene,
                preferred_species=preferred_species,
                candidate_gene=x[0],
                candidate_species=x[1],
            ),
            reverse=True,
        )
        best_gene, best_species = ranked[0]
        out.append((seq, best_gene, best_species))
    return out


def _maybe_rescue_left_stitch_alignment(
    *,
    expanded_query: str,
    center_sequence: str,
    left_add: str,
    region_annotation: str,
    preferred_gene: Optional[str],
    preferred_species: Optional[str],
    current_ref_sequence: str,
    current_mutations: list[Mutation],
    cdb,
    inn_id: str,
    chain_id: str,
    detail_order_idx: str,
) -> tuple[str, list[Mutation], Optional[str], Optional[str]]:
    """Use prefix-only guide scoring to reposition boundary deletions on current ref.

    IMPORTANT: this does not change the selected reference/gene/species for CH2/CH3.
    """
    if not left_add:
        return (
            current_ref_sequence,
            current_mutations,
            preferred_gene,
            preferred_species,
        )

    query_prefix = _gapless(left_add).upper()
    if not query_prefix:
        return (
            current_ref_sequence,
            current_mutations,
            preferred_gene,
            preferred_species,
        )

    window = max(1, len(_gapless(left_add)))
    current_boundary = _boundary_indel_count(current_mutations, window)
    if current_boundary == 0:
        return (
            current_ref_sequence,
            current_mutations,
            preferred_gene,
            preferred_species,
        )

    candidates = _collect_region_reference_candidates(
        cdb, region_annotation, preferred_gene, preferred_species
    )
    if not candidates:
        return (
            current_ref_sequence,
            current_mutations,
            preferred_gene,
            preferred_species,
        )

    def _find_boundary_anchor_pos(ref_sequence: str, center: str) -> Optional[int]:
        ref_gl = _gapless(ref_sequence).upper()
        center_gl = _gapless(center).upper()
        if len(center_gl) < 3:
            return None
        motif = center_gl[:3]
        idx = ref_gl.find(motif)
        if idx <= 0:
            return None
        return idx

    def _ref_aa_at_gapless_pos(ref_sequence: str, pos: int) -> Optional[str]:
        ref_gl = _gapless(ref_sequence)
        if pos < 1 or pos > len(ref_gl):
            return None
        return ref_gl[pos - 1]

    current_prefix_score = _candidate_prefix_score(
        query_prefix=query_prefix,
        ref_sequence=current_ref_sequence,
        preferred_gene=preferred_gene,
        preferred_species=preferred_species,
        candidate_gene=preferred_gene,
        candidate_species=preferred_species,
    )

    best_choice: Optional[tuple[float, Optional[str], Optional[str]]] = None
    for seq, gene, species in candidates:
        score = _candidate_prefix_score(
            query_prefix=query_prefix,
            ref_sequence=seq,
            preferred_gene=preferred_gene,
            preferred_species=preferred_species,
            candidate_gene=gene,
            candidate_species=species,
        )
        logger.info(
            "Context stitch guide candidate inn=%s chain=%s detail=%s region=%s gene=%s species=%s "
            "prefix_only_score=%.4f",
            inn_id,
            chain_id,
            detail_order_idx,
            region_annotation,
            gene,
            species,
            score,
        )
        if best_choice is None or score > best_choice[0]:
            best_choice = (score, gene, species)

    if best_choice is None or best_choice[0] <= current_prefix_score:
        logger.info(
            "Context stitch guide: no better prefix guide for inn=%s chain=%s detail=%s region=%s "
            "(current_boundary_indels=%d current_prefix_only_score=%.4f)",
            inn_id,
            chain_id,
            detail_order_idx,
            region_annotation,
            current_boundary,
            current_prefix_score,
        )
        return (
            current_ref_sequence,
            current_mutations,
            preferred_gene,
            preferred_species,
        )

    anchor_pos = _find_boundary_anchor_pos(current_ref_sequence, center_sequence)
    if anchor_pos is None:
        logger.info(
            "Context stitch guide: unable to locate center motif anchor for inn=%s chain=%s detail=%s region=%s; "
            "keeping original boundary calls",
            inn_id,
            chain_id,
            detail_order_idx,
            region_annotation,
        )
        return (
            current_ref_sequence,
            current_mutations,
            preferred_gene,
            preferred_species,
        )

    anchor_imgt_pos = _gapless_to_imgt_pos(anchor_pos, current_ref_sequence)
    anchor_ref_aa = _ref_aa_at_gapless_pos(current_ref_sequence, anchor_pos)
    if anchor_ref_aa is None:
        logger.info(
            "Context stitch guide: anchor ref residue missing at pos=%s for inn=%s chain=%s detail=%s region=%s; "
            "keeping original boundary calls",
            anchor_pos,
            inn_id,
            chain_id,
            detail_order_idx,
            region_annotation,
        )
        return (
            current_ref_sequence,
            current_mutations,
            preferred_gene,
            preferred_species,
        )

    leading_ins_present = any(
        m.mutation_type == "insertion" and int(m.position) == 0
        for m in current_mutations
    )
    has_early_del = any(
        m.mutation_type == "deletion" and int(m.position) <= window
        for m in current_mutations
    )
    if not leading_ins_present and not has_early_del:
        logger.info(
            "Context stitch guide: no early deletion/leading insertion to reposition for inn=%s chain=%s detail=%s region=%s",
            inn_id,
            chain_id,
            detail_order_idx,
            region_annotation,
        )
        return (
            current_ref_sequence,
            current_mutations,
            preferred_gene,
            preferred_species,
        )

    rewritten_prefix_muts = _rewrite_boundary_block_on_current_ref(
        query_prefix=query_prefix,
        current_ref_sequence=current_ref_sequence,
        anchor_gapless_pos=anchor_pos,
        drop_first_query_char=leading_ins_present,
    )
    if not rewritten_prefix_muts:
        logger.info(
            "Context stitch guide: boundary rewrite yielded no mutations for inn=%s chain=%s detail=%s region=%s; "
            "keeping original boundary calls",
            inn_id,
            chain_id,
            detail_order_idx,
            region_annotation,
        )
        return (
            current_ref_sequence,
            current_mutations,
            preferred_gene,
            preferred_species,
        )

    adjusted = [
        m
        for m in current_mutations
        if not (
            (
                m.mutation_type in {"substitution", "deletion"}
                and int(m.position) <= anchor_imgt_pos
            )
            or (m.mutation_type == "insertion" and int(m.position) == 0)
        )
    ]
    adjusted.extend(rewritten_prefix_muts)
    adjusted.sort(key=lambda m: (int(m.position), m.mutation_type))

    before_boundary = _boundary_indel_count(current_mutations, window)
    after_boundary = _boundary_indel_count(adjusted, window)
    guide_score, guide_gene, guide_species = best_choice
    logger.info(
        "Context stitch guide applied inn=%s chain=%s detail=%s region=%s "
        "guide_gene=%s guide_species=%s guide_prefix_only_score=%.4f current_prefix_only_score=%.4f "
        "boundary_indels_before=%d boundary_indels_after=%d anchor_pos=%d "
        "reference_kept=%s",
        inn_id,
        chain_id,
        detail_order_idx,
        region_annotation,
        guide_gene,
        guide_species,
        guide_score,
        current_prefix_score,
        before_boundary,
        after_boundary,
        anchor_imgt_pos,
        preferred_gene,
    )
    return current_ref_sequence, adjusted, preferred_gene, preferred_species


def realign_constant_domains_with_context(
    collection,
    bundle,
    target_regions: set[str],
    min_match_alignment: float = 0.5,
    cn=None,
    cdb=None,
    flank_regions: Optional[set[str]] = None,
    target_inn: Optional[str] = None,
    require_trigger: bool = True,
) -> ConstantContextRealignResult:
    """Run context-aware second-pass constant re-alignment for CH1/CL constant regions.

    Parameters
    ----------
    target_regions : set[str]
        Required set of region annotations to context-realign (for example,
        {"CH1", "C-KAPPA", "C-LAMBDA"}).
    cdb : ConstantDB
        Required. This function force-runs baseline realignment for
        perfect-match rows in ``target_regions`` before context expansion.

    This function:
    1) materializes `bundle` onto `collection` for ``target_regions`` only
       (via update_with_realignment),
    2) optionally force-aligns target perfect-match rows when ``cdb`` is set,
    3) selects target constant regions with ND/HINGE-style flank context,
    4) expands query using flank residues only when terminal deletions indicate
       missing 5' and/or 3' residues,
    5) realigns expanded (or unchanged) query against the same best reference,
    6) returns both context-labeled rows and update-ready rows.
    """
    from .operations import update_with_realignment

    if not target_regions:
        raise ValueError("target_regions is required and must be non-empty")
    if cdb is None:
        raise ValueError("cdb is required for context realignment baseline forcing")
    flank_regions = flank_regions or {
        "N.D",
        "HINGE-REGION",
        "H1",
        "H2",
        "H3",
        "H4",
        "H-CH2",
    }

    def _is_target_row(inn_id: str, region_annotation: str) -> bool:
        if region_annotation not in target_regions:
            return False
        if target_inn is None:
            return True
        return str(inn_id) == str(target_inn)

    scoped_bundle = ConstantRealignedCollection(
        collection=collection,
        mutations=[
            m
            for m in bundle.mutations
            if _is_target_row(str(m.inn_id), m.region_annotation)
        ],
    )
    updated = update_with_realignment(
        collection, scoped_bundle, min_match_alignment=min_match_alignment, cn=cn
    )
    validation_res = validate_constant_domains(updated, cdb)
    scoped_validation = ValidationResults(
        skipped_feature_types=validation_res.skipped_feature_types,
        perfect_matches=[
            row
            for row in validation_res.perfect_matches
            if _is_target_row(str(row[0]), row[2])
        ],
        to_align=[
            t
            for t in validation_res.to_align
            if _is_target_row(str(t.inn_id), t.region_annotation)
        ],
        no_gene=[
            row for row in validation_res.no_gene if _is_target_row(str(row[0]), row[2])
        ],
        unparseable=[
            row
            for row in validation_res.unparseable
            if _is_target_row(str(row[0]), row[2])
        ],
    )
    forced = list(scoped_validation.perfect_matches)
    if forced:
        logger.info(
            "Context prepass: forcing baseline realignment for %d perfect-match target rows",
            len(forced),
        )
        scoped_validation.no_gene.extend(forced)
        scoped_validation.perfect_matches = []
        forced_bundle = realign_constant_domains(
            scoped_validation,
            collection=updated,
            cdb=cdb,
            expand_on_mismatch=True,
        )
        updated = update_with_realignment(
            updated,
            forced_bundle,
            min_match_alignment=min_match_alignment,
            cn=cn,
        )

    requests: list[ConstantContextRealignRequest] = []
    seen: set[tuple[str, str, str, str]] = set()
    lookup: dict[str, tuple[list, int, object]] = {}

    for result in updated.results:
        if target_inn is not None and str(result.inn_id) != str(target_inn):
            continue

        by_chain_id: dict[str, list] = {}
        for idx, chain in enumerate(result.chains):
            by_chain_id.setdefault(chain.chain_id, []).append((idx, chain))

        for chain_id, items in by_chain_id.items():
            for local_i, (idx, chain) in enumerate(items):
                if chain.detail_order_idx in lookup:
                    raise RuntimeError(
                        f"Duplicate detail_order_idx: {chain.detail_order_idx}"
                    )
                lookup[chain.detail_order_idx] = (items, local_i, chain)

                if chain.region_annotation not in target_regions:
                    continue
                if not chain.sequence:
                    raise RuntimeError(f"Missing sequence for {chain.detail_order_idx}")

                prev_region = (
                    items[local_i - 1][1].region_annotation if local_i > 0 else None
                )
                next_region = (
                    items[local_i + 1][1].region_annotation
                    if local_i + 1 < len(items)
                    else None
                )
                if (
                    prev_region not in flank_regions
                    and next_region not in flank_regions
                ):
                    continue

                if chain.realigned_mutations is None:
                    raise RuntimeError(
                        f"realigned_mutations missing for {chain.detail_order_idx}"
                    )
                if not chain.realigned_germline_sequence:
                    raise RuntimeError(
                        f"realigned_germline_sequence missing for {chain.detail_order_idx}"
                    )

                seq_len = len(_gapless(chain.sequence))
                gl_len = len(_gapless(chain.realigned_germline_sequence))
                n_mut = len(chain.realigned_mutations)
                trigger_nmut = n_mut > 0
                trigger_short = seq_len < gl_len
                if require_trigger and not (trigger_nmut or trigger_short):
                    continue

                prev_lbl = prev_region if prev_region is not None else "START"
                next_lbl = next_region if next_region is not None else "END"
                flank_context = f"{prev_lbl}->{chain.region_annotation}->{next_lbl}"
                req = ConstantContextRealignRequest(
                    inn_id=str(chain.inn_id),
                    chain_id=chain.chain_id,
                    idx_in_result_chains=idx,
                    detail_order_idx=chain.detail_order_idx,
                    region_annotation=chain.region_annotation,
                    region_annotation_description=chain.region_annotation_description,
                    flank_context=flank_context,
                    prev_region_annotation=prev_region,
                    next_region_annotation=next_region,
                    sequence_len_gapless=seq_len,
                    germline_len_gapless=gl_len,
                    n_mut=n_mut,
                    trigger_nmut_gt_0=trigger_nmut,
                    trigger_seq_shorter_than_germline=trigger_short,
                )
                k = (req.inn_id, req.chain_id, req.detail_order_idx, req.flank_context)
                if k in seen:
                    raise RuntimeError(
                        f"Duplicate context realignment request key: {k}"
                    )
                seen.add(k)
                requests.append(req)
                logger.info(
                    "Context candidate inn=%s chain=%s detail=%s region=%s context=%s "
                    "n_mut=%d seq_len=%d gl_len=%d",
                    req.inn_id,
                    req.chain_id,
                    req.detail_order_idx,
                    req.region_annotation,
                    req.flank_context,
                    req.n_mut,
                    req.sequence_len_gapless,
                    req.germline_len_gapless,
                )

    context_realigned: list[RegionMutations] = []
    context_for_update: list[RegionMutations] = []
    context_rows: list[dict] = []

    for req in requests:
        if req.detail_order_idx not in lookup:
            raise RuntimeError(
                f"Context request detail_order_idx not found: {req.detail_order_idx}"
            )
        items, local_i, center = lookup[req.detail_order_idx]

        if not center.sequence:
            raise RuntimeError(f"Missing center sequence for {center.detail_order_idx}")
        if center.realigned_mutations is None:
            raise RuntimeError(
                f"Missing center realigned_mutations for {center.detail_order_idx}"
            )
        if not center.realigned_germline_sequence:
            raise RuntimeError(
                f"Missing center realigned_germline_sequence for {center.detail_order_idx}"
            )

        best_ref = center.realigned_germline_sequence
        _, baseline_muts = _best_sw(center.sequence, [best_ref])
        stored_n_mut = len(center.realigned_mutations or [])
        if stored_n_mut != len(baseline_muts):
            logger.warning(
                "Context baseline mismatch inn=%s chain=%s detail=%s region=%s: "
                "stored_n_mut=%d recomputed_n_mut=%d (using recomputed for expansion)",
                req.inn_id,
                req.chain_id,
                req.detail_order_idx,
                req.region_annotation,
                stored_n_mut,
                len(baseline_muts),
            )

        lead_del, trail_del = _terminal_deletion_counts(
            baseline_muts,
            best_ref,
        )
        left_neighbor = items[local_i - 1][1] if local_i > 0 else None
        right_neighbor = items[local_i + 1][1] if local_i + 1 < len(items) else None

        left_add, right_add = "", ""
        notes: list[str] = []
        if lead_del > 0:
            if (
                left_neighbor is None
                or left_neighbor.region_annotation not in flank_regions
            ):
                notes.append(f"left_flank_missing_for_{lead_del}")
            else:
                left_gl = _gapless(left_neighbor.sequence or "")
                if len(left_gl) < lead_del:
                    notes.append(f"left_flank_short_{len(left_gl)}<{lead_del}")
                else:
                    left_add = left_gl[-lead_del:]
        if trail_del > 0:
            if (
                right_neighbor is None
                or right_neighbor.region_annotation not in flank_regions
            ):
                notes.append(f"right_flank_missing_for_{trail_del}")
            else:
                right_gl = _gapless(right_neighbor.sequence or "")
                if len(right_gl) < trail_del:
                    notes.append(f"right_flank_short_{len(right_gl)}<{trail_del}")
                else:
                    right_add = right_gl[:trail_del]

        expanded_query = f"{left_add}{center.sequence}{right_add}"
        _, ctx_muts = _best_sw(expanded_query, [best_ref])
        best_ref, ctx_muts, rescued_gene, rescued_species = (
            _maybe_rescue_left_stitch_alignment(
                expanded_query=expanded_query,
                center_sequence=center.sequence,
                left_add=left_add,
                region_annotation=req.region_annotation,
                preferred_gene=center.realigned_gene or center.gene_name,
                preferred_species=center.realigned_species,
                current_ref_sequence=best_ref,
                current_mutations=ctx_muts,
                cdb=cdb,
                inn_id=req.inn_id,
                chain_id=req.chain_id,
                detail_order_idx=req.detail_order_idx,
            )
        )
        ctx_muts = _skip_boundary_leading_insertions(
            ctx_muts,
            req.prev_region_annotation,
            inn_id=req.inn_id,
            chain_id=req.chain_id,
            region_annotation=req.region_annotation,
            detail_order_idx=req.detail_order_idx,
        )

        status = (
            "REALIGNED_EXPANDED"
            if expanded_query != center.sequence
            else "REALIGNED_UNCHANGED_QUERY"
        )
        logger.info(
            "Context re-aligned inn=%s chain=%s detail=%s region=%s status=%s "
            "lead_del=%d trail_del=%d left_added=%d right_added=%d old_n_mut=%d new_n_mut=%d notes=%s",
            req.inn_id,
            req.chain_id,
            req.detail_order_idx,
            req.region_annotation,
            status,
            lead_del,
            trail_del,
            len(left_add),
            len(right_add),
            len(baseline_muts),
            len(ctx_muts),
            ";".join(notes),
        )

        ctx_rm = RegionMutations(
            inn_id=req.inn_id,
            chain_id=req.chain_id,
            region_annotation=f"{req.region_annotation}_CTX",
            category=1,
            germline_gene=rescued_gene or center.realigned_gene or center.gene_name,
            germline_species=rescued_species or center.realigned_species,
            mutations=ctx_muts,
            method="smith-waterman",
            query_sequence=expanded_query,
            ref_sequence=best_ref,
            region_annotation_description=req.region_annotation_description,
        )
        context_realigned.append(ctx_rm)
        context_for_update.append(
            RegionMutations(
                inn_id=req.inn_id,
                chain_id=req.chain_id,
                region_annotation=req.region_annotation,
                category=1,
                germline_gene=ctx_rm.germline_gene,
                germline_species=ctx_rm.germline_species,
                mutations=ctx_rm.mutations,
                method=ctx_rm.method,
                query_sequence=ctx_rm.query_sequence,
                ref_sequence=ctx_rm.ref_sequence,
                region_annotation_description=ctx_rm.region_annotation_description,
            )
        )
        context_rows.append(
            {
                "inn_id": req.inn_id,
                "chain_id": req.chain_id,
                "detail_order_idx": req.detail_order_idx,
                "region_annotation": req.region_annotation,
                "status": status,
                "notes": ";".join(notes),
                "lead_del": lead_del,
                "trail_del": trail_del,
                "left_added_len": len(left_add),
                "right_added_len": len(right_add),
                "old_n_mut_center": len(baseline_muts),
                "new_n_mut_context": len(ctx_muts),
            }
        )

    context_bundle_for_update = ConstantRealignedCollection(
        collection=collection, mutations=context_for_update
    )
    return ConstantContextRealignResult(
        updated_collection=updated,
        requests=requests,
        context_realigned=context_realigned,
        context_for_update=context_for_update,
        context_bundle_for_update=context_bundle_for_update,
        context_rows=context_rows,
    )


# merge_realigned removed — use INNStructureCollection.update_with_realignment() instead.

# merge_realigned removed — use INNStructureCollection.update_with_realignment() instead.

# merge_realigned removed — use INNStructureCollection.update_with_realignment() instead.
