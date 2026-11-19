"""Validate constant-domain gene assignments from scraped IMGT 2D structure data.

Usage
-----
from inn_analysis_ops.models.inn_structure.constant_validator import (
    build_constant_db, validate_constant_domains,
)
db = build_constant_db(pd.read_excel(..., sheet_name="annotation_with_imgt_gaps"))
results = validate_constant_domains(collection, db)
"""

import logging
import re
from dataclasses import dataclass, field
from typing import Optional

import pandas as pd

from ._region_config import REGION_SPECS
from ._utils import (
    _GAP_CHARS,
    _SPECIES_MAP,
    _gapless,
    _gapless_end_pos,
    _leading_gap_count,
    _stitch_query,
)

logger = logging.getLogger(__name__)

_GENE_RE = re.compile(r"([A-Z0-9*]+)\s+\((\d+(?:\.\d+)?)%\)\(([^)]+)\)")


def _parse_gene_entries(
    gene_name: Optional[str],
) -> tuple[list[tuple[str, float, str]], list[str]]:
    """Split gene_name by ',' then parse each token with the regex.

    Returns:
        parsed   — [(gene, pct, species), ...] for tokens that matched
        bad      — raw token strings that had content but didn't match
    """
    if not gene_name:
        return [], []
    parsed, bad = [], []
    for token in gene_name.split(","):
        token = token.strip()
        if not token:
            continue
        m = _GENE_RE.search(token)
        if m:
            parsed.append((m.group(1), float(m.group(2)), m.group(3)))
        else:
            bad.append(token)
    return parsed, bad


@dataclass
class AlignTarget:
    inn_id: str
    chain_id: str
    region_annotation: str
    query_sequence: Optional[str]  # scraped sequence to align
    db_sequences: list[str]  # reference sequences from DB
    db_gene_labels: list[Optional[str]] = field(
        default_factory=list
    )  # allele per db_sequence (None if unknown)
    db_species_labels: list[Optional[str]] = field(
        default_factory=list
    )  # species per db_sequence
    gene_name: Optional[str] = None  # full original string
    all_entries: list = field(default_factory=list)  # [(gene, pct, species), ...]
    matched_gene: Optional[str] = (
        None  # gene that had a DB hit (None for skip_gene regions)
    )
    matched_index: Optional[int] = None  # index of matched_gene in all_entries
    region_annotation_description: Optional[str] = (
        None  # e.g. "CH2 (520-629) [D5]" for duplicate disambiguation
    )
    nd_prefix: Optional[str] = (
        None  # gapless N.D. sequence immediately before this region (hinge only)
    )
    nd_suffix: Optional[str] = (
        None  # gapless N.D. sequence immediately after this region (hinge only)
    )


@dataclass
class ValidationResults:
    skipped_feature_types: dict = field(
        default_factory=dict
    )  # {(annotation, cft): [chain_ids]}
    perfect_matches: list = field(default_factory=list)  # all entries 100%
    to_align: list[AlignTarget] = field(default_factory=list)
    no_gene: list = field(default_factory=list)  # absent gene_name or no DB hit
    unparseable: list = field(
        default_factory=list
    )  # gene_name present but no tokens matched


@dataclass
class ConstantDB:
    """Pre-built index from annotation_with_imgt_gaps sheet. Build once, reuse."""

    gene_idx: dict  # {(gene_lower, region_label): [(species, seq)]}
    region_idx: dict  # {region_label: [seq]}
    split_points: dict = field(
        default_factory=dict
    )  # {(gene, compound_label): {primary_ref, extra_ref}}


def build_constant_db(constant_db_seqs: pd.DataFrame) -> ConstantDB:
    """Build and return a ConstantDB from the annotation_with_imgt_gaps DataFrame."""
    from ._utils import (
        _build_composite_seqs,
        _build_db_index,
        _parse_compound_split_points,
    )

    gene_idx, region_idx = _build_db_index(constant_db_seqs)
    _build_composite_seqs(gene_idx, region_idx, REGION_SPECS)
    split_points = _parse_compound_split_points(gene_idx)
    return ConstantDB(
        gene_idx=gene_idx, region_idx=region_idx, split_points=split_points
    )


def _inn_known_families(res: "ValidationResults") -> dict:
    """Return {inn_id: set[gene_family_lower]} from already-resolved entries."""
    families: dict[str, set[str]] = {}
    for t in res.to_align:
        if t.matched_gene:
            families.setdefault(t.inn_id, set()).add(
                t.matched_gene.split("*")[0].lower()
            )
    for inn_id, _, _, gene_name in res.perfect_matches:
        for gene, _, _ in _parse_gene_entries(gene_name)[0]:
            families.setdefault(inn_id, set()).add(gene.split("*")[0].lower())
    return families


def _build_seq_map(collection) -> dict:
    return {
        (result.inn_id, chain.chain_id, chain.region_annotation): chain.sequence
        for result in collection.results
        for chain in result.chains
    }


def _resolve_missing(
    entries: list,
    seq_map: dict,
    known_families: dict,
    cdb: ConstantDB,
    use_original_gene_name: bool,
) -> list[AlignTarget]:
    """Core logic for resolve_unparseable and resolve_no_gene."""
    targets: list[AlignTarget] = []
    for inn_id, chain_id, region_annotation, gene_name in entries:
        spec = REGION_SPECS.get(region_annotation)
        if spec is None or spec.skip_gene:
            continue
        query_seq = seq_map.get((inn_id, chain_id, region_annotation))
        if not query_seq:
            continue
        q_gl = _gapless(query_seq)
        inn_fams = known_families.get(inn_id, set())
        stored_gene_name = gene_name if use_original_gene_name else None

        # Stitch CHS sibling unconditionally when spec has extra_annotation and sibling exists.
        # This ensures the compound CH3-CHS (or CH4-CHS) alignment gets the full query
        # even when CH3 has mutations (preventing false CHS deletions from unmatched GK).
        stitched_query = query_seq
        if spec.extra_annotation:
            extra_seq = seq_map.get((inn_id, chain_id, spec.extra_annotation))
            if extra_seq:
                compound_label = f"{region_annotation}-{spec.extra_annotation}"
                bridge_refs = cdb.region_idx.get(compound_label, [])
                if bridge_refs:
                    stitched_query = _stitch_query(query_seq, extra_seq, bridge_refs[0])
                else:
                    stitched_query = query_seq.rstrip(".-") + extra_seq.lstrip(".-")

        exact: list[tuple[int, str, str, str]] = []
        substring: list[tuple[int, str, str, str]] = []
        for (gene, region_lbl), db_entries in cdb.gene_idx.items():
            if region_lbl not in spec.db_labels:
                continue
            score = 0 if gene.split("*")[0] in inn_fams else 1
            for _sp, db_seq in db_entries:
                db_gl = _gapless(db_seq)
                if db_gl == q_gl:
                    exact.append((score, gene, _sp, db_seq))
                elif q_gl in db_gl:
                    substring.append((score, gene, _sp, db_seq))

        hits = sorted(exact or substring, key=lambda x: x[0])
        if hits:
            seen: set[str] = set()
            ordered_seqs, ordered_labels, ordered_species = [], [], []
            for _, gene, sp, db_seq in hits:
                if db_seq not in seen:
                    seen.add(db_seq)
                    ordered_seqs.append(db_seq)
                    ordered_labels.append(gene)
                    ordered_species.append(sp)
            targets.append(
                AlignTarget(
                    inn_id=inn_id,
                    chain_id=chain_id,
                    region_annotation=region_annotation,
                    query_sequence=stitched_query,
                    db_sequences=ordered_seqs,
                    db_gene_labels=ordered_labels,
                    db_species_labels=ordered_species,
                    gene_name=stored_gene_name,
                    matched_gene=hits[0][1],
                )
            )
        else:
            all_seqs, all_labels, all_species = [], [], []
            for (gene, region_lbl), db_entries in cdb.gene_idx.items():
                if region_lbl in spec.db_labels:
                    for _sp, db_seq in db_entries:
                        all_seqs.append(db_seq)
                        all_labels.append(gene)
                        all_species.append(_sp)
            targets.append(
                AlignTarget(
                    inn_id=inn_id,
                    chain_id=chain_id,
                    region_annotation=region_annotation,
                    query_sequence=stitched_query,
                    db_sequences=all_seqs,
                    db_gene_labels=all_labels,
                    db_species_labels=all_species,
                    gene_name=stored_gene_name,
                )
            )
    return targets


def resolve_unparseable(
    res: "ValidationResults",
    cdb: ConstantDB,
    collection,
    known_families: dict = None,
) -> list[AlignTarget]:
    """Resolve unparseable gene_name entries via gapless DB search."""
    if known_families is None:
        known_families = _inn_known_families(res)
    return _resolve_missing(
        res.unparseable,
        _build_seq_map(collection),
        known_families,
        cdb,
        use_original_gene_name=True,
    )


def resolve_no_gene(
    res: "ValidationResults",
    cdb: ConstantDB,
    collection,
    known_families: dict = None,
) -> list[AlignTarget]:
    """Resolve entries with no gene_name via gapless DB search."""
    if known_families is None:
        known_families = _inn_known_families(res)
    return _resolve_missing(
        res.no_gene,
        _build_seq_map(collection),
        known_families,
        cdb,
        use_original_gene_name=False,
    )


def validate_constant_domains(
    collection, db: "ConstantDB | pd.DataFrame"
) -> ValidationResults:
    """Validate constant-domain gene assignments in a scraped INNStructureCollection.

    Args:
        collection: INNStructureCollection from scraper/CSV
        db: ConstantDB (from build_constant_db) or raw DataFrame (auto-converted)
    """
    from ._utils import _build_composite_seqs, _build_db_index

    if isinstance(db, pd.DataFrame):
        gene_idx, region_idx = _build_db_index(db)
        _build_composite_seqs(gene_idx, region_idx, REGION_SPECS)
    else:
        gene_idx, region_idx = db.gene_idx, db.region_idx

    out = ValidationResults()
    _HINGE_FLANK_ANNS = {"N.D", "LINKER"}

    for result in collection.results:
        chain_seq_map = {
            (c.chain_id, c.region_annotation): c.sequence for c in result.chains
        }

        # Build ordered chain list per chain_id for N.D. adjacency lookups
        by_chain_id: dict = {}
        for c in result.chains:
            by_chain_id.setdefault(c.chain_id, []).append(c)

        for chain in result.chains:
            spec = REGION_SPECS.get(chain.region_annotation)
            if spec is None:
                key = (chain.region_annotation, chain.chain_feature_type or None)
                out.skipped_feature_types.setdefault(key, []).append(chain.chain_id)
                continue

            # skip_gene regions (hinges): go straight to to_align using all db_label seqs
            if spec.skip_gene:
                db_seqs, db_labels, db_species = [], [], []
                for lbl in spec.db_labels:
                    for (gene, region_lbl), entries in gene_idx.items():
                        if region_lbl == lbl:
                            for _sp, seq in entries:
                                db_seqs.append(seq)
                                db_labels.append(gene)
                                db_species.append(_sp)
                    # fall back to region_idx for any seqs not in gene_idx
                    for s in region_idx.get(lbl, []):
                        if s not in db_seqs:
                            db_seqs.append(s)
                            db_labels.append(None)
                            db_species.append(None)
                # Find adjacent N.D. or Linker chains for hinge flanking
                chain_list = by_chain_id.get(chain.chain_id, [])
                idx = next((i for i, c in enumerate(chain_list) if c is chain), -1)
                nd_prefix = None
                nd_suffix = None
                if (
                    idx > 0
                    and chain_list[idx - 1].region_annotation in _HINGE_FLANK_ANNS
                ):
                    nd_prefix = _gapless(chain_list[idx - 1].sequence or "")
                if (
                    idx >= 0
                    and idx < len(chain_list) - 1
                    and chain_list[idx + 1].region_annotation in _HINGE_FLANK_ANNS
                ):
                    nd_suffix = _gapless(chain_list[idx + 1].sequence or "")
                out.to_align.append(
                    AlignTarget(
                        inn_id=result.inn_id,
                        chain_id=chain.chain_id,
                        region_annotation=chain.region_annotation,
                        query_sequence=chain.sequence,
                        db_sequences=db_seqs,
                        db_gene_labels=db_labels,
                        db_species_labels=db_species,
                        gene_name=chain.gene_name or None,
                        region_annotation_description=chain.region_annotation_description,
                        nd_prefix=nd_prefix or None,
                        nd_suffix=nd_suffix or None,
                    )
                )
                continue

            if not chain.gene_name:
                out.no_gene.append(
                    (result.inn_id, chain.chain_id, chain.region_annotation, None)
                )
                continue

            parsed, bad_tokens = _parse_gene_entries(chain.gene_name)
            if not parsed:
                out.unparseable.append(
                    (
                        result.inn_id,
                        chain.chain_id,
                        chain.region_annotation,
                        chain.gene_name,
                    )
                )
                logger.debug(
                    "Unparseable inn=%s chain=%s gene=%r",
                    result.inn_id,
                    chain.chain_id,
                    chain.gene_name,
                )
                continue
            if bad_tokens:
                logger.debug(
                    "Partially unparseable inn=%s chain=%s bad=%r",
                    result.inn_id,
                    chain.chain_id,
                    bad_tokens,
                )

            if all(pct == 100.0 for _, pct, _ in parsed):
                # If this region has a CHS component AND a sibling CHS chain exists,
                # fall through to the compound alignment path instead of skipping —
                # that way _split_compound handles CHS correctly even when CH3 is perfect.
                has_chs_sibling = bool(
                    spec.extra_annotation
                    and chain_seq_map.get((chain.chain_id, spec.extra_annotation))
                )
                if not has_chs_sibling:
                    out.perfect_matches.append(
                        (
                            result.inn_id,
                            chain.chain_id,
                            chain.region_annotation,
                            chain.gene_name,
                        )
                    )
                    continue
                # else: fall through to compound alignment below

            matched = None
            for entry_idx, (gene, pct, species) in enumerate(parsed):
                hits_with_labels: list[
                    tuple[str, str, str]
                ] = []  # (allele, db_species, seq)
                matched_label = None
                gene_prefix = gene.lower().split("*")[0]
                sp_prefix = _SPECIES_MAP.get(species.lower(), "") if species else ""
                for lbl in spec.db_labels:
                    for (g, rl), entries in gene_idx.items():
                        if rl != lbl or g.split("*")[0] != gene_prefix:
                            continue
                        for db_sp, seq in entries:
                            if not sp_prefix or db_sp.startswith(sp_prefix):
                                hits_with_labels.append((g, db_sp, seq))
                    if hits_with_labels:
                        matched_label = lbl
                        break
                if hits_with_labels:
                    # Put the exact matched allele first, then sort the rest by allele name
                    hits_with_labels.sort(key=lambda x: (x[0] != gene.lower(), x[0]))
                    hits = [seq for _, _, seq in hits_with_labels]
                    hit_labels = [g for g, _, _ in hits_with_labels]
                    hit_species = [sp for _, sp, _ in hits_with_labels]
                    query_seq = chain.sequence
                    if spec.extra_annotation and matched_label and "-" in matched_label:
                        extra_seq = chain_seq_map.get(
                            (chain.chain_id, spec.extra_annotation)
                        )
                        if extra_seq:
                            query_seq = _stitch_query(
                                chain.sequence or "", extra_seq, hits[0]
                            )
                    matched = AlignTarget(
                        inn_id=result.inn_id,
                        chain_id=chain.chain_id,
                        region_annotation=chain.region_annotation,
                        query_sequence=query_seq,
                        db_sequences=hits,
                        db_gene_labels=hit_labels,
                        db_species_labels=hit_species,
                        gene_name=chain.gene_name,
                        all_entries=parsed,
                        matched_gene=gene,
                        matched_index=entry_idx,
                        region_annotation_description=chain.region_annotation_description,
                    )
                    break

            if matched:
                out.to_align.append(matched)
            else:
                out.no_gene.append(
                    (
                        result.inn_id,
                        chain.chain_id,
                        chain.region_annotation,
                        chain.gene_name,
                    )
                )
                logger.warning(
                    "No DB hit for any gene in %r (inn=%s chain=%s)",
                    chain.gene_name,
                    result.inn_id,
                    chain.chain_id,
                )
    return out
