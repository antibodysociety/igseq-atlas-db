"""Analytics models and parsing operations for IMGT 2D structure scraping results.

Usage
-----
from inn_analysis_ops.models.inn_structure.analytics import parse_collection

chains, full_length, mutations = parse_collection(
    bundle_collection.results,
    inn_map=imgt_inn_number_to_inn_map,
)
observed_chains_df = chains.to_dataframe()
observed_full_length_df = full_length.to_dataframe()
mutational_profile_df = mutations.to_dataframe()
"""

from collections import defaultdict
from typing import Annotated, Dict, List, Optional, Set, Tuple, Union

import pandas as pd
from pydantic import BaseModel, Field
from pydantic.functional_validators import BeforeValidator

from .models import (
    INNChainDetail,
    INNStructureResult,
    InnId,
    MutationRecord,
    NumberedMutationRecord,
)

_CONSTANTS_FEATURES = {"C-DOMAIN", "HINGE-REGION", "CHS"}
_VARIABLE_FEATURES = {"V-DOMAIN"}
_IGNORE_REGIONS = {"C-ALPHA", "C-BETA", "C-BETA-2"}
_OBJECT_COLS = {
    "mutations_object",
    "scraped_mutations_object",
    "realigned_mutations",
}


_PIVOT_COLUMNS = [
    "sequence",
    "sequence_gapped",
    "germline",
    "c_allele",
    "c_allele_raw",
    "c_allele_species",
    "alignment_fraction",
    "mutations",
    "constant_chain_mutations_IMGT",
    "constant_chain_mutations_EU",
    "constant_chain_mutations_KABAT",
]
_PIVOT_ORDER = [
    "CH1/C-KAPPA/C-LAMBDA",
    "HINGE-REGION",
    "CH2",
    "CH3",
    "CH4",
    "CHS",
]
_CH1_CKL_GROUP = {"CH1", "C-KAPPA", "C-LAMBDA"}
_CK_GROUP = {"C-KAPPA"}
_CL_GROUP = {"C-LAMBDA"}
_V_GROUP = {"CH1", "CH2", "CH3", "CH4", "HINGE-REGION", "CHS"}

_C_CHAIN_CLASSIFIERS = {
    "CH1": "CH1",
    "C-KAPPA": "CL",
    "C-LAMBDA": "CL",
    "CH2": "FC",
    "CH3": "FC",
    "CH4": "FC",
    "HINGE-REGION": "HINGE",
    "CHS": "FC",
}

# ---------------------------------------------------------------------------
# Record models
# ---------------------------------------------------------------------------


class ObservedChain(BaseModel):
    """One domain-region record for a single chain per INN result."""

    inn: str
    inn_id: InnId
    chain_id: str
    chain_desc: str
    chain_type: str
    chain_detail_idx: str
    chain_feature_type: Optional[str] = None
    region: str
    sequence: str
    sequence_gapped: str
    total_mutations: int
    germline: Optional[str] = None
    v_allele: Optional[str] = None
    j_allele: Optional[str] = None
    c_allele: Optional[str] = None
    c_allele_raw: Optional[str] = None
    c_allele_species: str = ""
    failed_realignment: Optional[bool] = None
    alignment_fraction: Optional[float] = None
    mutations: str = ""
    mutations_object: List[MutationRecord] = Field(default_factory=list)
    scraped_mutations_object: Optional[List[MutationRecord]] = None
    realigned_mutations: Optional[List[MutationRecord]] = None
    constant_chain_mutations_IMGT: Optional[str] = None
    constant_chain_mutations_EU: Optional[str] = None
    constant_chain_mutations_KABAT: Optional[str] = None
    chain_region_iteration: Optional[int] = None
    variable_chain_species: str = ""
    constant_chain_species: str = ""


class ObservedFullLengthChain(BaseModel):
    """Aggregated full-length chain summary per chain_id per INN."""

    inn: str
    inn_id: InnId
    chain_id: str
    chain_type: str
    sequence: str
    variable_chain_species: str = ""
    constant_chain_species: str = ""
    chain_description: str = ""
    germline_sequence: Optional[str] = None
    v_allele: Optional[str] = None
    j_allele: Optional[str] = None
    c_allele: Optional[str] = None
    total_mutations: int = 0
    total_constant_mutations: int = 0
    total_mutations_unclassified_features: int = 0
    variable_mutations: str = ""
    constant_chain_mutations_IMGT: Optional[str] = None
    constant_chain_mutations_EU: Optional[str] = None
    constant_chain_mutations_KABAT: Optional[str] = None


class IndividualMutationInfo(BaseModel):
    """One mutation within a constant-domain chain region."""

    inn: str
    inn_id: InnId
    chain_id: str
    chain_type: str
    region: str
    region_classification: str
    region_iteration: str
    position: Union[int, str, float]
    mutation_type: str
    germline_seq: str
    mutated_seq: str
    IMGT: Optional[Union[int, str, float]] = None
    EU: Optional[Union[int, str, float]] = None
    KABAT: Optional[Union[int, str, float]] = None
    EXON: Optional[Union[int, str, float]] = None
    allele: Optional[str] = None
    gene: Optional[str] = None
    gene_species: str = ""


class ChainAggregation(BaseModel):
    """Per-chain accumulator built during result traversal (mirrors chains_by_id)."""

    chain_id: str
    chain_types: List[str] = Field(default_factory=list)
    total_constant: List[MutationRecord] = Field(default_factory=list)
    total_variable: List[MutationRecord] = Field(default_factory=list)
    other_features: List[str] = Field(default_factory=list)
    other_mutations: List[MutationRecord] = Field(default_factory=list)
    variable_alleles: List[str] = Field(
        default_factory=list
    )  # raw v_gene strings for species extraction
    v_alleles: List[str] = Field(default_factory=list)
    j_alleles: List[str] = Field(default_factory=list)
    constant_alleles: List[str] = Field(
        default_factory=list
    )  # raw constant gene strings for species extraction
    c_alleles: List[str] = Field(default_factory=list)
    constant_chain_mutations_by_numeric_system: Dict[str, List[MutationRecord]] = Field(
        default_factory=dict
    )
    mutational_info_by_region: Dict[str, List[MutationRecord]] = Field(
        default_factory=dict
    )
    full_chain: Optional[INNChainDetail] = None
    # True when at least one substantive constant domain (CH1/CH2/CH3/CHS/CL/TCR)
    # was seen — False means only HINGE-REGION contributed to total_constant/c_alleles,
    # so c_allele and total_constant_mutations should be nulled on the full-length record.
    has_substantive_constant: bool = False


# ---------------------------------------------------------------------------
# Collection models
# ---------------------------------------------------------------------------


class ObservedChainCollection(BaseModel):
    """Collection of ObservedChain records serialisable to a flat DataFrame."""

    records: List[ObservedChain] = Field(default_factory=list)

    def to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame([r.model_dump(exclude=_OBJECT_COLS) for r in self.records])


class ObservedFullLengthCollection(BaseModel):
    records: List[ObservedFullLengthChain] = Field(default_factory=list)

    def to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame([r.model_dump() for r in self.records])


class IndividualMutationCollection(BaseModel):
    records: List[IndividualMutationInfo] = Field(default_factory=list)

    def to_dataframe(self) -> pd.DataFrame:
        df = pd.DataFrame([r.model_dump() for r in self.records])
        if df.empty:
            return df
        return df.set_index(
            [
                "inn",
                "inn_id",
                "chain_id",
                "chain_type",
                "gene",
                "allele",
                "gene_species",
                "region_iteration",
                "region_classification",
                "region",
                "position",
            ]
        ).sort_index()


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _mutation_str(mutations: List[MutationRecord], imgt_vars: bool = False) -> str:
    def _sort_key(m: MutationRecord):
        rc = m.repeat_region_count or 0
        parts = str(m.position).split(".")
        keys: list = [rc]
        for i in range(10):
            if i >= len(parts):
                keys.append("00000")
                continue
            p = parts[i]
            if i == 1 and imgt_vars:
                keys.append(str(-1 * int(p)).zfill(5))
            elif p.isdigit():
                keys.append(str(int(p)).zfill(5))
            else:
                keys.append(p.zfill(5))
        return tuple(keys)

    def _fmt(m: MutationRecord) -> str:
        if not hasattr(m, "mutation_type"):
            if not hasattr(m, "mutated_seq") or m.mutated_seq is None:
                return f"{m.position}{m.germline_seq}>-"
            # Legacy scraped records: mutated_seq already encodes the insertion marker.
            return f"{m.position}{m.germline_seq}>{m.mutated_seq}"
        if m.mutation_type == "deletion":
            return f"{m.position}{m.germline_seq}>-"
        if m.mutation_type == "insertion":
            # mutated_seq already carries "^<seq>" or "+" — use as-is.
            return f"{m.position}{m.germline_seq}>{m.mutated_seq}"
        return f"{m.position}{m.germline_seq}>{m.mutated_seq}"

    return ", ".join(_fmt(m) for m in sorted(mutations, key=_sort_key))


def _first_allele(raw: Optional[str]) -> Optional[str]:
    """Extract primary gene name from a scraped allele string like 'IGHV1-3*01 (89%)(human)...'."""
    if not raw:
        return None
    return raw.split(" (")[0].split(",")[0].strip() or None


_SPECIES_NORMALIZATION = {
    "homo sapiens": "human",
    "house mouse": "mouse",
    "mus musculus": "mouse",
    "domestic cat": "cat",
    "canis lupus familiaris": "dog",
}


def _normalize_species(raw: str) -> str:
    key = raw.lower().split("_")[0].strip()
    return _SPECIES_NORMALIZATION.get(key, raw)


def _extract_species(alleles: List[str]) -> List[str]:
    """Extract and normalize species labels from raw allele strings containing '%'."""
    cleaned = [x.split(",")[0] for x in alleles if x]
    return sorted(
        {
            _normalize_species(
                (x.split("%")[1].lstrip("(), ").split(")")[0]).strip("(), ")
            )
            for x in cleaned
            if "%" in x
        }
    )


def _fmt_numbered_muts(
    numbered: List[NumberedMutationRecord],
    raw: List[MutationRecord],
    system: str = "imgt",
) -> Optional[str]:
    """Format mutations using numbered positions when available, falling back to raw for IMGT."""
    attr = f"{system}_position"
    if numbered:
        muts = [getattr(nm, attr) or nm.position for nm in numbered]
        muts = [m for m in muts if m is not None]
    elif system == "imgt":
        muts = list(raw)
    else:
        return None
    seen: Set[Tuple] = set()
    deduped = [
        m
        for m in muts
        if (k := (str(m.position), m.germline_seq, m.mutated_seq)) not in seen
        and not seen.add(k)
    ]
    return _mutation_str(deduped) or None


# ---------------------------------------------------------------------------
# Mutation merging helpers
# ---------------------------------------------------------------------------


def _merge_sub_and_insertion(
    mutations: List[MutationRecord],
    inn_name: str = "",
    inn_id: str = "",
    chain_id: str = "",
    region: str = "",
) -> List[MutationRecord]:
    """Merge a substitution and an insertion that share the same position into one record.

    The realigner emits these as two separate ``MutationRecord`` entries when a
    query residue both replaces the germline AA *and* is followed immediately by
    inserted residues (alignment columns: ref=T/-/-, query=L/S/S).  The correct
    representation is a single record with:

    * ``mutation_type`` = ``"insertion"``
    * ``germline_seq``  = germline AA from the substitution (e.g. ``"T"``)
    * ``mutated_seq``   = substituted AA + ``^`` + inserted sequence
                          (e.g. ``"L^SS"``)

    Any position with exactly one substitution **and** one insertion is merged.
    Positions with any other combination are left unchanged.
    """
    import logging

    by_pos: Dict = defaultdict(list)
    for m in mutations:
        by_pos[m.position].append(m)

    result: List[MutationRecord] = []
    for pos, group in by_pos.items():
        subs = [m for m in group if m.mutation_type == "substitution"]
        ins = [m for m in group if m.mutation_type == "insertion"]
        rest = [
            m for m in group if m.mutation_type not in ("substitution", "insertion")
        ]

        if len(subs) == 1 and len(ins) == 1 and not rest:
            sub, ins_m = subs[0], ins[0]
            ins_seq = ins_m.insertion_str or ins_m.mutated_seq.lstrip("^")
            merged_mutated = f"{sub.mutated_seq}^{ins_seq}"
            merged = MutationRecord(
                position=pos,
                mutated_seq=merged_mutated,
                germline_seq=sub.germline_seq,
                mutation_type="insertion",
                insertion_str=ins_seq,
                mutation_region=sub.mutation_region or ins_m.mutation_region,
                repeat_region_count=sub.repeat_region_count,
                numbering_system=sub.numbering_system,
                original_position=sub.original_position,
            )
            logging.warning(
                "parse_collection [inn=%s inn_id=%s chain_id=%s region=%s]: "
                "merged substitution %s>%s and insertion %s>%s at position %s "
                "into single insertion record %s>%s",
                inn_name,
                inn_id,
                chain_id,
                region,
                sub.germline_seq,
                sub.mutated_seq,
                ins_m.germline_seq,
                ins_m.mutated_seq,
                pos,
                merged.germline_seq,
                merged.mutated_seq,
            )
            result.append(merged)
        else:
            result.extend(group)

    # Preserve original ordering
    pos_order = list(dict.fromkeys(m.position for m in mutations))
    result.sort(key=lambda m: pos_order.index(m.position))
    return result


# ---------------------------------------------------------------------------
# Main operation
# ---------------------------------------------------------------------------


def parse_collection(
    results: List[INNStructureResult],
    inn_map: Dict,
    constants_features: Optional[List[str]] = None,
    variable_features: Optional[List[str]] = None,
    ignore_regions: Optional[List[str]] = None,
) -> Tuple[
    ObservedChainCollection,
    ObservedFullLengthCollection,
    IndividualMutationCollection,
]:
    """Parse INNStructureResult objects into typed analytics collections.

    Numbered mutation positions (IMGT/EU/Kabat) are read directly from
    ``chain.mutations_by_individual_numbering_systems``, which is populated
    by ``update_with_realignment``.

    Args:
        results: From bundle_collection.results.
        inn_map: {inn_number (float): inn_name (str)}.
        constants_features: Feature types counted as constant. Default: C-DOMAIN, HINGE-REGION, CHS.
        variable_features: Feature types counted as variable. Default: V-DOMAIN.
        ignore_regions: Region annotations to skip in individual mutation output.

    Returns:
        (ObservedChainCollection, ObservedFullLengthCollection, IndividualMutationCollection)
    """
    constants_set = (
        set(constants_features)
        if constants_features is not None
        else _CONSTANTS_FEATURES
    )
    variable_set = (
        set(variable_features) if variable_features is not None else _VARIABLE_FEATURES
    )
    ignore_set = set(ignore_regions) if ignore_regions is not None else _IGNORE_REGIONS

    chains_rec: List[ObservedChain] = []
    full_length_rec: List[ObservedFullLengthChain] = []
    mutation_rec_constant_chains: List[IndividualMutationInfo] = []

    for result in results:
        inn_name = inn_map[float(result.inn_id)]
        aggregations: Dict[str, ChainAggregation] = {}
        unique_region_count: Dict[str, Dict[str, int]] = defaultdict(
            lambda: defaultdict(int)
        )
        chains_start = len(chains_rec)

        for chain in result.chains:
            mutations = _merge_sub_and_insertion(
                chain.final_mutations,
                inn_name=inn_name,
                inn_id=str(result.inn_id),
                chain_id=chain.chain_id,
                region=chain.region_annotation,
            )
            gene = (
                chain.realigned_gene
                if chain.realigned_mutations is not None and not chain.realigned_failed
                else chain.gene_name or chain.realigned_gene
            )
            germline_seq = (
                chain.realigned_germline_sequence
                if chain.realigned_germline_sequence and not chain.realigned_failed
                else chain.germline_sequence
            )

            unique_region_count[chain.chain_id][chain.region_annotation] += 1
            region_iter = unique_region_count[chain.chain_id][chain.region_annotation]

            if chain.chain_id not in aggregations:
                aggregations[chain.chain_id] = ChainAggregation(chain_id=chain.chain_id)
            agg = aggregations[chain.chain_id]
            agg.mutational_info_by_region.setdefault(chain.region_annotation, [])
            agg.mutational_info_by_region[chain.region_annotation].extend(mutations)

            if chain.chain_feature_type == "Full Length":
                if agg.full_chain is not None:
                    raise AssertionError(
                        f"Multiple Full Length chains for INN {result.inn_id} chain {chain.chain_id}"
                    )
                agg.full_chain = chain
            elif chain.chain_feature_type in constants_set:
                agg.total_constant.extend(mutations)
                agg.chain_types.append(chain.chain_type)
                raw_gene = (
                    chain.gene_name or gene
                )  # prefer raw IMGT string (has %species%) for extraction
                if raw_gene:
                    agg.constant_alleles.append(raw_gene)
                if ca := _first_allele(gene):
                    agg.c_alleles.append(ca)
                # HINGE-REGION alone is not a substantive constant domain — only
                # C-DOMAIN (CH1/CH2/CH3/TCR) and CHS qualify.
                if chain.chain_feature_type != "HINGE-REGION":
                    agg.has_substantive_constant = True
            elif chain.chain_feature_type in variable_set:
                agg.total_variable.extend(mutations)
                agg.chain_types.append(chain.chain_type)
                if chain.v_gene:
                    agg.variable_alleles.append(
                        chain.v_gene
                    )  # raw for species extraction
                if va := _first_allele(chain.v_gene):
                    agg.v_alleles.append(va)
                if ja := _first_allele(chain.j_gene):
                    agg.j_alleles.append(ja)
            else:
                if chain.chain_feature_type:
                    agg.other_features.append(chain.chain_feature_type)
                agg.other_mutations.extend(mutations)

            # Read pre-computed numbered mutations from chain directly.
            # Keep the full list for pos_lookup (EU/KABAT position resolution in
            # IndividualMutationInfo) but filter out merged positions for string
            # formatting so _fmt_numbered_muts falls back to the merged mutation string.
            _all_numbered = chain.mutations_by_individual_numbering_systems or []
            _chain_merged_pos = {
                int(m.position)
                for m in mutations
                if m.mutation_type == "insertion"
                and "^" in (m.mutated_seq or "")
                and m.mutated_seq[0] != "^"
            }
            numbered = [
                nm
                for nm in _all_numbered
                if int(nm.position.position) not in _chain_merged_pos
            ]
            if numbered:
                for ns_key in ("IMGT", "EU", "KABAT"):
                    agg.constant_chain_mutations_by_numeric_system.setdefault(
                        ns_key, []
                    )
                agg.constant_chain_mutations_by_numeric_system["IMGT"].extend(numbered)
                agg.constant_chain_mutations_by_numeric_system["EU"].extend(numbered)
                agg.constant_chain_mutations_by_numeric_system["KABAT"].extend(numbered)

            chains_rec.append(
                ObservedChain(
                    inn=inn_name,
                    inn_id=str(result.inn_id),
                    chain_id=chain.chain_id,
                    chain_detail_idx=chain.detail_order_idx,
                    chain_desc=chain.chain_description,
                    chain_type=chain.chain_type,
                    chain_feature_type=(
                        "C-DOMAIN"
                        if chain.chain_feature_type in {"CHS", "HINGE-REGION"}
                        else chain.chain_feature_type
                    ),
                    region=chain.region_annotation,
                    sequence=chain.sequence.replace(".", ""),
                    sequence_gapped=chain.sequence,
                    germline=germline_seq,
                    total_mutations=len(mutations),
                    v_allele=(
                        _first_allele(chain.v_gene)
                        if chain.chain_feature_type in variable_set
                        else None
                    ),
                    j_allele=(
                        _first_allele(chain.j_gene)
                        if chain.chain_feature_type in variable_set
                        else None
                    ),
                    c_allele=(
                        _first_allele(gene)
                        if chain.chain_feature_type in constants_set
                        else None
                    ),
                    c_allele_raw=(
                        chain.gene_name
                        if chain.chain_feature_type in constants_set
                        else None
                    ),
                    c_allele_species=(
                        ",".join(
                            _extract_species(
                                [chain.gene_name] if chain.gene_name else []
                            )
                        )
                        or _normalize_species(chain.realigned_species)
                        if chain.realigned_species
                        else ("" if chain.chain_feature_type in constants_set else "")
                    ),
                    failed_realignment=chain.realigned_failed,
                    alignment_fraction=(
                        chain.realigned_fraction_score
                        if not chain.realigned_failed
                        else None
                    ),
                    mutations=_mutation_str(mutations),
                    mutations_object=list(mutations),
                    scraped_mutations_object=chain.scraped_mutations,
                    realigned_mutations=chain.realigned_mutations,
                    constant_chain_mutations_IMGT=_fmt_numbered_muts(
                        numbered, mutations, "imgt"
                    ),
                    constant_chain_mutations_EU=_fmt_numbered_muts(
                        numbered, mutations, "eu"
                    ),
                    constant_chain_mutations_KABAT=_fmt_numbered_muts(
                        numbered, mutations, "kabat"
                    ),
                    chain_region_iteration=(region_iter if region_iter > 1 else None),
                )
            )

            # Individual mutation rows (constant domains only)
            if (
                chain.chain_feature_type in constants_set
                and chain.region_annotation not in ignore_set
            ):
                # Build position lookup: original position → {system: MutationRecord}
                # Use the full unfiltered numbered list so merged positions (sub+ins)
                # still resolve to their EU/KABAT coordinates.
                pos_lookup: Dict[int, Dict[str, object]] = defaultdict(dict)
                for nm in _all_numbered:
                    try:
                        orig = int(nm.position.position)
                    except (ValueError, TypeError):
                        continue
                    for sys_key, attr in (
                        ("IMGT", "imgt_position"),
                        ("EU", "eu_position"),
                        ("KABAT", "kabat_position"),
                        ("EXON", "exon_position"),
                    ):
                        mr = getattr(nm, attr)
                        if mr is not None:
                            pos_lookup[orig][sys_key] = mr.position

                for mutation in mutations:
                    try:
                        pos_int = int(mutation.position)
                    except (ValueError, TypeError):
                        pos_int = None

                    ns_pos = pos_lookup.get(pos_int, {}) if pos_int is not None else {}
                    mutation_rec_constant_chains.append(
                        IndividualMutationInfo(
                            inn=inn_name,
                            inn_id=str(result.inn_id),
                            chain_id=chain.chain_id,
                            chain_type=chain.chain_type,
                            region=chain.region_annotation,
                            region_classification=_C_CHAIN_CLASSIFIERS.get(
                                chain.region_annotation, ""
                            ),
                            region_iteration=str(
                                region_iter - 1 if region_iter > 1 else ""
                            ),
                            position=mutation.position,
                            mutation_type=mutation.mutation_type,
                            germline_seq=mutation.germline_seq,
                            mutated_seq=mutation.mutated_seq or "",
                            IMGT=ns_pos.get("IMGT"),
                            EU=ns_pos.get("EU"),
                            KABAT=ns_pos.get("KABAT"),
                            EXON=ns_pos.get("EXON"),
                            allele=_first_allele(gene),
                            gene=(_first_allele(gene) or "").split("*")[0],
                            gene_species=",".join(
                                _extract_species(
                                    [chain.gene_name] if chain.gene_name else []
                                )
                            )
                            or (
                                _normalize_species(chain.realigned_species)
                                if chain.realigned_species
                                else ""
                            ),
                        )
                    )

        # Back-fill species into chain + mutation records for this result
        for chain_id, agg in aggregations.items():
            var_sp = ",".join(_extract_species(agg.variable_alleles))
            const_sp = ",".join(_extract_species(agg.constant_alleles))
            for rec in chains_rec[chains_start:]:
                if rec.chain_id == chain_id:
                    rec.variable_chain_species = var_sp
                    rec.constant_chain_species = const_sp

        # Full-length aggregated records (one per chain_id)
        for chain_id, agg in aggregations.items():
            if agg.full_chain is None:
                raise AssertionError(
                    f"No Full Length chain for INN {result.inn_id} chain {chain_id}"
                )
            # For full-length, collect all numbered records across all regions for this chain.
            # Exclude numbered records for positions that were merged (sub+ins → single
            # insertion) so _fmt_numbered_muts falls back to the merged mutation string.
            merged_positions = {
                int(m.position)
                for m in agg.total_constant
                if m.mutation_type == "insertion"
                and "^" in (m.mutated_seq or "")
                and m.mutated_seq[0] != "^"  # starts with sub AA, not bare insertion
            }
            all_numbered = [
                nm
                for chain in result.chains
                if chain.chain_id == chain_id
                for nm in (chain.mutations_by_individual_numbering_systems or [])
                if int(nm.position.position) not in merged_positions
            ]
            # Null out c_allele and constant mutation counts when the only
            # constant-type domains were HINGE-REGION (no CH1/CH2/CH3/CHS/CL/TCR).
            substantive_c_allele = (
                ",".join(sorted(set(agg.c_alleles))) or None
                if agg.has_substantive_constant
                else None
            )
            substantive_constant_muts = (
                agg.total_constant if agg.has_substantive_constant else []
            )
            full_length_rec.append(
                ObservedFullLengthChain(
                    inn=inn_name,
                    inn_id=str(result.inn_id),
                    chain_id=agg.full_chain.chain_id,
                    chain_type=(
                        "-".join(sorted(set(agg.chain_types)))
                        if agg.chain_types
                        else agg.full_chain.chain_type
                    ),
                    sequence=agg.full_chain.sequence.replace(".", ""),
                    variable_chain_species=",".join(
                        _extract_species(agg.variable_alleles)
                    ),
                    constant_chain_species=",".join(
                        _extract_species(agg.constant_alleles)
                    ),
                    chain_description=agg.full_chain.chain_description.split("=")[
                        0
                    ].strip(),
                    germline_sequence=agg.full_chain.germline_sequence,
                    v_allele=",".join(sorted(set(agg.v_alleles))) or None,
                    j_allele=",".join(sorted(set(agg.j_alleles))) or None,
                    c_allele=substantive_c_allele,
                    total_mutations=len(substantive_constant_muts)
                    + len(agg.total_variable)
                    + len(agg.other_mutations),
                    total_constant_mutations=len(substantive_constant_muts),
                    total_mutations_unclassified_features=len(agg.other_mutations),
                    variable_mutations=_mutation_str(agg.total_variable),
                    constant_chain_mutations_IMGT=_fmt_numbered_muts(
                        all_numbered, substantive_constant_muts, "imgt"
                    ),
                    constant_chain_mutations_EU=_fmt_numbered_muts(
                        all_numbered, substantive_constant_muts, "eu"
                    ),
                    constant_chain_mutations_KABAT=_fmt_numbered_muts(
                        all_numbered, substantive_constant_muts, "kabat"
                    ),
                )
            )

    return (
        ObservedChainCollection(records=chains_rec),
        ObservedFullLengthCollection(records=full_length_rec),
        IndividualMutationCollection(records=mutation_rec_constant_chains),
    )


def build_constant_chain_pivot(
    observed_chains: "pd.DataFrame",
    columns: Optional[List[str]] = None,
    pivot_order: Optional[List[str]] = None,
    constants_features: Optional[List[str]] = None,
    ignore_regions: Optional[List[str]] = None,
) -> "pd.DataFrame":
    """Filter observed_chains to constant domains and pivot into a wide per-INN table.

    Args:
        observed_chains: DataFrame from ObservedChainCollection.to_dataframe().
        columns: Sequence columns to include per region group. Defaults to the standard 9-column set.
        pivot_order: Top-level column order in the output. Defaults to CH1/CL, HINGE, CH2-CH4, CHS.
        constants_features: Feature types to include. Defaults to C-DOMAIN, HINGE-REGION, CHS.
        ignore_regions: Region annotations to exclude. Defaults to C-ALPHA, C-BETA, C-BETA-2.

    Returns:
        Multi-column DataFrame indexed by [inn, inn_id, chain_id, chain_feature_type, region,
        chain_region_iteration], columns = (region_group, sequence_field).
    """
    constants_set = (
        set(constants_features)
        if constants_features is not None
        else _CONSTANTS_FEATURES
    )
    ignore_set = set(ignore_regions) if ignore_regions is not None else _IGNORE_REGIONS
    cols = columns if columns is not None else _PIVOT_COLUMNS
    order = pivot_order if pivot_order is not None else _PIVOT_ORDER

    df = observed_chains[
        observed_chains["chain_feature_type"].isin(constants_set)
        & ~observed_chains["region"].isin(ignore_set)
    ].copy()

    df["group_of_interest"] = df["region"].apply(
        lambda x: "CH1/C-KAPPA/C-LAMBDA" if x in _CH1_CKL_GROUP else x
    )
    df["sub_domain"] = df["region"].apply(
        lambda region: (
            "CK" if region in _CK_GROUP else ("CL" if region in _CL_GROUP else "V")
        )
    )

    pivot = pd.concat(
        {
            group: gdf.set_index(
                [
                    "inn",
                    "inn_id",
                    "chain_id",
                    "chain_type",
                    "sub_domain",
                    "chain_region_iteration",
                    "variable_chain_species",
                    "constant_chain_species",
                ]
            )[cols]
            for group, gdf in df.groupby("group_of_interest", dropna=False)
        },
        axis=1,
    )

    return pivot.reindex(columns=order, level=0).sort_index()


def motif_profile_table(
    mutational_profile_df: "pd.DataFrame",
    group_level: List[str] = [
        "chain_id",
        "chain_type",
        "region_iteration",
        "region_classification",
    ],
) -> "pd.DataFrame":
    """Collapse per-position mutation rows into a motif string per region group.

    Consecutive deletions are merged (e.g. 216E(del),217P(del) → 216EP(del)).
    Substitutions are formatted as {ref}{pos}{alt} (e.g. K87R).
    Substitutions are formatted as {ref}{pos}{alt} (e.g. K87R).
    Insertions are formatted as {ref}{pos}^{ins_seq}(ins) (e.g. K447^PTLYN(ins)).

    Args:
        mutational_profile_df: From IndividualMutationCollection.to_dataframe().
        group_level: Index levels to group by. Defaults to
            ["chain_id", "chain_type", "region_iteration", "region_classification"].

    Returns:
        DataFrame with the original grouped columns and an additional column
        "constant_chain_mutational_motif" containing the motif string.
    """

    grouped = mutational_profile_df.groupby(level=group_level, dropna=False).agg(
        {
            "EU": list,
            "mutation_type": list,
            "germline_seq": list,
            "mutated_seq": list,
        }
    )
    motif = grouped.apply(_collapse_mutation_group, axis=1).rename(
        "constant_chain_mutational_motif"
    )
    table = pd.concat([grouped, motif], axis=1)
    table["num_mutations"] = table["mutated_seq"].apply(len)
    return table


def _collapse_mutation_group(row: "pd.Series") -> str:
    """Collapse one group of per-position mutations into a motif string."""
    subs: List[Tuple] = []
    insertions: Dict = defaultdict(list)
    deletions: List[Tuple] = []

    for eu_pos, mut_type, germ, mut in zip(
        row["EU"],
        row["mutation_type"],
        row["germline_seq"],
        row["mutated_seq"],
    ):
        sort_key = _eu_sort_key(eu_pos)
        if mut_type == "substitution":
            subs.append((sort_key, f"{germ}{eu_pos}{mut}"))
        elif mut_type == "deletion":
            deletions.append((sort_key, eu_pos, germ))
        elif mut_type == "insertion":
            insertions[(sort_key, eu_pos, germ)].append(mut)

    # merge consecutive deletions
    del_motifs: List[Tuple] = []
    current_key = current_pos_str = None
    current_last = -10.0
    del_seq = ""
    for sort_key, eu_pos, germ in sorted(deletions, key=lambda x: x[0]):
        numeric = float(sort_key[0])
        if current_key is not None and numeric - current_last <= 1:
            del_seq += germ
        else:
            if current_key is not None:
                del_motifs.append((current_key, f"{current_pos_str}{del_seq}(del)"))
            current_key, current_pos_str, del_seq = sort_key, eu_pos, germ
        current_last = numeric
    if current_key is not None:
        del_motifs.append((current_key, f"{current_pos_str}{del_seq}(del)"))

    ins_motifs = [
        (
            sort_key,
            f"{germ}{eu_pos}^{''.join(s.lstrip('^') if s.startswith('^') else s for s in seqs)}(ins)",
        )
        for (sort_key, eu_pos, germ), seqs in insertions.items()
    ]

    all_muts = sorted(subs + del_motifs + ins_motifs, key=lambda x: x[0])
    return ",".join(m[1] for m in all_muts)


def _eu_sort_key(eu_pos) -> Tuple:
    s = str(eu_pos)
    if s.isdigit():
        return (int(s), "")
    if ".i" in s:
        base, ins = s.split(".i")
        return (float(base), ins)
    return (float(s), "")
