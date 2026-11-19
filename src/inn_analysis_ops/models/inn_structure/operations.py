"""Business logic operating on INNStructure data models.

Separated from inn_analysis_ops.models.py so that imports from realigner.py and
numbering modules can be added without creating circular imports.
"""

import re
from collections import defaultdict, deque
from pathlib import Path
from typing import Deque, Dict, List, Optional, Set, Tuple, Union

import pandas as pd

from ..mutation_models import (
    MutationRecord,
    NumberedMutationRecord,
)
from ..numbering.conversion import convert_mutation_positioning
from ..numbering.core import (
    ConstantChainNumbering,
    NumberingTypes,
)

from .analytics import _fmt_numbered_muts, _mutation_str
from .chain_classifier import (
    ArmBinderType,
    VhDomainType,
    VlDomainType,
    classify_arm_binder_type,
    classify_vh_domain,
    classify_vl_domain,
    compute_region_ownership,
)
from .models import (
    BinderChain,
    BinderGroup,
    BindingArm,
    INNChainDetail,
    INNStructureCollection,
    INNStructureResult,
    RealignedBundle,
)
from .realigner import RegionMutations

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_VL_SET = {"V-KAPPA", "V-LAMBDA"}
_CL_SET = {"C-KAPPA", "C-LAMBDA"}
_SIG = _VL_SET | _CL_SET | {"VH", "CH1", "CH2", "CH3"}

_VH_ANN = "VH"
_VL_ANNS = {"V-KAPPA", "V-LAMBDA"}
_CL_ANNS = {"C-KAPPA", "C-LAMBDA"}
_REGION_TO_COL: Dict[str, str] = {
    "CH1": "ch1",
    "CH2": "ch2",
    "CH3": "ch3",
    "CH4": "ch4",
    "CHS": "chs",
    "Linker": "linker",
    "HINGE-REGION": "hinge_region",
    "C-ALPHA": "tcr_alpha",
    "C-BETA-2": "tcr_beta",
}
_HEAVY_CONSTANT_ANNS = {
    "CH1",
    "CH2",
    "CH3",
    "CH4",
    "HINGE-REGION",
    "CHS",
    "C-ALPHA",
    "C-BETA-2",
}
_CONST_TRACK_ANNS = (
    (_HEAVY_CONSTANT_ANNS | _CL_ANNS)
    - {
        "HINGE-REGION",
        "CHS",
        "C-ALPHA",  # TCR constant — excluded from coverage validation (ownership via VH group)
        "C-BETA-2",
    }
)

_ALL_NUMBERING: List[NumberingTypes] = [
    NumberingTypes.IMGT,
    NumberingTypes.EU,
    NumberingTypes.KABAT,
    NumberingTypes.EXON,
]
_HINGE_GENE_RE = re.compile(r"([0-9])[a-zA-Z]+$")

# Type aliases used in _build_binding_arms
_OwnershipEntry = Tuple[Optional[INNChainDetail], int]
_OwnershipMap = Dict[int, _OwnershipEntry]
_ArmSpec = Tuple[
    Optional[INNChainDetail],
    Optional[INNChainDetail],
    Optional[str],
    Optional[str],
]


# ---------------------------------------------------------------------------
# Numbering helpers
# ---------------------------------------------------------------------------


def _numbering_region_key(
    chain: INNChainDetail, cn: ConstantChainNumbering
) -> Optional[str]:
    if chain.region_annotation == "HINGE-REGION":
        gene = chain.realigned_gene or chain.gene_name
        if not gene:
            return None
        gene_class = _HINGE_GENE_RE.sub(r"\1", gene.split("*")[0].upper())
        key = f"HINGE-{gene_class}"
        return key if key in cn.regions else None
    return chain.region_annotation if chain.region_annotation in cn.regions else None


def _build_numbered_mutations(
    mutations: List[MutationRecord],
    region: str,
    cn: ConstantChainNumbering,
    region_repeat: int,
) -> Optional[List[NumberedMutationRecord]]:
    by_system = convert_mutation_positioning(
        region, mutations, _ALL_NUMBERING, cn, region_repeat
    )
    if not by_system:
        return None
    result = []
    for orig in mutations:
        kwargs: Dict[str, Optional[MutationRecord]] = {"position": orig}
        for nt, mut_list in by_system.items():
            matched = next(
                (m for m in mut_list if m.original_position == int(orig.position)),
                None,
            )
            kwargs[f"{str(nt).lower()}_position"] = matched
        result.append(NumberedMutationRecord(**kwargs))
    return result or None


# ---------------------------------------------------------------------------
# Arm-building helpers
# ---------------------------------------------------------------------------

_ClassifyResult = Tuple[
    List[Tuple[INNChainDetail, INNChainDetail]],
    List[INNChainDetail],
    List[INNChainDetail],
]


def _classify_subunits(chains: List[INNChainDetail]) -> _ClassifyResult:
    def _next_sig(start: int) -> Optional[int]:
        j = start
        while j < len(chains) and chains[j].region_annotation not in _SIG:
            j += 1
        return j if j < len(chains) else None

    _CH_SIG = {"CH1", "CH2", "CH3"}
    scfv_pairs: List[Tuple[INNChainDetail, INNChainDetail]] = []
    anchored_vls: List[INNChainDetail] = []
    orphan_vhs: List[INNChainDetail] = []
    used: Set[int] = set()
    for i, c in enumerate(chains):
        if i in used:
            continue
        ann = c.region_annotation
        if ann == "VH":
            j = _next_sig(i + 1)
            if j is not None and chains[j].region_annotation in _CH_SIG:
                orphan_vhs.append(c)
                used.add(i)
            elif j is not None and chains[j].region_annotation in _VL_SET:
                k = _next_sig(j + 1)
                if k is not None and chains[k].region_annotation in _CL_SET:
                    orphan_vhs.append(c)
                    anchored_vls.append(chains[j])
                    used |= {i, j, k}
                else:
                    scfv_pairs.append((c, chains[j]))
                    used |= {i, j}
            else:
                orphan_vhs.append(c)
                used.add(i)
        elif ann in _VL_SET:
            j = _next_sig(i + 1)
            if j is not None and chains[j].region_annotation in _CL_SET:
                anchored_vls.append(c)
                used |= {i, j}
            elif j is not None and chains[j].region_annotation == "VH":
                scfv_pairs.append((chains[j], c))
                used |= {i, j}
            else:
                anchored_vls.append(c)
                used.add(i)
    return scfv_pairs, anchored_vls, orphan_vhs


def _owned_annotations(
    var_detail: INNChainDetail,
    chain_id: str,
    details_by_chain_id: Dict[str, List[INNChainDetail]],
    ownership_map: _OwnershipMap,
) -> Set[str]:
    _, grp = ownership_map.get(id(var_detail), (None, -1))
    if grp < 0:
        return set()
    result: Set[str] = set()
    for d in details_by_chain_id.get(chain_id, []):
        if d is var_detail:
            continue
        _, dgrp = ownership_map.get(id(d), (None, -1))
        if dgrp == grp:
            result.add(d.region_annotation)
    return result


def _build_binding_arms(
    inn_result: INNStructureResult,
) -> List[BindingArm]:
    _VhT = VhDomainType
    _VlT = VlDomainType

    _CONST_GENE_RE = re.compile(r"^IG[LK]C", re.IGNORECASE)
    _VL_TO_CL = {"V-LAMBDA": "C-LAMBDA", "V-KAPPA": "C-KAPPA"}

    _CANONICAL_VL = {_VlT.VKAPPA, _VlT.VLAMBDA, _VlT.VK_CL, _VlT.VL_CK}
    _FAB_VH = {_VhT.VH_FAB, _VhT.VH_CK, _VhT.VH_CL}
    _FAB_VL = {
        _VlT.VKAPPA,
        _VlT.VLAMBDA,
        _VlT.VK_CL,
        _VlT.VL_CK,
        _VlT.VK_CH1,
        _VlT.VL_CH1,
    }
    _SWAPPED_VL = {_VlT.VK_CH1_FC, _VlT.VL_CH1_FC}
    _SWAPPED_VH = {_VhT.VH_CK, _VhT.VH_CL}
    _ORPHAN_VH = {_VhT.VH_ORPHAN}
    _ORPHAN_VL = {_VlT.VK_ONLY, _VlT.VL_ONLY}
    _arm_skip = {
        "VH",
        "V-KAPPA",
        "V-LAMBDA",
        "V-ALPHA",  # TCR variable — not an Ig variable domain; skip from constant tracking
        "V-BETA",
        "Full Length",
        "N.D",
        "LINKER",
    }
    _ignore_chain_type = {
        "G-ALPHA1-LIKE",
        "G-BETA-LIKE",
        "G-GAMMA-LIKE",
        "CFH",
        "CYTOPLASMIC-REGION",
        "RLN2",
        "SGSH",
        "NRG1",
        "TNFSF13B",
    }

    def _vht(d: INNChainDetail) -> Optional[VhDomainType]:
        return vh_domain_types.get(id(d))

    def _vlt(d: INNChainDetail) -> Optional[VlDomainType]:
        return vl_domain_types.get(id(d))

    def _pair_pass(
        vh_filter: Set[VhDomainType],
        vl_filter: Set[VlDomainType],
        cross_chain_only: bool = True,
    ) -> None:
        for vi, (vh_d, vh_cid) in enumerate(free_vhs):
            if vi in paired_vh or _vht(vh_d) not in vh_filter:
                continue
            for li, (vl_d, vl_cid) in enumerate(free_vls):
                if li in paired_vl or _vlt(vl_d) not in vl_filter:
                    continue
                if cross_chain_only and vh_cid == vl_cid:
                    continue
                cross_chain_arms.append((vh_d, vl_d, vh_cid, vl_cid))
                if not has_shared_heavy:
                    paired_vh.add(vi)
                if not has_shared_vl:
                    paired_vl.add(li)
                if not (has_shared_heavy or has_shared_vl):
                    break

    def _make_binder_chain(var_d: INNChainDetail, cid: str) -> BinderChain:
        omap = ownership_by_chain_id[cid]
        _, grp = omap.get(id(var_d), (None, -1))
        return BinderChain(
            chain_id=cid,
            variable_domain_id=var_d.detail_order_idx,
            variable_type=("VH" if var_d.region_annotation == "VH" else "VL"),
            constant_domain_idxs=[
                d.detail_order_idx
                for d in details_by_chain_id[cid]
                if omap.get(id(d), (None, -1))[1] == grp
                and d is not var_d
                and d.region_annotation not in _arm_skip
            ],
            variable_sequence=var_d.sequence,
        )

    def _arm_sort_key(arm: _ArmSpec) -> str:
        vh_d, vl_d, _, _ = arm
        d = vh_d or vl_d
        return d.detail_order_idx if d else ""

    def _gapless_or_none(seq: Optional[str]) -> Optional[str]:
        return seq.replace(".", "") if seq else None

    # -- Step 1: Group chain details by physical polypeptide ----------------
    # currently inn chain detail are domains mapped to a polypeptide chain
    details_by_chain_id: Dict[str, List[INNChainDetail]] = defaultdict(list)
    for detail in inn_result.chains:
        details_by_chain_id[detail.chain_id].append(detail)

    # Fix mislabeled V-regions with constant-gene names
    for details in details_by_chain_id.values():
        for i, d in enumerate(details):
            if (
                d.region_annotation in _VL_TO_CL
                and d.gene_name
                and _CONST_GENE_RE.search(d.gene_name)
            ):
                details[i] = d.model_copy(
                    update={"region_annotation": _VL_TO_CL[d.region_annotation]}
                )

    # -- Step 2: Blueprint + ownership per chain_id -------------------------
    ownership_by_chain_id: Dict[str, _OwnershipMap] = {}
    for chain_id, details in details_by_chain_id.items():
        ownership_by_chain_id[chain_id] = compute_region_ownership(details)

    # -- Step 3: Classify variable domains per chain_id ---------------------
    scfv_arms: List[_ArmSpec] = []
    free_vls: List[Tuple[INNChainDetail, str]] = []
    free_vhs: List[Tuple[INNChainDetail, str]] = []

    for chain_id, details in details_by_chain_id.items():
        scfv_pairs, anchored, orphans = _classify_subunits(details)
        scfv_arms.extend((vh, vl, chain_id, chain_id) for vh, vl in scfv_pairs)
        free_vls.extend((vl, chain_id) for vl in anchored)
        free_vhs.extend((vh, chain_id) for vh in orphans)

    # -- Step 3b: Classify individual VH/VL domain types --------------------
    vh_domain_types: Dict[int, VhDomainType] = {}
    vl_domain_types: Dict[int, VlDomainType] = {}
    for vh_d, vl_d, cid, _ in scfv_arms:
        vh_domain_types[id(vh_d)] = classify_vh_domain(
            _owned_annotations(
                vh_d, cid, details_by_chain_id, ownership_by_chain_id[cid]
            ),
            is_scfv=True,
        )
        vl_domain_types[id(vl_d)] = classify_vl_domain(
            vl_d.region_annotation,
            _owned_annotations(
                vl_d, cid, details_by_chain_id, ownership_by_chain_id[cid]
            ),
            is_scfv=True,
        )
    for vh_d, cid in free_vhs:
        vh_domain_types[id(vh_d)] = classify_vh_domain(
            _owned_annotations(
                vh_d, cid, details_by_chain_id, ownership_by_chain_id[cid]
            )
        )
    for vl_d, cid in free_vls:
        vl_domain_types[id(vl_d)] = classify_vl_domain(
            vl_d.region_annotation,
            _owned_annotations(
                vl_d, cid, details_by_chain_id, ownership_by_chain_id[cid]
            ),
        )

    # -- Step 4: Pair free VHs with free VLs --------------------------------

    _chains_with_vh = {cid for _, cid in free_vhs} | {cid for _, _, cid, _ in scfv_arms}
    free_vls.sort(key=lambda x: (x[1] in _chains_with_vh, x[0].detail_order_idx))

    paired_vh: Set[int] = set()
    paired_vl: Set[int] = set()
    cross_chain_arms: List[_ArmSpec] = []

    _vh_cids = {cid for _, cid in free_vhs}
    _vl_cids = {cid for _, cid in free_vls}
    has_shared_heavy = len(free_vhs) == 1 and len(free_vls) >= 2 and len(_vl_cids) >= 2
    has_shared_vl = len(free_vls) == 1 and len(free_vhs) >= 2 and len(_vh_cids) >= 2

    _pair_pass({_VhT.IG_FULL_LENGTH}, _CANONICAL_VL)
    _pair_pass(_FAB_VH, _FAB_VL)
    _pair_pass(_SWAPPED_VH, _SWAPPED_VL)
    _pair_pass(_ORPHAN_VH, _ORPHAN_VL)

    for vi, (vh_d, vh_cid) in enumerate(free_vhs):
        if vi not in paired_vh:
            cross_chain_arms.append((vh_d, None, vh_cid, None))
    for li, (vl_d, vl_cid) in enumerate(free_vls):
        if li not in paired_vl:
            cross_chain_arms.append((None, vl_d, None, vl_cid))

    cross_chain_arms.sort(key=_arm_sort_key)
    all_arms: List[_ArmSpec] = scfv_arms + cross_chain_arms

    # -- Build BinderGroups -----------------------------------------------
    detail_by_idx: Dict[str, INNChainDetail] = {
        d.detail_order_idx: d
        for details in details_by_chain_id.values()
        for d in details
    }

    binder_groups: List[BinderGroup] = [
        BinderGroup(
            inn_id=inn_result.inn_id,
            chains=[
                _make_binder_chain(var_d, cid)
                for var_d, cid in [(vh_d, vh_cid), (vl_d, vl_cid)]
                if var_d is not None
            ],
        )
        for vh_d, vl_d, vh_cid, vl_cid in all_arms
    ]

    # Validate: all CH/CL on chains-with-variables appear in some BinderGroup
    _chains_with_vars = {
        cid
        for cid, details in details_by_chain_id.items()
        if any(d.region_annotation in {_VH_ANN} | _VL_ANNS for d in details)
    }
    _all_tracked = {
        d.detail_order_idx
        for cid in _chains_with_vars
        for d in details_by_chain_id[cid]
        if d.region_annotation in _CONST_TRACK_ANNS
    }
    _covered = {
        idx
        for bg in binder_groups
        for bc in bg.chains
        for idx in bc.constant_domain_idxs
        if detail_by_idx[idx].region_annotation in _CONST_TRACK_ANNS
    }
    if _all_tracked - _covered:
        raise ValueError(
            f"INN {inn_result.inn_id}: constant domains not in any binder group: "
            f"{_all_tracked - _covered}"
        )

    # -- Step 5: Emit one raw row per arm -----------------------------------
    raw_rows: List[Dict[str, object]] = []

    for bg in binder_groups:
        # at this point binder group repressents a compilation of vh/vl or domains that recognize a unique antigen, so we can classify the arm binder type based on the domain types of the variable domains
        # for reach binder group we have a vh/vl and a list of constant chain ids that should be associated with it
        # so now wwe can loop through each binder chain and emit a row of information about this fv combination
        vh_d = detail_by_idx[bg.vh.variable_domain_id] if bg.vh else None
        vl_d = detail_by_idx[bg.vl.variable_domain_id] if bg.vl else None
        vh_chain_id = bg.vh.chain_id if bg.vh else None
        vl_chain_id = bg.vl.chain_id if bg.vl else None
        is_scfv = vh_chain_id is not None and vh_chain_id == vl_chain_id
        vh_first = is_scfv and bool(
            vh_d and vl_d and vh_d.detail_order_idx < vl_d.detail_order_idx
        )
        vh_dt = vh_domain_types.get(id(vh_d)) if vh_d else None
        vl_dt = vl_domain_types.get(id(vl_d)) if vl_d else None
        arm_bt = classify_arm_binder_type(
            vh_dt, vl_dt, is_scfv=is_scfv, vh_first=vh_first
        )

        all_const_details = [
            detail_by_idx[idx] for bc in bg.chains for idx in bc.constant_domain_idxs
        ]
        _ch_details = [
            d for d in all_const_details if d.region_annotation in _HEAVY_CONSTANT_ANNS
        ]
        _cl_details = [d for d in all_const_details if d.region_annotation in _CL_ANNS]
        _other_details = [
            d
            for d in all_const_details
            if d.region_annotation in _REGION_TO_COL
            and d.region_annotation not in _HEAVY_CONSTANT_ANNS | _CL_ANNS
        ]
        _fl_details: List[Tuple[INNChainDetail, bool]] = [
            (d, bc.variable_type == "VH")
            for bc in bg.chains
            for d in details_by_chain_id[bc.chain_id]
            if d.region_annotation == "Full Length"
        ]
        unrecognised_annotations: Set[str] = {
            d.region_annotation
            for d in all_const_details
            if d.region_annotation
            not in _HEAVY_CONSTANT_ANNS | _CL_ANNS | set(_REGION_TO_COL)
        }

        for d in (
            ([vh_d] if vh_d else []) + ([vl_d] if vl_d else []) + all_const_details
        ):
            if not d.sequence:
                if d.region_annotation in _ignore_chain_type:
                    continue
                raise AssertionError(
                    f"INN {inn_result.inn_id}: empty sequence for {d.region_annotation} "
                    f"(detail {d.detail_order_idx})"
                )

        row: Dict[str, object] = {
            "inn_id": inn_result.inn_id,
            "arm_binder_type": arm_bt.value,
            "vh_domain_type": vh_dt.value if vh_dt else None,
            "vl_domain_type": vl_dt.value if vl_dt else None,
            "shared_vl": has_shared_vl,
            "shared_heavy": has_shared_heavy,
            "cl_chain_detail_order_idxs": [],
            "ch_chain_detail_order_idxs": [],
        }

        if vh_d:
            row.update(
                {
                    "vh": vh_d.sequence,
                    "vh_v_gene": vh_d.v_gene,
                    "vh_d_gene": vh_d.d_gene,
                    "vh_j_gene": vh_d.j_gene,
                    "vh_dga_url": vh_d.domain_gap_align_url,
                    "vh_total_mutations": len(vh_d.final_mutations) or None,
                    "vh_mutations": _mutation_str(vh_d.final_mutations),
                    "vh_chain_id": vh_d.chain_id,
                    "vh_chain_id_order_idx": vh_d.detail_order_idx,
                }
            )
        if vl_d:
            row.update(
                {
                    "vl": vl_d.sequence,
                    "vl_v_gene": vl_d.v_gene,
                    "vl_j_gene": vl_d.j_gene,
                    "vl_dga_url": vl_d.domain_gap_align_url,
                    "vl_total_mutations": len(vl_d.final_mutations) or None,
                    "vl_mutations": _mutation_str(vl_d.final_mutations),
                    "vl_chain_id": vl_d.chain_id,
                    "vl_chain_id_order_idx": vl_d.detail_order_idx,
                }
            )

        const_seqs: Dict[str, List[str]] = defaultdict(list)
        _ch_raw_muts: List[MutationRecord] = []
        _ch_numbered_muts: List[NumberedMutationRecord] = []
        _cl_raw_muts: List[MutationRecord] = []
        _cl_numbered_muts: List[NumberedMutationRecord] = []

        for d in _ch_details:
            col = _REGION_TO_COL.get(d.region_annotation)
            if col:
                const_seqs[col].append(d.sequence)
                if d.gene_name:
                    row[f"{col}_gene"] = d.gene_name
                mut_count = len(d.final_mutations) or None
                if mut_count is not None:
                    row[f"{col}_total_mutations"] = mut_count
            row["ch_chain_detail_order_idxs"].append(d.detail_order_idx)
            _ch_raw_muts.extend(d.final_mutations)
            if d.mutations_by_individual_numbering_systems:
                _ch_numbered_muts.extend(d.mutations_by_individual_numbering_systems)

        # If _ch_details contains only HINGE-REGION (no CH1/CH2/CH3/CHS/CL/TCR),
        # null out the aggregate mutation lists — mirroring the analytics.py rule.
        # The hinge sequence and hinge_region_total_mutations fields are kept.
        _has_substantive_ch = any(
            d.region_annotation != "HINGE-REGION" for d in _ch_details
        )
        if not _has_substantive_ch:
            _ch_raw_muts = []
            _ch_numbered_muts = []

        for d in _cl_details:
            const_seqs["cl"].append(d.sequence)
            row["cl_chain_detail_order_idxs"].append(d.detail_order_idx)
            if "cl_chain_id" not in row:
                row["cl_chain_id"] = d.chain_id
                row["cl_type"] = (
                    "kappa" if d.region_annotation == "C-KAPPA" else "lambda"
                )
                row["cl_realigned_gene"] = d.realigned_gene
                row["cl_realigned_failed"] = d.realigned_failed
                row["cl_realigned_fraction_score"] = d.realigned_fraction_score
            if "cl_total_mutations" not in row:
                row["cl_total_mutations"] = len(d.final_mutations) or None
            _cl_raw_muts.extend(d.final_mutations)
            if d.mutations_by_individual_numbering_systems:
                _cl_numbered_muts.extend(d.mutations_by_individual_numbering_systems)

        for d, is_heavy in _fl_details:
            row["full_length_heavy" if is_heavy else "full_length_light"] = d.sequence

        for d in _other_details:
            const_seqs[_REGION_TO_COL[d.region_annotation]].append(d.sequence)

        row["_const_seqs"] = dict(const_seqs)
        row["_ch_raw_muts"] = _ch_raw_muts
        row["_ch_numbered_muts"] = _ch_numbered_muts
        row["_cl_raw_muts"] = _cl_raw_muts
        row["_cl_numbered_muts"] = _cl_numbered_muts
        row["_unrecognised"] = unrecognised_annotations
        raw_rows.append(row)

    # -- Step 6: Group by (VH seq, VL seq) and merge -----------------------
    # Cases where two structurally independent BinderGroups happen to have the same VH+VL sequences.
    # The classic case is a symmetric IgG: two identical HC+LC pairs. They produce two separate binder groups (different physical chains A+B vs C+D), but both emit a row with the same VH and VL sequences. Without Step 6 you'd get two identical rows in the output.
    # Step 6 collapses them into one row and merges the ch_chain_detail_order_idxs / cl_chain_detail_order_idxs lists, so the final BindingArm records that both copies of CH1/CH2/CH3 exist.
    # The absorbed_keys re.sub-block handles a related case: an arm that only has VH (or only VL) gets absorbed into the paired arm that shares the same sequence, rather than emitting as a standalone half-arm row.

    groups: Dict[Tuple[Optional[str], Optional[str]], List[Dict[str, object]]] = (
        defaultdict(list)
    )
    for row in raw_rows:
        key = (
            _gapless_or_none(row.get("vh")),
            _gapless_or_none(row.get("vl")),
        )
        groups[key].append(row)

    absorbed_keys: Set[Tuple[Optional[str], Optional[str]]] = set()
    for key in list(groups):
        vh_k, vl_k = key
        if vh_k is not None and vl_k is None:
            target = next(
                (k for k in groups if k[0] == vh_k and k[1] is not None),
                None,
            )
            if target:
                groups[target].extend(groups[key])
                absorbed_keys.add(key)
        elif vh_k is None and vl_k is not None:
            target = next(
                (k for k in groups if k[1] == vl_k and k[0] is not None),
                None,
            )
            if target:
                groups[target].extend(groups[key])
                absorbed_keys.add(key)
    for key in absorbed_keys:
        del groups[key]

    rows: List[BindingArm] = []
    arm_slot_idx = 0
    for group_rows in groups.values():
        primary = group_rows[0]

        merged_const: Dict[str, List[str]] = defaultdict(list)
        merged_ch_idxs: List[str] = []
        merged_cl_idxs: List[str] = []
        merged_unrecognised: Set[str] = set()

        for r in group_rows:
            for col, seqs in r.get("_const_seqs", {}).items():
                merged_const[col].extend(seqs)
            merged_ch_idxs.extend(r["ch_chain_detail_order_idxs"])
            merged_cl_idxs.extend(r["cl_chain_detail_order_idxs"])
            merged_unrecognised.update(r.get("_unrecognised", set()))

        out: Dict[str, object] = {
            "inn_id": primary["inn_id"],
            "arm_slot": chr(ord("a") + arm_slot_idx),
            "arm_binder_type": primary.get("arm_binder_type"),
            "vh_domain_type": primary.get("vh_domain_type"),
            "vl_domain_type": primary.get("vl_domain_type"),
            "shared_vl": primary.get("shared_vl", False),
            "shared_heavy": primary.get("shared_heavy", False),
            "duplicate_vh": len(group_rows) > 1,
        }

        for k in (
            "vh",
            "vh_v_gene",
            "vh_d_gene",
            "vh_j_gene",
            "vh_dga_url",
            "vh_total_mutations",
            "vh_mutations",
            "vh_chain_id",
            "vh_chain_id_order_idx",
            "vl",
            "vl_v_gene",
            "vl_j_gene",
            "vl_dga_url",
            "vl_total_mutations",
            "vl_mutations",
            "vl_chain_id",
            "vl_chain_id_order_idx",
            "cl_type",
            "cl_total_mutations",
            "cl_chain_id",
            "cl_realigned_gene",
            "cl_realigned_failed",
            "cl_realigned_fraction_score",
            "ch1_total_mutations",
            "ch1_gene",
            "ch2_total_mutations",
            "ch2_gene",
            "ch3_total_mutations",
            "ch3_gene",
            "chs_total_mutations",
            "chs_gene",
            "tcr_alpha_total_mutations",
            "tcr_alpha_gene",
            "tcr_beta_total_mutations",
            "tcr_beta_gene",
            "hinge_region_total_mutations",
            "full_length_heavy",
            "full_length_light",
        ):
            # Primary row first; fall back to any other row in the group that
            # has this key.  This matters for tandem-VHH-Fc formats where VH1
            # (primary, no constants) and VH2 (has CH2/CH3) share the same VH
            # sequence and get collapsed — without the fallback, per-domain
            # gene/mutation metadata from VH2 would be silently dropped.
            val = primary.get(k)
            if val is None:
                for r in group_rows[1:]:
                    val = r.get(k)
                    if val is not None:
                        break
            if val is not None:
                out[k] = val

        for col in (
            "ch1",
            "ch2",
            "ch3",
            "chs",
            "tcr_alpha",
            "tcr_beta",
            "hinge_region",
            "cl",
        ):
            seqs = merged_const.get(col)
            if not seqs:
                continue
            unique = list(dict.fromkeys(seqs))
            out[col] = "::".join(unique)
            out[f"unique_{col}_count"] = len(unique)
        if merged_const.get("linker"):
            out["linker"] = merged_const["linker"][-1]

        out["ch_chain_detail_order_idxs"] = ";".join(sorted(set(merged_ch_idxs)))
        out["cl_chain_detail_order_idxs"] = ";".join(sorted(set(merged_cl_idxs)))
        if merged_unrecognised:
            out["other_annotations_skipped"] = ";".join(sorted(merged_unrecognised))

        all_ch_raw: List[MutationRecord] = [
            m for r in group_rows for m in r.get("_ch_raw_muts", [])
        ]
        all_ch_num: List[NumberedMutationRecord] = [
            m for r in group_rows for m in r.get("_ch_numbered_muts", [])
        ]
        all_cl_raw: List[MutationRecord] = [
            m for r in group_rows for m in r.get("_cl_raw_muts", [])
        ]
        all_cl_num: List[NumberedMutationRecord] = [
            m for r in group_rows for m in r.get("_cl_numbered_muts", [])
        ]
        out["ch_mutations_imgt"] = _fmt_numbered_muts(
            all_ch_num, all_ch_raw, system="imgt"
        )
        out["ch_mutations_eu"] = _fmt_numbered_muts(all_ch_num, all_ch_raw, system="eu")
        out["ch_mutations_kabat"] = _fmt_numbered_muts(
            all_ch_num, all_ch_raw, system="kabat"
        )
        out["cl_mutations_imgt"] = _fmt_numbered_muts(
            all_cl_num, all_cl_raw, system="imgt"
        )
        out["cl_mutations_eu"] = _fmt_numbered_muts(all_cl_num, all_cl_raw, system="eu")
        out["cl_mutations_kabat"] = _fmt_numbered_muts(
            all_cl_num, all_cl_raw, system="kabat"
        )
        out["ch_mutations"] = _mutation_str(all_ch_raw) or None
        out["cl_mutations"] = _mutation_str(all_cl_raw) or None
        rows.append(BindingArm(**out))
        arm_slot_idx += 1

    return rows


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def _build_no_fv_arm_row(inn_result: INNStructureResult) -> Optional[Dict]:
    """Build a single arm_df-schema row for an INN that has no VH/VL variable domains.

    Used when include_sequences_without_fv=True in collapse_to_vh_vl_binder_info.
    Only called when _build_binding_arms returns [] for an INN.

    Populates constant-domain sequences, genes, and mutation counts from
    INNChainDetail objects, mirroring the per-domain logic inside _build_binding_arms.
    VH/VL fields are left None.  arm_binder_type is set to 'fc_only'.
    """
    chains_by_id: Dict[str, List[INNChainDetail]] = defaultdict(list)
    for d in inn_result.chains:
        chains_by_id[d.chain_id].append(d)

    # Pick the heavy chain first (chain_type=='h'), fall back to any chain
    primary_cid = next(
        (
            cid
            for cid, ds in chains_by_id.items()
            if any(d.chain_type == "h" for d in ds)
        ),
        next(iter(chains_by_id), None),
    )
    light_cid = next(
        (
            cid
            for cid, ds in chains_by_id.items()
            if cid != primary_cid and any(d.chain_type == "l" for d in ds)
        ),
        None,
    )

    if primary_cid is None:
        return None

    row: Dict[str, object] = {
        "inn_id": inn_result.inn_id,
        "arm_slot": "a",
        "arm_binder_type": "fc_only",
        "vh_domain_type": None,
        "vl_domain_type": None,
        "shared_vl": False,
        "shared_heavy": False,
        "duplicate_vh": False,
        "ch_chain_detail_order_idxs": [],
        "cl_chain_detail_order_idxs": [],
    }

    const_seqs: Dict[str, List[str]] = defaultdict(list)
    ch_raw_muts: List[MutationRecord] = []
    ch_numbered_muts: List[NumberedMutationRecord] = []
    cl_raw_muts: List[MutationRecord] = []
    cl_numbered_muts: List[NumberedMutationRecord] = []

    for d in chains_by_id[primary_cid]:
        ann = d.region_annotation
        if ann == "Full Length":
            row["full_length_heavy"] = (
                d.sequence.replace(".", "") if d.sequence else None
            )
            continue
        col = _REGION_TO_COL.get(ann)
        if col:
            const_seqs[col].append(d.sequence)
            gene = d.realigned_gene or d.gene_name
            if gene:
                row.setdefault(f"{col}_gene", gene)
            n_muts = len(d.final_mutations) or None
            if n_muts:
                row.setdefault(f"{col}_total_mutations", n_muts)
            row["ch_chain_detail_order_idxs"].append(d.detail_order_idx)
            ch_raw_muts.extend(d.final_mutations)
            if d.mutations_by_individual_numbering_systems:
                ch_numbered_muts.extend(d.mutations_by_individual_numbering_systems)
        elif ann in _CL_ANNS:
            const_seqs["cl"].append(d.sequence)
            row["cl_chain_detail_order_idxs"].append(d.detail_order_idx)
            if "cl_chain_id" not in row:
                row["cl_chain_id"] = d.chain_id
                row["cl_type"] = "kappa" if ann == "C-KAPPA" else "lambda"
                row["cl_realigned_gene"] = d.realigned_gene
                row["cl_realigned_failed"] = d.realigned_failed
                row["cl_realigned_fraction_score"] = d.realigned_fraction_score
                row["cl_total_mutations"] = len(d.final_mutations) or None
            cl_raw_muts.extend(d.final_mutations)
            if d.mutations_by_individual_numbering_systems:
                cl_numbered_muts.extend(d.mutations_by_individual_numbering_systems)

    if light_cid:
        fl_light = next(
            (
                d
                for d in chains_by_id[light_cid]
                if d.region_annotation == "Full Length"
            ),
            None,
        )
        if fl_light and fl_light.sequence:
            row["full_length_light"] = fl_light.sequence.replace(".", "")

    for col, seqs in const_seqs.items():
        unique = list(dict.fromkeys(seqs))
        row[col] = "::".join(unique)
        row[f"unique_{col}_count"] = len(unique)

    row["ch_chain_detail_order_idxs"] = ";".join(
        sorted(set(row["ch_chain_detail_order_idxs"]))
    )
    row["cl_chain_detail_order_idxs"] = ";".join(
        sorted(set(row["cl_chain_detail_order_idxs"]))
    )
    row["ch_mutations"] = _mutation_str(ch_raw_muts) or None
    row["cl_mutations"] = _mutation_str(cl_raw_muts) or None
    row["ch_mutations_imgt"] = _fmt_numbered_muts(
        ch_numbered_muts, ch_raw_muts, system="imgt"
    )
    row["ch_mutations_eu"] = _fmt_numbered_muts(
        ch_numbered_muts, ch_raw_muts, system="eu"
    )
    row["ch_mutations_kabat"] = _fmt_numbered_muts(
        ch_numbered_muts, ch_raw_muts, system="kabat"
    )
    row["cl_mutations_imgt"] = _fmt_numbered_muts(
        cl_numbered_muts, cl_raw_muts, system="imgt"
    )
    row["cl_mutations_eu"] = _fmt_numbered_muts(
        cl_numbered_muts, cl_raw_muts, system="eu"
    )
    row["cl_mutations_kabat"] = _fmt_numbered_muts(
        cl_numbered_muts, cl_raw_muts, system="kabat"
    )

    return row


def collapse_to_vh_vl_binder_info(
    collection: INNStructureCollection,
    include_sequences_without_fv: bool = False,
) -> pd.DataFrame:
    """One row per binding arm per INN.

    Args:
        collection: INNStructureCollection after update_with_realignment.
        include_sequences_without_fv: When True, INNs that produce no arm rows
            (no VH/VL — e.g. Fc-fusions, receptor-Ig fusions) are included as a
            single row with arm_binder_type='fc_only'.  All VH/VL fields are None;
            constant-domain sequences, genes, and mutation data are populated where
            available.  A ``_no_fv`` boolean column marks these rows (True).
    """
    arms: List[BindingArm] = []
    no_fv_rows: List[Dict] = []

    for r in collection.results:
        result_arms = _build_binding_arms(r)
        if result_arms:
            arms.extend(result_arms)
        elif include_sequences_without_fv:
            row = _build_no_fv_arm_row(r)
            if row is not None:
                no_fv_rows.append(row)

    df = pd.DataFrame([arm.model_dump() for arm in arms])

    if not include_sequences_without_fv:
        return df

    df["_no_fv"] = False

    if no_fv_rows:
        no_fv_df = pd.DataFrame(no_fv_rows)
        for col in df.columns:
            if col not in no_fv_df.columns:
                no_fv_df[col] = None
        no_fv_df = no_fv_df[df.columns]
        no_fv_df["_no_fv"] = True
        df = pd.concat([df, no_fv_df], ignore_index=True)

    return df


def collapse_from_flat_csv(path: Union[str, Path]) -> pd.DataFrame:
    return collapse_to_vh_vl_binder_info(
        INNStructureCollection.from_flat_dataframe(pd.read_csv(path))
    )


def update_with_realignment(
    collection: INNStructureCollection,
    bundle: RealignedBundle,
    min_match_alignment: float = 0.5,
    cn: Optional[ConstantChainNumbering] = None,
) -> INNStructureCollection:
    """Return a new collection with realigned mutations and (optionally) numbered positions."""
    exact: Dict[Tuple[str, str, str, str], RegionMutations] = {}
    fallback: Dict[Tuple[str, str, str], Deque[RegionMutations]] = defaultdict(deque)
    for m in bundle.mutations:
        if m.region_annotation_description:
            exact[
                (
                    m.inn_id,
                    m.chain_id,
                    m.region_annotation,
                    m.region_annotation_description,
                )
            ] = m
        else:
            fallback[(m.inn_id, m.chain_id, m.region_annotation)].append(m)

    def _find_rm(
        inn_id: str,
        chain_id: str,
        region_annotation: str,
        region_annotation_description: Optional[str],
    ) -> Optional[RegionMutations]:
        if region_annotation_description:
            rm = exact.get(
                (
                    inn_id,
                    chain_id,
                    region_annotation,
                    region_annotation_description,
                )
            )
            if rm is not None:
                return rm
        q = fallback.get((inn_id, chain_id, region_annotation))
        if not q:
            return None
        return q[0] if len(q) == 1 else q.popleft()

    new_results: List[INNStructureResult] = []
    for result in collection.results:
        new_chains: List[INNChainDetail] = []
        region_iter_counter: Dict[str, int] = defaultdict(int)
        for chain in result.chains:
            rm = _find_rm(
                str(chain.inn_id),
                chain.chain_id,
                chain.region_annotation,
                chain.region_annotation_description,
            )
            updates: Dict[str, object] = {}
            if rm is not None:
                if rm.ref_sequence and rm.query_sequence:
                    non_mutated = min(
                        len(rm.ref_sequence), len(rm.query_sequence)
                    ) - len(rm.mutations or [])
                    frac = non_mutated / len(rm.query_sequence)
                    updates["realigned_fraction_score"] = frac
                    updates["realigned_failed"] = frac < min_match_alignment
                updates["realigned_gene"] = rm.germline_gene
                updates["realigned_species"] = rm.germline_species
                updates["realigned_germline_sequence"] = rm.ref_sequence
                updates["realigned_mutations"] = [
                    MutationRecord(
                        position=m.position,
                        mutated_seq=m.query_aa,
                        germline_seq=m.ref_aa,
                        mutation_type=m.mutation_type,
                        insertion_str=m.insertion_str,
                    )
                    for m in rm.mutations
                ]
            new_chain = chain.model_copy(update=updates) if updates else chain
            update = {"mutations_by_individual_numbering_systems": None}
            if cn is not None:
                rk = _numbering_region_key(new_chain, cn)
                if rk and new_chain.final_mutations:
                    region_iter_counter[rk] += 1
                    numbered = _build_numbered_mutations(
                        new_chain.final_mutations,
                        rk,
                        cn,
                        region_iter_counter[rk] - 1,
                    )
                    if numbered:
                        update = {"mutations_by_individual_numbering_systems": numbered}

            new_chain = new_chain.model_copy(update=update)
            new_chains.append(new_chain)
        new_results.append(INNStructureResult(inn_id=result.inn_id, chains=new_chains))

    return INNStructureCollection(
        results=new_results,
        failures=collection.failures,
        missing=collection.missing,
    )
