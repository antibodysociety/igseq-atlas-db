"""Parse IMGT chain descriptions into structural blueprints and classify binder types.

Converts the IMGT 2D structure chain_description label (e.g. "H-GAMMA-1",
"SCFV-KAPPA-HEAVY-H-CH2-CH3") into a normalized list of expected domain roles,
then classifies each chain into a binder type (classical, scFv, CrossMab, etc.).
"""

import re
from enum import Enum
from typing import Dict, List, Optional, Set, Tuple

from pydantic import BaseModel

# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class DomainRole(str, Enum):
    VH = "VH"
    VL_KAPPA = "V-KAPPA"
    VL_LAMBDA = "V-LAMBDA"
    CH1 = "CH1"
    CH2 = "CH2"
    CH3 = "CH3"
    CH4 = "CH4"
    CL_KAPPA = "C-KAPPA"
    CL_LAMBDA = "C-LAMBDA"
    HINGE = "HINGE-REGION"
    CHS = "CHS"
    UNKNOWN = "UNKNOWN"


class BinderType(str, Enum):
    """Per-chain binder classification (based on chain_description parsing)."""

    CLASSICAL_VH_VL = "classical_vh_vl"
    VH_ONLY = "vh_only"
    VL_ONLY = "vl_only"
    CONSTANT_ONLY = "constant_only"
    SCFV = "scfv"
    SCFV_FC = "scfv_fc"
    CROSSMAB = "crossmab"
    NON_CANONICAL = "non_canonical"


# ---------------------------------------------------------------------------
# Per-domain classification (individual VH / VL within an ownership group)
# ---------------------------------------------------------------------------


class VhDomainType(str, Enum):
    """Classification of an individual VH domain based on its owned constants."""

    IG_FULL_LENGTH = "ig_full_length"  # VH + CH1 + Fc (CH2/CH3/CH4)
    VH_FAB = "vh_fab"  # VH + CH1 (no Fc)
    VH_CK = "vh_ck"  # VH + C-KAPPA
    VH_CL = "vh_cl"  # VH + C-LAMBDA
    VH_SCFV = "vh_scfv"  # VH from an scFv pair
    VH_ORPHAN = "vh_orphan"  # VH by itself
    UNKNOWN = "unknown"


class VlDomainType(str, Enum):
    """Classification of an individual VL domain based on its owned constants."""

    VKAPPA = "vkappa"  # V-KAPPA + C-KAPPA
    VLAMBDA = "vlambda"  # V-LAMBDA + C-LAMBDA
    VK_CL = "vk_cl"  # V-KAPPA + C-LAMBDA
    VL_CK = "vl_ck"  # V-LAMBDA + C-KAPPA
    VK_CH1 = "vk_ch1"  # V-KAPPA + CH1
    VK_CH1_FC = "vk_ch1_fc"  # V-KAPPA + CH1 + Fc
    VL_CH1 = "vl_ch1"  # V-LAMBDA + CH1
    VL_CH1_FC = "vl_ch1_fc"  # V-LAMBDA + CH1 + Fc
    VK_SCFV = "vk_scfv"  # V-KAPPA from scFv
    VL_SCFV = "vl_scfv"  # V-LAMBDA from scFv
    VK_FC = "vk_fc"  # V-KAPPA + Fc (no CL, no CH1)
    VL_FC = "vl_fc"  # V-LAMBDA + Fc (no CL, no CH1)
    VK_ONLY = "vk_only"  # V-KAPPA by itself
    VL_ONLY = "vl_only"  # V-LAMBDA by itself
    UNKNOWN = "unknown"


_FC_ROLES = {
    DomainRole.CH2,
    DomainRole.CH3,
    DomainRole.CH4,
    DomainRole.HINGE,
    DomainRole.CHS,
}


def classify_vh_domain(
    owned_annotations: Set[str], is_scfv: bool = False
) -> VhDomainType:
    """Classify a VH domain by what constant regions it owns."""
    if is_scfv:
        return VhDomainType.VH_SCFV
    has_ch1 = "CH1" in owned_annotations
    has_fc = bool(owned_annotations & {"CH2", "CH3", "CH4", "HINGE-REGION", "CHS"})
    has_ck = "C-KAPPA" in owned_annotations
    has_cl = "C-LAMBDA" in owned_annotations
    if has_ch1 and has_fc:
        return VhDomainType.IG_FULL_LENGTH
    if has_ch1:
        return VhDomainType.VH_FAB
    if has_ck:
        return VhDomainType.VH_CK
    if has_cl:
        return VhDomainType.VH_CL
    if has_fc:
        return VhDomainType.IG_FULL_LENGTH  # VH + Fc without CH1 still full-length-like
    return VhDomainType.VH_ORPHAN


def classify_vl_domain(
    vl_annotation: str, owned_annotations: Set[str], is_scfv: bool = False
) -> VlDomainType:
    """Classify a VL domain by its type (kappa/lambda) and owned constants."""
    is_kappa = vl_annotation == "V-KAPPA"
    if is_scfv:
        return VlDomainType.VK_SCFV if is_kappa else VlDomainType.VL_SCFV
    has_ck = "C-KAPPA" in owned_annotations
    has_cl = "C-LAMBDA" in owned_annotations
    has_ch1 = "CH1" in owned_annotations
    has_fc = bool(owned_annotations & {"CH2", "CH3", "CH4", "HINGE-REGION", "CHS"})
    if is_kappa:
        if has_ck:
            return VlDomainType.VKAPPA
        if has_cl:
            return VlDomainType.VK_CL
        if has_ch1 and has_fc:
            return VlDomainType.VK_CH1_FC
        if has_ch1:
            return VlDomainType.VK_CH1
        if has_fc:
            return VlDomainType.VK_FC
        return VlDomainType.VK_ONLY
    else:
        if has_cl:
            return VlDomainType.VLAMBDA
        if has_ck:
            return VlDomainType.VL_CK
        if has_ch1 and has_fc:
            return VlDomainType.VL_CH1_FC
        if has_ch1:
            return VlDomainType.VL_CH1
        if has_fc:
            return VlDomainType.VL_FC
        return VlDomainType.VL_ONLY


# ---------------------------------------------------------------------------
# Arm-level binder classification (paired VH+VL → arm type)
# ---------------------------------------------------------------------------


class ArmBinderType(str, Enum):
    """Classification of a paired binding arm based on VH + VL domain types."""

    CANONICAL_HEAVY_KAPPA = "canonical_heavy_kappa"
    CANONICAL_HEAVY_LAMBDA = "canonical_heavy_lambda"
    SCFV_HEAVY_KAPPA = "scfv_heavy_kappa"  # VH→VK on chain
    SCFV_HEAVY_LAMBDA = "scfv_heavy_lambda"  # VH→VL on chain
    SCFV_KAPPA_HEAVY = "scfv_kappa_heavy"  # VK→VH on chain
    SCFV_LAMBDA_HEAVY = "scfv_lambda_heavy"  # VL→VH on chain
    FAB_HEAVY_KAPPA = "fab_heavy_kappa"
    FAB_HEAVY_LAMBDA = "fab_heavy_lambda"
    SWAPPED_CHAIN_KAPPA = "swapped_chain_kappa"  # vk_ch1_fc + vh_ck/vh_cl
    SWAPPED_CHAIN_LAMBDA = "swapped_chain_lambda"
    ORPHAN_PAIR = "orphan_pair"
    VH_ONLY = "vh_only"
    VL_ONLY = "vl_only"
    VL_FC_ONLY = "vl_fc_only"


# Valid (vh_domain_type, vl_domain_type) → ArmBinderType lookups
_CANONICAL_VL = {
    VlDomainType.VKAPPA,
    VlDomainType.VLAMBDA,
    VlDomainType.VK_CL,
    VlDomainType.VL_CK,
}
_FAB_VH = {VhDomainType.VH_FAB, VhDomainType.VH_CK, VhDomainType.VH_CL}
_FAB_VL = {
    VlDomainType.VKAPPA,
    VlDomainType.VLAMBDA,
    VlDomainType.VK_CL,
    VlDomainType.VL_CK,
    VlDomainType.VK_CH1,
    VlDomainType.VL_CH1,
}
_SWAPPED_VL = {VlDomainType.VK_CH1_FC, VlDomainType.VL_CH1_FC}
_SWAPPED_VH = {VhDomainType.VH_CK, VhDomainType.VH_CL}
_ORPHAN_VH = {VhDomainType.VH_ORPHAN}
_ORPHAN_VL = {VlDomainType.VK_ONLY, VlDomainType.VL_ONLY}


def _is_kappa_vl(vl_type: VlDomainType) -> bool:
    return vl_type in {
        VlDomainType.VKAPPA,
        VlDomainType.VK_CL,
        VlDomainType.VK_CH1,
        VlDomainType.VK_CH1_FC,
        VlDomainType.VK_SCFV,
        VlDomainType.VK_ONLY,
        VlDomainType.VK_FC,
    }


def classify_arm_binder_type(
    vh_type: Optional[VhDomainType],
    vl_type: Optional[VlDomainType],
    is_scfv: bool = False,
    vh_first: bool = True,
) -> ArmBinderType:
    """Classify a paired arm from its VH and VL domain types.

    Raises ValueError for invalid/unexpected pairings.
    """
    # Solo arms
    if vh_type is not None and vl_type is None:
        return ArmBinderType.VH_ONLY
    if vh_type is None and vl_type is not None:
        if vl_type in {VlDomainType.VK_FC, VlDomainType.VL_FC}:
            return ArmBinderType.VL_FC_ONLY
        return ArmBinderType.VL_ONLY

    # scFv
    if is_scfv:
        kappa = _is_kappa_vl(vl_type)
        if vh_first:
            return (
                ArmBinderType.SCFV_HEAVY_KAPPA
                if kappa
                else ArmBinderType.SCFV_HEAVY_LAMBDA
            )
        return (
            ArmBinderType.SCFV_KAPPA_HEAVY if kappa else ArmBinderType.SCFV_LAMBDA_HEAVY
        )

    # Canonical: ig_full_length + vkappa/vlambda
    if vh_type == VhDomainType.IG_FULL_LENGTH and vl_type in _CANONICAL_VL:
        kappa = _is_kappa_vl(vl_type)
        return (
            ArmBinderType.CANONICAL_HEAVY_KAPPA
            if kappa
            else ArmBinderType.CANONICAL_HEAVY_LAMBDA
        )

    # Fab: vh_fab/vh_ck/vh_cl + standard VL
    if vh_type in _FAB_VH and vl_type in _FAB_VL:
        kappa = _is_kappa_vl(vl_type)
        return (
            ArmBinderType.FAB_HEAVY_KAPPA if kappa else ArmBinderType.FAB_HEAVY_LAMBDA
        )

    # Swapped chain: vk_ch1_fc/vl_ch1_fc + vh_ck/vh_cl
    if vh_type in _SWAPPED_VH and vl_type in _SWAPPED_VL:
        kappa = _is_kappa_vl(vl_type)
        return (
            ArmBinderType.SWAPPED_CHAIN_KAPPA
            if kappa
            else ArmBinderType.SWAPPED_CHAIN_LAMBDA
        )

    # Orphan pair
    if vh_type in _ORPHAN_VH and vl_type in _ORPHAN_VL:
        return ArmBinderType.ORPHAN_PAIR

    raise ValueError(f"Unexpected arm pairing: vh_type={vh_type}, vl_type={vl_type}")


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------


class ParsedDomainSlot(BaseModel):
    role: DomainRole
    slot_index: int


class ChainBlueprint(BaseModel):
    """Parsed expected structure for one chain_id."""

    chain_id: str
    chain_description_label: str
    expected_slots: List[ParsedDomainSlot]
    binder_type: BinderType
    is_canonical: bool


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_VH_ROLES = {DomainRole.VH}
_VL_ROLES = {DomainRole.VL_KAPPA, DomainRole.VL_LAMBDA}
_VARIABLE_ROLES = _VH_ROLES | _VL_ROLES
_CL_ROLES = {DomainRole.CL_KAPPA, DomainRole.CL_LAMBDA}
_CH_ROLES = {
    DomainRole.CH1,
    DomainRole.CH2,
    DomainRole.CH3,
    DomainRole.CH4,
    DomainRole.HINGE,
    DomainRole.CHS,
}
_CONSTANT_ROLES = _CH_ROLES | _CL_ROLES

_CANONICAL_RE = re.compile(
    r"((H\-GAMMA)|(L\-KAPPA)|(L\-LAMBDA))([\-0-9]+)?([A-Z]?)$", re.IGNORECASE
)

# Token → DomainRole mapping (after normalization)
_TOKEN_MAP: Dict[str, DomainRole] = {
    "VH": DomainRole.VH,
    "HEAVY": DomainRole.VH,
    "V.KAPPA": DomainRole.VL_KAPPA,
    "KAPPA": DomainRole.VL_KAPPA,
    "V.LAMBDA": DomainRole.VL_LAMBDA,
    "LAMBDA": DomainRole.VL_LAMBDA,
    "C.KAPPA": DomainRole.CL_KAPPA,
    "C.LAMBDA": DomainRole.CL_LAMBDA,
    "CH1": DomainRole.CH1,
    "CH2": DomainRole.CH2,
    "CH3": DomainRole.CH3,
    "CH4": DomainRole.CH4,
}

# scFv tokens get mapped to their underlying variable roles
_SCFV_TOKEN_MAP: Dict[str, DomainRole] = {
    "SCFV.KAPPA": DomainRole.VL_KAPPA,
    "SCFV.LAMBDA": DomainRole.VL_LAMBDA,
    "SCFV.VH": DomainRole.VH,
    "SCFV.V.KAPPA": DomainRole.VL_KAPPA,
    "SCFV.V.LAMBDA": DomainRole.VL_LAMBDA,
    "SCFV.HEAVY": DomainRole.VH,
}

# Chain-id–specific hardcoded overrides (data corrections)
_CHAIN_ID_OVERRIDES: Dict[str, str] = {
    "INN 9340_M": "+CH4",  # append CH4 to existing description
    "INN 13065_M": "CH2-CH3",  # replace entire description
    "INN 12998_H": "SCFV-KAPPA-HEAVY-CH2-CH3-SCFV-HEAVY-KAPPA",
}

# Description-level hardcoded overrides
_DESC_OVERRIDES: Dict[str, str] = {
    "H-GAMMA-1-GAMMA-2": "H-GAMMA",
    "H-GAMMA-1-C-KAPPA": "VH-C-KAPPA-CH2-CH3",
}


# ---------------------------------------------------------------------------
# Normalization pipeline
# ---------------------------------------------------------------------------


def _normalize_label(label: str, chain_id: str, detailed_desc: str = "") -> str:
    """Apply the regex normalization pipeline to a chain description label."""
    # Chain-id overrides
    if chain_id in _CHAIN_ID_OVERRIDES:
        override = _CHAIN_ID_OVERRIDES[chain_id]
        label = (label + override) if override.startswith("+") else override

    # Description-level overrides
    if label in _DESC_OVERRIDES:
        label = _DESC_OVERRIDES[label]

    # V-KAPPA with C-KAPPA in detailed description but not in label
    if label == "V-KAPPA" and detailed_desc and "+ C-KAP" in detailed_desc:
        label = "V.KAPPA-C.KAPPA"

    # Typo fix
    label = label.replace("KAPA", "KAPPA")

    # Strip isotype numbers: GAMMA-1, KAPPA-2, LAMBDA-1-2 etc.
    label = re.sub(r"(GAMMA|KAPPA|LAMBDA)-\d(-\d)?", r"\1", label, flags=re.IGNORECASE)

    # Collapse scFv compound: SCFV-X-Y(-H) → SCFV.X-Y
    label = re.sub(
        r"SCFV-([^-]+)-([^-]+)(-H)?", r"SCFV.\1-\2", label, flags=re.IGNORECASE
    )

    # Protect V/L/H-gamma/kappa/lambda from being split: V-KAPPA → V.KAPPA
    label = re.sub(
        r"(V|L|H)-(gamma|kappa|lambda)", r"\1.\2", label, flags=re.IGNORECASE
    )

    # Protect FUSION-
    label = re.sub(r"FUSION-", "Fusion.", label, flags=re.IGNORECASE)

    # Protect C-KAPPA/C-LAMBDA
    label = re.sub(r"C-(Kappa|Lambda)", r"C.\1", label, flags=re.IGNORECASE)

    # Expand canonical heavy: .GAMMA → VH-CH1-CH2-CH3
    label = re.sub(
        r"[\.\-]GAMMA(\d+)?", "-VH-CH1-CH2-CH3", label, flags=re.IGNORECASE
    ).strip("-")

    # Expand IgM/IgE: h-(mu|epsilon) → VH-CH1-CH2-CH3-CH4
    label = re.sub(
        r"h-(mu|epsilon)", "-VH-CH1-CH2-CH3-CH4", label, flags=re.IGNORECASE
    ).strip("-")

    # 1c-like → V.lambda-c.kappa
    label = re.sub(r"1c-like", "V.lambda-c.kappa", label, flags=re.IGNORECASE).strip(
        "-"
    )

    # VH1, VH2, etc. → VH
    label = re.sub(r"vh\d", "VH", label, flags=re.IGNORECASE).strip("-")

    # Expand canonical light: L.KAPPA → V.KAPPA-C.KAPPA, L.LAMBDA → V.LAMBDA-C.LAMBDA
    label = re.sub(r"L\.(Kappa|Lambda)", r"V.\1-C.\1", label, flags=re.IGNORECASE)

    # Expand IGHG references: IGHG1(CH1) → CH1
    label = re.sub(r"IGHG\d+\(([0-9a-z\-)]+)\)", r"\1", label, flags=re.IGNORECASE)

    return label


def _tokenize(label: str) -> List[str]:
    """Split normalized label into domain tokens, filtering noise."""
    tokens: List[str] = label.split("-")
    # Filter out bare "H" (heavy chain prefix) and empty tokens
    return [t for t in tokens if t and t.upper() not in ("H",)]


def _map_token(token: str) -> DomainRole:
    """Map a single normalized token to a DomainRole."""
    upper = token.upper()
    if upper in _TOKEN_MAP:
        return _TOKEN_MAP[upper]
    if upper in _SCFV_TOKEN_MAP:
        return _SCFV_TOKEN_MAP[upper]
    # Strip trailing parens/digits from partial matches
    cleaned = re.sub(r"[()0-9]+$", "", upper)
    if cleaned in _TOKEN_MAP:
        return _TOKEN_MAP[cleaned]
    if cleaned in _SCFV_TOKEN_MAP:
        return _SCFV_TOKEN_MAP[cleaned]
    return DomainRole.UNKNOWN


# ---------------------------------------------------------------------------
# Classification
# ---------------------------------------------------------------------------


def _classify_binder_type(slots: List[DomainRole], label: str) -> BinderType:
    """Classify binder type from the sequence of domain roles."""
    role_set = set(slots)
    has_vh = bool(role_set & _VH_ROLES)
    has_vl = bool(role_set & _VL_ROLES)
    has_cl = bool(role_set & _CL_ROLES)
    has_ch = bool(role_set & _CH_ROLES)
    is_scfv = "SCFV" in label.upper()

    # Constant-only (no variable domains at all)
    if not has_vh and not has_vl:
        if has_ch or has_cl:
            return BinderType.CONSTANT_ONLY
        return BinderType.NON_CANONICAL

    # scFv-based
    if is_scfv:
        return BinderType.SCFV_FC if has_ch else BinderType.SCFV

    # CrossMab: VH paired with CL but no VL, or VL paired with CH1 but no VH
    if has_vh and has_cl and not has_vl:
        return BinderType.CROSSMAB
    if has_vl and DomainRole.CH1 in role_set and not has_vh:
        return BinderType.CROSSMAB

    # Classical VH+VL
    if has_vh and has_vl:
        return BinderType.CLASSICAL_VH_VL

    # VH only (nanobody/VHH or heavy chain without external light on this chain)
    if has_vh and not has_vl:
        return BinderType.VH_ONLY

    # VL only
    if has_vl and not has_vh:
        return BinderType.VL_ONLY

    return BinderType.NON_CANONICAL


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def parse_chain_description(
    chain_description: str,
    chain_id: str,
) -> ChainBlueprint:
    """Parse an IMGT chain_description into a ChainBlueprint.

    Args:
        chain_description: Full chain_description from INNChainDetail
            (e.g. "H-GAMMA-1= VH (1-124) [D1] + CH1 ...")
        chain_id: The chain_id for hardcoded override lookups.

    Returns:
        ChainBlueprint with expected domain slots and binder type.
    """
    # Split on "=" to get label vs detailed description
    parts = chain_description.split("=", 1)
    label = parts[0].strip()
    detailed = parts[1].strip() if len(parts) > 1 else ""

    is_canonical = bool(_CANONICAL_RE.match(label))
    normalized = _normalize_label(label, chain_id, detailed)
    tokens = _tokenize(normalized)

    # Expand repeated VH from detailed description when label is just "VH"
    if normalized.upper() == "VH" and detailed and detailed.count(" + VH ") >= 2:
        n_vh = detailed.count(" + VH ") + (
            1 if detailed.strip().startswith("VH ") else 0
        )
        tokens = ["VH"] * max(n_vh, 1)

    roles = [_map_token(t) for t in tokens]
    # Filter out UNKNOWN roles
    roles = [r for r in roles if r != DomainRole.UNKNOWN]

    slots = [ParsedDomainSlot(role=r, slot_index=i) for i, r in enumerate(roles)]
    binder_type = _classify_binder_type(roles, normalized)

    return ChainBlueprint(
        chain_id=chain_id,
        chain_description_label=label,
        expected_slots=slots,
        binder_type=binder_type,
        is_canonical=is_canonical,
    )


# Type alias for ownership map entries: (owner_chain_or_None, group_index)
OwnershipEntry = Tuple[Optional[object], int]
OwnershipMap = Dict[int, OwnershipEntry]  # keyed by id(chain_detail)


def compute_region_ownership(
    chains: list,
) -> OwnershipMap:
    """Assign each INNChainDetail to its owning variable-domain slot index.

    Walks the actual regions in order and assigns each constant/non-variable
    region to the nearest preceding variable domain's slot group.

    Args:
        chains: List of INNChainDetail for this chain_id, in detail_order_idx order.

    Returns:
        dict mapping id(INNChainDetail) → (owner_variable_chain, slot_group_index)
            where owner_variable_chain is the INNChainDetail of the VH/VL that owns it,
            or None for shared/pre-anchor regions.
            slot_group_index groups constants with their preceding variable domain.
    """
    _SKIP_ANNOTATIONS: Set[str] = {"Full Length", "N.D", "LINKER"}

    # Walk chains sequentially; track current variable-domain anchor
    current_anchor: Optional[object] = None
    current_group: int = -1
    ownership: OwnershipMap = {}

    for c in chains:
        ann = c.region_annotation
        if ann in _SKIP_ANNOTATIONS:
            ownership[id(c)] = (None, -1)
            continue

        # Is this a variable domain (VH or VL)?
        is_variable = ann in ("VH",) or ann in ("V-KAPPA", "V-LAMBDA")
        if is_variable:
            current_anchor = c
            current_group += 1
            ownership[id(c)] = (c, current_group)
        else:
            # Constant or other region → owned by current anchor
            ownership[id(c)] = (current_anchor, current_group)

    return ownership
