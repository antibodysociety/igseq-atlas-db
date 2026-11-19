import ast
from collections import defaultdict
from typing import (
    TYPE_CHECKING,
    Annotated,
    Dict,
    List,
    Literal,
    Optional,
    Protocol,
    Union,
)

import pandas as pd
from ..mutation_models import (  # noqa: F401 — re-exported
    MutationRecord,
    NumberedMutationRecord,
)
from pydantic import (
    BaseModel,
    Field,
    PrivateAttr,
    field_serializer,
    field_validator,
    model_validator,
)
from pydantic.functional_validators import BeforeValidator


from .chain_type import infer_chain_type
from .realigner import RegionMutations

# Coercing type alias: always stores inn_id as str regardless of input type.
# Using Annotated + BeforeValidator ensures coercion happens during both
# model_validate() and normal construction — avoids Pydantic v2 serialization
# warnings when int values are passed as inn_id.
InnId = Annotated[str, BeforeValidator(str)]


class RealignedBundle(Protocol):
    """Structural type for ConstantRealignedCollection — avoids circular import."""

    mutations: List[RegionMutations]


class BinderChain(BaseModel):
    """One variable domain and its owned constant regions on a physical chain."""

    chain_id: str
    variable_domain_id: str
    variable_type: Literal["VH", "VL"]
    constant_domain_idxs: List[str] = Field(default_factory=list)
    variable_sequence: str


class BinderGroup(BaseModel):
    """A complete binding unit: VH+VL, VH-only, or VL-only."""

    inn_id: InnId
    chains: List[BinderChain]

    @model_validator(mode="after")
    def _validate(self) -> "BinderGroup":
        n = len(self.chains)
        if n > 2:
            raise ValueError(f"INN {self.inn_id}: BinderGroup has {n} chains, max is 2")
        if n == 2:
            types = sorted(c.variable_type for c in self.chains)
            if types != ["VH", "VL"]:
                raise ValueError(
                    f"INN {self.inn_id}: two-chain group must be VH+VL, got {types}"
                )
        return self

    @property
    def vh(self) -> Optional[BinderChain]:
        return next((c for c in self.chains if c.variable_type == "VH"), None)

    @property
    def vl(self) -> Optional[BinderChain]:
        return next((c for c in self.chains if c.variable_type == "VL"), None)


class BindingArm(BaseModel):
    """One antigen-binding arm extracted from an INN structure."""

    inn_id: InnId
    arm_slot: str

    # Classification
    arm_binder_type: Optional[str] = None
    vh_domain_type: Optional[str] = None
    vl_domain_type: Optional[str] = None
    shared_vl: bool = False
    shared_heavy: bool = False
    duplicate_vh: bool = False

    # VH
    vh: Optional[str] = None
    vh_v_gene: Optional[str] = None
    vh_d_gene: Optional[str] = None
    vh_j_gene: Optional[str] = None
    vh_dga_url: Optional[str] = None
    vh_total_mutations: Optional[float] = None
    vh_mutations: Optional[str] = None
    vh_chain_id: Optional[str] = None
    vh_chain_id_order_idx: Optional[str] = None

    # VL
    vl: Optional[str] = None
    vl_v_gene: Optional[str] = None
    vl_j_gene: Optional[str] = None
    vl_dga_url: Optional[str] = None
    vl_total_mutations: Optional[float] = None
    vl_mutations: Optional[str] = None
    vl_chain_id: Optional[str] = None
    vl_chain_id_order_idx: Optional[str] = None

    # CH mutations
    ch_mutations: Optional[str] = None
    cl_mutations: Optional[str] = None
    ch_mutations_imgt: Optional[str] = None
    ch_mutations_eu: Optional[str] = None
    ch_mutations_kabat: Optional[str] = None
    cl_mutations_imgt: Optional[str] = None
    cl_mutations_eu: Optional[str] = None
    cl_mutations_kabat: Optional[str] = None

    # CL
    cl: Optional[str] = None
    cl_type: Optional[str] = None
    cl_total_mutations: Optional[int] = None
    cl_chain_id: Optional[str] = None
    cl_realigned_gene: Optional[str] = None
    cl_realigned_failed: Optional[bool] = None
    cl_realigned_fraction_score: Optional[float] = None
    unique_cl_count: Optional[int] = None

    # CH1
    ch1: Optional[str] = None
    ch1_total_mutations: Optional[int] = None
    ch1_gene: Optional[str] = None
    unique_ch1_count: Optional[int] = None

    # CH2
    ch2: Optional[str] = None
    ch2_total_mutations: Optional[int] = None
    ch2_gene: Optional[str] = None
    unique_ch2_count: Optional[int] = None

    # CH3
    ch3: Optional[str] = None
    ch3_total_mutations: Optional[int] = None
    ch3_gene: Optional[str] = None
    unique_ch3_count: Optional[int] = None

    # CHS (C-terminal tail)
    chs: Optional[str] = None
    chs_total_mutations: Optional[int] = None
    chs_gene: Optional[str] = None
    unique_chs_count: Optional[int] = None

    # TCR constant chains
    tcr_alpha: Optional[str] = None
    tcr_alpha_total_mutations: Optional[int] = None
    tcr_alpha_gene: Optional[str] = None
    unique_tcr_alpha_count: Optional[int] = None

    tcr_beta: Optional[str] = None
    tcr_beta_total_mutations: Optional[int] = None
    tcr_beta_gene: Optional[str] = None
    unique_tcr_beta_count: Optional[int] = None

    # Hinge
    hinge_region: Optional[str] = None
    hinge_region_total_mutations: Optional[int] = None

    # Full-length sequences (when no domain-level data available)
    full_length_heavy: Optional[str] = None
    full_length_light: Optional[str] = None

    # Bookkeeping
    ch_chain_detail_order_idxs: str = ""
    cl_chain_detail_order_idxs: str = ""
    other_annotations_skipped: Optional[str] = None


class INNChainDetail(BaseModel):
    """Detail for one chain region/domain from IMGT 2D structure page."""

    inn_id: InnId
    chain_id: str
    chain_type: str
    detail_order_idx: str
    chain_length: int = 0
    chain_description: Optional[str] = None
    chain_feature_type: Optional[str] = None
    region_annotation: str
    region_annotation_description: Optional[str] = None
    sequence: Optional[str] = None
    germline_sequence: Optional[str] = None
    gene_name: Optional[str] = None
    v_gene: Optional[str] = None
    d_gene: Optional[str] = None
    j_gene: Optional[str] = None
    total_mutations: Optional[int] = None
    domain_gap_align_url: Optional[str] = None
    scraped_mutations: Optional[List[MutationRecord]] = None
    realigned_mutations: Optional[List[MutationRecord]] = None
    realigned_failed: Optional[bool] = None
    realigned_gene: Optional[str] = None
    realigned_germline_sequence: Optional[str] = None
    realigned_fraction_score: Optional[float] = None
    realigned_species: Optional[str] = None
    mutations_by_individual_numbering_systems: Optional[
        List[NumberedMutationRecord]
    ] = None

    @property
    def final_mutations(self) -> List[MutationRecord]:
        """Resolved mutations: realigned if available and not failed, else scraped."""
        if self.realigned_mutations is not None and not self.realigned_failed:
            return self.realigned_mutations
        return self.scraped_mutations or []


class INNStructureResult(BaseModel):
    """All chain details for a single INN entry."""

    inn_id: InnId
    chains: List[INNChainDetail] = Field(default_factory=list)

    def to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame([c.model_dump() for c in self.chains])

    def retype_chains(self) -> "INNStructureResult":
        new_chains = [
            c.model_copy(
                update={"chain_type": infer_chain_type(c.chain_description or "")}
            )
            for c in self.chains
        ]
        return self.model_copy(update={"chains": new_chains})


class INNStructureCollection(BaseModel):
    """Collection of INN structure results with lookup helpers."""

    results: List[INNStructureResult] = Field(default_factory=list)
    failures: Dict[str, str] = Field(default_factory=dict)
    missing: List[str] = Field(default_factory=list)

    _by_domain: Optional[Dict[str, "INNChainDetail"]] = PrivateAttr(default=None)
    _by_inn: Optional[Dict[str, "INNStructureResult"]] = PrivateAttr(default=None)

    def _build_chain_map(self) -> Dict[str, "INNChainDetail"]:
        by_idx_map: Dict[str, INNChainDetail] = {}
        for result in self.results:
            for chain in result.chains:
                if chain.detail_order_idx in by_idx_map:
                    raise ValueError(
                        f"Duplicate inn_id/detail_order_idx: {result.inn_id}/{chain.detail_order_idx}"
                    )
                by_idx_map[chain.detail_order_idx] = chain
        return by_idx_map

    @property
    def chain_map(self) -> Dict[str, "INNChainDetail"]:
        if self._by_domain is None:
            self._by_domain = self._build_chain_map()
        return self._by_domain

    @property
    def inn_map(self) -> Dict[str, "INNStructureResult"]:
        if self._by_inn is None:
            self._by_inn = {r.inn_id: r for r in self.results}
        return self._by_inn

    def __getitem__(self, inn_id: str) -> "INNStructureResult":
        return self.inn_map[str(inn_id)]

    def get_specific_domain(self, detail_order_idx: str) -> "INNChainDetail":
        return self.chain_map[detail_order_idx]

    @property
    def count(self) -> int:
        return len(self.results)

    def retype_chains(self) -> "INNStructureCollection":
        return self.model_copy(
            update={"results": [r.retype_chains() for r in self.results]}
        )

    def to_dataframe(self) -> pd.DataFrame:
        """One row per INN with chain count summary."""
        rows = [
            {
                "inn_id": r.inn_id,
                "chain_count": len(r.chains),
                "heavy_chains": sum(1 for c in r.chains if c.chain_type == "h"),
                "light_chains": sum(1 for c in r.chains if c.chain_type == "l"),
            }
            for r in self.results
        ]
        return pd.DataFrame(rows)

    def to_flat_dataframe(self) -> pd.DataFrame:
        """One row per chain detail across all INNs."""
        return pd.concat(
            [r.to_dataframe() for r in self.results if r.chains],
            ignore_index=True,
        )

    @classmethod
    def from_flat_dataframe(cls, df: pd.DataFrame) -> "INNStructureCollection":
        """Reconstruct an INNStructureCollection from a flat per-chain DataFrame."""
        _LIST_FIELDS = (
            "scraped_mutations",
            "realigned_mutations",
            "mutations_by_individual_numbering_systems",
        )
        groups: Dict[str, List[INNChainDetail]] = defaultdict(list)
        for _, row in df.iterrows():
            kwargs: Dict[str, object] = {
                k: v
                for k, v in row.items()
                if not (isinstance(v, float) and pd.isna(v))
            }
            for field in _LIST_FIELDS:
                if field in kwargs and isinstance(kwargs[field], str):
                    try:
                        kwargs[field] = ast.literal_eval(kwargs[field])
                    except Exception:
                        kwargs.pop(field, None)
            groups[str(kwargs["inn_id"])].append(INNChainDetail.model_validate(kwargs))
        return cls(
            results=[INNStructureResult(inn_id=k, chains=v) for k, v in groups.items()]
        )
