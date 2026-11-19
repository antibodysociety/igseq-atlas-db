"""Pydantic models for the consolidated merged antibody database."""

import re
from typing import TYPE_CHECKING, Dict, List, Literal, Optional

import pandas as pd
from pydantic import BaseModel, Field

SOURCE = Literal["therasabdab", "imgt", "mabsolve", "derived"]


class ColumnSource(BaseModel):
    column_name: str
    source: SOURCE
    description: str = ""


# ── INN metadata table ─────────────────────────────────────────────────


class MergedINNRecord(BaseModel):
    """One row per canonical INN — metadata only, no sequences."""

    inn_name: str
    inn_mol_name: Optional[str] = None
    sources: str = Field("", description="Comma-sep contributing databases")
    n_binders_therasabdab: Optional[int] = Field(
        None, description="Distinct VH/VL arms from TheraSAbDab"
    )
    n_binders_imgt: Optional[int] = Field(
        None, description="Distinct VH/VL arms from IMGT 2D"
    )
    n_binders_mismatch: Optional[bool] = Field(
        None,
        description="True when TheraSAbDab and IMGT arm counts differ",
    )

    # TheraSAbDab
    target: Optional[str] = None
    format: Optional[str] = None
    isotype: Optional[str] = None
    highest_clinical_trial: Optional[str] = None
    estimated_status: Optional[str] = None
    has_100_si_struc: Optional[str] = None
    has_99_si_struc: Optional[str] = None
    has_95_98_si_struc: Optional[str] = None
    pdb_100: Optional[str] = None
    pdb_99: Optional[str] = None
    pdb_95_98: Optional[str] = None
    dev_technology: Optional[str] = None
    companies: Optional[str] = None
    year_proposed: Optional[str] = None
    year_recommended: Optional[str] = None
    year_received: Optional[str] = None
    year_proposed_imgt: Optional[str] = None
    year_recommended_imgt: Optional[str] = None
    conditions_approved: Optional[str] = None
    conditions_active: Optional[str] = None
    conditions_discontinued: Optional[str] = None
    notes: Optional[str] = None

    # IMGT mAb-DB
    inn_number: Optional[str] = None
    imgt_id: Optional[str] = None
    structure_2d: Optional[str] = None
    structure_3d: Optional[str] = None
    receptor_type: Optional[str] = None
    specificity: Optional[str] = None
    species: Optional[str] = None
    conjugated: Optional[str] = None
    origin_species: Optional[str] = None
    origin_clone: Optional[str] = None
    expression_system: Optional[str] = None
    mechanism_of_action: Optional[str] = None
    clinical_indication: Optional[str] = None
    clinical_domain: Optional[str] = None
    clinical_trials: Optional[str] = None
    engineered_variant: Optional[str] = None
    development_status: Optional[str] = None
    biosimilars: Optional[str] = None

    # mAbSolve
    mabsolve_format: Optional[str] = None
    effector_function: Optional[str] = None
    half_life: Optional[str] = None
    heterodimerization: Optional[bool] = None
    site_specific_conj: Optional[bool] = None
    stabilized: Optional[bool] = None
    c_term_truncation: Optional[bool] = None
    other_fc_mod: Optional[bool] = None

    # IMGT 2D chain counts
    imgt_2d_chain_count: Optional[int] = None
    imgt_2d_heavy_count: Optional[int] = None
    imgt_2d_light_count: Optional[int] = None
    imgt_2d_inn_id: Optional[str] = None


INN_COLUMN_SOURCES: list[dict] = [
    {
        "column_name": "inn_name",
        "source": "derived",
        "description": "Normalized INN name (join key)",
    },
    {
        "column_name": "inn_mol_name",
        "source": "derived",
        "description": "Molecule-level INN root (first token)",
    },
    {
        "column_name": "sources",
        "source": "derived",
        "description": "Comma-sep contributing databases",
    },
    {
        "column_name": "n_binders_therasabdab",
        "source": "therasabdab",
        "description": "Distinct VH/VL arms from TheraSAbDab",
    },
    {
        "column_name": "n_binders_imgt",
        "source": "imgt",
        "description": "Distinct VH/VL arms from IMGT 2D",
    },
    {
        "column_name": "n_binders_mismatch",
        "source": "derived",
        "description": "True when TheraSAbDab and IMGT arm counts differ",
    },
    {
        "column_name": "target",
        "source": "therasabdab",
        "description": "Target antigen",
    },
    {
        "column_name": "format",
        "source": "therasabdab",
        "description": "Molecule format",
    },
    {
        "column_name": "isotype",
        "source": "therasabdab",
        "description": "Antibody isotype",
    },
    {
        "column_name": "highest_clinical_trial",
        "source": "therasabdab",
        "description": "Highest clinical trial phase",
    },
    {
        "column_name": "estimated_status",
        "source": "therasabdab",
        "description": "Estimated status",
    },
    {
        "column_name": "has_100_si_struc",
        "source": "therasabdab",
        "description": "Has 100% seqID Fv structure",
    },
    {
        "column_name": "has_99_si_struc",
        "source": "therasabdab",
        "description": "Has 99% seqID Fv structure",
    },
    {
        "column_name": "has_95_98_si_struc",
        "source": "therasabdab",
        "description": "Has 95-98% seqID Fv structure",
    },
    {
        "column_name": "pdb_100",
        "source": "therasabdab",
        "description": "PDB codes 100% seqID",
    },
    {
        "column_name": "pdb_99",
        "source": "therasabdab",
        "description": "PDB codes 99% seqID",
    },
    {
        "column_name": "pdb_95_98",
        "source": "therasabdab",
        "description": "PDB codes 95-98% seqID",
    },
    {
        "column_name": "dev_technology",
        "source": "therasabdab",
        "description": "Developmental technology",
    },
    {
        "column_name": "companies",
        "source": "therasabdab",
        "description": "Developing companies",
    },
    {
        "column_name": "year_proposed",
        "source": "therasabdab",
        "description": "Year INN proposed",
    },
    {
        "column_name": "year_recommended",
        "source": "therasabdab",
        "description": "Year INN recommended",
    },
    {
        "column_name": "year_received",
        "source": "therasabdab",
        "description": "Year INN received",
    },
    {
        "column_name": "year_proposed_imgt",
        "source": "imgt",
        "description": "Year extracted from IMGT prop_list (#(YYYY))",
    },
    {
        "column_name": "year_recommended_imgt",
        "source": "imgt",
        "description": "Year extracted from IMGT rec_list (#(YYYY))",
    },
    {
        "column_name": "conditions_approved",
        "source": "therasabdab",
        "description": "Approved indications",
    },
    {
        "column_name": "conditions_active",
        "source": "therasabdab",
        "description": "Active indications",
    },
    {
        "column_name": "conditions_discontinued",
        "source": "therasabdab",
        "description": "Discontinued indications",
    },
    {
        "column_name": "notes",
        "source": "therasabdab",
        "description": "Notes",
    },
    {
        "column_name": "inn_number",
        "source": "imgt",
        "description": "IMGT INN number",
    },
    {
        "column_name": "imgt_id",
        "source": "imgt",
        "description": "IMGT/mAb-DB identifier",
    },
    {
        "column_name": "structure_2d",
        "source": "imgt",
        "description": "IMGT/2Dstructure-DB ID",
    },
    {
        "column_name": "structure_3d",
        "source": "imgt",
        "description": "IMGT/3Dstructure-DB ID",
    },
    {
        "column_name": "receptor_type",
        "source": "imgt",
        "description": "IMGT receptor type",
    },
    {
        "column_name": "specificity",
        "source": "imgt",
        "description": "Target specificity",
    },
    {
        "column_name": "species",
        "source": "imgt",
        "description": "Antibody species",
    },
    {
        "column_name": "conjugated",
        "source": "imgt",
        "description": "Conjugation info",
    },
    {
        "column_name": "origin_species",
        "source": "imgt",
        "description": "Origin clone species",
    },
    {
        "column_name": "origin_clone",
        "source": "imgt",
        "description": "Origin clone name",
    },
    {
        "column_name": "expression_system",
        "source": "imgt",
        "description": "Expression system",
    },
    {
        "column_name": "mechanism_of_action",
        "source": "imgt",
        "description": "Mechanism of action",
    },
    {
        "column_name": "clinical_indication",
        "source": "imgt",
        "description": "Clinical indication",
    },
    {
        "column_name": "clinical_domain",
        "source": "imgt",
        "description": "Clinical domain",
    },
    {
        "column_name": "clinical_trials",
        "source": "imgt",
        "description": "Clinical trial NCT IDs",
    },
    {
        "column_name": "engineered_variant",
        "source": "imgt",
        "description": "IMGT engineered variant",
    },
    {
        "column_name": "development_status",
        "source": "imgt",
        "description": "IMGT development status",
    },
    {
        "column_name": "biosimilars",
        "source": "imgt",
        "description": "Biosimilar information",
    },
    {
        "column_name": "mabsolve_format",
        "source": "mabsolve",
        "description": "Fc format",
    },
    {
        "column_name": "effector_function",
        "source": "mabsolve",
        "description": "Effector function modification",
    },
    {
        "column_name": "half_life",
        "source": "mabsolve",
        "description": "Half-life modification",
    },
    {
        "column_name": "heterodimerization",
        "source": "mabsolve",
        "description": "Heterodimerization technology",
    },
    {
        "column_name": "site_specific_conj",
        "source": "mabsolve",
        "description": "Site-specific conjugation",
    },
    {
        "column_name": "stabilized",
        "source": "mabsolve",
        "description": "Stabilized Fc",
    },
    {
        "column_name": "c_term_truncation",
        "source": "mabsolve",
        "description": "C-terminal truncation",
    },
    {
        "column_name": "other_fc_mod",
        "source": "mabsolve",
        "description": "Other Fc modification",
    },
    {
        "column_name": "imgt_2d_chain_count",
        "source": "imgt",
        "description": "Total chain count",
    },
    {
        "column_name": "imgt_2d_heavy_count",
        "source": "imgt",
        "description": "Heavy chain count",
    },
    {
        "column_name": "imgt_2d_light_count",
        "source": "imgt",
        "description": "Light chain count",
    },
    {
        "column_name": "imgt_2d_inn_id",
        "source": "imgt",
        "description": "IMGT 2D INN ID(s) observed in arm table",
    },
]


class MergedINNCollection(BaseModel):
    records: List[MergedINNRecord] = Field(default_factory=list)
    column_sources: List[ColumnSource] = Field(default_factory=list)

    @property
    def count(self) -> int:
        return len(self.records)

    def to_dataframe(
        self,
        include_sequences_without_fv: bool = False,
        collection=None,  # INNStructureCollection
        inn_id_to_name: Optional[Dict[str, str]] = None,
    ) -> pd.DataFrame:
        """Return a DataFrame of merged INN records.

        Args:
            include_sequences_without_fv: When True, backfills ``n_binders_imgt``,
                ``imgt_2d_chain_count``, ``imgt_2d_heavy_count``, ``imgt_2d_light_count``,
                and ``imgt_2d_inn_id`` for INNs where those fields are NaN or 0 (i.e. the
                arm pipeline found no VH/VL arms, as is the case for Fc-fusions).  All
                INNs must already be present in the DataFrame — this never adds new rows.
                Requires ``collection`` and ``inn_id_to_name``.
            collection: INNStructureCollection after update_with_realignment.
            inn_id_to_name: Mapping inn_id (str) → inn_name (str).

        Returns:
            DataFrame.  Updated rows are marked with ``_binders_updated=True`` when
            ``include_sequences_without_fv=True``; all other rows get ``_binders_updated=False``.
        """
        from .merger import _backfill_inn_binder_counts

        df = pd.DataFrame([r.model_dump() for r in self.records])
        if not include_sequences_without_fv:
            return df
        if collection is None or inn_id_to_name is None:
            raise ValueError(
                "include_sequences_without_fv=True requires both "
                "'collection' and 'inn_id_to_name' to be provided."
            )
        return _backfill_inn_binder_counts(df, collection, inn_id_to_name)

    def to_schema_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame([cs.model_dump() for cs in self.column_sources])


# ── Sequence table ─────────────────────────────────────────────────────


class MergedSequenceRecord(BaseModel):
    """One row per unique VH/VL arm per canonical INN.

    vh/vl hold the consensus sequence (TheraSAbDab preferred, falls back to IMGT).
    When the two sources disagree, sequence_mismatch=True and the four
    source-specific columns are populated; otherwise they are None.
    """

    inn_name: str
    inn_mol_name: Optional[str] = Field(
        None, description="Molecule-level INN root (first token)"
    )
    inn_number: Optional[str] = Field(None, description="IMGT INN number")
    molecule_slot: str = Field(..., description="Arm label (a, b, c, …)")
    sources: str = Field("", description="Comma-sep sources contributing this arm")

    vh: Optional[str] = None
    vl: Optional[str] = None

    sequence_mismatch: bool = False
    sequence_truncated: bool = False
    therasabdab_vh_mismatch: Optional[str] = Field(
        None, description="TheraSAbDab VH when it differs from IMGT"
    )
    therasabdab_vl_mismatch: Optional[str] = Field(
        None, description="TheraSAbDab VL when it differs from IMGT"
    )
    imgt_vh_mismatch: Optional[str] = Field(
        None, description="IMGT VH when it differs from TheraSAbDab"
    )
    imgt_vl_mismatch: Optional[str] = Field(
        None, description="IMGT VL when it differs from TheraSAbDab"
    )

    imgt_vh_v_gene: Optional[str] = None
    imgt_vh_j_gene: Optional[str] = None
    imgt_vl_v_gene: Optional[str] = None
    imgt_vl_j_gene: Optional[str] = None
    imgt_vh_total_mutations: Optional[int] = None
    imgt_vh_mutations: Optional[str] = None
    imgt_vl_total_mutations: Optional[int] = None
    imgt_vl_mutations: Optional[str] = None

    # IMGT constant chain / full-length sequences
    imgt_ch1: Optional[str] = None
    imgt_ch1_gene: Optional[str] = None
    imgt_ch1_total_mutations: Optional[int] = None
    imgt_ch2: Optional[str] = None
    imgt_ch2_gene: Optional[str] = None
    imgt_ch2_total_mutations: Optional[int] = None
    imgt_ch3: Optional[str] = None
    imgt_ch3_gene: Optional[str] = None
    imgt_ch3_total_mutations: Optional[int] = None
    imgt_chs: Optional[str] = None
    imgt_chs_gene: Optional[str] = None
    imgt_chs_total_mutations: Optional[int] = None
    imgt_tcr_alpha: Optional[str] = None
    imgt_tcr_alpha_gene: Optional[str] = None
    imgt_tcr_alpha_total_mutations: Optional[int] = None
    imgt_tcr_beta: Optional[str] = None
    imgt_tcr_beta_gene: Optional[str] = None
    imgt_tcr_beta_total_mutations: Optional[int] = None
    imgt_hinge_region: Optional[str] = None
    imgt_hinge_region_total_mutations: Optional[int] = None
    imgt_cl: Optional[str] = None
    imgt_cl_type: Optional[str] = None
    imgt_cl_total_mutations: Optional[int] = None

    # Mutation strings (from BindingArm / collapse_to_vh_vl_binder_info)
    imgt_ch_mutations: Optional[str] = None
    imgt_ch_mutations_imgt: Optional[str] = None
    imgt_ch_mutations_eu: Optional[str] = None
    imgt_ch_mutations_kabat: Optional[str] = None
    imgt_cl_mutations: Optional[str] = None
    imgt_cl_mutations_imgt: Optional[str] = None
    imgt_cl_mutations_eu: Optional[str] = None
    imgt_cl_mutations_kabat: Optional[str] = None

    imgt_full_length_heavy: Optional[str] = None
    imgt_full_length_light: Optional[str] = None
    imgt_linker: Optional[str] = None
    imgt_other_annotations: Optional[str] = None

    imgt_vh_chain_id: Optional[str] = None
    imgt_vl_chain_id: Optional[str] = None
    imgt_cl_chain_id: Optional[str] = None
    imgt_ch_detail_order_idxs: Optional[str] = None
    imgt_cl_detail_order_idxs: Optional[str] = None

    target: Optional[str] = None
    row_sources: Optional[str] = Field(
        None,
        description="Original TheraSAbDab row names (semicolon-joined)",
    )


SEQ_COLUMN_SOURCES: list[dict] = [
    {
        "column_name": "inn_name",
        "source": "derived",
        "description": "Canonical INN name",
    },
    {
        "column_name": "inn_mol_name",
        "source": "derived",
        "description": "Molecule-level INN root (first token)",
    },
    {
        "column_name": "inn_number",
        "source": "imgt",
        "description": "IMGT INN number",
    },
    {
        "column_name": "molecule_slot",
        "source": "derived",
        "description": "Arm label within INN",
    },
    {
        "column_name": "sources",
        "source": "derived",
        "description": "Contributing databases",
    },
    {
        "column_name": "vh",
        "source": "derived",
        "description": "Consensus VH sequence",
    },
    {
        "column_name": "vl",
        "source": "derived",
        "description": "Consensus VL sequence",
    },
    {
        "column_name": "sequence_mismatch",
        "source": "derived",
        "description": "True if VH or VL is genuinely different between TheraSAbDab and IMGT",
    },
    {
        "column_name": "sequence_truncated",
        "source": "derived",
        "description": "True if one source's VH or VL is a substring of the other (length difference only)",
    },
    {
        "column_name": "therasabdab_vh_mismatch",
        "source": "therasabdab",
        "description": "TheraSAbDab VH (only when differs from IMGT VH)",
    },
    {
        "column_name": "therasabdab_vl_mismatch",
        "source": "therasabdab",
        "description": "TheraSAbDab VL (only when differs from IMGT VL)",
    },
    {
        "column_name": "imgt_vh_mismatch",
        "source": "imgt",
        "description": "IMGT VH (only when differs from TheraSAbDab VH)",
    },
    {
        "column_name": "imgt_vl_mismatch",
        "source": "imgt",
        "description": "IMGT VL (only when differs from TheraSAbDab VL)",
    },
    {
        "column_name": "imgt_vh_v_gene",
        "source": "imgt",
        "description": "IMGT 2D VH V-gene allele",
    },
    {
        "column_name": "imgt_vh_d_gene",
        "source": "imgt",
        "description": "IMGT 2D VH D-gene allele",
    },
    {
        "column_name": "imgt_vh_j_gene",
        "source": "imgt",
        "description": "IMGT 2D VH J-gene allele",
    },
    {
        "column_name": "imgt_vl_v_gene",
        "source": "imgt",
        "description": "IMGT 2D VL V-gene allele",
    },
    {
        "column_name": "imgt_vl_j_gene",
        "source": "imgt",
        "description": "IMGT 2D VL J-gene allele",
    },
    {
        "column_name": "imgt_vh_dga_url",
        "source": "imgt",
        "description": "VH DomainGapAlign URL",
    },
    {
        "column_name": "imgt_vl_dga_url",
        "source": "imgt",
        "description": "VL DomainGapAlign URL",
    },
    {
        "column_name": "imgt_vh_total_mutations",
        "source": "imgt",
        "description": "VH total mutations from germline",
    },
    {
        "column_name": "imgt_vh_mutations",
        "source": "imgt",
        "description": "VH positional mutation string",
    },
    {
        "column_name": "imgt_vl_total_mutations",
        "source": "imgt",
        "description": "VL total mutations from germline",
    },
    {
        "column_name": "imgt_vl_mutations",
        "source": "imgt",
        "description": "VL positional mutation string",
    },
    {
        "column_name": "imgt_ch1",
        "source": "imgt",
        "description": "CH1 constant domain sequence",
    },
    {
        "column_name": "imgt_ch1_gene",
        "source": "imgt",
        "description": "CH1 gene allele",
    },
    {
        "column_name": "imgt_ch1_total_mutations",
        "source": "imgt",
        "description": "CH1 total mutations from germline",
    },
    {
        "column_name": "imgt_ch2",
        "source": "imgt",
        "description": "CH2 constant domain sequence",
    },
    {
        "column_name": "imgt_ch2_gene",
        "source": "imgt",
        "description": "CH2 gene allele",
    },
    {
        "column_name": "imgt_ch2_total_mutations",
        "source": "imgt",
        "description": "CH2 total mutations from germline",
    },
    {
        "column_name": "imgt_ch3",
        "source": "imgt",
        "description": "CH3 constant domain sequence",
    },
    {
        "column_name": "imgt_ch3_gene",
        "source": "imgt",
        "description": "CH3 gene allele",
    },
    {
        "column_name": "imgt_ch3_total_mutations",
        "source": "imgt",
        "description": "CH3 total mutations from germline",
    },
    {
        "column_name": "imgt_chs",
        "source": "imgt",
        "description": "CHS (C-terminal tail) sequence",
    },
    {
        "column_name": "imgt_chs_gene",
        "source": "imgt",
        "description": "CHS gene allele",
    },
    {
        "column_name": "imgt_chs_total_mutations",
        "source": "imgt",
        "description": "CHS total mutations from germline",
    },
    {
        "column_name": "imgt_tcr_alpha",
        "source": "imgt",
        "description": "TCR C-ALPHA constant domain sequence",
    },
    {
        "column_name": "imgt_tcr_alpha_gene",
        "source": "imgt",
        "description": "TCR C-ALPHA gene allele",
    },
    {
        "column_name": "imgt_tcr_alpha_total_mutations",
        "source": "imgt",
        "description": "TCR C-ALPHA total mutations from germline",
    },
    {
        "column_name": "imgt_tcr_beta",
        "source": "imgt",
        "description": "TCR C-BETA-2 constant domain sequence",
    },
    {
        "column_name": "imgt_tcr_beta_gene",
        "source": "imgt",
        "description": "TCR C-BETA-2 gene allele",
    },
    {
        "column_name": "imgt_tcr_beta_total_mutations",
        "source": "imgt",
        "description": "TCR C-BETA-2 total mutations from germline",
    },
    {
        "column_name": "imgt_hinge_region",
        "source": "imgt",
        "description": "Hinge region sequence",
    },
    {
        "column_name": "imgt_hinge_region_total_mutations",
        "source": "imgt",
        "description": "Hinge region total mutations from germline",
    },
    {
        "column_name": "imgt_cl",
        "source": "imgt",
        "description": "CL constant domain sequence",
    },
    {
        "column_name": "imgt_cl_type",
        "source": "imgt",
        "description": "CL type (kappa or lambda)",
    },
    {
        "column_name": "imgt_cl_total_mutations",
        "source": "imgt",
        "description": "CL total mutations from germline",
    },
    {
        "column_name": "imgt_ch_mutations",
        "source": "imgt",
        "description": "CH (all regions) raw positional mutations",
    },
    {
        "column_name": "imgt_ch_mutations_imgt",
        "source": "derived",
        "description": "CH mutations (IMGT numbered, all CH regions)",
    },
    {
        "column_name": "imgt_ch_mutations_eu",
        "source": "derived",
        "description": "CH mutations (EU numbered, all CH regions)",
    },
    {
        "column_name": "imgt_ch_mutations_kabat",
        "source": "derived",
        "description": "CH mutations (Kabat numbered, all CH regions)",
    },
    {
        "column_name": "imgt_cl_mutations",
        "source": "imgt",
        "description": "CL raw positional mutations",
    },
    {
        "column_name": "imgt_cl_mutations_imgt",
        "source": "derived",
        "description": "CL mutations (IMGT numbered)",
    },
    {
        "column_name": "imgt_cl_mutations_eu",
        "source": "derived",
        "description": "CL mutations (EU numbered)",
    },
    {
        "column_name": "imgt_cl_mutations_kabat",
        "source": "derived",
        "description": "CL mutations (Kabat numbered)",
    },
    {
        "column_name": "imgt_full_length_heavy",
        "source": "imgt",
        "description": "Full-length heavy chain sequence",
    },
    {
        "column_name": "imgt_full_length_light",
        "source": "imgt",
        "description": "Full-length light chain sequence",
    },
    {
        "column_name": "imgt_linker",
        "source": "imgt",
        "description": "Linker sequence (e.g. scFv)",
    },
    {
        "column_name": "imgt_other_annotations",
        "source": "imgt",
        "description": "Other region annotations (semicolon-joined)",
    },
    {
        "column_name": "imgt_vh_chain_id",
        "source": "imgt",
        "description": "VH chain ID",
    },
    {
        "column_name": "imgt_vl_chain_id",
        "source": "imgt",
        "description": "VL chain ID",
    },
    {
        "column_name": "imgt_cl_chain_id",
        "source": "imgt",
        "description": "CL chain ID",
    },
    {
        "column_name": "imgt_ch_detail_order_idxs",
        "source": "imgt",
        "description": "CH domain detail order indices (semicolon-joined)",
    },
    {
        "column_name": "imgt_cl_detail_order_idxs",
        "source": "imgt",
        "description": "CL domain detail order indices (semicolon-joined)",
    },
    {
        "column_name": "target",
        "source": "therasabdab",
        "description": "Target antigen for this arm",
    },
    {
        "column_name": "row_sources",
        "source": "therasabdab",
        "description": "Original TheraSAbDab row names",
    },
]


class MergedSequenceCollection(BaseModel):
    records: List[MergedSequenceRecord] = Field(default_factory=list)
    column_sources: List[ColumnSource] = Field(default_factory=list)

    @property
    def count(self) -> int:
        return len(self.records)

    def to_dataframe(
        self,
        include_sequences_without_fv: bool = False,
        collection=None,  # INNStructureCollection — required when include_sequences_without_fv=True
        inn_id_to_name: (dict[str, str] | None) = None,  # inn_id → inn_name map
    ) -> pd.DataFrame:
        """Return a DataFrame of merged sequence records.

        Args:
            include_sequences_without_fv: When True, appends rows for INNs that have
                constant chain data in the IMGT 2D collection but no VH/VL arm (e.g.
                Fc-fusions like abatacept).  Requires ``collection`` and ``inn_id_to_name``.
            collection: INNStructureCollection after update_with_realignment.
            inn_id_to_name: Mapping from inn_id (str) → inn_name (str).  In the notebook
                this is ``imgt_inn_number_to_inn_map``.

        Returns:
            DataFrame.  When ``include_sequences_without_fv=True``, a ``_supplemented``
            boolean column is added: False for standard arm-pipeline rows, True for
            the appended Fc-fusion rows.
        """
        from .merger import (
            _supplement_from_collection,  # avoid circular at module level
        )

        df = pd.DataFrame([r.model_dump() for r in self.records])
        if not include_sequences_without_fv:
            return df
        if collection is None or inn_id_to_name is None:
            raise ValueError(
                "include_sequences_without_fv=True requires both "
                "'collection' and 'inn_id_to_name' to be provided."
            )
        return _supplement_from_collection(df, collection, inn_id_to_name)

    def to_schema_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame([cs.model_dump() for cs in self.column_sources])


# ── Combined output ────────────────────────────────────────────────────


class MergedDatabase(BaseModel):
    inn: MergedINNCollection = Field(default_factory=MergedINNCollection)
    sequences: MergedSequenceCollection = Field(
        default_factory=MergedSequenceCollection
    )

    # Merge tracking: maps original names to canonical names
    fuzzy_aliases: Dict[str, str] = Field(
        default_factory=dict,
        description="Mol-name fallback mappings used during source joins ({inn_name or inn|source: matched_mol_name})",
    )
    inn_id_consolidations: Dict[str, List[str]] = Field(
        default_factory=dict,
        description="Merged INN names with multiple INN IDs across sources",
    )
    excluded_imgt: Dict[str, Dict[str, str]] = Field(
        default_factory=dict,
        description="Reserved merge-time exclusion tracking",
    )
    reverse_map: Dict[str, str] = Field(
        default_factory=dict,
        description="Reverse fuzzy fallback map: matched_mol_name → inn|source",
    )
    imgt_year_parse_issues: Dict[str, List[str]] = Field(
        default_factory=lambda: {"prop_list": [], "rec_list": []},
        description="Unique invalid IMGT year-source strings without parenthesized years, grouped by source column",
    )
