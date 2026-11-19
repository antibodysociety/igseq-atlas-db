import re

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field


def _to_snake(name: str) -> str:
    """Convert a column header to snake_case."""
    s = re.sub(r"[/\-\s]+", "_", name)
    s = re.sub(r"[^\w]", "", s)
    s = re.sub(r"_+", "_", s).strip("_").lower()
    return s


# Explicit mapping for known IMGT mAb-DB column headers -> field names
IMGT_COLUMN_MAP = {
    "IMGT/mAb-DB ID": "imgt_id",
    "INN": "inn",
    "INN Number": "inn_number",
    "INN Prop. List": "prop_list",
    "INN Rec. List": "rec_list",
    "Common name": "common_name",
    "Proprietary name": "proprietary_name",
    "IMGT receptor type": "receptor_type",
    "Species": "species",
    "Radiolabelled / Conjugated / Fused": "conjugated",
    "Format": "format",
    "Receptor identification": "receptor_id",
    "IMGT/2Dstructure-DB": "structure_2d",
    "IMGT/3Dstructure-DB": "structure_3d",
    "Origin clone species": "origin_species",
    "Origin clone name": "origin_clone",
    "Specificity target name and species": "specificity",
    "Clinical indication": "clinical_indication",
    "IMGT engineered variant": "engineered_variant",
    "Development status": "development_status",
    "Regulatory / external references": "external_refs",
    "Company": "company",
    "Expression system": "expression_system",
    "Development Technology": "dev_technology",
    "Authority decisions": "authority_decisions",
    "References": "references",
    "Application": "application",
    "Clinical domain": "clinical_domain",
    "Mechanism of action": "mechanism_of_action",
    "Clinical trials": "clinical_trials",
    "Biosimilars": "biosimilars",
}


class IMGTRecord(BaseModel):
    """Single antibody record from IMGT mAb-DB."""

    model_config = ConfigDict(extra="allow")

    inn: str = Field("", description="International Nonproprietary Name")
    inn_number: str = Field("", description="INN Number")
    imgt_id: str = Field("", description="IMGT/mAb-DB identifier")
    structure_2d: str = Field("", description="IMGT/2Dstructure-DB ID")
    structure_3d: str = Field("", description="IMGT/3Dstructure-DB ID")
    receptor_type: str = Field("", description="IMGT receptor type")
    format: str = Field("", description="Antibody format")
    specificity: str = Field("", description="Target specificity")
    company: str = Field("", description="Developing company")
    development_status: str = Field("", description="Clinical development status")
    clinical_domain: str = Field("", description="Clinical domain")

    @classmethod
    def from_series(cls, row: pd.Series, column_map: dict) -> "IMGTRecord":
        data = {}
        for col, val in row.items():
            field = column_map.get(col, _to_snake(str(col)))
            data[field] = "" if pd.isna(val) else str(val)
        return cls(**data)


class IMGTCollection(BaseModel):
    """Container for IMGT mAb-DB search results."""

    records: list[IMGTRecord] = Field(default_factory=list)

    @property
    def count(self) -> int:
        return len(self.records)

    def to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame([r.model_dump() for r in self.records])
