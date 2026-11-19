from typing import List, Optional

from pydantic import BaseModel, Field


class TheraSAbDabDetail(BaseModel):
    """Detailed info from a single TheraSAbDab molecule detail page."""

    therapeutic: str = Field("")
    target: str = Field("")
    heavy_chain: str = Field("")
    light_chain: str = Field("")
    pdb_100: str = Field("", description="100% seqID Fv structures")
    pdb_99: str = Field("", description="99% seqID Fv structures")
    pdb_95_98: str = Field("", description="95-98% seqID Fv structures")
    format: str = Field("")
    isotype: str = Field("")
    highest_clinical_trial: str = Field("")
    estimated_status: str = Field("")
    dev_technology: str = Field("")
    year_proposed: str = Field("")
    year_recommended: str = Field("")
    companies: str = Field("")
    conditions_approved: str = Field("")
    conditions_active: str = Field("")
    conditions_discontinued: str = Field("")
    notes: str = Field("")

    @classmethod
    def from_sections(cls, structural: dict[str, str], metadata: dict[str, str]) -> "TheraSAbDabDetail":
        return cls(
            therapeutic=structural.get("Therapeutic", ""),
            target=structural.get("Target", ""),
            heavy_chain=structural.get("Heavy Chain", ""),
            light_chain=structural.get("Light Chain", ""),
            pdb_100=structural.get("100% seqID Fv Structure", ""),
            pdb_99=structural.get("99% seqID Fv Structure", ""),
            pdb_95_98=structural.get("95-98% seqID Fv Structure", ""),
            format=metadata.get("Format", ""),
            isotype=metadata.get("Isotype", ""),
            highest_clinical_trial=metadata.get("Highest Clinical Trial (Feb '25)", ""),
            estimated_status=metadata.get("Estimated Status (Feb '25)", ""),
            dev_technology=metadata.get("Recorded Developmental Technology", ""),
            year_proposed=metadata.get("INN Year Proposed", ""),
            year_recommended=metadata.get("INN Year Recommended", ""),
            companies=metadata.get("Companies Involved", ""),
            conditions_approved=metadata.get("Conditions Approved", ""),
            conditions_active=metadata.get("Conditions Active", ""),
            conditions_discontinued=metadata.get("Conditions Discontinued", ""),
            notes=metadata.get("Notes", ""),
        )


class TheraSAbDabRecord(BaseModel):
    """Single antibody row from the TheraSAbDab search table."""

    name: str = Field(..., description="INN therapeutic name")
    format: str = Field(..., description="e.g. Whole mAb, Fab, Bispecific mAb")
    highest_clinical_trial: str = Field("")
    estimated_status: str = Field("")
    target: str = Field("")
    year_proposed: Optional[int] = Field(None)
    has_100_si_struc: bool = Field(False)
    has_99_si_struc: bool = Field(False)
    has_95_98_si_struc: bool = Field(False)
    isotype: str = Field("")
    vh1: str = Field("")
    vl1: str = Field("")
    vh2: str = Field("")
    vl2: str = Field("")
    companies: str = Field("")
    year_received: Optional[str] = Field("")
    detail: Optional[TheraSAbDabDetail] = Field(None)
    detail_url: str = Field("")

    @staticmethod
    def _parse_bool(val: str) -> bool:
        return val.strip().upper() == "YES"

    @staticmethod
    def _parse_year(val: str) -> Optional[int]:
        try:
            return int(val.strip())
        except (ValueError, TypeError):
            return None

    @classmethod
    def from_row(cls, table_cells: list[str], url_params: dict, detail_url: str = "") -> "TheraSAbDabRecord":
        return cls(
            name=table_cells[0],
            format=table_cells[1],
            highest_clinical_trial=table_cells[2],
            estimated_status=table_cells[3],
            target=table_cells[4],
            year_proposed=cls._parse_year(table_cells[5]),
            has_100_si_struc=cls._parse_bool(table_cells[6]),
            has_99_si_struc=cls._parse_bool(table_cells[7]),
            has_95_98_si_struc=cls._parse_bool(table_cells[8]),
            isotype=url_params.get("isotype", [""])[0],
            vh1=url_params.get("heavy1", [""])[0],
            vl1=url_params.get("light1", [""])[0],
            vh2=url_params.get("heavy2", ["na"])[0],
            vl2=url_params.get("light2", ["na"])[0],
            companies=url_params.get("companies", [""])[0],
            year_received=url_params.get("yearrec", [""])[0],
            detail_url=detail_url,
        )


class TheraSAbDabCollection(BaseModel):
    """Container for TheraSAbDabDetail objects from bulk fetch."""

    therapeutics: list[TheraSAbDabDetail] = Field(default_factory=list)

    @property
    def count(self) -> int:
        return len(self.therapeutics)

    def to_dataframe(self):
        import pandas as pd

        return pd.DataFrame([t.model_dump() for t in self.therapeutics])


class TheraSAbDabMergedRecord(BaseModel):
    """Search table + detail page merged into a single flat record."""

    name: str
    target: str
    format: str
    isotype: str
    vh1: str = ""
    vl1: str = ""
    vh2: str = ""
    vl2: str = ""
    highest_clinical_trial: str = ""
    estimated_status: str = ""
    has_100_si_struc: bool = False
    has_99_si_struc: bool = False
    has_95_98_si_struc: bool = False
    pdb_100: str = ""
    pdb_99: str = ""
    pdb_95_98: str = ""
    dev_technology: str = ""
    companies: str = ""
    year_proposed: Optional[int] = None
    year_recommended: str = ""
    year_received: Optional[str] = ""
    conditions_approved: str = ""
    conditions_active: str = ""
    conditions_discontinued: str = ""
    notes: str = ""

    @classmethod
    def from_record(cls, t: TheraSAbDabRecord) -> "TheraSAbDabMergedRecord":
        if t.detail is None:
            raise ValueError(f"No detail fetched for {t.name}. Call fetch_detail first.")
        d = t.detail
        checks = [
            ("name", "therapeutic", "name"),
            ("target", "target", "target"),
            ("format", "format", "format"),
            ("isotype", "isotype", "isotype"),
            ("vh1", "heavy_chain", "heavy chain"),
            ("vl1", "light_chain", "light chain"),
        ]
        for t_field, d_field, label in checks:
            t_val = str(getattr(t, t_field, "")).strip().lower()
            d_val = str(getattr(d, d_field, "")).strip().lower()
            if t_val and d_val and t_val not in ("na", "") and d_val not in ("na", ""):
                if t_val != d_val:
                    raise ValueError(
                        f"Mismatch in {label} for {t.name}: "
                        f"search='{getattr(t, t_field)}' vs detail='{getattr(d, d_field)}'"
                    )
        return cls(
            name=t.name,
            target=t.target,
            format=t.format,
            isotype=t.isotype or d.isotype,
            vh1=t.vh1 or d.heavy_chain,
            vl1=t.vl1 or d.light_chain,
            vh2=t.vh2,
            vl2=t.vl2,
            highest_clinical_trial=t.highest_clinical_trial or d.highest_clinical_trial,
            estimated_status=t.estimated_status or d.estimated_status,
            has_100_si_struc=t.has_100_si_struc,
            has_99_si_struc=t.has_99_si_struc,
            has_95_98_si_struc=t.has_95_98_si_struc,
            pdb_100=d.pdb_100,
            pdb_99=d.pdb_99,
            pdb_95_98=d.pdb_95_98,
            dev_technology=d.dev_technology,
            companies=d.companies or t.companies,
            year_proposed=t.year_proposed,
            year_recommended=d.year_recommended,
            year_received=t.year_received,
            conditions_approved=d.conditions_approved,
            conditions_active=d.conditions_active,
            conditions_discontinued=d.conditions_discontinued,
            notes=d.notes,
        )


class TheraSAbDabINNRecord(BaseModel):
    """One row per canonical INN. Metadata fields that differ across trailing-number
    rows of the same antibody are semicolon-joined. n_binders = distinct VH/VL arms."""

    inn_name: str
    n_binders: int = Field(..., description="Number of distinct VH/VL binding arms")
    format: Optional[str] = None
    isotype: Optional[str] = None
    highest_clinical_trial: Optional[str] = None
    estimated_status: Optional[str] = None
    has_100_si_struc: Optional[str] = None
    has_99_si_struc: Optional[str] = None
    has_95_98_si_struc: Optional[str] = None
    companies: Optional[str] = None
    year_proposed: Optional[str] = None
    year_received: Optional[str] = None


class TheraSAbDabINNCollection(BaseModel):
    records: List[TheraSAbDabINNRecord] = Field(default_factory=list)

    @property
    def count(self) -> int:
        return len(self.records)

    def to_dataframe(self):
        import pandas as pd

        return pd.DataFrame([r.model_dump() for r in self.records])


class TheraSAbDabSequenceRecord(BaseModel):
    """One row per unique VH/VL arm within an INN."""

    inn_name: str
    molecule_slot: str = Field(..., description="Arm label (a, b, c, …)")
    vh: Optional[str] = None
    vl: Optional[str] = None
    target: Optional[str] = Field(None, description="Semicolon-joined unique targets for this arm")
    row_sources: List[str] = Field(
        default_factory=list,
        description="Original TheraSAbDab row names that contributed this arm",
    )


class TheraSAbDabSequenceCollection(BaseModel):
    records: List[TheraSAbDabSequenceRecord] = Field(default_factory=list)

    @property
    def count(self) -> int:
        return len(self.records)

    def to_dataframe(self):
        import pandas as pd

        rows = [{**r.model_dump(exclude={"row_sources"}), "row_sources": ";".join(r.row_sources)} for r in self.records]
        return pd.DataFrame(rows)
