"""Pandera schema for the raw IMGT mAb-DB search-table export.

Validates CSV exports such as ``notebooks/db_sources/IMGT_mabdb_033026.csv``
before they are parsed via :mod:`imgt_mabdb` (see
``src/inn_analysis_ops/models/imgt_mabdb.py`` / ``IMGTMAbDBScraper``) into
:class:`models.imgt_mabdb.IMGTRecord` objects. Column names/types below were
derived directly from that export.

Usage
-----
    import pandas as pd
    from inn_analysis_ops.data_io.imgt_mab_dab_schema import IMGTMabDbCsvSchema

    df = pd.read_csv("notebooks/db_sources/IMGT_mabdb_033026.csv")
    validated = IMGTMabDbCsvSchema.validate(df)
"""

import pandera.pandas as pa
from pandera.typing import Series

# receptor_type is a small controlled vocabulary of IMGT receptor categories:
# IG (immunoglobulin), TR (T-cell receptor), CPCA/RPI/FPIA (chimeric/fusion
# protein and receptor-Ig constructs).
_RECEPTOR_TYPE_VALUES = {"IG", "TR", "CPCA", "RPI", "FPIA"}

# development_status is a comma-joined combination of clinical phases;
# individual tokens observed in the data (validated per-token below, not as a
# whole-cell isin, since cells frequently combine multiple phases e.g.
# "Phase I,Phase II,Phase I/II,Phase III", sometimes with stray blank tokens
# from double commas).
_DEVELOPMENT_STATUS_TOKENS = {
    "Discontinued",
    "Phase I",
    "Phase I/II",
    "Phase II",
    "Phase II/III",
    "Phase III",
    "Phase IV",
    "Phase M",
    "Preclinical",
    "Proof of concept",
    "Withdrawn",
}


class IMGTMabDbCsvSchema(pa.DataFrameModel):
    """Schema for the flat IMGT mAb-DB search-table CSV export (one row per receptor)."""

    # IMGT/mAb-DB identifier — the only column guaranteed unique/non-null;
    # `inn` is NOT unique (e.g. "interferon alfa" spans multiple imgt_id rows).
    imgt_id: Series[int] = pa.Field(unique=True, gt=0)

    inn: Series[str] = pa.Field(nullable=True)
    inn_number: Series[float] = pa.Field(nullable=True, ge=0)

    # Equal to inn_number whenever both are present (2D structure DB is keyed
    # by the same numeric ID); kept as its own column because it can be null
    # independently of inn_number in a handful of rows.
    structure_2d: Series[float] = pa.Field(nullable=True, ge=0)
    structure_3d: Series[str] = pa.Field(nullable=True)  # PDB code, e.g. "1qnz"

    receptor_type: Series[str] = pa.Field(isin=_RECEPTOR_TYPE_VALUES)

    # Always null in current exports; kept as float (pandas' inferred dtype
    # for an all-NaN column) rather than str so validation doesn't force a
    # premature type decision before real data is observed.
    format: Series[float] = pa.Field(nullable=True)

    specificity: Series[str] = pa.Field(nullable=True)
    company: Series[str] = pa.Field(nullable=True)

    development_status: Series[str] = pa.Field(nullable=True)
    clinical_domain: Series[str] = pa.Field(
        nullable=True
    )  # semicolon-joined clinical domain tags

    prop_list: Series[str] = pa.Field(nullable=True)  # INN Proposed List number(year)
    rec_list: Series[str] = pa.Field(nullable=True)  # INN Recommended List number(year)

    common_name: Series[str] = pa.Field(nullable=True)
    proprietary_name: Series[str] = pa.Field(nullable=True)
    species: Series[str] = pa.Field(
        nullable=True
    )  # e.g. "Homo sapiens", "Humanized", "Chimeric"
    receptor_id: Series[str] = pa.Field(
        nullable=True
    )  # e.g. "IgG1 - kappa", "Fab - G1 - kappa"
    conjugated: Series[str] = pa.Field(
        nullable=True
    )  # free-text conjugation/fusion description
    dev_technology: Series[str] = pa.Field(nullable=True)
    origin_species: Series[str] = pa.Field(nullable=True)
    origin_clone: Series[str] = pa.Field(nullable=True)
    expression_system: Series[str] = pa.Field(nullable=True)
    application: Series[str] = pa.Field(
        nullable=True
    )  # e.g. "Therapeutic", "Diagnostic"
    mechanism_of_action: Series[str] = pa.Field(nullable=True)
    clinical_indication: Series[str] = pa.Field(nullable=True)

    # Raw IMGT header text retained verbatim (not snake_cased) in this export.
    regulatory_agency_status_and_year: Series[str] = pa.Field(
        alias="Regulatory agency status and year", nullable=True
    )
    clinical_trials: Series[str] = pa.Field(nullable=True)
    authority_decisions: Series[str] = pa.Field(nullable=True)
    external_links: Series[str] = pa.Field(alias="External links", nullable=True)
    citations_notes_references: Series[str] = pa.Field(
        alias="Citations/ Notes/ References", nullable=True
    )
    biosimilars: Series[str] = pa.Field(nullable=True)

    class Config:
        strict = True
        coerce = True

    @pa.check("development_status")
    def _valid_development_status_tokens(cls, series: Series[str]) -> Series[bool]:
        """Every comma-separated token in a non-null cell must be a known status."""

        def _valid(cell) -> bool:
            if cell is None or (
                isinstance(cell, float) and cell != cell
            ):  # NaN check w/o pandas import
                return True
            tokens = {t.strip() for t in str(cell).split(",")} - {""}
            return tokens <= _DEVELOPMENT_STATUS_TOKENS

        return series.apply(_valid)
