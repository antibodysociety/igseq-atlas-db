import asyncio
import re
from collections import defaultdict
from string import ascii_lowercase
from typing import Optional
from urllib.parse import parse_qs, urlparse

import aiohttp
import pandas as pd
import requests
from bs4 import BeautifulSoup

from inn_analysis_ops.models.therasabdab import (
    TheraSAbDabINNCollection,
    TheraSAbDabINNRecord,
    TheraSAbDabSequenceCollection,
    TheraSAbDabSequenceRecord,
)


import pandera.pandas as pa
from pandera.typing import Series, DataFrame

# Amino-acid sequence or the literal placeholder "na" used by TheraSAbDab
# when a chain (e.g. vh2/vl2 for non-bispecific entries) is not applicable.
_SEQUENCE_OR_NA = r"^(na|[A-Za-z]+)$"

_HIGHEST_CLINICAL_TRIAL_VALUES = {
    "TBC",
    "Unknown",
    "Preclinical",
    "Phase-I",
    "Phase-I/II",
    "Phase-I/Phase-II",
    "Phase-II",
    "Phase-II/III",
    "Phase-III",
    "Preregistration",
    "Preregistration (w)",
    "Approved",
    "Approved (w)",
    "Approved (withdrawn)",
}

_ESTIMATED_STATUS_VALUES = {
    "Active",
    "Discontinued",
    "TBC",
    "NFD",
    "Approved",
    "ND",
}


class TheraSAbDabCsvSchema(pa.DataFrameModel):
    """Schema for the flat TheraSAbDab search-table CSV export (one row per INN)."""

    name: Series[str] = pa.Field(unique=True, str_length={"min_value": 1})
    target: Series[str] = pa.Field(nullable=False)
    format: Series[str] = pa.Field(nullable=False)
    isotype: Series[str] = pa.Field(nullable=False)

    vh1: Series[str] = pa.Field(str_matches=_SEQUENCE_OR_NA)
    vl1: Series[str] = pa.Field(str_matches=_SEQUENCE_OR_NA)
    vh2: Series[str] = pa.Field(str_matches=_SEQUENCE_OR_NA)
    vl2: Series[str] = pa.Field(str_matches=_SEQUENCE_OR_NA)

    highest_clinical_trial: Series[str] = pa.Field(isin=_HIGHEST_CLINICAL_TRIAL_VALUES)
    estimated_status: Series[str] = pa.Field(isin=_ESTIMATED_STATUS_VALUES)

    has_100_si_struc: Series[bool] = pa.Field(nullable=False)
    has_99_si_struc: Series[bool] = pa.Field(nullable=False)
    has_95_98_si_struc: Series[bool] = pa.Field(nullable=False)

    # "[Fvs: HL]"-style PDB hit lists; blank when no structure at that
    # sequence-identity tier exists.
    pdb_100: Series[str] = pa.Field(nullable=True)
    pdb_99: Series[str] = pa.Field(nullable=True)
    pdb_95_98: Series[str] = pa.Field(nullable=True)

    dev_technology: Series[str] = pa.Field(nullable=True)
    companies: Series[str] = pa.Field(nullable=False)

    year_proposed: Series[int] = pa.Field(ge=1980, le=2100)
    year_recommended: Series[float] = pa.Field(ge=1980, le=2100, nullable=True)
    year_received: Series[float] = pa.Field(ge=1980, le=2100, nullable=True)

    conditions_approved: Series[str] = pa.Field(nullable=True)
    conditions_active: Series[str] = pa.Field(nullable=True)
    conditions_discontinued: Series[str] = pa.Field(nullable=True)
    notes: Series[str] = pa.Field(nullable=True)

    class Config:
        strict = True
        coerce = True


def _clean_seq(val) -> Optional[str]:
    if pd.isna(val):
        return None
    s = str(val).strip()
    return None if s.lower() == "na" or not s else s


def _target_part(target_str: Optional[str], idx: int) -> Optional[str]:
    if not target_str:
        return None
    parts = [p.strip() for p in target_str.split(";")]
    return parts[idx] if idx < len(parts) else None


_META_COLS = [
    "format",
    "isotype",
    "highest_clinical_trial",
    "estimated_status",
    "has_100_si_struc",
    "has_99_si_struc",
    "has_95_98_si_struc",
    "companies",
    "year_proposed",
    "year_received",
]


def _base_name(name: str) -> str:
    return re.sub(r"\d+$", "", name).strip()


def _join_col(series: pd.Series) -> Optional[str]:
    """Semicolon-join unique non-null values; return None if all null."""
    vals = [str(v) for v in series.dropna().unique() if str(v).strip()]
    return ";".join(vals) if vals else None


def _arms_from_group(group: pd.DataFrame) -> list[dict]:
    """Extract unique (VH, VL) arms from a group of rows sharing a base INN name."""
    pair_info: dict = defaultdict(lambda: {"targets": set(), "sources": []})
    for _, row in group.iterrows():
        tgt_str = None if pd.isna(row.get("target")) else str(row["target"])
        for arm_idx, (vh_col, vl_col) in enumerate([("vh1", "vl1"), ("vh2", "vl2")]):
            vh = _clean_seq(row.get(vh_col))
            vl = _clean_seq(row.get(vl_col))
            if vh is None and vl is None:
                continue
            key = (vh, vl)
            t = _target_part(tgt_str, arm_idx)
            if t:
                pair_info[key]["targets"].add(t)
            pair_info[key]["sources"].append(row["name"])
    return [
        {
            "key": key,
            "targets": info["targets"],
            "sources": list(dict.fromkeys(info["sources"])),
        }
        for key, info in pair_info.items()
    ]


def build_inn_table(thera_df: pd.DataFrame) -> TheraSAbDabINNCollection:
    """One record per canonical INN. Metadata columns that differ across
    trailing-number rows of the same antibody are semicolon-joined."""
    df = thera_df.copy()
    df["_base"] = df["name"].apply(_base_name)

    records: list[TheraSAbDabINNRecord] = []
    for base, group in df.groupby("_base", sort=False):
        arms = _arms_from_group(group)
        meta = {
            col: _join_col(group[col]) for col in _META_COLS if col in group.columns
        }
        records.append(TheraSAbDabINNRecord(inn_name=base, n_binders=len(arms), **meta))

    return TheraSAbDabINNCollection(records=records)


def build_sequence_table(
    thera_df: DataFrame[TheraSAbDabCsvSchema],
) -> TheraSAbDabSequenceCollection:
    """One record per unique VH/VL arm per canonical INN."""
    df = thera_df.copy()
    df["_base"] = df["name"].apply(_base_name)

    records: list[TheraSAbDabSequenceRecord] = []
    for base, group in df.groupby("_base", sort=False):
        for slot_idx, arm in enumerate(_arms_from_group(group)):
            (vh, vl), targets, sources = arm["key"], arm["targets"], arm["sources"]
            slot_label = ascii_lowercase[slot_idx] if slot_idx < 26 else str(slot_idx)
            records.append(
                TheraSAbDabSequenceRecord(
                    inn_name=base,
                    molecule_slot=slot_label,
                    vh=vh,
                    vl=vl,
                    target=";".join(sorted(targets)) or None,
                    row_sources=sources,
                )
            )

    return TheraSAbDabSequenceCollection(records=records)
