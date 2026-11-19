from typing import Optional

import pandas as pd
from pydantic import BaseModel, Field


class MAbSolveRecord(BaseModel):
    """Single antibody INN record from mAbsolve INN Analysis table."""

    inn: str = Field(..., description="International Nonproprietary Name")
    format: str = Field("", description="Antibody format (IgG, Fab, Fc fusion, etc.)")
    antigen: str = Field("", description="Target antigen")
    effector_function: str = Field(
        "", description="Effector function modification (↓/↑)"
    )
    half_life: str = Field("", description="Half-life modification (↑)")
    heterodimerization: bool = Field(False, description="Heterodimerization technology")
    site_specific_conj: bool = Field(False, description="Site-specific conjugation")
    stabilized: bool = Field(False, description="Stabilized Fc")
    c_term_truncation: bool = Field(False, description="C-terminal truncation")
    other: bool = Field(False, description="Other Fc modification")

    @classmethod
    def from_row(cls, cells: list[str], headers: list[str]) -> "MAbSolveRecord":
        """Build from a list of cell text values and column headers."""

        def _get(idx: int) -> str:
            return cells[idx].strip() if idx < len(cells) else ""

        def _bool(idx: int) -> bool:
            return _get(idx).lower() in ("yes", "✓", "✔", "true", "x")

        # First 3 columns are always name, format, antigen
        data = {"inn": _get(0), "format": _get(1), "antigen": _get(2)}

        # Map remaining columns dynamically by header text
        col_field_map = _build_column_map(headers)
        for idx, header in enumerate(headers[3:], start=3):
            field = col_field_map.get(idx)
            if not field:
                continue
            field_info = cls.model_fields.get(field)
            if field_info and field_info.annotation is bool:
                data[field] = _bool(idx)
            else:
                data[field] = _get(idx)

        return cls(**data)


# Fuzzy matching of header text to model fields
_HEADER_KEYWORDS = {
    "effector_function": ["effector"],
    "half_life": ["half"],
    "heterodimerization": ["heterodimer"],
    "site_specific_conj": ["site specific"],
    "stabilized": ["stabiliz"],
    "c_term_truncation": ["truncat"],
    "other": ["other"],
}


def _build_column_map(headers: list[str]) -> dict[int, str]:
    """Map column index -> field name using fuzzy header matching."""
    col_map: dict[int, str] = {}
    for idx, header in enumerate(headers):
        if idx < 3:
            continue
        h_lower = header.lower().strip()
        for field, keywords in _HEADER_KEYWORDS.items():
            if any(kw in h_lower for kw in keywords):
                col_map[idx] = field
                break
    return col_map


class MAbSolveCollection(BaseModel):
    """Container for mAbsolve INN Analysis results."""

    records: list[MAbSolveRecord] = Field(default_factory=list)

    @property
    def count(self) -> int:
        return len(self.records)

    def to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame([r.model_dump() for r in self.records])
