"""Pandera schema for the raw mAbsolve  wilkinson and hale INN Analysis table export"""

import pandera.pandas as pa
from pandera.typing import Series

# Fixed vocabulary of antibody format labels used by mAbsolve's INN Analysis
# table (includes numbered "(type N)" sub-variants of several formats).
_FORMAT_VALUES = {
    "Bispecific IgG (type 1)",
    "Bispecific IgG (type 2)",
    "Bispecific IgG (type 3)",
    "Bispecific IgG (type 4)",
    "Bispecific IgG (type 5)",
    "Bispecific IgG (type 6)",
    "Diabody",
    "Diabody-Fc (type 1)",
    "Diabody-Fc (type 2)",
    "F(ab')2",
    "Fab",
    "Fab-IgG (type 1)",
    "Fab-IgG (type 2)",
    "Fab-IgG (type 3)",
    "Fab-protein fusion (type 1)",
    "Fab-protein fusion (type 2)",
    "Fc domain",
    "Fc fusion protein (type 1)",
    "Fc fusion protein (type 2)",
    "Fc fusion protein (type 3)",
    "Fc fusion protein (type 4)",
    "Fc fusion protein (type 5)",
    "Fc fusion protein (type 6)",
    "Fc fusion protein (type 7)",
    "Fv-IgG",
    "Fv-protein fusion",
    "Heterodimeric Fab-Fc",
    "Heterodimeric Fab/scFv-Fc",
    "IgG",
    "IgG-protein fusion (type 1)",
    "IgG-protein fusion (type 2)",
    "IgG-protein fusion (type 3)",
    "IgG-scFv (type 1)",
    "IgG-scFv (type 2)",
    "IgG-scFv (type 3)",
    "IgG-scFv (type 4)",
    "IgM",
    "IgM with J chain fusion",
    "Minibody",
    "Protease activated IgG",
    "Protein conjugated Fc",
    "Small molecule conjugated Fc",
    "Tandem dAb (type 1)",
    "Tandem dAb (type 2)",
    "Tandem dAb-Fc",
    "Tandem scFv",
    "Tandem scFv-scFc",
    "Triple dAb (type 1)",
    "Triple dAb (type 2)",
    "Triple dAb (type 3)",
    "Unconfirmed",
    "dAb",
    "dAb-Fc",
    "di-Fab'",
    "scFv",
    "scFv-Fc",
    "scFv-TCR fusion",
    "scFv-protein fusion",
}

# effector_function / half_life cells hold a directional-change glyph when a
# modification is annotated, and are blank/NaN otherwise.
_DIRECTION_VALUES = {"\u2191", "\u2193"}  # ↑ (increased), ↓ (decreased)


class WilkHaleCsvSchema(pa.DataFrameModel):
    """Schema for the flat mAbsolve INN Analysis CSV export (one row per INN)."""

    inn: Series[str] = pa.Field(unique=True, str_length={"min_value": 1})
    format: Series[str] = pa.Field(isin=_FORMAT_VALUES)
    antigen: Series[str] = pa.Field(nullable=False, str_length={"min_value": 1})

    effector_function: Series[str] = pa.Field(isin=_DIRECTION_VALUES, nullable=True)
    half_life: Series[str] = pa.Field(isin=_DIRECTION_VALUES, nullable=True)

    heterodimerization: Series[bool] = pa.Field(nullable=False)
    site_specific_conj: Series[bool] = pa.Field(nullable=False)
    stabilized: Series[bool] = pa.Field(nullable=False)
    c_term_truncation: Series[bool] = pa.Field(nullable=False)
    other: Series[bool] = pa.Field(nullable=False)

    class Config:
        strict = True
        coerce = True
