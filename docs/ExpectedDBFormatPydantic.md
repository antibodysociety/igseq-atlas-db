# Pydantic Format of Database Object

This document defines the exact Pydantic/pandera object contracts expected from each raw data source in this repo, and how each source's tabular export maps onto its corresponding model.


---

# Expected IMGT 2D Structure Parse Format

This section defines the exact Pydantic object contract expected from a description of INN sequence information with mutational data, as produced by the IMGT 2D structure scraper.

## Canonical top-level object

`INNStructureCollection` shape

```python
class INNStructureCollection(BaseModel):
    results: list[INNStructureResult] = []
    failures: dict[str, str] = {}
    missing: list[str] = []
```

- `results`: successful parses only (one `INNStructureResult` per INN with a parsed 2D page).
- `missing`: INN IDs with no IMGT 2D entry (page parsed but no `div#main`).
- `failures`: INN IDs that raised parser/network/runtime errors (`{inn_id: error_message}`).

---

## Per-INN object

```python
class INNStructureResult(BaseModel):
    inn_id: str
    chains: list[INNChainDetail] = []
```

- `inn_id` is always normalized to `str`.
- `chains` includes:
  1. domain-level rows (VH/CH1/CH2/... etc), and
  2. one synthetic full-chain row with:
     - `chain_feature_type="Full Length"`
     - `region_annotation="Full Length"`
     - `detail_order_idx="<Chain ID>.full_chain"`

---

## Per-chain/domain object (exact downstream contract)

`INNChainDetail` objects are the core rows consumed by realignment/collapse steps:

```python
class INNChainDetail(BaseModel):
    inn_id: str
    chain_id: str
    chain_type: str
    detail_order_idx: str
    chain_length: int = 0
    chain_description: str | None = None
    chain_feature_type: str | None = None
    region_annotation: str
    region_annotation_description: str | None = None
    sequence: str | None = None
    germline_sequence: str | None = None
    gene_name: str | None = None
    v_gene: str | None = None
    d_gene: str | None = None
    j_gene: str | None = None
    total_mutations: int | None = None
    domain_gap_align_url: str | None = None
    scraped_mutations: list[MutationRecord] | None = None
    realigned_mutations: list[MutationRecord] | None = None
    realigned_failed: bool | None = None
    realigned_gene: str | None = None
    realigned_germline_sequence: str | None = None
    realigned_fraction_score: float | None = None
    realigned_species: str | None = None
    mutations_by_individual_numbering_systems: list[NumberedMutationRecord] | None = (
        None
    )
```

### Required invariants for notebook compatibility

1. `detail_order_idx` must be non-null for every chain row.
   - Domain rows are generated as `"{chain_id}.{idx}"`.
   - Full-length row is `"{chain_id}.full_chain"`.
2. `region_annotation` must contain the normalized IMGT label (e.g., `VH`, `CH2`, `C-KAPPA`, `HINGE-REGION`, `Full Length`).
3. `sequence` should be present for analyzable regions; some rows can be `None`.
4. `scraped_mutations` must be a list of `MutationRecord` (or `None` if unavailable).

---

## Mutation objects

```python
class MutationRecord(BaseModel):
    position: int | str | float
    mutated_seq: str
    germline_seq: str
    mutation_type: str = "substitution"  # substitution|insertion|deletion
    insertion_str: str | None = None
    mutation_region: str | None = None
    repeat_region_count: int | None = None
    numbering_system: str | None = None
    original_position: int | None = None
```

For numbered exports (optional enrichment):

```python
class NumberedMutationRecord(BaseModel):
    position: MutationRecord
    imgt_position: MutationRecord | None = None
    eu_position: MutationRecord | None = None
    kabat_position: MutationRecord | None = None
    exon_position: MutationRecord | None = None
```

---

Minimal JSON-like example (`model_dump()` style)

```json
{
  "results": [
    {
      "inn_id": "8495",
      "chains": [
        {
          "inn_id": "8495",
          "chain_id": "INN 8495_H",
          "chain_type": "h",
          "detail_order_idx": "INN 8495_H.0",
          "chain_length": 447,
          "chain_description": "H-GAMMA-1",
          "chain_feature_type": "V-DOMAIN",
          "region_annotation": "VH",
          "region_annotation_description": "VH (1-129) [D1]",
          "sequence": "QVQL...",
          "germline_sequence": "QVQL...",
          "gene_name": "IGHV...",
          "v_gene": "IGHV...",
          "d_gene": null,
          "j_gene": "IGHJ...",
          "total_mutations": 7,
          "domain_gap_align_url": "https://imgt.org/3Dstructure-DB/cgi/DomainGapAlign.cgi?...",
          "scraped_mutations": [
            {
              "position": 31,
              "mutated_seq": "S",
              "germline_seq": "N",
              "mutation_type": "substitution",
              "insertion_str": null,
              "mutation_region": null,
              "repeat_region_count": null,
              "numbering_system": null,
              "original_position": null
            }
          ],
          "realigned_mutations": null,
          "realigned_failed": null,
          "realigned_gene": null,
          "realigned_germline_sequence": null,
          "realigned_fraction_score": null,
          "realigned_species": null,
          "mutations_by_individual_numbering_systems": null
        }
      ]
    }
  ],
  "failures": {},
  "missing": []
}
```
---

# Expected IMGT mAb-DB Format

This section defines the expected tabular format for the raw IMGT mAb-DB search-table export the pandera schema that validates it, and the pydantic objects it is parsed into downstream.

## Raw CSV shape (one row per receptor)

The source file is a flat CSV with **one row per IMGT mAb-DB receptor entry** (32 columns; `imgt_id` is the unique key — `inn` is *not* unique, since a single INN such as `"interferon alfa"` can span multiple `imgt_id` rows for different receptor variants). It is validated by `IMGTMabDbCsvSchema` (a `pandera.pandas.DataFrameModel`) defined in `src/inn_analysis_ops/data_io/imgt_mab_dab.py`.

```python
import pandera.pandas as pa
from pandera.typing import Series


class IMGTMabDbCsvSchema(pa.DataFrameModel):
    imgt_id: Series[int]  # unique, > 0 — the true primary key
    inn: Series[str]  # nullable; NOT unique
    inn_number: Series[float]  # nullable

    structure_2d: Series[float]  # nullable; equals inn_number when both present
    structure_3d: Series[str]  # nullable; PDB code, e.g. "1qnz"

    receptor_type: Series[str]  # enumerated: IG, TR, CPCA, RPI, FPIA

    format: Series[float]  # nullable; always null in current exports

    specificity: Series[str]  # nullable
    company: Series[str]  # nullable
    development_status: Series[
        str
    ]  # nullable; comma-joined phase tokens (validated per-token)
    clinical_domain: Series[str]  # nullable; semicolon-joined domain tags

    prop_list: Series[str]  # nullable; INN Proposed List number(year)
    rec_list: Series[str]  # nullable; INN Recommended List number(year)

    common_name: Series[str]  # nullable
    proprietary_name: Series[str]  # nullable
    species: Series[str]  # nullable; e.g. "Homo sapiens", "Humanized", "Chimeric"
    receptor_id: Series[str]  # nullable; e.g. "IgG1 - kappa", "Fab - G1 - kappa"
    conjugated: Series[str]  # nullable; free-text conjugation/fusion description
    dev_technology: Series[str]  # nullable
    origin_species: Series[str]  # nullable
    origin_clone: Series[str]  # nullable
    expression_system: Series[str]  # nullable
    application: Series[str]  # nullable; e.g. "Therapeutic", "Diagnostic"
    mechanism_of_action: Series[str]  # nullable
    clinical_indication: Series[str]  # nullable

    # Raw IMGT header text retained verbatim (not snake_cased) in this export.
    regulatory_agency_status_and_year: Series[
        str
    ]  # aliased from "Regulatory agency status and year"; nullable
    clinical_trials: Series[str]  # nullable
    authority_decisions: Series[str]  # nullable
    external_links: Series[str]  # aliased from "External links"; nullable
    citations_notes_references: Series[
        str
    ]  # aliased from "Citations/ Notes/ References"; nullable
    biosimilars: Series[str]  # nullable

    class Config:
        strict = True
        coerce = True
```

### Required invariants

1. `imgt_id` is the unique primary key (`inn` is intentionally **not** unique-constrained — the same INN can legitimately appear across multiple `imgt_id` rows).
2. `receptor_type` is constrained to the known enumerated set `{IG, TR, CPCA, RPI, FPIA}`.
3. `development_status` cells may combine multiple comma-separated phase tokens (e.g. `"Phase I,Phase II,Phase I/II,Phase III"`, sometimes with stray blank tokens from double commas); every non-blank token must belong to the known phase vocabulary (`Discontinued`, `Phase I`…`Phase IV`, `Phase M`, `Preclinical`, `Proof of concept`, `Withdrawn`).
4. `structure_2d` equals `inn_number` whenever both are present (both come from the same underlying numeric ID).
5. `format` is nullable and currently always empty in real exports — do not assume it is populated.
6. Free-text columns (`specificity`, `conjugated`, `species`, `receptor_id`, etc.) intentionally have no `isin` constraint — their content is too varied/compositional to enumerate.

### Validating the raw file

```python
import pandas as pd
from inn_analysis_ops.data_io.imgt_mab_dab import IMGTMabDbCsvSchema

df = pd.read_csv("mabdb.csv")
validated = IMGTMabDbCsvSchema.validate(
    df, lazy=True
)  # raises pa.errors.SchemaErrors on violation
```


---

# Expected TheraSAbDab Format

This section defines the expected tabular format for the raw TheraSAbDab search-table export
 the pandera schema that validates it, and the pydantic objects it is parsed into downstream.

## Raw CSV shape (one row per INN)

The source file is a flat CSV with **one row per antibody name** (26 columns). It is validated by `TheraSAbDAbCsvSchema` (a `pandera.pandas.DataFrameModel`) defined in `src/inn_analysis_ops/data_io/therasabdab.py`:

```python
import pandera.pandas as pa
from pandera.typing import Series


class TheraSAbDabCsvSchema(pa.DataFrameModel):
    name: Series[str]  # unique INN/therapeutic name
    target: Series[str]
    format: Series[str]  # e.g. "Whole mAb", "Fab", "Bispecific mAb"
    isotype: Series[str]  # e.g. "G1", "G4", "G1;G1", "na"

    vh1: Series[str]  # amino-acid sequence, or literal "na"
    vl1: Series[str]
    vh2: Series[str]  # second arm; "na" for non-bispecifics
    vl2: Series[str]

    highest_clinical_trial: Series[
        str
    ]  # enumerated: Phase-I..Phase-III, Approved, TBC, ...
    estimated_status: Series[
        str
    ]  # enumerated: Active, Discontinued, TBC, NFD, Approved, ND

    has_100_si_struc: Series[bool]
    has_99_si_struc: Series[bool]
    has_95_98_si_struc: Series[bool]

    pdb_100: Series[str]  # nullable; "[Fvs: HL]"-style PDB hit list
    pdb_99: Series[str]  # nullable
    pdb_95_98: Series[str]  # nullable

    dev_technology: Series[str]  # nullable
    companies: Series[str]

    year_proposed: Series[int]  # non-nullable
    year_recommended: Series[float]  # nullable
    year_received: Series[float]  # nullable

    conditions_approved: Series[str]  # nullable
    conditions_active: Series[str]  # nullable
    conditions_discontinued: Series[str]  # nullable
    notes: Series[str]  # nullable

    class Config:
        strict = True
        coerce = True
```

### Required invariants

1. `name` is unique per row (one row = one canonical therapeutic name; trailing-numbered duplicates like `abelacimab2` are distinct rows, not the same INN — see `_base_name` grouping below).
2. `vh1`/`vl1`/`vh2`/`vl2` must be either a pure amino-acid string (`^[A-Za-z]+$`) or the literal placeholder `"na"` — never blank/`NaN`.
3. `highest_clinical_trial` and `estimated_status` are constrained to their known enumerated value sets (see schema `isin(...)` checks); unexpected values should fail validation rather than silently pass through.
4. `has_100_si_struc` / `has_99_si_struc` / `has_95_98_si_struc` are always populated booleans (never null).
5. `year_proposed` is always present; `year_recommended` and `year_received` may be null.

### Validating the raw file

```python
import pandas as pd
from inn_analysis_ops.data_io.therasabdab import TheraSAbDabCsvSchema

df = pd.read_csv("therasabdab.csv")
validated = TheraSAbDabCsvSchema.validate(
    df, lazy=True
)  # raises pa.errors.SchemaErrors on violation
```


---

# Expected Wilkinsone and hale Format

This section defines the expected tabular format for the raw wilkinsone and hale INN Analysis table export the pandera schema that validates it, and the pydantic objects it is parsed into downstream.

## Raw CSV shape (one row per INN)

The source file is a flat CSV with **one row per INN** (10 columns; `inn` is unique). It is validated by `WilkHaleCsvSchema` (a `pandera.pandas.DataFrameModel`) defined in data_io/wilk_hale.py:

```python
import pandera.pandas as pa
from pandera.typing import Series


class WilkHaleCsvSchema(pa.DataFrameModel):
    inn: Series[str]  # unique, non-empty INN name
    format: Series[
        str
    ]  # enumerated antibody format (56 known values, incl. "(type N)" variants)
    antigen: Series[str]  # non-empty target antigen description

    effector_function: Series[str]  # nullable; "↑" or "↓" only
    half_life: Series[str]  # nullable; "↑" or "↓" only

    heterodimerization: Series[bool]  # non-nullable
    site_specific_conj: Series[bool]  # non-nullable
    stabilized: Series[bool]  # non-nullable
    c_term_truncation: Series[bool]  # non-nullable
    other: Series[bool]  # non-nullable

    class Config:
        strict = True
        coerce = True
```

### Required invariants

1. `inn` is unique per row (one row per canonical INN — unlike TheraSAbDab, there is no trailing-number duplication scheme here).
2. `format` is constrained to the fixed vocabulary of ~56 antibody-format labels observed in the source table (e.g. `IgG`, `Fab`, `Fc fusion protein (type 1)`…`(type 7)`, `Bispecific IgG (type 1)`…`(type 6)`, `Unconfirmed`). New/unrecognized format strings should fail validation rather than pass silently, since downstream logic switches on these exact labels.
3. `effector_function` and `half_life` are nullable, but when present must be exactly the directional glyph `↑` (increased) or `↓` (decreased) — no other text is valid.
4. The five boolean flag columns (`heterodimerization`, `site_specific_conj`, `stabilized`, `c_term_truncation`, `other`) are always populated (never null); absence of a modification is `False`, not `NaN`.

### Validating the raw file

```python
import pandas as pd
from inn_analysis_ops.data_io.wilk_hale import WilkHaleCsvSchema

df = pd.read_csv("whale_inn_analysis.csv")
validated = WilkHaleCsvSchema.validate(
    df, lazy=True
)  # raises pa.errors.SchemaErrors on violation
```

---
