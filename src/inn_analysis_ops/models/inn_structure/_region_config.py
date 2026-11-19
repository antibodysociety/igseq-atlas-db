"""Declarative mapping from scraped region_annotation to DB lookup strategy."""

from dataclasses import dataclass


@dataclass(frozen=True)
class RegionSpec:
    db_labels: tuple
    extra_annotation: object  # str | None
    skip_gene: bool = False
    compound_build: tuple = ()


REGION_SPECS = {
    "CH1": RegionSpec(("CH1",), extra_annotation=None),
    "CH2": RegionSpec(("CH2",), extra_annotation=None),
    "CH3": RegionSpec(("CH3", "CH3-CHS"), extra_annotation="CHS"),
    "CH4": RegionSpec(("CH4", "CH4-CHS"), extra_annotation="CHS"),
    # CHS is a sub-component stitched onto CH3/CH4 via extra_annotation above.
    # It has no standalone gene_name in the scraped data, so omit here to avoid no_gene noise.
    "C-KAPPA": RegionSpec(("C-REGION",), extra_annotation=None),
    "C-LAMBDA": RegionSpec(("C-REGION",), extra_annotation=None),
    "HINGE-REGION": RegionSpec(
        ("H", "H1+H2+H3+H4", "H1+H2+H3", "H1+H2"),
        extra_annotation=None,
        skip_gene=True,
        compound_build=(("H1", "H2", "H3", "H4"), ("H1", "H2", "H3"), ("H1", "H2")),
    ),
    "H1": RegionSpec(("H1",), extra_annotation=None, skip_gene=True),
    "H2": RegionSpec(("H2",), extra_annotation=None, skip_gene=True),
    "H3": RegionSpec(("H3",), extra_annotation=None, skip_gene=True),
    "H4": RegionSpec(("H4",), extra_annotation=None, skip_gene=True),
    # H-CH2: fused hinge+CH2 label found in DB — treat as skip_gene like other hinges
    "H-CH2": RegionSpec(("H-CH2",), extra_annotation=None, skip_gene=True),
}
