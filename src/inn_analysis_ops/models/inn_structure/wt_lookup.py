from __future__ import annotations

import re
from typing import Union

import pandas as pd

from ..numbering.core import (
    ConstantChainNumbering,
    NumberingTypes,
)

from ._utils import _GAP_CHARS, _SPECIES_MAP
from .constant_validator import ConstantDB

_REGION_TO_DB_LABEL: dict[str, str] = {
    "CH1": "CH1",
    "CH2": "CH2",
    "CH3": "CH3",
    "CH4": "CH4",
    "CHS": "CHS",
    "C-KAPPA": "C-REGION",
    "C-LAMBDA": "C-REGION",
    "HINGE-REGION": "H",
}

_REGION_TO_CN_REGION: dict[str, str] = {
    "CH1": "CH1",
    "CH2": "CH2",
    "CH3": "CH3",
    "CH4": "CH4",
    "CHS": "CHS",
    "C-KAPPA": "C-KAPPA",
    "C-LAMBDA": "C-LAMBDA",
}

# Fallback combined DB labels when a standalone region label is absent.
# Maps requested region → [(combined_db_label, cn_region_to_keep), ...]
_REGION_COMBINED_FALLBACKS: dict[str, list[tuple[str, str]]] = {
    "CH3": [("CH3-CHS", "CH3")],
    "CHS": [("CH3-CHS", "CHS"), ("CH4-CHS", "CHS")],
    "CH4": [("CH4-CHS", "CH4")],
}

_GENE_TO_HINGE_CN: dict[str, str] = {
    "ighg1": "HINGE-IGHG1",
    "ighg2": "HINGE-IGHG2",
    "ighg3": "HINGE-IGHG3",
    "ighg4": "HINGE-IGHG4",
    "ighm": "HINGE-IGHM",
    "ighd": "HINGE-IGHD",
    "igha": "HINGE-IGHA",
}

_DEFAULT_TABLE_REGIONS: list[str] = [
    "CH1",
    "CH2",
    "CH3",
    "CH4",
    "CHS",
    "C-KAPPA",
    "C-LAMBDA",
    "HINGE-REGION",
]


def _all_known_species(cdb: ConstantDB) -> set[str]:
    """Return the set of raw species strings present anywhere in *cdb*."""
    return {sp for entries in cdb.gene_idx.values() for sp, _ in entries}


def _resolve_species_filter(cdb: ConstantDB, species_filter: str) -> str:
    """Normalize *species_filter* and assert it matches a known species in *cdb*.

    ``constant_chains``/``individual_mutation_profile`` store normalized
    species labels (e.g. ``"human"``), while ``cdb`` stores the raw species
    strings from the source annotation sheet (e.g. ``"Homo sapiens"``).
    :data:`models.inn_structure._utils._SPECIES_MAP` translates
    between the two.

    If neither the normalized form nor the raw *species_filter* string
    matches any species actually present in *cdb*, raises
    ``AssertionError`` — this means the species name is unrecognized and
    ``_SPECIES_MAP`` needs a new entry, rather than silently falling back
    to picking an arbitrary (possibly wrong-species) sequence.

    Parameters
    ----------
    cdb :
        :class:`ConstantDB` to validate against.
    species_filter :
        Species string as supplied by the caller (normalized label like
        ``"human"``, or already a raw species string like ``"Homo sapiens"``).

    Returns
    -------
    str
        The resolved raw species prefix to use for ``str.startswith`` matching.

    Raises
    ------
    AssertionError
        If *species_filter* cannot be resolved to any species present in *cdb*.
    """
    candidate = _SPECIES_MAP.get(species_filter.lower(), species_filter)
    known = _all_known_species(cdb)
    if not any(sp.startswith(candidate) for sp in known):
        raise AssertionError(
            f"get_wt_residues: species {species_filter!r} (resolved to {candidate!r}) "
            f"does not match any species present in cdb. Known species: {sorted(known)}. "
            f"Add an entry to _SPECIES_MAP in "
            f"db_query/imgt/inn_structure/_utils.py mapping {species_filter!r} to the "
            f"correct raw species string."
        )
    return candidate


def get_wt_residues(
    cdb: ConstantDB,
    cn: ConstantChainNumbering,
    gene: str,
    regions: list[str],
    allele_index: int = 0,
    species_filter: str | None = "Homo sapiens",
) -> dict[Union[int, str], str]:
    """Return {eu_position: wt_residue} for all non-gap positions across regions.

    Parameters
    ----------
    cdb : ConstantDB
        Built from build_constant_db(constant_db_seqs).
    cn : ConstantChainNumbering
        Loaded with VH, IGL, IGK, VH-HINGE sheets from IMGT_EU_NUMBERING.xlsx.
    gene : str
        Gene+allele e.g. "ighg1*01" (case-insensitive).
    regions : list[str]
        Annotation region labels e.g. ["CH2", "CH3"].
    allele_index : int
        Fallback index into DB entries when species_filter finds no match.
    species_filter : str | None
        Prefer entries whose species starts with this string (default "Homo sapiens").

    Returns
    -------
    dict[int | str, str]
        Maps EU position (int for regular, str like "234.i1" for insertions) to WT residue.
    """
    result: dict[Union[int, str], str] = {}
    gene_lower = gene.lower()
    if "*" not in gene_lower:
        gene_lower = f"{gene_lower}*01"
    gene_family = gene_lower.split("*")[0]

    _resolved_species_filter = (
        _resolve_species_filter(cdb, species_filter) if species_filter else None
    )

    for region in regions:
        db_label = _REGION_TO_DB_LABEL.get(region, region)

        if region == "HINGE-REGION":
            cn_region = _GENE_TO_HINGE_CN.get(gene_family)
        else:
            cn_region = _REGION_TO_CN_REGION.get(region, region)

        if cn_region is None or cn_region not in cn.regions:
            continue

        entries = cdb.gene_idx.get((gene_lower, db_label), [])

        # Fallback: gene_lower may be family-only (e.g. 'ighg1') while gene_idx
        # keys carry full alleles (e.g. 'ighg1*01').  Scan for a family match.
        if not entries:
            for (g, lbl), v in cdb.gene_idx.items():
                if lbl == db_label and g.split("*")[0] == gene_lower:
                    entries = v
                    break

        # Fallback: try combined DB labels (e.g. "CH3-CHS") and filter by CN region boundary
        combined_cn_filter: str | None = None
        combined_cn_regions: list[str] | None = None
        if not entries and region in _REGION_COMBINED_FALLBACKS:
            for (
                combined_label,
                keep_cn_region,
            ) in _REGION_COMBINED_FALLBACKS[region]:
                entries = cdb.gene_idx.get((gene_lower, combined_label), [])
                if entries:
                    # Derive sub-region list from the combined label (e.g. "CH3-CHS" → ["CH3","CHS"])
                    combined_cn_regions = [
                        r for r in combined_label.split("-") if r in cn.regions
                    ]
                    combined_cn_filter = keep_cn_region
                    break

        if not entries:
            continue

        gapped_seq = _pick_entry(entries, allele_index, _resolved_species_filter)
        if not gapped_seq:
            continue

        # Build EU allowlist when splitting a combined sequence (e.g. CH3 out of CH3-CHS)
        eu_allowset: set | None = None
        if combined_cn_filter is not None and combined_cn_filter in cn.regions:
            try:
                _filter_df = cn.filled_position_df[combined_cn_filter].dropna(how="all")
                eu_allowset = set()
                for _v in _filter_df[NumberingTypes.EU].dropna():
                    try:
                        eu_allowset.add(int(float(_v)))
                    except (ValueError, TypeError):
                        pass
            except (KeyError, Exception):
                eu_allowset = None

        # cn_df is indexed by total sequential position (1-based, including gap rows)
        if combined_cn_regions:
            # Concatenate sub-region tables and re-index sequentially for the combined sequence
            _parts = [
                cn.filled_position_df[r].dropna(how="all")
                for r in combined_cn_regions
                if r in cn.regions
            ]
            if not _parts:
                continue
            cn_df = pd.concat(_parts, ignore_index=True)
            cn_df.index = range(1, len(cn_df) + 1)
        else:
            cn_df = cn.filled_position_df[cn_region].dropna(how="all")

        total_pos = 0
        for char in gapped_seq:
            total_pos += 1
            if char in _GAP_CHARS:
                continue
            try:
                eu_val = cn_df.loc[total_pos, NumberingTypes.EU]
            except KeyError:
                continue
            if eu_val is None or (isinstance(eu_val, float) and pd.isna(eu_val)):
                continue
            try:
                eu_key: Union[int, str] = int(float(eu_val))
            except (ValueError, TypeError):
                eu_key = str(eu_val)
            # When splitting a combined region, only keep positions in the target cn region
            if (
                eu_allowset is not None
                and isinstance(eu_key, int)
                and eu_key not in eu_allowset
            ):
                continue
            result[eu_key] = char

    return result


def get_eu_to_region_map(
    cn: ConstantChainNumbering,
) -> dict:
    """Build a complete ``{eu_position: region}`` map from *cn*.

    Iterates every region in ``cn.filled_position_df`` and reads the ``EU``
    column directly.  The result is a flat dict suitable for O(1) lookup of
    which constant-chain region a given EU position belongs to.

    Parameters
    ----------
    cn :
        :class:`~models.numbering.core.ConstantChainNumbering` loaded
        from ``IMGT_EU_NUMBERING.xlsx``.

    Returns
    -------
    dict[int | str, str]
        Maps EU position (``int`` for regular, ``str`` like ``"234.i1"`` for
        insertions) to region name (e.g. ``"CH2"``).
    """
    eu_to_region: dict = {}
    for region in cn.regions:
        try:
            region_df = cn.filled_position_df[region].dropna(how="all")
        except KeyError:
            continue
        if NumberingTypes.EU not in region_df.columns:
            continue
        for eu_val in region_df[NumberingTypes.EU].dropna():
            try:
                eu_key: int | str = int(float(eu_val))
            except (ValueError, TypeError):
                eu_key = str(eu_val)
            if eu_key not in eu_to_region:
                eu_to_region[eu_key] = str(region)
    return eu_to_region


def get_region_for_eu_position(
    cn: ConstantChainNumbering,
    eu_position: "int | str",
    eu_to_region_map: "dict | None" = None,
) -> str:
    """Return the constant-chain region name for *eu_position*.

    Parameters
    ----------
    cn :
        :class:`~models.numbering.core.ConstantChainNumbering`.
    eu_position :
        EU position — ``int`` for regular positions, ``str`` like ``"234.i1"``
        for insertions.
    eu_to_region_map :
        Pre-built map from :func:`get_eu_to_region_map`.  Pass it to avoid
        rebuilding on every call.  When ``None`` a fresh map is built.

    Returns
    -------
    str
        Region name (e.g. ``"CH2"``).

    Raises
    ------
    KeyError
        If *eu_position* is not found in any region.
    """
    _map = (
        eu_to_region_map if eu_to_region_map is not None else get_eu_to_region_map(cn)
    )
    if eu_position not in _map:
        raise KeyError(
            f"EU position {eu_position!r} not found in any region of the "
            f"numbering table. Available regions: {sorted(set(_map.values()))}."
        )
    return _map[eu_position]


def get_germline_at_insertion_pos(
    cdb: ConstantDB,
    cn: ConstantChainNumbering,
    gene: str,
    region: str,
    eu_anchor_pos: Union[int, str, float],
    allele_index: int = 0,
    species_filter: str | None = "Homo sapiens",
) -> str | None:
    """Return the WT residue at the anchor EU position of an insertion.

    For insertions, ``germline_seq`` is always ``"-"`` because no germline
    residue exists at the inserted position itself.  The ``EU`` column on an
    insertion row records the *anchor* position — the germline residue
    immediately before the insertion — which is a regular EU-numbered position
    resolvable by ``get_wt_residues``.

    The anchor EU value may arrive as a plain integer (``230``) or as an
    insertion-notation string (``"230.i1"``); both are handled by extracting
    the leading integer.

    The ``region`` value may be the display label ``"HINGE"`` (as produced by
    analytics) or the canonical label ``"HINGE-REGION"``; both are accepted.

    Parameters
    ----------
    cdb : ConstantDB
        Built from ``build_constant_db``.
    cn : ConstantChainNumbering
        Loaded with the relevant region sheets from IMGT_EU_NUMBERING.xlsx.
    gene : str
        Gene (+ optional allele) e.g. ``"ighg1"`` or ``"ighg1*01"``.
    region : str
        Annotation region label e.g. ``"CH2"``, ``"HINGE"``, or ``"HINGE-REGION"``.
    eu_anchor_pos : int | str | float
        The EU position from the insertion row (the ``EU`` column value), i.e.
        the position *after which* the insertion occurs in the germline alignment.
        Accepts plain integers or strings like ``"230.i1"``.
    allele_index : int
        Passed through to ``get_wt_residues``.
    species_filter : str | None
        Passed through to ``get_wt_residues``.

    Returns
    -------
    str | None
        The single-letter WT residue at the anchor position, or ``None`` if
        the position cannot be resolved (unknown gene / region / EU value).

    Examples
    --------
    >>> row = inn_summary.query('mutation_type == "insertion"').iloc[0]
    >>> get_germline_at_insertion_pos(cdb, cn, row["gene"], row["region"], row["EU"])
    'K'
    >>> get_germline_at_insertion_pos(cdb, cn, 'ighg1', 'HINGE', '230.i1')
    'P'
    """
    # "HINGE" is the analytics display label; get_wt_residues needs the canonical form.
    canonical_region = "HINGE-REGION" if region.upper() == "HINGE" else region

    wt = get_wt_residues(
        cdb,
        cn,
        gene,
        [canonical_region],
        allele_index=allele_index,
        species_filter=species_filter,
    )

    # Extract the integer anchor from values like 230, 230.0, or "230.i1".
    m = re.match(r"(\d+)", str(eu_anchor_pos))
    if m is None:
        return None
    key: int = int(m.group(1))
    return wt.get(key)


def build_germline_numbering_table(
    cdb: ConstantDB,
    cn: ConstantChainNumbering,
    regions: list[str] | None = None,
) -> pd.DataFrame:
    """Return a WT lookup table across all species/gene/allele/region combinations.

    The output includes one row per resolved WT residue at each EU position with
    aligned IMGT/KABAT/EXON numbering.

    Notes
    -----
    - ``allele`` is preserved as the full ``gene*allele`` label and uppercased.
    - ``gene`` is the gene family portion (before ``*``), uppercased.
    """
    selected_regions = regions if regions is not None else _DEFAULT_TABLE_REGIONS
    cn_regions = set(cn.regions)

    def _eu_key(value) -> int | str | None:
        if pd.isna(value):
            return None
        try:
            return int(float(value))
        except (ValueError, TypeError):
            return str(value)

    def _cn_region_for_gene(region: str, gene_allele: str) -> str | None:
        if region != "HINGE-REGION":
            return region if region in cn_regions else None
        gene_family = gene_allele.split("*")[0]
        candidate = _GENE_TO_HINGE_CN.get(gene_family)
        if candidate in cn_regions:
            return candidate
        return None

    numbering_by_region: dict[str, dict[int | str, dict[str, object]]] = {}
    for cn_region in cn_regions:
        region_df = cn.filled_position_df[cn_region].dropna(how="all")
        eu_lookup: dict[int | str, dict[str, object]] = {}
        for _, row in region_df.iterrows():
            eu_pos = _eu_key(row.get(NumberingTypes.EU))
            if eu_pos is None:
                continue
            eu_lookup[eu_pos] = {
                "imgt_position": row.get(NumberingTypes.IMGT),
                "kabat_position": row.get(NumberingTypes.KABAT),
                "exon_position": row.get(NumberingTypes.EXON),
            }
        numbering_by_region[cn_region] = eu_lookup

    gene_species_pairs = {
        (gene_allele, species)
        for (gene_allele, _region), entries in cdb.gene_idx.items()
        for species, _ in entries
    }

    rows: list[dict[str, object]] = []
    for gene_allele, species in sorted(gene_species_pairs):
        gene_family = gene_allele.split("*")[0].upper()
        allele_label = gene_allele.upper()

        for region in selected_regions:
            cn_region = _cn_region_for_gene(region, gene_allele)
            if cn_region is None:
                continue

            wt_map = get_wt_residues(
                cdb=cdb,
                cn=cn,
                gene=gene_allele,
                regions=[region],
                species_filter=species,
            )
            if not wt_map:
                continue

            region_numbering = numbering_by_region.get(cn_region, {})
            for eu_pos, wt_residue in wt_map.items():
                numbering = region_numbering.get(eu_pos, {})
                rows.append(
                    {
                        "species": species,
                        "gene": gene_family,
                        "region": region,
                        "allele": allele_label,
                        "eu_position": eu_pos,
                        "imgt_position": numbering.get("imgt_position"),
                        "kabat_position": numbering.get("kabat_position"),
                        "exon_position": numbering.get("exon_position"),
                        "germline_sequence": wt_residue,
                    }
                )

    table = pd.DataFrame(rows)
    if table.empty:
        return table

    return table.sort_values(
        by=["species", "gene", "allele", "region", "eu_position"],
        kind="stable",
    ).reset_index(drop=True)


def _pick_entry(
    entries: list[tuple[str, str]],
    index: int,
    species_filter: str | None,
) -> str | None:
    if species_filter:
        for sp, seq in entries:
            if sp.startswith(species_filter):
                return seq
    if index < len(entries):
        return entries[index][1]
    return None
