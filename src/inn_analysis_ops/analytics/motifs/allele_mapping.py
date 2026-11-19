"""Chain/EU-position → allele mapping for cross-chain germline-reversion checks.

Two lookup structures, built once per pipeline run and reused across every
bispecific chain pair in :mod:`analytics.motifs.enumerate`:

``ind_mut_allele_map`` (:func:`_infer_chain_id_mut_to_allele_map`)
    ``{(chain_id, EU_position): mutation_allele_mapper}`` — for a specific
    mutation observed on a specific chain, tells you which gene/allele/region
    that mutation was originally attributed to (from
    ``ds.individual_mutation_profile``).

``chain_id_region_allele_map`` (:func:`_infer_chain_id_region_to_allele_map`)
    ``ChainIdRegionAlleleMapper`` — for any ``(chain_id, region)`` pair,
    tells you the resolved allele + species for that chain at that region
    (from ``ds.constant_chains``), split into ``heavy_chain`` / ``light_chain``
    dicts.

Together these let :func:`~analytics.motifs.enumerate._is_germline_reversion`
answer: "chain A has mutation X at EU position P in region R — what does
chain B's germline look like at that same P/R?" without re-deriving the
region from the numbering table (avoiding the fragile EU→region inference
that caused repeated bugs).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, NamedTuple

import pandas as pd

if TYPE_CHECKING:
    from inn_analysis_ops.data_io.merge.loader import INNAnalysisDataset

_log = logging.getLogger(__name__)


class ChainID_EU_Key(NamedTuple):
    """Lookup key: a specific chain at a specific EU position."""

    chain_id: str
    eu_pos: str | float | int


class Region_Allele_Key(NamedTuple):
    """A resolved (region, allele) pair — output of sub-domain inference."""

    region: str
    allele: str | None


class AlleleSpecies(NamedTuple):
    allele: str | None
    species: str


class ChainID_Region_Key(NamedTuple):
    """Lookup key: a specific chain at a specific constant-chain region."""

    chain_id: str
    region: str
    region_iteration: str | None


class ChainIDWithIteration(NamedTuple):
    chain_id: str
    region_iteration: str | None


@dataclass
class MutationAlleleMapper:
    """Gene/allele/region a specific mutation was attributed to on its chain."""

    allele: str
    gene: str
    region: str


@dataclass
class ChainIdRegionAlleleMapper:
    """``{(chain_id, region): (allele, species)}`` split by heavy/light chain."""

    heavy_chain: dict[ChainID_Region_Key, AlleleSpecies] = field(default_factory=dict)
    light_chain: dict[ChainID_Region_Key, AlleleSpecies] = field(default_factory=dict)


def _infer_chain_id_mut_to_allele_map(
    ds: "INNAnalysisDataset | pd.DataFrame",
) -> dict[ChainID_EU_Key, MutationAlleleMapper]:
    """Build ``{(chain_id, EU_pos): MutationAlleleMapper}`` from mutation-level data.

    Parameters
    ----------
    ds :
        An :class:`INNAnalysisDataset`, or the raw
        ``individual_mutation_profile`` DataFrame directly.  Must have
        ``chain_id``, ``allele``, ``gene``, ``region`` as index levels and
        ``EU`` as a column.

    Returns
    -------
    dict[ChainID_EU_Key, MutationAlleleMapper]
        One entry per unique ``(chain_id, EU_pos)``.  If the same key is
        seen twice with conflicting gene/allele/region info, the first
        mapping wins and the conflict is logged (not raised).
    """
    mutation_pos_map_to_allele: dict[ChainID_EU_Key, MutationAlleleMapper] = {}
    checks: list[tuple] = []

    mut_table = (
        ds.copy() if isinstance(ds, pd.DataFrame) else ds.individual_mutation_profile
    )

    _chain_id_pos = list(mut_table.index.names).index("chain_id")
    _allele_id_pos = list(mut_table.index.names).index("allele")
    _gene_id_pos = list(mut_table.index.names).index("gene")
    _region_id_pos = list(mut_table.index.names).index("region")

    for idx, row in mut_table.iterrows():
        chain_id = idx[_chain_id_pos]
        gene = idx[_gene_id_pos]
        allele = idx[_allele_id_pos]
        region = idx[_region_id_pos]
        key = ChainID_EU_Key(chain_id, row["EU"])
        value = MutationAlleleMapper(allele=allele, gene=gene, region=region)
        if (
            key in mutation_pos_map_to_allele
            and mutation_pos_map_to_allele[key] != value
        ):
            checks.append((value, key, mutation_pos_map_to_allele[key]))
            continue

        mutation_pos_map_to_allele[key] = value

    if checks:
        for ck in checks:
            _log.warning(
                "We did not map this second iteration of an eu pos to gene info: %s",
                ck,
            )
    return mutation_pos_map_to_allele


def _infer_chain_id_region_to_allele_map(
    ds: "INNAnalysisDataset",
) -> ChainIdRegionAlleleMapper:
    """Build ``{(chain_id, region): (allele, species)}`` from ``ds.constant_chains``.

    Heavy and light chain regions are kept in separate dicts
    (``.heavy_chain`` / ``.light_chain``) so callers can pick the correct
    mapper based on which region a given mutation belongs to.

    ``CH1/C-KAPPA/C-LAMBDA`` is a single combined column in
    ``ds.constant_chains`` — this function deconvolutes it into the real
    sub-domain (``CH1``, ``C-KAPPA``, or ``C-LAMBDA``) by inspecting the
    gene family of the allele actually stored there.

    Raises
    ------
    AssertionError
        If any chain has no allele info at all across every region, or if
        a chain's region is assigned conflicting alleles across the
        ``constant_chains`` table that cannot be resolved.
    """

    def _infer_sub_domain(allele: str | None, raw_sub: str) -> list[Region_Allele_Key]:
        """Return [(resolved_sub_domain, gene_or_None)] entries for this raw sub-domain."""
        expected_region_labels = [
            "CH1/C-KAPPA/C-LAMBDA",
            "CH2",
            "CH3",
            "CH4",
            "CHS",
            "HINGE-REGION",
        ]

        if raw_sub not in expected_region_labels:
            raise ValueError(
                f"region definition {raw_sub} is not recognized by the expected column layout "
                "of 'innanalysisdataset.constant_chains'"
                f"expected one of: {expected_region_labels}"
            )
        if raw_sub != "CH1/C-KAPPA/C-LAMBDA":
            return [Region_Allele_Key(raw_sub, allele)]
        # Null → all three map to None
        if allele is None:
            return [
                Region_Allele_Key("CH1", None),
                Region_Allele_Key("C-KAPPA", None),
                Region_Allele_Key("C-LAMBDA", None),
            ]
        allele = allele.upper()
        gene = allele.upper().split("*")[0]
        if gene.startswith("IGK"):
            return [
                Region_Allele_Key("C-KAPPA", allele),
                Region_Allele_Key("CH1", None),
                Region_Allele_Key("C-LAMBDA", None),
            ]
        elif gene.startswith("IGL"):
            return [
                Region_Allele_Key("C-LAMBDA", allele),
                Region_Allele_Key("CH1", None),
                Region_Allele_Key("C-KAPPA", None),
            ]
        else:  # IGH, IGA, IGE, IGM → CH1 since should be heavy
            return [
                Region_Allele_Key("CH1", allele),
                Region_Allele_Key("C-KAPPA", None),
                Region_Allele_Key("C-LAMBDA", None),
            ]

    chain_id_region_to_allele_map = ChainIdRegionAlleleMapper(
        heavy_chain={}, light_chain={}
    )

    alleles_by_chain = ds.constant_chains.swaplevel(axis=1)["c_allele"].copy()

    if alleles_by_chain.isnull().all(axis=1).sum() > 0:
        debug = alleles_by_chain[alleles_by_chain.isnull().all(axis=1)]
        raise AssertionError(
            f"constant chains table contains info that has no allele info: {debug.index}"
        )

    _chain_id_pos = list(alleles_by_chain.index.names).index("chain_id")
    _h_or_l_pos = list(alleles_by_chain.index.names).index("chain_type")
    _species_pos = list(alleles_by_chain.index.names).index("constant_chain_species")
    _region_iteration_pos = list(alleles_by_chain.index.names).index(
        "chain_region_iteration"
    )
    debug: list = []

    for idx, allele_info in alleles_by_chain.iterrows():
        chain_id = idx[_chain_id_pos]
        chain_type = idx[_h_or_l_pos]
        species = idx[_species_pos]
        raw_iteration = idx[_region_iteration_pos]
        iteration = None if pd.isnull(raw_iteration) else str(raw_iteration)

        for region_col, allele in allele_info.items():
            _resolved_region_allele_arr = _infer_sub_domain(
                allele if not pd.isnull(allele) else None, region_col
            )
            for _resolved_region, _resolved_allele in _resolved_region_allele_arr:
                if _resolved_allele:
                    _resolved_allele = _resolved_allele.upper()

                if _resolved_region in ("C-KAPPA", "C-LAMBDA"):
                    _mapper = chain_id_region_to_allele_map.light_chain
                elif _resolved_region in (
                    "CH1",
                    "CH2",
                    "CH3",
                    "CH4",
                    "CHS",
                    "HINGE-REGION",
                ):
                    _mapper = chain_id_region_to_allele_map.heavy_chain
                else:
                    raise AssertionError(
                        f"operation 'infer_sub_domain' returned unexpected region {_resolved_region}"
                    )

                key = ChainID_Region_Key(chain_id, _resolved_region, iteration)
                current = _mapper.get(key)
                current_allele = current.allele if current else None
                if current is None or current.allele is None:
                    _mapper[key] = AlleleSpecies(_resolved_allele, species)
                elif _resolved_allele is None:
                    # doesn't add any info
                    continue
                elif current_allele != _resolved_allele:
                    # mismatch between multiple chains
                    # hack this might happen only once in current data
                    rescue = (
                        alleles_by_chain.query("chain_id==@chain_id")
                        .stack()
                        .value_counts()
                        .index[0]
                    )
                    _mapper[key] = AlleleSpecies(rescue.upper(), species)
                    debug.append(
                        (
                            _mapper[key],
                            key,
                            _resolved_allele,
                            chain_type,
                            rescue.upper(),
                        )
                    )
                    continue
                _mapper[key] = AlleleSpecies(_resolved_allele, species)

    if debug:
        for _d in debug:
            _log.warning("chain_id_region_allele_map: resolved allele mismatch %s", _d)

    return chain_id_region_to_allele_map


def region_to_subdomain(region: str) -> str:
    """Map a constant-chain ``region`` label to its ``sub_domain`` category.

    ``ds.constant_chains`` carries a ``sub_domain`` index level with only
    three possible values — ``"CK"`` (C-KAPPA), ``"CL"`` (C-LAMBDA), or
    ``"V"`` (everything else: CH1, CH2, CH3, CH4, CHS, HINGE-REGION —
    i.e. every heavy-chain constant region).
    """
    if region == "C-KAPPA":
        return "CK"
    if region == "C-LAMBDA":
        return "CL"
    return "V"


# Regions considered part of the Fc — everything after CH1/hinge.
_FC_REGIONS = frozenset({"CH2", "CH3", "CH4", "CHS"})


def chain_has_fc_region(
    chain_id: str,
    chain_id_region_allele_map: "ChainIdRegionAlleleMapper",
) -> bool:
    """Return ``True`` if *chain_id* has a resolved allele at any Fc region.

    Looks up ``chain_id_region_allele_map.heavy_chain`` for any
    ``ChainID_Region_Key`` belonging to *chain_id* whose region is in
    :data:`_FC_REGIONS` (``CH2``, ``CH3``, ``CH4``, ``CHS``) with a
    non-``None`` resolved allele — i.e. this chain actually carries Fc
    sequence, not just a Fab/CH1-only heavy domain.
    """
    return any(
        key.chain_id == chain_id and key.region in _FC_REGIONS and value.allele
        for key, value in chain_id_region_allele_map.heavy_chain.items()
    )


def _classify_by_sub_domain(grp: pd.DataFrame) -> dict[str, list[ChainIDWithIteration]]:
    """Bucket the chain_ids of one INN group by sub_domain.

    ``CK``/``CL`` chain_ids are both filed under *both* the ``"CK"`` and
    ``"CL"`` buckets — light-chain-to-light-chain comparisons (kappa vs
    lambda) are allowed; only light-to-heavy (``V``) mixing is excluded.
    ``V`` (heavy-chain) chain_ids go in the ``"V"`` bucket only.
    """
    domain: dict[str, list[ChainIDWithIteration]] = {"CK": [], "CL": [], "V": []}
    for _, row in grp.iterrows():
        sub_domain = str(row["sub_domain"].values[0])
        chain = str(row["chain_id"].values[0])
        raw_iteration = row["chain_region_iteration"].values[0]
        iteration = None if pd.isnull(raw_iteration) else str(raw_iteration)

        if sub_domain in ("CK", "CL"):
            domain["CK"].append(ChainIDWithIteration(chain, iteration))
            domain["CL"].append(ChainIDWithIteration(chain, iteration))
        elif sub_domain == "V":
            domain["V"].append(ChainIDWithIteration(chain, iteration))
        else:
            raise AssertionError(f"Unknown domain {sub_domain}")
    return domain


def _infer_inn_chain_domain_map(
    ds: "INNAnalysisDataset",
) -> dict[str, dict[str, list[str]]]:
    """Build ``{inn_id: {"CK"|"CL"|"V": [chain_id, ...]}}`` from ``ds.constant_chains``.

    For any ``(inn_id, region)`` pair, tells you every chain_id *within
    that same INN* that is a valid germline-reversion comparison candidate
    for a mutation observed at that region (via :func:`region_to_subdomain`).
    """
    flat = ds.constant_chains.reset_index()
    return {
        str(inn_id): _classify_by_sub_domain(grp)
        for inn_id, grp in flat.groupby(by="inn_id", dropna=False)
    }


def find_compatible_chain_ids(
    chain_id_arm: str,
    inn_id: str,
    region: str,
    inn_chain_domain_map: dict[str, dict[str, list[ChainIDWithIteration]]],
) -> list[ChainIDWithIteration]:
    """Return the other chain_ids in *inn_id* comparable with *chain_id_arm* at *region*.

    Restricts candidates to the ``sub_domain`` bucket (:func:`region_to_subdomain`)
    that *region* belongs to, within this exact INN — never across INNs,
    and never mixing an unrelated sub_domain in.
    """
    domain = inn_chain_domain_map.get(inn_id)
    if domain is None:
        return []
    target_subdomain = region_to_subdomain(region)
    return [c for c in domain.get(target_subdomain, []) if c.chain_id != chain_id_arm]


def find_multi_chain_inns_by_subdomain(
    ds: "INNAnalysisDataset",
) -> tuple[dict[str, dict[str, list[str]]], dict[str, list[tuple[str, ...]]]]:
    """Group chains for cross-chain pairing by ``sub_domain``, not whole-chain ``chain_type``.

    Replaces the previous ``chain_type`` (h/l/scfv/fusion)-based grouping
    from :func:`~analytics._helpers.find_multi_chain_inns`, which incorrectly
    grouped chains by their *overall* chain_type label — causing e.g. a
    C-KAPPA/Fc fusion chain (labeled ``chain_type="l"`` because of its
    C-KAPPA domain) to only ever be pairwise-compared against other
    ``"l"``-labeled chains, even though its Fc/CH2-CH3 region should be
    compared against true heavy (``"V"`` sub_domain) chains instead.

    Grouping buckets are:

    * ``"V"`` — every heavy-chain constant sub_domain (CH1, CH2, CH3, CH4,
      CHS, HINGE-REGION).
    * ``"light"`` — C-KAPPA and C-LAMBDA pooled together (kappa vs lambda
      comparisons are allowed — see :func:`_classify_by_sub_domain`).

    Only chains with at least one constant-domain mutation (present in
    ``ds.individual_mutation_profile``) are included, and only INN/bucket
    combinations with ≥2 qualifying chains are kept — mirroring
    :func:`~analytics._helpers.find_multi_chain_inns`'s filtering.

    Returns
    -------
    multi_chain_inns : dict[str, dict[str, list[str]]]
        ``{"INN {inn_id}": {"V"|"light": [sorted chain_ids]}}``.
    same_chain_type_groups : dict[str, list[tuple[str, ...]]]
        ``{inn_id: [tuple_of_sorted_chain_ids, ...]}``.
    """
    mutated_chain_ids: set[str] = set(
        ds.individual_mutation_profile.index.get_level_values("chain_id")
    )

    flat = ds.constant_chains.reset_index()
    flat = flat[flat["chain_id"].isin(mutated_chain_ids)]

    bucket_of = {"CK": "light", "CL": "light", "V": "V"}
    flat = flat.assign(_bucket=flat["sub_domain"].map(bucket_of))
    if flat["_bucket"].isna().any():
        bad = sorted(flat.loc[flat["_bucket"].isna(), "sub_domain"].unique())
        raise AssertionError(f"Unknown sub_domain value(s) encountered: {bad}")

    multi_chain_inns: dict[str, dict[str, list[str]]] = {}
    same_chain_type_groups: dict[str, list[tuple[str, ...]]] = {}

    for (inn_id, bucket), grp in flat.groupby(["inn_id", "_bucket"], dropna=False):
        chain_ids = tuple(sorted(set(grp["chain_id"])))
        if len(chain_ids) < 2:
            continue
        inn_id = str(inn_id)
        multi_chain_inns.setdefault(f"INN {inn_id}", {})[bucket] = list(chain_ids)
        same_chain_type_groups.setdefault(inn_id, [])
        if chain_ids not in same_chain_type_groups[inn_id]:
            same_chain_type_groups[inn_id].append(chain_ids)

    return multi_chain_inns, same_chain_type_groups
