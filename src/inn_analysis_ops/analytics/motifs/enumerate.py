"""Chain-level motif enumeration.

Models
------
ChainEnumerationRow
    Typed input schema — one row per (INN, chain).
ChainEnumerationRowAnnotated
    ``ChainEnumerationRow`` extended with parsed ``unique_alleles`` and
    ``unique_genes`` derived from the raw ``c_allele`` string.

Functions
---------
annotate_constant_gene
    Derive ``unique_alleles`` / ``unique_genes`` from a ``ChainEnumerationRow``.
enumerate_chain_motifs
    Enumerate all non-empty mutation sub-motifs from per-chain EU strings.
step_filter_remove_motif
    Keep only rows where remove_motif=False (moved from chain.py).
step_restore_other_gene_hits
    Re-insert motifs from original table sharing same pos→mut signature
    across gene groups (moved from chain.py).
"""

from __future__ import annotations

import logging
import re
from collections import defaultdict
from functools import reduce
from itertools import combinations, product
from math import comb as _comb
from operator import pos
from typing import Literal

import pandas as pd
from pydantic import BaseModel, field_validator
from tqdm.auto import tqdm
from tqdm.auto import tqdm as _tqdm

from inn_analysis_ops.analytics._helpers import (
    _parse_mutation_pe_into_parts,
    build_token_ignore_set,
    eu_mutations_to_tokens,
    format_motif_label,
    token_matches_ignore_set,
)
from inn_analysis_ops.analytics.motifs.gene_assignment import (
    _create_inn_map_of_mutation_gene_info,
)
from inn_analysis_ops.analytics.motifs.allele_mapping import (
    ChainID_EU_Key,
    ChainID_Region_Key,
    ChainIdRegionAlleleMapper,
    MutationAlleleMapper,
    chain_has_fc_region,
    find_compatible_chain_ids,
)
from inn_analysis_ops.analytics.motifs.models import (
    ChainMotifStep,  # noqa: PLC0415
    ChainMotifStepResult,
    _RestoreOtherGeneHitsIn,
    _RestoreOtherGeneHitsOut,
)

_log = logging.getLogger(__name__)
_EU_TOKEN_RE = re.compile(r"^(\d+(?:\.[A-Za-z0-9]+)?)([A-Za-z\-]+)>([\^A-Za-z\-+]+)$")

# ---------------------------------------------------------------------------
# Input schema
# ---------------------------------------------------------------------------

GroupByField = Literal[
    "chain_type",
    "constant_chain_species",
    "variable_chain_species",
    "c_allele",
    "unique_alleles",
    "unique_genes",
]


def _parse_and_filter_tokens(
    eu_mutation_string: str | None,
    token_ignore_set: frozenset[str] | None = None,
) -> frozenset:
    raw_token_set = eu_mutations_to_tokens(eu_mutation_string)
    if not token_ignore_set:
        return raw_token_set
    return frozenset(
        token
        for token in raw_token_set
        if not token_matches_ignore_set(token, token_ignore_set)
    )


def _find_excessive_consecutive_deletions(
    eu_mutation_string: str | None,
    max_consecutive_deletions: int | None,
) -> frozenset[str]:
    """Return deletion tokens to drop when a consecutive run exceeds the threshold."""
    if max_consecutive_deletions is None:
        return frozenset()
    assert max_consecutive_deletions > 0, "max_consecutive_deletions must be > 0"
    if eu_mutation_string is None:
        return frozenset()
    s = str(eu_mutation_string).strip()
    if not s:
        return frozenset()

    raw_tokens = [tok.strip() for tok in s.split(",") if tok.strip()]
    if not raw_tokens:
        return frozenset()

    deletion_entries: list[tuple[int, str]] = []
    for tok in raw_tokens:
        m = _EU_TOKEN_RE.match(tok)
        if m is None:
            continue
        pos, _wt, mt = m.group(1), m.group(2), m.group(3)
        if mt != "-" or "." in pos:
            continue
        deletion_entries.append((int(pos), tok))

    if not deletion_entries:
        return frozenset()

    deletion_entries = sorted(set(deletion_entries), key=lambda x: (x[0], x[1]))
    dropped: set[str] = set()
    run: list[tuple[int, str]] = []

    def _finalize_run() -> None:
        if len(run) > max_consecutive_deletions:
            dropped.update(tok for _pos, tok in run)

    for entry in deletion_entries:
        if not run:
            run = [entry]
            continue
        if entry[0] == run[-1][0] + 1:
            run.append(entry)
            continue
        _finalize_run()
        run = [entry]
    _finalize_run()

    return frozenset(dropped)


def _find_resolved_parent(
    motif: str,
    resolved_cache: dict[str, list[str]],
    not_consumed: set[str],
    parents_map: dict[str, list[str]],
) -> list[str]:
    if motif in not_consumed:
        return [motif]
    if motif in resolved_cache:
        return resolved_cache[motif]
    result = sorted(
        set(
            new_p
            for p in parents_map[motif]
            for new_p in _find_resolved_parent(
                p, resolved_cache, not_consumed, parents_map
            )
        )
    )
    resolved_cache[motif] = result
    return result


def _collapse_to_shortest(
    motifs: list[str],
    pe_token_map: dict[str, set[str]],
) -> list[str]:
    unique: dict[str, set[str]] = {}
    for motif in sorted(motifs, key=len):
        motif_tokens = pe_token_map[motif]
        if not any(t.issubset(motif_tokens) for t in unique.values()):
            unique[motif] = motif_tokens
    return sorted(unique)


def _individual_profile_table_to_unique_regions(
    individual_mutation_profile: pd.DataFrame,
) -> dict[Literal["simple", "gene_specific"], dict[str, list[tuple[str, str]]]]:
    individual_mutation_profile["eu_mutation_name"] = individual_mutation_profile[
        ["germline_seq", "EU", "mutated_seq"]
    ].apply(lambda x: f"{x['germline_seq']}{x['EU']}{x['mutated_seq']}", axis=1)
    unique_gene_region_associations = (
        individual_mutation_profile.reset_index()
        .groupby(by=["eu_mutation_name"])
        .apply(
            lambda x: sorted(
                set(tuple(zip(*[x[f] for f in ["gene", "region_classification"]])))
            )
        )
        .to_dict()
    )

    classify_by_gene_dependency = {"simple": {}, "gene_specific": {}}

    for (
        mutation,
        gene_region_pairs,
    ) in unique_gene_region_associations.items():
        if len(gene_region_pairs) == 1:
            classify_by_gene_dependency["simple"][mutation] = gene_region_pairs[0][1]
            classify_by_gene_dependency["gene_specific"][
                (mutation, gene_region_pairs[0][0])
            ] = gene_region_pairs[0][1]
        else:
            unique_regions = []
            for gene, region in gene_region_pairs:
                unique_regions.append(region)
                classify_by_gene_dependency["gene_specific"][(mutation, gene)] = region
            unique_regions = sorted(set(unique_regions))
            if len(unique_regions) == 1:
                classify_by_gene_dependency["simple"][mutation] = unique_regions[0]

    return classify_by_gene_dependency


def _chain_key_set_to_inn_chains_dict(
    chain_key_set: frozenset,
) -> dict[str, list[str]]:
    """Convert ``frozenset[(inn, chain_id)]`` → ``{inn: [sorted chain_ids]}``."""
    inn_chains: dict[str, list[str]] = {}
    for inn, chain_id in chain_key_set:
        inn_chains.setdefault(inn, []).append(chain_id)
    return {inn: sorted(chains) for inn, chains in sorted(inn_chains.items())}


def _gen_inn_chain_mut_string_map(
    chain_motif_df: pd.DataFrame,
) -> dict[tuple[str, str], list[str]]:
    """Build ``{(inn, chain_id): [motif_string_pe sorted by length]}`` from *chain_motif_df*.

    For each row in *chain_motif_df*, every ``(inn, chain_id)`` pair in
    ``unique_inn_chains`` is mapped to its motif's ``mutation_string_pe``.
    The per-pair list is sorted shortest-first so ``subset[-1]`` gives the
    longest (most specific) motif for that chain.
    """
    inn_chain_to_motifs: dict[tuple[str, str], list[str]] = {}
    for _, row in _tqdm(
        chain_motif_df.iterrows(),
        total=len(chain_motif_df),
        desc="Building inn-chain motif map",
        leave=False,
    ):
        pe = row["mutation_string_pe"]
        for chain_id in row["unique_inn_chains"]:
            inn_id = chain_id.split("_")[0]
            key = (inn_id, chain_id)
            inn_chain_to_motifs.setdefault(key, []).append(pe)
    return {key: sorted(motifs, key=len) for key, motifs in inn_chain_to_motifs.items()}


def _remove_subsumed_motifs(motif_info: list[dict]) -> list[dict]:

    grouped_by_types = {}
    for new_motif in motif_info:
        if new_motif["type"] not in grouped_by_types:
            grouped_by_types[new_motif["type"]] = []
        grouped_by_types[new_motif["type"]].append(new_motif)
    final_filtered_groups = []
    for subset in grouped_by_types.values():
        all_mut_strings = sorted(
            subset,
            key=lambda x: -len(x["tokens"]),
        )

        current_set_by_mut_string = {}
        for mut in all_mut_strings:
            if mut["mutation_string_pe"] not in current_set_by_mut_string:
                current_set_by_mut_string[mut["mutation_string_pe"]] = set()
            current_set_by_mut_string[mut["mutation_string_pe"]].add(mut["inn"])

        token_tracking = {}
        for mut in all_mut_strings:
            ref = mut["mutation_string_pe"]
            token = mut["tokens"]
            inn_set = current_set_by_mut_string[ref]
            unique_element = True
            for selected_data in token_tracking.values():
                token_selected = selected_data["tokens"]
                inn_subset = selected_data["inn"]
                if token.issubset(token_selected) and inn_set.issubset(inn_subset):
                    unique_element = False
                    break
            if unique_element:
                token_tracking[ref] = {
                    "tokens": token,
                    "inn": inn_set,
                }
        for mut in all_mut_strings:
            if mut["mutation_string_pe"] in token_tracking:
                final_filtered_groups.append(mut)
    return final_filtered_groups


_COMPOUND_REGION_FALLBACK: dict[str, list[str]] = {
    "CH3": ["CH3", "CHS", "CH3-CHS"],
    "CH4": ["CH4", "CH4-CHS"],
    "CHS": ["CHS", "CH3", "CH3-CHS"],
    "HINGE": [
        "HINGE",
        "HINGE-REGION",
        "H1+H2+H3+H4",
        "H1+H2+H3",
        "H1+H2",
        "HINGESYN",
        "H",
        "H1",
        "H2",
        "H3",
        "H4",
        "H1+H2",
        "H-CH2",
    ],
    "C-KAPPA": ["C-REGION", "C-KAPPA"],
    "C-LAMBDA": ["C-REGION", "C-LAMBDA"],
    "CH2": ["CH2", "H-CH2"],
}


def _get_wt_with_fallback(
    cdb: "ConstantDB",
    cn: "ConstantChainNumbering",
    gene: str,
    sub_domain: str,
    species_filter: str | None = None,
) -> dict:
    """Return germline WT residues for *gene* at *sub_domain*, with compound fallback.

    For some genes (e.g. IGHG4) CH3 is stored as ``"CH3-CHS"`` in *cdb*.
    :func:`~models.inn_structure.wt_lookup.get_wt_residues` has an
    internal fallback for this, but may still miss a position if the EU
    allowset filtering is too narrow.  When the primary lookup is empty,
    this function retries with the sibling region(s) included so every
    position in the compound region is covered.

    Parameters
    ----------
    cdb, cn :
        Constant-chain database and numbering objects.
    gene :
        Gene or allele string e.g. ``"IGHG4"`` / ``"IGHG4*01"``.
    sub_domain :
        Primary sub-domain to query e.g. ``"CH3"``.
    species_filter :
        Passed through to :func:`get_wt_residues`.  ``None`` disables
        species preference and picks the first available entry.

    Returns
    -------
    dict[int | str, str]
        ``{eu_position: wt_residue}``.

    Raises
    ------
    AssertionError
        If no germline sequence can be found for *gene* at *sub_domain*
        (or any of its compound fallbacks).
    """
    from inn_analysis_ops.models.inn_structure.wt_lookup import get_wt_residues

    wt = get_wt_residues(cdb, cn, gene, [sub_domain], species_filter=species_filter)
    if not wt and sub_domain in _COMPOUND_REGION_FALLBACK:
        wt = get_wt_residues(
            cdb,
            cn,
            gene,
            _COMPOUND_REGION_FALLBACK[sub_domain],
            species_filter=species_filter,
        )
    if not wt:
        raise AssertionError(
            f"Cannot determine wt sequence for {gene}, sub_domain={_COMPOUND_REGION_FALLBACK.get(sub_domain, sub_domain)}"
        )

    return wt


def _is_germline_reversion(
    token: str,
    chain_id_arm: str,
    chain_id_other: str,
    chain_region_iteration: str | None,
    ind_mut_allele_map: "dict[ChainID_EU_Key, MutationAlleleMapper]",
    chain_id_region_allele_map: "ChainIdRegionAlleleMapper",
    cdb: "ConstantDB",
    cn: "ConstantChainNumbering",
) -> "bool | None":
    """Check whether *token* is a germline reversion relative to *chain_id_other*.

    A token observed on *chain_id_arm* is a germline reversion if its
    mutated residue equals the wild-type germline residue of
    *chain_id_other* at the same EU position and region.

    Unlike the earlier implementation, the region is **not** inferred from
    the EU numbering table — it comes directly from
    ``ind_mut_allele_map[(chain_id_arm, eu_pos)].region``, i.e. the region
    this exact mutation was already attributed to on this exact chain.  This
    avoids the fragile EU→region inference that previously caused mismatches
    (e.g. HINGE naming, CH3/CHS compound boundaries, light/light-chain EU
    collisions).

    Parameters
    ----------
    token :
        PE mutation string e.g. ``"L234F"``.
    chain_id_arm :
        Chain ID whose arm contains *token*.
    chain_id_other :
        The other bispecific chain to compare against.
    ind_mut_allele_map :
        Output of :func:`~analytics.motifs.allele_mapping._infer_chain_id_mut_to_allele_map`.
    chain_id_region_allele_map :
        Output of :func:`~analytics.motifs.allele_mapping._infer_chain_id_region_to_allele_map`.
    cdb, cn :
        Constant-chain database and numbering objects.

    Returns
    -------
    bool
        ``True`` if the token is a germline reversion into *chain_id_other*'s
        germline, ``False`` if not, ``None`` if the check could not be
        performed (missing mutation/region/allele info).
    """
    _pos_int, _pos_sfx, _germ, _mut = _parse_mutation_pe_into_parts(token)
    eu_key: int | str = f"{_pos_int}.{_pos_sfx}" if _pos_sfx else _pos_int

    mut_info = ind_mut_allele_map.get(ChainID_EU_Key(chain_id_arm, eu_key))
    if mut_info is None:
        _log.warning(
            "Germline-reversion: no mutation-allele info for chain '%s' at EU %r "
            "(token '%s') — skipping.",
            chain_id_arm,
            eu_key,
            token,
        )
        return None

    region = mut_info.region
    is_light = region in ("C-KAPPA", "C-LAMBDA")
    mapper = (
        chain_id_region_allele_map.light_chain
        if is_light
        else chain_id_region_allele_map.heavy_chain
    )

    # Light-chain regions may legitimately differ in subtype between arms
    # (kappa vs lambda) — CK-to-CL and CL-to-CK comparisons are allowed, so
    # try the arm's own region first, then the other light-chain subtype.
    regions_to_try = (
        (region, "C-LAMBDA" if region == "C-KAPPA" else "C-KAPPA")
        if is_light
        else (region,)
    )

    other_entry = None
    other_region = region
    for candidate_region in regions_to_try:
        other_entry = mapper.get(
            ChainID_Region_Key(chain_id_other, candidate_region, chain_region_iteration)
        )
        if other_entry is not None and other_entry[0] is not None:
            other_region = candidate_region
            break
        other_entry = None

    if other_entry is None:
        _log.debug(
            "Germline-reversion: chain '%s' has no allele resolved at region(s) %s "
            "iteration %r (token '%s' on chain '%s') — skipping this candidate.",
            chain_id_other,
            regions_to_try,
            chain_region_iteration,
            token,
            chain_id_arm,
        )
        return None

    other_allele, other_species = other_entry

    wt = _get_wt_with_fallback(
        cdb, cn, other_allele, other_region, species_filter=other_species
    )
    if eu_key not in wt:
        _log.debug(
            "Germline-reversion: EU position %r (token '%s') not found in germline "
            "of allele '%s' region '%s' — skipping this candidate.",
            eu_key,
            token,
            other_allele,
            other_region,
        )
        return None

    return wt[eu_key].upper() == _mut.upper()


def _partition_arm_tokens(
    arm_tokens: set[str],
    chain_id_arm: str,
    inn_id: str,
    inn_chain_domain_map: "dict[str, dict[str, list[str]]]",
    ind_mut_allele_map: "dict[ChainID_EU_Key, MutationAlleleMapper]",
    chain_id_region_allele_map: "ChainIdRegionAlleleMapper",
    cdb: "ConstantDB",
    cn: "ConstantChainNumbering",
) -> tuple[set[str], set[str]]:
    """Partition *arm_tokens* into exclusive and germline-reversion sets.

    For each token, the region it belongs to (from ``ind_mut_allele_map``)
    is used to look up the set of other chain_ids *within this same INN*
    that are valid comparison candidates (:func:`~analytics.motifs.allele_mapping.find_compatible_chain_ids`
    — only chains sharing the token's ``sub_domain`` category, i.e.
    V-to-V or CK/CL-to-CK/CL). :func:`_is_germline_reversion` is then
    tried against each candidate in turn; the token is rescued as a
    reversion if *any* candidate's germline matches.

    Parameters
    ----------
    arm_tokens :
        The arm-exclusive token set to partition (e.g. ``m1_only``).
    chain_id_arm :
        Chain whose arm the tokens belong to.
    inn_id :
        The INN these chains belong to — used to scope candidate lookup
        to chains of the *same* INN only.
    inn_chain_domain_map :
        Output of :func:`~analytics.motifs.allele_mapping._infer_inn_chain_domain_map`.
    ind_mut_allele_map, chain_id_region_allele_map, cdb, cn :
        Passed through to :func:`_is_germline_reversion`.

    Returns
    -------
    exclusive : set[str]
        Tokens that remain arm-exclusive.
    reversion : set[str]
        Tokens rescued as germline reversions.
    """
    exclusive: set[str] = set()
    reversion: set[str] = set()
    for token in arm_tokens:
        _pos_int, _pos_sfx, _germ, _mut = _parse_mutation_pe_into_parts(token)
        eu_key: int | str = f"{_pos_int}.{_pos_sfx}" if _pos_sfx else _pos_int
        mut_info = ind_mut_allele_map.get(ChainID_EU_Key(chain_id_arm, eu_key))
        if mut_info is None:
            exclusive.add(token)
            continue

        compatible_others = find_compatible_chain_ids(
            chain_id_arm, inn_id, mut_info.region, inn_chain_domain_map
        )
        if not compatible_others:
            _log.warning(
                "Germline-reversion: no sub_domain-compatible chain found for "
                "chain '%s' (INN %s) region '%s' (token '%s') — "
                "treating as arm-exclusive.",
                chain_id_arm,
                inn_id,
                mut_info.region,
                token,
            )
            exclusive.add(token)
            continue

        is_reversion = False
        any_checked = False
        for chain_id_other, chain_region_iteration in compatible_others:
            result = _is_germline_reversion(
                token,
                chain_id_arm,
                chain_id_other,
                chain_region_iteration,
                ind_mut_allele_map,
                chain_id_region_allele_map,
                cdb,
                cn,
            )
            if result is not None:
                any_checked = True
            if result is True:
                is_reversion = True
                break

        if not is_reversion and not any_checked:
            _log.warning(
                "Germline-reversion: chain '%s' (INN %s) region '%s' (token '%s') "
                "— none of the %d candidate(s) %s had a resolvable allele/germline "
                "— treating as arm-exclusive.",
                chain_id_arm,
                inn_id,
                mut_info.region,
                token,
                len(compatible_others),
                [c.chain_id for c in compatible_others],
            )

        if is_reversion:
            reversion.add(token)
        else:
            exclusive.add(token)
    return exclusive, reversion


def _build_cross_chain_rows(
    inn: str,
    chain_id_1: str,
    chain_id_2: str,
    m1_only: set[str],
    m2_only: set[str],
    common: set[str],
    m1_rescued: set[str],
    m2_rescued: set[str],
    both_have_fc: bool = True,
) -> list[dict]:
    """Build the cross-chain motif rows for one chain pair.

    Returns up to five rows:

    * ``bispecific_ind_branch`` for chain 1 (arm-exclusive tokens only)
    * ``bispecific_ind_branch`` for chain 2
    * ``bispecific_anchor`` (strict token intersection)
    * ``ind_chain_full_motif`` for chain 1 (``m1_rescued ∪ common``)
    * ``ind_chain_full_motif`` for chain 2 (``m2_rescued ∪ common``)

    Parameters
    ----------
    both_have_fc :
        Whether *both* chain_id_1 and chain_id_2 have a resolved Fc
        (CH2/CH3/CH4/CHS) allele — see
        :func:`~analytics.motifs.allele_mapping.chain_has_fc_region`.  If
        ``False``, this pairing has no real Fc-vs-Fc comparison (e.g. one
        arm is Fab-only / CH1-only), so the arm-exclusive rows are tagged
        ``bispecific_ind_branch_not_fc`` instead of ``bispecific_ind_branch``
        — these tokens are not a true bispecific-pairing feature, since
        there is no Fc partner to compare/rescue them against.
    """
    rows: list[dict] = []
    ind_branch_type = (
        "bispecific_ind_branch" if both_have_fc else "bispecific_ind_branch_not_fc"
    )

    def _row(
        chain: str, tokens: set[str], rtype: str, rescued: "set[str] | None" = None
    ) -> dict:
        r: dict = {
            "inn": inn,
            "chain": chain,
            "mutation_string_pe": "/".join(
                sorted(tokens, key=_parse_mutation_pe_into_parts)
            ),
            "tokens": tokens,
            "type": rtype,
        }
        if rescued:
            r["rescued_germline_pair"] = rescued
        return r

    if m1_only:
        rows.append(_row(chain_id_1, m1_only, ind_branch_type, m1_rescued or None))
    if m2_only:
        rows.append(_row(chain_id_2, m2_only, ind_branch_type, m2_rescued or None))
    if common:
        rows.append(_row(f"{chain_id_1},{chain_id_2}", common, "bispecific_anchor"))
    if m1_rescued and common:
        rows.append(_row(chain_id_1, m1_rescued | common, "ind_chain_full_motif"))
    if m2_rescued and common:
        rows.append(_row(chain_id_2, m2_rescued | common, "ind_chain_full_motif"))

    return rows


def _collect_global_rescued_tokens(motif_rows: list[dict]) -> set[str]:
    """Union every ``rescued_germline_pair`` across ALL rows into one flat set.

    Scope is intentionally global — across every chain and every INN — not
    scoped to a single ``(inn, chain)``. If a mutation token has been
    confirmed as a germline reversion anywhere in *motif_rows* (i.e. it
    matched the germline of some compatible partner chain at least once),
    it is treated as a universally-recognized reversion token: any other
    row whose main tokens still contain that literal token string is very
    likely the same underlying germline-reversion event, just observed on
    a different chain/pairing that didn't independently confirm it (e.g.
    no compatible partner chain was available in that specific pairing).
    """
    rescued: set[str] = set()
    for row in motif_rows:
        if row["type"] not in ("bispecific_ind_branch", "bispecific_ind_branch_not_fc"):
            continue
        row_rescued = row.get("rescued_germline_pair")
        if row_rescued:
            rescued.update(row_rescued)
    return rescued


def _collect_anchor_tokens_by_chain(
    motif_rows: list[dict],
) -> dict[tuple[str, str], set[str]]:
    """Build ``{(inn, chain): {anchor_token, ...}}`` from every ``bispecific_anchor`` row.

    ``bispecific_anchor`` rows are keyed by a comma-joined ``"chain_1,chain_2"``
    pair; this unions the anchor tokens each individual chain participated
    in, across every pairing it appears in — used as the "common" component
    when constructing a new ``ind_chain_full_motif`` row for a chain that
    previously had no rescued tokens (and therefore no such row) until the
    global reconciliation pass rescues one.
    """
    anchor_tokens: dict[tuple[str, str], set[str]] = defaultdict(set)
    for row in motif_rows:
        if row["type"] != "bispecific_anchor":
            continue
        for chain_id in row["chain"].split(","):
            anchor_tokens[(row["inn"], chain_id.strip())].update(row["tokens"])
    return dict(anchor_tokens)


def _reconcile_global_rescues(motif_rows: list[dict]) -> list[dict]:
    """Second-pass rescue reconciliation across the ENTIRE cross-chain result set.

    Scope is global: if a mutation token was confirmed as a germline
    reversion (present in *any* row's ``rescued_germline_pair``, for any
    chain, in any INN), that literal token is treated as a universally
    recognized reversion. Any other row — for any chain, in any INN —
    whose main tokens still contain that same token string is almost
    certainly the same underlying reversion event; it is stripped from
    that row's main tokens too, purely by mutation-token identity.

    For every ``bispecific_ind_branch``/``bispecific_ind_branch_not_fc`` row
    whose main tokens still contain a globally-recognized rescued token:

    * that token is moved out of the row's main tokens and into its
      ``rescued_germline_pair`` (row dropped entirely if no main tokens
      remain);
    * the row's ``mutation_string_pe`` is regenerated from the updated
      main tokens;
    * the sibling ``ind_chain_full_motif`` row for that ``(inn, chain)`` is
      updated to include the newly-rescued token (created from
      ``rescued ∪ anchor tokens`` — see :func:`_collect_anchor_tokens_by_chain`
      — if no such row existed yet).

    ``bispecific_anchor`` rows are never modified — a token there is a
    literal shared mutation between two specific chains, which is already
    unambiguous regardless of the global reversion-token set.

    Parameters
    ----------
    motif_rows :
        The full, un-deduplicated list of per-pairing rows produced by
        :func:`_build_cross_chain_rows` across every chain pair in every
        INN (i.e. ``new_motif_info`` before :func:`_remove_subsumed_motifs`).

    Returns
    -------
    list[dict]
        The reconciled row list — same shape as *motif_rows*, with some
        rows updated/dropped and (rarely) new ``ind_chain_full_motif`` rows
        added.
    """
    global_rescued = _collect_global_rescued_tokens(motif_rows)
    if not global_rescued:
        return motif_rows

    anchor_tokens_by_inn_chain = _collect_anchor_tokens_by_chain(motif_rows)

    full_motif_positions: dict[tuple[str, str], int] = {
        (row["inn"], row["chain"]): i
        for i, row in enumerate(motif_rows)
        if row["type"] == "ind_chain_full_motif"
    }

    result_rows: list[dict] = list(motif_rows)
    newly_rescued_by_inn_chain: dict[tuple[str, str], set[str]] = defaultdict(set)
    rows_updated = 0
    rows_dropped = 0

    for i, row in enumerate(motif_rows):
        if row["type"] not in ("bispecific_ind_branch", "bispecific_ind_branch_not_fc"):
            continue

        own_rescued = set(row.get("rescued_germline_pair") or set())
        leaked = (row["tokens"] & global_rescued) - own_rescued
        if not leaked:
            continue

        key = (row["inn"], row["chain"])
        new_tokens = row["tokens"] - leaked
        new_rescued = own_rescued | leaked

        new_row = dict(row)
        new_row["tokens"] = new_tokens
        new_row["mutation_string_pe"] = "/".join(
            sorted(new_tokens, key=_parse_mutation_pe_into_parts)
        )
        new_row["rescued_germline_pair"] = new_rescued
        result_rows[i] = new_row if new_tokens else None  # type: ignore[assignment]

        newly_rescued_by_inn_chain[key].update(leaked)

        if new_tokens:
            rows_updated += 1
            _log.info(
                "Global rescue reconciliation: chain '%s' (INN %s) — moved %s "
                "from main tokens to rescued_germline_pair (was already "
                "rescued elsewhere for this chain). Row '%s' -> '%s'.",
                row["chain"],
                row["inn"],
                sorted(leaked),
                row["mutation_string_pe"],
                new_row["mutation_string_pe"],
            )
        else:
            rows_dropped += 1
            _log.info(
                "Global rescue reconciliation: chain '%s' (INN %s) — row '%s' "
                "fully consumed by rescue (%s); dropping row.",
                row["chain"],
                row["inn"],
                row["mutation_string_pe"],
                sorted(leaked),
            )

    rows_full_motif_updated = 0
    rows_full_motif_created = 0
    for key, leaked in newly_rescued_by_inn_chain.items():
        inn, chain = key
        fm_idx = full_motif_positions.get(key)
        if fm_idx is not None:
            fm_row = dict(result_rows[fm_idx])
            fm_tokens = fm_row["tokens"] | leaked
            fm_row["tokens"] = fm_tokens
            fm_row["mutation_string_pe"] = "/".join(
                sorted(fm_tokens, key=_parse_mutation_pe_into_parts)
            )
            result_rows[fm_idx] = fm_row
            rows_full_motif_updated += 1
            _log.info(
                "Global rescue reconciliation: chain '%s' (INN %s) — added %s "
                "to existing ind_chain_full_motif row.",
                chain,
                inn,
                sorted(leaked),
            )
            continue

        # No existing ind_chain_full_motif row for this chain — build one
        # from the newly-rescued tokens plus whatever anchor tokens this
        # chain has ever shared with a partner.
        anchor_tokens = anchor_tokens_by_inn_chain.get(key, set())
        full_tokens = leaked | anchor_tokens
        if not full_tokens:
            continue
        result_rows.append(
            {
                "inn": inn,
                "chain": chain,
                "mutation_string_pe": "/".join(
                    sorted(full_tokens, key=_parse_mutation_pe_into_parts)
                ),
                "tokens": full_tokens,
                "type": "ind_chain_full_motif",
            }
        )
        rows_full_motif_created += 1
        _log.info(
            "Global rescue reconciliation: chain '%s' (INN %s) — created new "
            "ind_chain_full_motif row '%s' (no prior row existed).",
            chain,
            inn,
            full_tokens
            and "/".join(sorted(full_tokens, key=_parse_mutation_pe_into_parts)),
        )

    if (
        rows_updated
        or rows_dropped
        or rows_full_motif_updated
        or rows_full_motif_created
    ):
        _log.info(
            "Global rescue reconciliation summary: %d bispecific_ind_branch row(s) "
            "trimmed, %d fully consumed/dropped, %d ind_chain_full_motif row(s) "
            "updated, %d ind_chain_full_motif row(s) created.",
            rows_updated,
            rows_dropped,
            rows_full_motif_updated,
            rows_full_motif_created,
        )

    return [row for row in result_rows if row is not None]


def _extract_multi_chain_overlaps(
    multi_chain_inns: dict[str, dict[str, list[str]]],
    same_chain_type_groups: dict[str, list[tuple[str, ...]]],
    inn_chain_to_motifs: dict[tuple[str, str], list[str]],
    pe_token_map: dict[str, set[str]],
    ind_mut_allele_map: "dict[ChainID_EU_Key, MutationAlleleMapper] | None" = None,
    chain_id_region_allele_map: "ChainIdRegionAlleleMapper | None" = None,
    inn_chain_domain_map: "dict[str, dict[str, list[str]]] | None" = None,
    cdb: "ConstantDB | None" = None,
    cn: "ConstantChainNumbering | None" = None,
) -> tuple[list[dict], list[tuple]]:
    """Derive bispecific arm and anchor motifs from cross-chain pairing.

    Delegates per-pair logic to:
    - :func:`_partition_arm_tokens` — split arm tokens into exclusive / reversion
    - :func:`_build_cross_chain_rows` — construct the output rows

    Parameters
    ----------
    ind_mut_allele_map, chain_id_region_allele_map :
        Outputs of :mod:`analytics.motifs.allele_mapping`.  Required
        (together with ``cdb``/``cn``) to enable the germline-reversion
        check.  ``None`` disables it entirely.
    inn_chain_domain_map :
        Output of :func:`~analytics.motifs.allele_mapping._infer_inn_chain_domain_map`.
        Required (together with the above) to restrict germline-reversion
        comparisons to chains, within the same INN, sharing the token's
        sub_domain category (V-to-V or CK/CL-to-CK/CL only).
    cdb, cn :
        Constant-chain database and numbering objects.  Required for the
        germline-reversion check.

    Returns
    -------
    new_motif_info : list[dict]
    skipped : list[tuple]
    """
    skipped: list[tuple] = []
    new_motif_info: list[dict] = []

    _do_reversion = (
        ind_mut_allele_map is not None
        and chain_id_region_allele_map is not None
        and inn_chain_domain_map is not None
        and cdb is not None
        and cn is not None
    )

    known_chain_groups: dict[str, set[tuple]] = {
        inn: set(groups) for inn, groups in same_chain_type_groups.items()
    }

    for inn, chain_types in _tqdm(
        multi_chain_inns.items(),
        total=len(multi_chain_inns),
        desc=f"Cross-chain pairing ({len(multi_chain_inns)} multi-chain INNs)",
        leave=False,
    ):
        for chain_type, chains in chain_types.items():
            inn_id = inn.replace("INN", "").strip()
            if tuple(chains) not in known_chain_groups[inn_id]:
                skipped.append(
                    (inn, inn_id, chain_type, chains, known_chain_groups[inn_id])
                )
                continue

            longest_per_chain: list[tuple[str, str]] = []
            for chain_id in chains:
                motif_list = inn_chain_to_motifs.get((inn, chain_id))
                if motif_list:
                    longest_per_chain.append((motif_list[-1], chain_id))

            for idx1 in range(len(longest_per_chain)):
                for idx2 in range(idx1 + 1, len(longest_per_chain)):
                    m1, chain_id_1 = longest_per_chain[idx1]
                    m2, chain_id_2 = longest_per_chain[idx2]
                    if m1 == m2:
                        continue

                    m1_tokens = pe_token_map[m1]
                    m2_tokens = pe_token_map[m2]
                    m1_only: set[str] = m1_tokens - m2_tokens
                    m2_only: set[str] = m2_tokens - m1_tokens
                    common: set[str] = m1_tokens & m2_tokens

                    m1_rescued: set[str] = set()
                    m2_rescued: set[str] = set()

                    if (
                        ind_mut_allele_map is not None
                        or chain_id_region_allele_map is not None
                        or inn_chain_domain_map is not None
                    ):
                        if not _do_reversion:
                            raise ValueError(
                                f"ind_mut_allele_map/chain_id_region_allele_map/"
                                f"inn_chain_domain_map is set for INN {inn!r} but cdb, "
                                f"cn, or one of the maps were not provided. Pass all "
                                f"of them via MotifIndexConfig."
                            )
                        # Candidates = every other chain_id within this same INN
                        # sharing the token's sub_domain category — resolved
                        # per-token inside _partition_arm_tokens.
                        m1_only, m1_rescued = _partition_arm_tokens(
                            m1_only,
                            chain_id_1,
                            inn_id,
                            inn_chain_domain_map,
                            ind_mut_allele_map,
                            chain_id_region_allele_map,
                            cdb,
                            cn,
                        )
                        m2_only, m2_rescued = _partition_arm_tokens(
                            m2_only,
                            chain_id_2,
                            inn_id,
                            inn_chain_domain_map,
                            ind_mut_allele_map,
                            chain_id_region_allele_map,
                            cdb,
                            cn,
                        )

                    new_motif_info.extend(
                        _build_cross_chain_rows(
                            inn,
                            chain_id_1,
                            chain_id_2,
                            m1_only,
                            m2_only,
                            common,
                            m1_rescued,
                            m2_rescued,
                            both_have_fc=(
                                chain_has_fc_region(
                                    chain_id_1, chain_id_region_allele_map
                                )
                                and chain_has_fc_region(
                                    chain_id_2, chain_id_region_allele_map
                                )
                                if chain_id_region_allele_map is not None
                                else True
                            ),
                        )
                    )

    new_motif_info = _reconcile_global_rescues(new_motif_info)
    return _remove_subsumed_motifs(new_motif_info), skipped


class ChainEnumerationRow(BaseModel):
    """One (INN, chain) row supplied to :func:`enumerate_chain_motifs`.

    Maps directly from ``ObservedFullLengthChain`` in
    ``models.inn_structure.analytics``.

    ``constant_chain_mutations_EU`` uses the compact format produced by
    ``_fmt_numbered_muts``: comma-separated tokens such as ``"234L>A, 235L>A"``.
    ``None`` = germline (no constant-domain mutations); those rows are skipped.
    """

    inn: str
    chain_id: str
    chain_type: str
    constant_chain_species: str | None
    variable_chain_species: str | None
    c_allele: str | None
    v_allele: str | None
    j_allele: str | None
    total_constant_mutations: int
    constant_chain_mutations_EU: str

    @field_validator("inn", "chain_id", "chain_type")
    @classmethod
    def must_be_non_empty(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("must be a non-empty string")
        return value

    @field_validator("constant_chain_species")
    @classmethod
    def force_non_empty(cls, value: str | None) -> str:
        if value is None or not value.strip():
            return "NA"
        return value


class ChainEnumerationRowAnnotated(ChainEnumerationRow):
    """``ChainEnumerationRow`` with parsed constant-gene annotation.

    ``unique_alleles`` and ``unique_genes`` are derived from ``c_allele``
    by :func:`annotate_constant_gene`.  Use these as ``group_by`` fields
    instead of the raw ``c_allele`` string.
    """

    unique_alleles: tuple[str, ...]  # e.g. ("IGHG1*01", "IGHG1*03")
    unique_genes: tuple[str, ...]  # e.g. ("IGHG1",)


def chain_collection_to_enumeration_rows(
    collection: "ObservedFullLengthCollection",
) -> list[ChainEnumerationRowAnnotated]:
    """Convert an ``ObservedFullLengthCollection`` to annotated enumeration rows.

    Only chains with a non-empty ``constant_chain_mutations_EU`` value are
    included — germline chains carry no motif signal and are skipped.

    Parameters
    ----------
    collection :
        ``ds.full_length_chain_collection`` from a
        ``ConstantChainAnalysisResult``.

    Returns
    -------
    list[ChainEnumerationRowAnnotated]
        One entry per chain with at least one constant-domain EU mutation,
        with ``unique_alleles`` and ``unique_genes`` pre-computed from
        ``c_allele``.
    """
    return [
        annotate_constant_gene(ChainEnumerationRow.model_validate(record.model_dump()))
        for record in collection.records
        if record.constant_chain_mutations_EU
        and record.constant_chain_mutations_EU.strip()
    ]


def annotate_constant_gene(
    row: ChainEnumerationRow,
) -> ChainEnumerationRowAnnotated:
    """Derive ``unique_alleles`` and ``unique_genes`` from ``row.c_allele``.

    Parameters
    ----------
    row :
        Any :class:`ChainEnumerationRow` (or subclass).

    Returns
    -------
    :class:`ChainEnumerationRowAnnotated` — the original row plus:

    ``unique_alleles``
        Sorted unique uppercased allele strings split from ``c_allele`` on
        ``","`` e.g. ``"IGHG1*01,ighg1*03"`` → ``("IGHG1*01", "IGHG1*03")``.
    ``unique_genes``
        Sorted unique gene names (portion before ``"*"``) derived from
        ``unique_alleles``, e.g. ``("IGHG1",)``.

    Both fields are empty tuples when ``c_allele`` is ``None``.
    """
    if row.c_allele is None:
        unique_alleles: tuple[str, ...] = ()
        unique_genes: tuple[str, ...] = ()
    else:
        unique_alleles = tuple(
            sorted(
                {
                    token.strip().upper()
                    for token in row.c_allele.split(",")
                    if token.strip()
                }
            )
        )
        unique_genes = tuple(
            sorted({allele.split("*")[0] for allele in unique_alleles})
        )
    return ChainEnumerationRowAnnotated(
        **row.model_dump(exclude={"unique_alleles", "unique_genes"}),
        unique_alleles=unique_alleles,
        unique_genes=unique_genes,
    )


def _expand_to_manual_motifs(
    enum_df: pd.DataFrame,
    forced_manual_motif_strings: list[str],
    token_ignore_set: frozenset,
) -> pd.DataFrame:
    """Expand enumerated motifs to include manually curated motifs."""
    # Implementation goes here

    found_exact_motif = enum_df["mutation_string_pe"].isin(forced_manual_motif_strings)

    reinsert_exact_motif = found_exact_motif & enum_df["remove_motif"]
    if reinsert_exact_motif.sum():
        _log.info(f"Expanding motif set to include {reinsert_exact_motif.sum()} motifs")
        _log.info(
            f"Update new motifs with {enum_df.loc[reinsert_exact_motif, 'mutation_string_pe'].tolist()}"
        )

    enum_df.loc[reinsert_exact_motif, "remove_motif"] = False

    enum_df.loc[reinsert_exact_motif, "motif_selection_reason"] = (
        "manual_motif_curation"
    )

    enum_df.loc[reinsert_exact_motif, "motif_detail"] = (
        enum_df.loc[reinsert_exact_motif, "motif_detail"]
        .fillna("")
        .apply(lambda x: f"{x}; added back in because of manual curation".strip("; "))
    )

    return enum_df


def enumerate_chain_motifs(
    chain_rows: (
        list[ChainEnumerationRow]
        | list[ChainEnumerationRowAnnotated]
        | "ObservedFullLengthCollection"
    ),
    individual_mutation_profile: pd.DataFrame,
    min_chain_count: int = 1,
    min_motif_size: int = 2,
    max_motif_size: int | None = None,
    mutations_to_ignore: dict | None = None,
    forced_manual_motif_strings: list[str] | None = None,
    max_consecutive_deletions: int | None = 5,
) -> pd.DataFrame:
    """Enumerate all non-empty sub-motifs from per-chain EU mutation strings.

    Parameters
    ----------
    chain_rows :
        Either a sequence of :class:`ChainEnumerationRow` (or
        :class:`ChainEnumerationRowAnnotated`), **or** an
        ``ObservedFullLengthCollection`` (i.e. ``ds.full_length_chain_collection``)
        — in which case :func:`chain_collection_to_enumeration_rows` is called
        automatically to extract and filter mutated chains.
    min_chain_count :
        Motifs carried by fewer distinct ``(inn, chain_id)`` pairs are excluded.
    min_motif_size :
        Minimum number of tokens per motif.  Default 2 — singletons are skipped.
        Set to 1 to include single-mutation motifs.
    max_motif_size :
        Hard cap on token-set size.  ``None`` = enumerate all sizes.
    mutations_to_ignore :
        ``MotifIndexConfig.mutations_to_ignore_for_motif_ops`` —
        ``{mutation_type: {region: [positions]}}``.  Matching tokens are
        stripped from every chain's profile before enumeration.  ``None``
        applies no filtering.
    max_consecutive_deletions :
        Maximum allowed run length of consecutive deletion tokens (integer EU
        positions) within one chain profile. If any run exceeds this threshold,
        deletion tokens in that run are excluded from enumeration while
        non-deletion mutations from the same chain are preserved. ``None``
        disables this safeguard.

    Returns
    -------
    DataFrame sorted (motif_length desc, inn_count desc):

    ``mutation_string``        – sorted ``"pos:change / ..."`` colon-notation string
    ``motif_length``           – number of tokens
    ``inn_count``              – distinct INNs carrying this motif
    ``chain_count``            – distinct ``(inn, chain_id)`` pairs
    ``remove_motif``           – ``True`` iff this motif should be excluded
    ``motif_selection_reason`` – short category when ``remove_motif=True``
    ``motif_detail``           – verbose explanation
    ``super_motifs``           – direct non-consumed parent motif PE strings
    ``motif_token_set``        – ``frozenset[str]`` of colon-notation tokens
    """
    # Accept ObservedFullLengthCollection directly — convert to annotated rows
    if hasattr(chain_rows, "records"):
        chain_rows = chain_collection_to_enumeration_rows(chain_rows)

    if not chain_rows:
        raise ValueError("enumerate_chain_motifs: no mutated chain rows found.")

    token_ignore_set: frozenset = (
        build_token_ignore_set(mutations_to_ignore)
        if mutations_to_ignore
        else frozenset()
    )

    chain_df: pd.DataFrame = pd.DataFrame(
        [row.model_dump() for row in chain_rows if row.total_constant_mutations > 0]
    )

    enum_df = _enumerate_single_stratum(
        chain_df,
        individual_mutation_profile=individual_mutation_profile,
        token_ignore_set=token_ignore_set,
        min_chain_count=min_chain_count,
        min_motif_size=min_motif_size,
        max_motif_size=max_motif_size,
        max_consecutive_deletions=max_consecutive_deletions,
        stratum_label="all",
    )

    if forced_manual_motif_strings is not None:
        enum_df = _expand_to_manual_motifs(
            enum_df, forced_manual_motif_strings, token_ignore_set
        )
    return enum_df


# ---------------------------------------------------------------------------
# _enumerate_single_stratum — private helpers
# ---------------------------------------------------------------------------

# Columns returned by _enumerate_single_stratum when the input has no mutated
# chains.  Used to return a schema-correct empty DataFrame for early exits.
_EMPTY_STRATUM_COLUMNS = [
    "mutation_string",
    "mutation_string_pe",
    "motif_length",
    "inn_count",
    "chain_count",
    "inns",
    "unique_inn_chains",
    "unique_inn_chain_counts",
    "remove_motif",
    "motif_selection_reason",
    "motif_detail",
    "super_motifs",
    "motif_token_set",
]

# Regions treated as secondary (non-primary) for the allele-call priority logic.
# Mutations in HINGE / CHS / HINGE-REGION are structurally ambiguous between
# genes and are used only as a fallback when no FC or primary-domain annotation
# is available.
_SECONDARY_REGIONS: frozenset[str] = frozenset({"HINGE", "CHS", "HINGE-REGION"})

# Constant-domain regions belonging to the heavy chain arm.
# Used to classify a motif's chain-arm type as "heavy", "light", or "unk".
_HEAVY_REGIONS: frozenset[str] = frozenset(
    {"FC", "HINGE", "HINGE-REGION", "CHS", "CH1", "CH2", "CH3"}
)
# Constant-domain regions belonging to the light chain arm.
_LIGHT_REGIONS: frozenset[str] = frozenset({"CL", "CK"})


def _validate_no_duplicate_chain_keys(chain_df: pd.DataFrame) -> None:
    """Raise ValueError if any (inn, chain_id) pair appears more than once.

    Each (inn, chain_id) must be unique so that the inverted index maps tokens
    to exactly the right set of chains — duplicates would inflate carrier counts
    and produce misleading motif frequencies.
    """
    dup_mask = chain_df.duplicated(subset=["inn", "chain_id"])
    if dup_mask.any():
        raise ValueError(
            "enumerate_chain_motifs: duplicate (inn, chain_id) pairs:\n"
            + chain_df.loc[dup_mask, ["inn", "chain_id"]].to_string()
        )


def _build_token_sets_and_inverted_index(
    chain_df: pd.DataFrame,
    token_ignore_set: frozenset,
    max_consecutive_deletions: int | None = None,
) -> tuple[
    pd.Series,  # chain_key_series (filtered to mutated chains only)
    pd.Series,  # token_set_series (filtered to mutated chains only)
    dict[str, frozenset[tuple[str, str]]],  # token → frozenset[(inn, chain_id)]
]:
    """Parse EU mutation strings into token sets and build an inverted index.

    Step 1 — build a (inn, chain_id) key series aligned with chain_df's index
    so that every downstream mask operation keeps the index in sync.

    Step 2 — parse each chain's EU mutation string into a frozenset of tokens,
    optionally dropping any token that matches the caller-supplied ignore set.
    A token looks like ``"234L>A"`` or ``"235L>A"`` (EU position notation).

    Step 3 — keep only chains that have at least one surviving token after
    filtering; germline chains (empty token set) carry no motif signal and
    would pollute the inverted index.

    Step 4 — build the inverted index by exploding each (chain_key, token_set)
    pair into one row per token, then grouping by token to collect the set of
    chain keys that carry it.  This lets the enumeration step look up carriers
    for any single token in O(1) instead of scanning every chain.

    Returns only rows/entries that survive the mutated-chain filter.
    """
    # Step 1: pair (inn, chain_id) tuples aligned with the DataFrame index
    chain_key_series = pd.Series(
        list(zip(chain_df["inn"], chain_df["chain_id"])),
        index=chain_df.index,
        name="chain_key",
    )

    # Step 2: parse + filter each EU mutation string → frozenset of tokens.
    # If a chain has an excessive run of consecutive deletions, drop only
    # those deletion tokens and keep the remaining mutation profile.
    token_sets: list[frozenset] = []
    for row in chain_df[["inn", "chain_id", "constant_chain_mutations_EU"]].itertuples(
        index=False
    ):
        excessive_deletions = _find_excessive_consecutive_deletions(
            row.constant_chain_mutations_EU,
            max_consecutive_deletions=max_consecutive_deletions,
        )
        eu_for_parse = row.constant_chain_mutations_EU
        if excessive_deletions:
            raw_tokens = [
                tok.strip()
                for tok in str(row.constant_chain_mutations_EU).split(",")
                if tok.strip()
            ]
            kept_tokens = [tok for tok in raw_tokens if tok not in excessive_deletions]
            eu_for_parse = ", ".join(kept_tokens)
            _log.warning(
                "Skipping deletion tokens from motif enumeration due to "
                "consecutive deletions > %d for inn=%s chain_id=%s "
                "skipped_deletion_tokens=%s mutation_string=%s",
                max_consecutive_deletions,
                row.inn,
                row.chain_id,
                sorted(excessive_deletions),
                row.constant_chain_mutations_EU,
            )
        token_sets.append(
            _parse_and_filter_tokens(
                eu_for_parse,
                token_ignore_set=token_ignore_set,
            )
        )
    token_set_series = pd.Series(token_sets, index=chain_df.index)

    # Step 3: drop chains whose entire token set was filtered out (germline-like)
    mutated_mask = token_set_series.map(len) > 0
    chain_key_series = chain_key_series.loc[mutated_mask]
    token_set_series = token_set_series.loc[mutated_mask]

    # Step 4: explode (chain_key, token_set) → long (chain_key, token) table,
    # then group by token to produce {token: frozenset[(inn, chain_id)]}
    token_to_chain_key_set: dict[str, frozenset[tuple[str, str]]] = (
        pd.DataFrame({"chain_key": chain_key_series, "token_set": token_set_series})
        .assign(token=lambda df: df["token_set"].map(sorted))  # sort for stable explode
        .explode("token")
        .drop(columns="token_set")
        .reset_index(drop=True)
        .groupby("token")["chain_key"]
        .agg(frozenset)  # one frozenset per token containing all its carrier chain keys
        .to_dict()
    )

    return chain_key_series, token_set_series, token_to_chain_key_set


def _enumerate_motif_carriers(
    token_set_series: pd.Series,
    token_to_chain_key_set: dict[str, frozenset],
    gene_info_map: dict[tuple, list],
    min_chain_count: int,
    max_motif_size: int | None,
    stratum_label: str,
) -> dict[frozenset, dict[tuple[str, str], dict[str, list]]]:
    """Enumerate all sub-motifs from unique mutation profiles and find their carriers.

    A *motif* is any non-empty subset of tokens drawn from a chain's mutation
    profile.  A chain *carries* a motif iff it has every token in that motif.

    Algorithm
    ---------
    1. Collect the distinct mutation profiles (unique token-set values).  Many
       chains may share the same profile (e.g., all IGHG1*01 chains with the
       same two substitutions), so deduplicating avoids re-enumerating identical
       sub-motif sets.

    2. **Pre-prune rare tokens** — a token carried by fewer than
       ``min_chain_count`` chains globally can never appear in a motif that
       passes the frequency threshold.  We strip such tokens from every profile
       before generating combinations.  This is the key optimisation that prevents
       combinatorial explosion on chains with many idiosyncratic mutations:
       a chain with 31 mutations where 28 are unique to it is reduced to a 3-token
       profile, slashing its C(31,k) fanout to C(3,k).

    3. Pre-compute the total combination count across the pruned profiles for the
       progress bar.  The ``motif_to_carriers`` cache ensures each unique motif
       token-set is processed only once even if it appears in many profiles.

    4. For each profile, iterate over every combination of the surviving tokens
       up to ``effective_max`` size.  Skip already-seen motifs.

    5. For a new motif, find its carrier chains by intersecting the inverted-
       index sets for each of its tokens.  Only chains that have *all* tokens
       survive the intersection — this is the key correctness invariant.

    6. Attach gene annotation per (inn, chain, token) from ``gene_info_map``
       so later steps can resolve allele/region information without another join.

    Returns
    -------
    motif_to_carriers :
        ``{motif_token_set: {(inn, chain_id): {token: gene_info_list}}}``
        where ``gene_info_list`` is the list of (region, gene, allele, species)
        tuples returned by ``_create_inn_map_of_mutation_gene_info``.
    full_length_motifs :
        A set of ``frozenset`` token-sets that represent complete profiles
        (full-length mutations for each unique profile).  These are enumerated
        first and should be labeled and filtered separately from sub-motifs.
    """
    # Step 1: deduplicate profiles — many chains share identical mutation profiles
    unique_profiles: list[frozenset] = list(token_set_series.unique())

    # Step 2: pre-prune tokens that are globally too rare to ever qualify.
    # A token appearing in fewer than min_chain_count chains can never be part
    # of a motif that passes the frequency filter, so enumerating it wastes work.
    qualifying_tokens: frozenset[str] = frozenset(
        tok
        for tok, carriers in token_to_chain_key_set.items()
        if len(carriers) >= min_chain_count
    )
    # For each profile, keep only tokens that can contribute to qualifying motifs.
    # Profiles that lose all their tokens are skipped entirely.
    pruned_profiles: list[frozenset] = [
        profile & qualifying_tokens
        for profile in unique_profiles
        if profile & qualifying_tokens  # skip profiles left empty after pruning
    ]

    pruned_count = sum(1 for p, pp in zip(unique_profiles, pruned_profiles) if p != pp)
    if pruned_count:
        _log.debug(
            f"[{stratum_label}] Pruned rare tokens from {pruned_count}/{len(unique_profiles)} "
            f"profiles ({len(qualifying_tokens)}/{len(token_to_chain_key_set)} tokens survive)"
        )

    # Step 3: upper-bound on loop iterations for the progress bar (post-prune)
    total_attempts = sum(
        sum(
            _comb(len(p), s)
            for s in range(
                1,
                (len(p) if max_motif_size is None else min(len(p), max_motif_size)) + 1,
            )
        )
        for p in pruned_profiles
    )

    # Step 4–6: enumerate combinations and find carrier chains
    # motif_ts (frozenset of tokens) → {(inn, chain_id): {token: gene_info}}
    motif_to_carriers: dict[frozenset, dict] = {}
    # Track which motifs are full-length (complete profile) so they can be
    # labeled differently and exempt from removal filters
    full_length_motifs: set[frozenset] = set()

    with _tqdm(
        total=total_attempts,
        desc=f"[{stratum_label}] Enumerating motifs ({len(pruned_profiles)} profiles)",
        leave=False,
    ) as pbar:
        for profile in pruned_profiles:
            sorted_tokens = sorted(profile)  # deterministic iteration order

            # Step 4a: Enumerate the full-length motif (the complete profile)
            # BEFORE breaking it down into sub-motifs. This ensures every unique
            # profile appears as a motif with reason='full_length_inn_mutation'.
            full_motif_ts = frozenset(sorted_tokens)
            if full_motif_ts not in motif_to_carriers:
                # Find carriers: chains that have every token in this profile
                carriers: frozenset = reduce(
                    frozenset.intersection,
                    (token_to_chain_key_set[t] for t in full_motif_ts),
                )
                motif_to_carriers[full_motif_ts] = {
                    (inn, chain): {
                        tok: gene_info_map[(inn, chain, format_motif_label(tok))]
                        for tok in full_motif_ts
                    }
                    for inn, chain in carriers
                }
                full_length_motifs.add(full_motif_ts)
                pbar.update(1)  # count the full-length enumeration

            # Step 4b: Enumerate sub-motifs using the min/max algorithm
            effective_max = (
                len(sorted_tokens)
                if max_motif_size is None
                else min(len(sorted_tokens), max_motif_size)
            )
            for size in range(1, effective_max + 1):
                for token_tuple in combinations(sorted_tokens, size):
                    motif_ts = frozenset(token_tuple)
                    pbar.update(1)

                    # Skip motifs already enumerated from a different profile
                    if motif_ts in motif_to_carriers:
                        continue

                    # Intersect per-token carrier sets: only chains carrying
                    # every token in this motif survive
                    carriers: frozenset = reduce(
                        frozenset.intersection,
                        (token_to_chain_key_set[t] for t in motif_ts),
                    )

                    # Attach gene annotation per carrier so
                    # _resolve_allele_combos_for_inn_chain can work without
                    # an extra join step
                    motif_to_carriers[motif_ts] = {
                        (inn, chain): {
                            tok: gene_info_map[(inn, chain, format_motif_label(tok))]
                            for tok in motif_ts
                        }
                        for inn, chain in carriers
                    }

    return motif_to_carriers, full_length_motifs


def _build_super_motif_index(
    motif_to_carriers: dict[frozenset, dict],
    stratum_label: str = "all",
) -> dict[frozenset, list[frozenset]]:
    """Map each motif to its list of direct super-motifs (motifs one token larger).

    A super-motif of M is any enumerated motif M' = M ∪ {t} for some token t.
    The index is used by ``_annotate_super_motifs_and_removal`` to decide
    whether a motif is *subsumed* — i.e., every chain that carries M also
    carries a strictly larger motif M', making M redundant.

    Algorithm (O(N·k) where N = number of motifs, k = average motif size):
    For every motif M, remove each of its k tokens in turn to produce each
    direct sub-motif M minus {t}.  If that sub-motif was enumerated, register M as
    one of its super-motifs.  Because we iterate over all M, every sub-motif
    accumulates all of its super-motifs by the time the loop finishes.
    """
    n = len(motif_to_carriers)
    # Initialise every enumerated motif with an empty super-motif list
    super_index: dict[frozenset, list[frozenset]] = {ts: [] for ts in motif_to_carriers}

    for motif_ts in _tqdm(
        motif_to_carriers,
        total=n,
        desc=f"[{stratum_label}] Building super-motif index ({n} motifs)",
        leave=False,
    ):
        for token in motif_ts:
            # sub_ts is motif_ts with one token removed
            sub_ts = motif_ts - {token}
            # Only register if sub_ts itself was enumerated (it may not be if
            # it fell below a size floor or had zero carriers)
            if sub_ts in super_index:
                super_index[sub_ts].append(motif_ts)

    return super_index


def _resolve_allele_combos_for_inn_chain(
    mut_data: dict[str, list],
) -> dict[str, list[str] | None]:
    """Resolve per-token gene info into classified allele path strings.

    Each token in a motif can be associated with multiple (region, gene,
    allele, species) tuples — e.g., a position may exist in different
    germline genes.  ``itertools.product`` generates every combination of
    gene-info tuples across all tokens so that every possible allele assignment
    for this (inn, chain, motif) is considered.

    Each combination is classified into three buckets:

    ``fc_keys``    — tuples whose region is exactly ``"FC"``
    ``unique_keys``— tuples whose region is NOT a secondary region
    ``other_keys`` — tuples whose region IS a secondary region
                     (HINGE / CHS / HINGE-REGION)

    Within each bucket, duplicate (species, gene) keys are collapsed (only the
    first allele seen for a key is kept per combination).  The bucket is then
    serialised as a pipe-separated string of ``"species_gene_allele"`` entries,
    sorted for stable output.

    The three sorted-unique lists produced across all combinations become the
    ``fc_only_allele_info``, ``allele_info``, and ``backup_allele_info`` fields
    that ``_coalesce_allele_call`` later resolves in priority order.

    Parameters
    ----------
    mut_data :
        ``{token: [(region, gene, allele, species), ...]}`` for one (inn, chain).

    Returns
    -------
    dict with keys:
        ``region``             – slash-joined sorted set of all region strings seen
        ``fc_only_allele_info``– sorted unique FC-region path strings, or None
        ``allele_info``        – sorted unique non-secondary-region path strings, or None
        ``backup_allele_info`` – sorted unique secondary-region path strings, or None
    """
    # Each value in mut_data is a list of (region, gene, allele, species) tuples.
    # product(*token_gene_lists) generates one combination per token assignment.
    token_gene_lists = list(mut_data.values())

    all_fc: list[str] = []
    all_unique: list[str] = []
    all_other: list[str] = []
    stitches: list[str] = []  # collects every region seen across all combinations

    for combo in product(*token_gene_lists):
        # One dict per bucket for this combination; keyed by (species, gene)
        # to collapse duplicate gene assignments within a single combination
        fc_keys: dict[tuple, list] = defaultdict(list)
        unique_keys: dict[tuple, list] = defaultdict(list)
        other_keys: dict[tuple, list] = defaultdict(list)

        for region, gene, allele, species in combo:
            stitches.append(region)
            key = (species, gene)

            # FC bucket: only FC-region entries (used as primary allele signal)
            if region == "FC" and key not in fc_keys:
                fc_keys[key].append(allele)

            # Secondary / backup bucket: HINGE, CHS, HINGE-REGION
            if region in _SECONDARY_REGIONS and key not in other_keys:
                other_keys[key].append(allele)
            # Primary non-secondary bucket: everything else (CH1, CH2, CH3, CL, …)
            elif region not in _SECONDARY_REGIONS and key not in unique_keys:
                unique_keys[key].append(allele)

        # Serialise each bucket as a stable pipe-separated string
        def _path(mapping: dict[tuple, list]) -> str:
            # Format: "species_gene_allele1/allele2" per (species, gene) key,
            # then pipe-join the sorted unique entries across all keys
            return "|".join(
                sorted({f"{k[0]}_{k[1]}_{'/'.join(v)}" for k, v in mapping.items()})
            )

        if fc_keys:
            all_fc.append(_path(fc_keys))
        if unique_keys:
            all_unique.append(_path(unique_keys))
        if other_keys:
            all_other.append(_path(other_keys))

    return {
        # Slash-joined sorted unique regions seen across all token/combo combinations
        "region": "/".join(sorted(set(stitches))) if stitches else None,
        # Deduplicated path strings per bucket (None when the bucket was never filled)
        "fc_only_allele_info": sorted(set(all_fc)) or None,
        "allele_info": sorted(set(all_unique)) or None,
        "backup_allele_info": sorted(set(all_other)) or None,
    }


def _build_per_motif_annotation_df(
    motif_to_carriers: dict[frozenset, dict],
    stratum_label: str = "all",
) -> pd.DataFrame:
    """Flatten carrier gene-annotation data into a long per-(motif, inn, chain) DataFrame.

    For every (motif, inn, chain) triple in ``motif_to_carriers``, calls
    ``_resolve_allele_combos_for_inn_chain`` to convert the raw per-token gene
    info into the four annotation fields (region, allele_info, …).

    The result is a *long* table — one row per (motif × carrier chain) — before
    any grouping.  Later steps coalesce allele calls and then group the long
    table back to one row per motif × allele-call combination.
    """
    n = len(motif_to_carriers)
    rows = []
    for motif_ts, carriers in _tqdm(
        motif_to_carriers.items(),
        total=n,
        desc=f"[{stratum_label}] Resolving gene annotation ({n} motifs)",
        leave=False,
    ):
        for (inn, chain_id), mut_data in carriers.items():
            # Resolve the gene-annotation product for this (inn, chain, motif)
            allele_fields = _resolve_allele_combos_for_inn_chain(mut_data)
            rows.append(
                {
                    "motif_token_set": motif_ts,
                    "inn": inn,
                    "chain_id": chain_id,
                    **allele_fields,  # region, fc_only_allele_info, allele_info, backup_allele_info
                }
            )
    return pd.DataFrame(rows)


def _coalesce_allele_call(dff: pd.DataFrame) -> pd.DataFrame:
    """Pick the best available allele call via priority fallback, then explode rows.

    Each (motif, inn, chain) row may have up to three allele-info columns, each
    a list of path strings representing different gene-region classifications:

    1. ``fc_only_allele_info``  — preferred: derived exclusively from FC region entries
    2. ``allele_info``          — fallback: from non-secondary regions (CH1, CH2, CH3, …)
    3. ``backup_allele_info``   — last resort: from secondary regions (HINGE, CHS, …)

    The coalescing loop fills ``mutation_inn_chain_allele_call`` from the
    highest-priority column and falls back to the next only where the result
    is still null.

    After coalescing, the column holds a *list* of path strings (one per
    distinct allele combination).  ``explode`` turns each list element into its
    own row so that downstream groupby aggregation treats each allele call as a
    separate category.
    """
    allele_priority = [
        "fc_only_allele_info",
        "allele_info",
        "backup_allele_info",
    ]
    dff = dff.copy()

    # Seed the call column from the highest-priority source
    dff["mutation_inn_chain_allele_call"] = dff[allele_priority[0]]

    # Fill nulls progressively from lower-priority sources
    for fallback in allele_priority[1:]:
        null_mask = dff["mutation_inn_chain_allele_call"].isnull()
        dff.loc[null_mask, "mutation_inn_chain_allele_call"] = dff.loc[
            null_mask, fallback
        ]

    # Drop rows where all three allele sources were None — these represent chains
    # with no resolvable gene annotation and cannot be grouped or split downstream
    dff = dff[dff["mutation_inn_chain_allele_call"].notna()].copy()

    # One row per distinct allele-call path string
    return dff.explode("mutation_inn_chain_allele_call")


def _add_gene_and_region_columns(
    dff: pd.DataFrame,
    stratum_label: str = "all",
) -> pd.DataFrame:
    """Derive gene_species / gene / alleles and heavy-vs-light region columns.

    ``mutation_inn_chain_allele_call`` is a pipe-separated string of
    ``"species_gene_allele"`` entries, e.g.
    ``"human_IGHG1_01|human_IGHG1_03"``.

    For each of the three underscore-delimited fields (species=0, gene=1,
    alleles=2) we split on ``"|"``, extract that index position from each
    entry, deduplicate, sort, and re-join with ``"|"`` to produce a compact
    summary column.

    ``h_l_crossed_regions`` classifies each slash-separated region token as
    ``"heavy"``, ``"light"``, or ``"unk"`` using the module-level region sets,
    then collapses to the sorted unique set of chain-arm labels.  This flags
    whether a motif spans only heavy chains, only light chains, or both.
    """
    dff = dff.copy()
    n = len(dff)

    # Extract species (idx=0), gene (idx=1), allele(s) (idx=2) from the
    # pipe-separated allele-call string — deduplicate and sort within each field
    for idx, field in enumerate(("gene_species", "gene", "alleles")):
        _tqdm.pandas(
            desc=f"[{stratum_label}] Extracting {field} ({n} rows)",
            leave=False,
        )
        dff[field] = dff["mutation_inn_chain_allele_call"].progress_apply(
            lambda x, i=idx: "|".join(
                sorted({grp.split("_")[i] for grp in x.split("|")})
            )
        )

    # Unique sorted list of region tokens for this row (e.g. ["CH2", "FC"])
    _tqdm.pandas(
        desc=f"[{stratum_label}] Classifying heavy/light regions ({n} rows)",
        leave=False,
    )
    dff["unique_regions"] = dff["region"].progress_apply(
        lambda x: sorted(set(x.split("/")))
    )
    dff["h_l_crossed_regions"] = dff["unique_regions"].apply(
        lambda regions: "/".join(
            sorted(
                {
                    (
                        "heavy"
                        if r in _HEAVY_REGIONS
                        else "light"
                        if r in _LIGHT_REGIONS
                        else "unk"
                    )
                    for r in regions
                }
            )
        )
    )

    # Serialise the list back to a slash-separated string for storage
    dff["unique_regions"] = dff["unique_regions"].apply("/".join)
    return dff


def _group_to_motif_df(dff: pd.DataFrame) -> pd.DataFrame:
    """Group the long annotation DataFrame to one row per motif × allele-call group.

    After exploding allele calls, ``dff`` has one row per (motif, inn, chain,
    allele-call path).  Different chains carrying the same motif under the same
    allele call should be aggregated together.  Grouping by
    (motif_token_set, region, h_l_crossed_regions, unique_regions,
     mutation_inn_chain_allele_call) collapses them while collecting the sets
    of INNs, chain IDs, and gene/allele summary values.

    The five ``by`` columns become the group identity — the row uniquely
    describes a motif as observed under a specific allele-call context.
    """
    return (
        dff.groupby(
            by=[
                "motif_token_set",
                "region",
                "h_l_crossed_regions",
                "unique_regions",
                "mutation_inn_chain_allele_call",
            ],
            dropna=False,
        )
        .agg(
            inn=("inn", lambda x: set(x)),  # set of INN names
            chain_id=("chain_id", lambda x: set(x)),  # set of chain IDs
            gene_species=("gene_species", lambda x: tuple(set(x))),
            gene=("gene", lambda x: tuple(set(x))),
            alleles=("alleles", lambda x: tuple(set(x))),
        )
        .reset_index()
    )


def _add_motif_label_columns(
    result_df: pd.DataFrame,
    full_length_motifs: set[frozenset],
    stratum_label: str = "all",
) -> pd.DataFrame:
    """Add human-readable label and count columns derived from motif_token_set / inn / chain_id.

    ``mutation_string``    — space-delimited tokens sorted alphabetically,
                             e.g. ``"234L>A / 235L>A"``
    ``mutation_string_pe`` — the PE (protein-engineering) label format produced
                             by ``format_motif_label``, spaces stripped,
                             e.g. ``"L234A/L235A"``
    ``motif_length``       — number of distinct mutation tokens in the motif
    ``inn_count``          — number of unique INNs carrying this motif
    ``chain_count``        — number of unique chain IDs carrying this motif
    ``motif_selection_reason`` — for full-length motifs, mark as 'full_length_inn_mutation'

    Parameters
    ----------
    full_length_motifs :
        Set of frozensets (token sets) that were enumerated as complete profiles
        before sub-motif enumeration. These are labeled explicitly so they can
        bypass min_chain_count / min_motif_size filters.
    """
    result_df = result_df.copy()
    n = len(result_df)

    _tqdm.pandas(
        desc=f"[{stratum_label}] Building mutation strings ({n} rows)",
        leave=False,
    )
    result_df["mutation_string"] = result_df["motif_token_set"].progress_map(
        lambda ts: " / ".join(sorted(ts))
    )
    # PE format: convert e.g. "234L>A" tokens to "L234A" notation then join
    _tqdm.pandas(
        desc=f"[{stratum_label}] Formatting PE labels ({n} rows)",
        leave=False,
    )
    result_df["mutation_string_pe"] = result_df["mutation_string"].progress_map(
        lambda ms: format_motif_label(ms).replace(" ", "")
    )
    result_df["motif_length"] = result_df["motif_token_set"].map(len)
    result_df["inn_count"] = result_df["inn"].map(len)
    result_df["chain_count"] = result_df["chain_id"].map(len)

    # Mark full-length motifs early so they're labeled distinctly
    result_df["motif_selection_reason"] = result_df["motif_token_set"].map(
        lambda ts: "full_length_inn_mutation" if ts in full_length_motifs else None
    )

    return result_df


def _annotate_super_motifs_and_removal(
    result_df: pd.DataFrame,
    motif_to_carriers: dict[frozenset, dict],
    super_index: dict[frozenset, list[frozenset]],
    stratum_label: str = "all",
) -> pd.DataFrame:
    """Attach super_motifs, remove_motif, and motif_detail columns.

    For each motif M we look up its super-motifs (motifs one token larger that
    were enumerated) and decide whether M should be flagged for removal:

    *Subsumed*: M is subsumed if at least one super-motif M' is carried by
    exactly the same (inn, chain) set as M.  That means M adds no new carriers
    — every chain that has M also has M', making M redundant; any analysis
    that picks M would equivalently pick M' with more information.

    *Independent*: if every super-motif M' has a strictly smaller carrier set
    (i.e. M explains more chains), then M is kept as an independent motif even
    though larger motifs exist.

    *Maximal*: if M has no super-motifs at all in the enumerated set, it is
    already the most specific motif for its carriers.

    ``motif_detail`` records a human-readable explanation for each decision.
    ``super_motifs`` lists the PE strings of direct super-motifs for reference.

    Note: carrier sets are retrieved from ``motif_to_carriers`` (the raw
    dict-of-dicts) rather than from ``result_df`` columns because the grouped
    DataFrame may have multiple rows per motif (one per allele-call context) and
    the frozenset comparison must be done on the original carrier keys.
    """
    result_df = result_df.copy()

    # Fast lookup: motif token-set → its PE string (needed to name super-motifs)
    token_set_to_pe: dict[frozenset, str] = dict(
        zip(result_df["motif_token_set"], result_df["mutation_string_pe"])
    )

    super_motifs_list: list[list[str]] = []
    remove_flags: list[bool] = []
    details: list[str] = []

    for _, row in _tqdm(
        result_df.iterrows(),
        total=len(result_df),
        desc=f"[{stratum_label}] Annotating super-motifs + removal flags ({len(result_df)} rows)",
        leave=False,
    ):
        motif_ts = row["motif_token_set"]

        # The carrier set for this motif: frozenset of (inn, chain_id) pairs
        carrier_set = frozenset(motif_to_carriers[motif_ts].keys())

        # Direct super-motifs from the pre-built index (may be empty for maximal motifs)
        super_token_sets = super_index.get(motif_ts, [])

        # Resolve PE label for each super-motif that was enumerated
        super_motif_strings = sorted(
            token_set_to_pe[s] for s in super_token_sets if s in token_set_to_pe
        )
        super_motifs_list.append(super_motif_strings)

        if not super_token_sets:
            # No larger motif exists — this motif is maximal for its carrier set
            remove_flags.append(False)
            details.append("maximal — no super-motifs in enumerated set")
            continue

        # Super-motifs that have IDENTICAL carriers → they subsume this motif
        subsuming = [
            token_set_to_pe[s]
            for s in super_token_sets
            if s in token_set_to_pe
            and frozenset(motif_to_carriers[s].keys()) == carrier_set
        ]

        if subsuming:
            # A larger motif covers the exact same chains — flag as redundant
            remove_flags.append(True)
            details.append("subsumed by: " + "; ".join(sorted(subsuming)))
        else:
            # All super-motifs have fewer carriers — this motif is genuinely independent
            remove_flags.append(False)
            details.append(
                f"independent — {len(super_token_sets)} super-motif(s) all have "
                f"fewer chains ({len(carrier_set)} chains not fully explained by any larger motif)"
            )

    result_df["super_motifs"] = super_motifs_list
    result_df["remove_motif"] = pd.array(remove_flags, dtype=bool)
    result_df["motif_detail"] = details
    return result_df


def _apply_filter_flags(
    result_df: pd.DataFrame,
    min_chain_count: int,
    min_motif_size: int,
) -> pd.DataFrame:
    """Flag motifs that fall below the caller-supplied frequency / size thresholds.

    This does **not** drop rows — it only sets ``remove_motif=True`` and
    updates ``motif_detail`` so callers can still inspect filtered motifs.
    If a motif was already flagged as subsumed, the new reason is appended to
    the existing detail string rather than replacing it.

    Two independent filters are applied in order:

    min_chain_count
        Motifs carried by fewer than ``min_chain_count`` distinct (inn, chain)
        pairs are considered too rare to be meaningful.  ``chain_count`` is the
        number of *unique chain IDs* that carry the motif (not INN count).

    min_motif_size
        Motifs with fewer than ``min_motif_size`` tokens are too short to
        represent a meaningful multi-residue pattern.

    After both filters, ``motif_selection_reason`` is set to a short category
    string so downstream callers can group/filter without parsing the free-text
    ``motif_detail``.
    """
    result_df = result_df.copy()

    # ── Full-length motifs: exempt from removal filters ────────────────────────
    # Motifs labeled 'full_length_inn_mutation' are enumerated profiles that
    # bypass size/frequency thresholds by design, so mark them as passed
    full_length_mask = (
        result_df.get("motif_selection_reason", "") == "full_length_inn_mutation"
    )
    result_df.loc[full_length_mask, "remove_motif"] = False
    result_df.loc[full_length_mask, "motif_detail"] = (
        "full-length profile, exempt from thresholds"
    )

    # ── min_chain_count filter ────────────────────────────────────────────────
    # Skip full-length motifs — they're already exempted above
    if min_chain_count > 1:
        rare = result_df["chain_count"] < min_chain_count
        # Exclude full-length motifs from the filter
        rare = rare & ~full_length_mask
        # Only update rows not already flagged (avoid overwriting a subsumed reason)
        new_rare = rare & ~result_df["remove_motif"]
        result_df.loc[new_rare, "remove_motif"] = True
        result_df.loc[new_rare, "motif_detail"] = result_df.loc[
            new_rare, "chain_count"
        ].map(lambda n: f"below min_chain_count={min_chain_count} ({n} chain(s))")
        # For rows already flagged for another reason, append the chain-count note
        result_df.loc[rare & result_df["remove_motif"] & ~new_rare, "motif_detail"] += (
            f"; also below min_chain_count={min_chain_count}"
        )

    # ── min_motif_size filter ─────────────────────────────────────────────────
    # Skip full-length motifs — they're already exempted above
    if min_motif_size > 1:
        short = result_df["motif_length"] < min_motif_size
        # Exclude full-length motifs from the filter
        short = short & ~full_length_mask
        new_short = short & ~result_df["remove_motif"]
        result_df.loc[new_short, "remove_motif"] = True
        result_df.loc[new_short, "motif_detail"] = result_df.loc[
            new_short, "motif_length"
        ].map(lambda n: f"below min_motif_size={min_motif_size} ({n} residue(s))")
        result_df.loc[
            short & result_df["remove_motif"] & ~new_short, "motif_detail"
        ] += f"; also below min_motif_size={min_motif_size}"

    # ── motif_selection_reason: short category derived from motif_detail text ─
    def _selection_reason(remove: bool, detail: str, existing_reason: str) -> str:
        """Map the free-text detail to a short category label.

        Preserve pre-set reasons like 'full_length_inn_mutation' — don't override them.
        """
        # If already labeled (e.g. as 'full_length_inn_mutation'), preserve it
        if (
            existing_reason
            and existing_reason != "None"
            and not pd.isna(existing_reason)
        ):
            return existing_reason

        # Otherwise derive from detail + remove flag
        if not remove:
            return "passed_original_filters"
        if "subsumed" in detail:
            return "subsumed"
        if "min_motif_size" in detail:
            return "motif_length_below_cutoff"
        if "min_chain_count" in detail:
            return "motif_count_below_cutoff"
        return "removed"

    result_df["motif_selection_reason"] = [
        _selection_reason(r, d, existing)
        for r, d, existing in zip(
            result_df["remove_motif"],
            result_df["motif_detail"],
            result_df.get("motif_selection_reason", [None] * len(result_df)),
        )
    ]
    return result_df


def _finalise_export(result_df: pd.DataFrame) -> pd.DataFrame:
    """Select canonical output columns, sort by motif size then frequency, reset index.

    Only columns that actually exist in ``result_df`` are included, so the
    function is tolerant of optional columns added by upstream steps.

    Sort order: longest motifs first (most specific), then by INN count
    descending (most common first) — giving a natural "most interesting motifs
    at the top" ordering for downstream inspection.
    """
    # export_cols = [
    #     "mutation_string",      # human-readable "A234L / L235A" form
    #     "mutation_string_pe",   # PE-notation "A234L/L235A" form
    #     "motif_length",         # number of mutation tokens
    #     "inn_count",            # distinct INNs carrying this motif
    #     "chain_count",          # distinct (inn, chain_id) pairs carrying this motif
    #     "inn",                  # set of INN names
    #     "chain_id",             # set of chain IDs
    #     "remove_motif",         # True if flagged for removal
    #     "motif_selection_reason",  # short category label
    #     "motif_detail",         # verbose explanation
    #     "super_motifs",         # PE strings of direct super-motifs
    #     "motif_token_set",      # frozenset of token strings
    # ]
    # available_cols = [c for c in export_cols if c in result_df.columns]
    out = result_df.sort_values(
        ["inn_count", "motif_length"], ascending=[False, True]
    ).reset_index(drop=True)
    return out


def _enumerate_single_stratum(
    chain_df: pd.DataFrame,
    individual_mutation_profile: pd.DataFrame,
    token_ignore_set: frozenset,
    min_chain_count: int,
    min_motif_size: int,
    max_motif_size: int | None,
    max_consecutive_deletions: int | None = 5,
    stratum_label: str = "all",
    coalesce_gene_allelic_keys_to_str: str = "|",
) -> pd.DataFrame:
    """Enumerate all mutation motifs for a single stratum of chains.

    This is the top-level orchestrator for motif discovery.  It calls the
    focused private helpers in order and returns a tidy output DataFrame.

    Pipeline
    --------
    1. Build gene info map  — query the individual mutation profile to get a
       (inn, chain_id, mutation_pe) → [(region, gene, allele, species)] lookup.
    2. Validate inputs      — guard against duplicate (inn, chain_id) rows.
    3. Build token index    — parse EU strings, filter ignored tokens, build
                              inverted index token → carrier chains.
    4. Early exit           — if no chain has any surviving token, return the
                              schema-correct empty DataFrame immediately.
    5. Enumerate motifs     — all token sub-combinations up to max_motif_size,
                              using the inverted index to find carriers per motif.
    6. Super-motif index    — map each motif to its direct larger-motif parents.
    7. Annotation           — flatten carriers → long per-(motif, inn, chain) table.
    8. Coalesce allele call — pick best allele-info source per row; explode lists.
    9. Add region columns   — heavy/light classification, unique_regions.
    10. Group               — collapse long table to one row per motif × allele group.
    11. Label columns       — mutation_string, mutation_string_pe, counts.
    12. Super-motif flags   — attach super_motifs list and remove_motif / detail.
    13. Filter flags        — flag motifs below min_chain_count / min_motif_size.
    14. Finalise export     — select canonical columns, sort, reset index.
    """
    # Step 1: build (inn, chain_id, mutation_pe) → gene-info lookup
    _, gene_info_map = _create_inn_map_of_mutation_gene_info(
        individual_mutation_profile.reset_index(), ["gene", "gene_species"]
    )

    # Step 2: guard — duplicate chain keys would corrupt the inverted index
    _validate_no_duplicate_chain_keys(chain_df)

    # Step 3: parse EU mutation strings → token sets + inverted index
    chain_key_series, token_set_series, token_to_chain_key_set = (
        _build_token_sets_and_inverted_index(
            chain_df.copy(),
            token_ignore_set,
            max_consecutive_deletions=max_consecutive_deletions,
        )
    )

    # Loud visibility for chains whose token fanout still exceeds max_motif_size.
    if max_motif_size is not None:
        oversize_mask = token_set_series.map(len) > int(max_motif_size)
        if oversize_mask.any():
            for (inn, chain_id), token_set in zip(
                chain_key_series.loc[oversize_mask],
                token_set_series.loc[oversize_mask],
            ):
                _log.error(
                    "Motif fanout warning: inn=%s chain_id=%s token_count=%d exceeds "
                    "max_motif_size=%d after filtering/capping; enumeration will still "
                    "consider subsets only up to max_motif_size",
                    inn,
                    chain_id,
                    len(token_set),
                    int(max_motif_size),
                )

    # Step 4: nothing to enumerate if all chains are germline after filtering
    if token_set_series.empty:
        return pd.DataFrame(columns=_EMPTY_STRATUM_COLUMNS)

    # Step 5: enumerate all motifs — full-length profiles first, then sub-motifs
    motif_to_carriers, full_length_motifs = _enumerate_motif_carriers(
        token_set_series,
        token_to_chain_key_set,
        gene_info_map,
        min_chain_count,
        max_motif_size,
        stratum_label,
    )

    # Step 6: map each motif to its direct super-motifs (one token larger)
    super_index = _build_super_motif_index(motif_to_carriers, stratum_label)

    # Steps 7–10: annotation pipeline → grouped one-row-per-motif DataFrame
    annotation_df = _build_per_motif_annotation_df(motif_to_carriers, stratum_label)
    annotation_df = _coalesce_allele_call(annotation_df)
    annotation_df = _add_gene_and_region_columns(annotation_df, stratum_label)
    result_df = _group_to_motif_df(annotation_df)

    # Steps 11–12: labelling and super-motif / removal flags
    result_df = _add_motif_label_columns(result_df, full_length_motifs, stratum_label)
    result_df = _annotate_super_motifs_and_removal(
        result_df, motif_to_carriers, super_index, stratum_label
    )
    result_df = _apply_filter_flags(result_df, min_chain_count, min_motif_size)

    if coalesce_gene_allelic_keys_to_str is not None:
        result_df[["gene", "gene_species", "alleles"]] = result_df[
            ["gene", "gene_species", "alleles"]
        ].map(
            lambda x: (
                coalesce_gene_allelic_keys_to_str.join(x)
                if not isinstance(x, str)
                else x
            )
        )

    return _finalise_export(result_df)


def _collapse_motif_df_results(
    motif_df_result: pd.DataFrame, group_by: list[str]
) -> pd.DataFrame:
    tqdm.pandas(desc="aggregating motifs")

    def flatten(x):
        if isinstance(x, str):
            yield x
        else:
            try:
                for item in x:
                    yield from flatten(item)
            except TypeError:
                yield x

    inn_id_key_in_df = list(motif_df_result.columns.intersection(["inn", "inns"]))
    if len(inn_id_key_in_df) != 1:
        raise AssertionError(
            f"Expected exactly one of 'inn' or 'unique_inns' in columns, found: {inn_id_key_in_df}"
        )

    allele_key = list(motif_df_result.columns.intersection({"allele", "alleles"}))
    if len(allele_key) != 1:
        raise AssertionError(
            f"Expected exactly one of 'allele' or 'alleles' in columns, found: {allele_key}"
        )
    allele_key = allele_key[0]  # unwrap from list to scalar string
    gene_key = "gene"
    species_key = "gene_species"

    chain_id_key_in_df = list(
        motif_df_result.columns.intersection(["chain_id", "unique_inn_chains"])
    )
    if len(chain_id_key_in_df) != 1:
        raise AssertionError(
            f"Expected exactly one of 'chain_id' or 'unique_inn_chains' in columns, found: {chain_id_key_in_df}"
        )

    inn_id_key = inn_id_key_in_df[0]
    chain_id_key = chain_id_key_in_df[0]

    _FIRST = [
        "mutation_string",
        "motif_token_set",
        "h_l_crossed_regions",
        "unique_regions",
        "mutation_string_pe",
        "super_motifs",
        "remove_motif",
        "motif_detail",
        "motif_selection_reason",
        "motif_length",
    ]
    _SORTED_SET = [
        gene_key,
        species_key,
        allele_key,
        "mutation_inn_chain_allele_call",
    ]

    agg_spec = {col: "first" for col in _FIRST}
    agg_spec.update(
        {
            col: lambda x: sorted(
                set(flatten([y if isinstance(y, str) else "|".join(y) for y in x]))
            )
            for col in _SORTED_SET
        }
    )
    agg_spec[inn_id_key] = lambda x: set(flatten(list(x)))
    agg_spec[chain_id_key] = lambda x: set(flatten(list(x)))

    group_by = ["mutation_string_pe"] + [
        grp for grp in group_by if grp != "mutation_string_pe"
    ]

    agg_spec_filtered = {k: v for k, v in agg_spec.items() if k not in group_by}

    agg_result = motif_df_result.groupby(group_by, sort=False, dropna=False).agg(
        agg_spec_filtered
    )

    agg_result = agg_result.rename(
        columns={
            inn_id_key: "inns",
            chain_id_key: "unique_inn_chains",
            allele_key: "allele",
            gene_key: "gene",
            species_key: "gene_species",
        }
    )
    agg_result["inn_count"] = agg_result["inns"].apply(len)
    agg_result["chain_count"] = agg_result["unique_inn_chains"].apply(len)

    return agg_result


# ---------------------------------------------------------------------------
# Step functions (moved from chain.py)
# ----------


def step_filter_remove_motif(
    step_in: ChainMotifStep,
) -> tuple[ChainMotifStepResult, pd.DataFrame]:
    """Keep only rows where ``remove_motif=False``.

    Input
    -----
    motif_df : full enumerated + cross-chain table.

    Output
    ------
    motif_df : rows with ``remove_motif=True`` dropped.
    rows_dropped : count of removed rows.
    """

    before = step_in.motif_df
    after = before[~before["remove_motif"]].copy()
    removed_rows = before[before["remove_motif"]].copy()
    return (
        ChainMotifStepResult(
            motif_df=after,
            step_name="filter_remove_motif",
            rows_added=0,
            rows_dropped=len(before) - len(after),
            detail=f"Dropped {len(before) - len(after)} remove_motif=True rows",
        ),
        removed_rows,
    )


def step_restore_other_gene_hits(
    step_in: _RestoreOtherGeneHitsIn,
) -> _RestoreOtherGeneHitsOut:
    """Re-insert motifs from original table sharing same pos→mut signature across gene groups.

    Matches by ``motif_signature_without_germline`` (ignores germline residue),
    restoring rows present in the pre-filter enumeration but absent from the
    current table.

    Input
    -----
    motif_df : filtered motif table (post ``step_filter_remove_motif``).
    original_motif_df : unfiltered enumerated table.
    min_count : minimum ``inn_count`` required for a row to be restored.

    Output
    ------
    motif_df : table with any recovered gene-hit rows appended and flagged
               ``motif_retention_reason="other_gene_hit"``.
    recovered_rows : number of rows restored.
    """

    current_df = step_in.motif_df.copy().reset_index(drop=True)
    original_df = step_in.original_motif_df.copy().reset_index(drop=True)

    if (
        current_df.empty
        or original_df.empty
        or "mutation_string_pe" not in current_df.columns
    ):
        return _RestoreOtherGeneHitsOut(
            motif_df=current_df,
            step_name="restore_other_gene_hits",
            rows_added=0,
            rows_dropped=0,
            recovered_rows=0,
        )

    for _df in (current_df, original_df):
        _df["_sig_posmut"] = _df["mutation_string_pe"].map(
            lambda pe: frozenset(
                tok.split("->", 1)[1].strip() if "->" in tok else tok
                for tok in pe.split("/")
                if tok.strip()
            )
        )
        key_cols = [
            c
            for c in (
                "mutation_string_pe",
                "chain_type",
                "constant_chain_species",
            )
            if c in _df.columns
        ]
        _df["_row_key"] = _df[key_cols].astype(str).agg("|".join, axis=1)

    current_sigs = set(current_df["_sig_posmut"].map(frozenset))
    current_keys = set(current_df["_row_key"])

    candidate = original_df[
        original_df["_sig_posmut"].map(frozenset).isin(current_sigs)
    ].copy()
    missing = candidate[~candidate["_row_key"].isin(current_keys)].copy()
    if "inn_count" in missing.columns:
        missing = missing[missing["inn_count"] >= step_in.min_count].copy()

    if missing.empty:
        current_df.drop(
            columns=["_sig_posmut", "_row_key"],
            inplace=True,
            errors="ignore",
        )
        return _RestoreOtherGeneHitsOut(
            motif_df=current_df.reset_index(drop=True),
            step_name="restore_other_gene_hits",
            rows_added=0,
            rows_dropped=0,
            recovered_rows=0,
        )

    missing["motif_retention_reason"] = "other_gene_hit"
    missing["motif_selection_reason"] = None
    if "motif_retention_reason" not in current_df.columns:
        current_df["motif_retention_reason"] = None

    union_cols = list(dict.fromkeys(list(current_df.columns) + list(missing.columns)))
    current_df = current_df.reindex(columns=union_cols)
    missing = missing.reindex(columns=union_cols)
    out_df = pd.concat([current_df, missing], ignore_index=True)
    out_df = out_df.drop_duplicates(subset=["_row_key"], keep="first").reset_index(
        drop=True
    )
    out_df.drop(columns=["_sig_posmut", "_row_key"], inplace=True, errors="ignore")

    return _RestoreOtherGeneHitsOut(
        motif_df=out_df,
        step_name="restore_other_gene_hits",
        rows_added=len(missing),
        rows_dropped=0,
        recovered_rows=len(missing),
    )
