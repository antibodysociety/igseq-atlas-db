"""Chain-level motif pipeline: cross-chain integration and statistical enrichment.

This module provides the orchestration layer for the chain-level motif pipeline.
The enumeration layer lives in :mod:`analytics.motif_pipeline.enumerate`.

Functions
---------
resolve_super_motifs
    Walk the raw super_motif DAG to non-consumed ancestors (final resolution step).
expand_motifs_by_cross_chain_pairing
    Derive bispecific arm/anchor rows from multi-chain INN pairwise comparison.
update_chain_motif_df_with_cross_chain
    Merge cross-chain rows into the enumerated motif table.
build_chain_motif_index
    End-to-end chain-level motif pipeline returning a :class:`ChainMotifIndexResult`.

Step Functions (typed, auditable)
----------------------------------
step_filter_remove_motif
    Keep only rows where remove_motif=False (imported from enumerate).
step_restore_other_gene_hits
    Re-insert motifs from original table (imported from enumerate).
step_add_statistics
    Möbius conditional stats, independence baseline, and pairwise co-mutation metrics
    (imported from clustering).

step_cluster_motifs
    Re-exported from clustering.

"""

from __future__ import annotations

import logging
import time
from collections import defaultdict
from copy import deepcopy
from dataclasses import fields
from typing import TYPE_CHECKING, Literal, Optional

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field
from scipy.stats import binomtest as _binom
from tqdm.auto import tqdm
from tqdm.auto import tqdm as _tqdm

from inn_analysis_ops.analytics._helpers import (
    motif_signature_without_germline,
    motif_tokens_set,
    tokens_to_mutation_string,
)
from inn_analysis_ops.analytics.co_mutation_analytics import build_marginals_table
from inn_analysis_ops.analytics.motif_schema import MotifStatsCols as C
from inn_analysis_ops.analytics.motifs.clustering import (
    step_add_statistics,  # noqa: E402
    step_build_cluster_info,
    step_filter_by_statistics,
)
from inn_analysis_ops.analytics.motifs.enumerate import (  # noqa: E402
    ChainEnumerationRowAnnotated,
    _collapse_motif_df_results,
    _collapse_to_shortest,
    _extract_multi_chain_overlaps,
    _find_resolved_parent,
    _gen_inn_chain_mut_string_map,
    chain_collection_to_enumeration_rows,
    enumerate_chain_motifs,
    step_filter_remove_motif,
    step_restore_other_gene_hits,
)
from inn_analysis_ops.analytics.motifs.gene_assignment import (
    FIELDS_FOR_NOMENCLATURE,
    _create_gene_map_for_inn_chains_by_fc_only,
    _create_inn_map_of_mutation_gene_info,
    _get_genes_associated_with_motif,
)
from inn_analysis_ops.analytics.motifs.models import (
    ChainIterativeClusters,
    ChainMotifIndexResult,
    ChainMotifStep,
    ChainMotifStepResult,
    _RestoreOtherGeneHitsIn,
)
from inn_analysis_ops.analytics.motifs.stat_functions import generate_submotifs

if TYPE_CHECKING:
    from inn_analysis_ops.data_io.merge.loader import INNAnalysisDataset

_log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Typed step IO models
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Helpers: step delta computation
# ---------------------------------------------------------------------------


def _step_row_key(df: pd.DataFrame) -> pd.Series:
    key_cols = [
        c
        for c in (
            "mutation_string_pe",
            "chain_type",
            "constant_chain_species",
        )
        if c in df.columns
    ]
    return df[key_cols].astype(str).agg("|".join, axis=1)


def _compute_step_delta(
    before_df: pd.DataFrame,
    after_df: pd.DataFrame,
    add_reason: str = "",
    remove_reason: str = "",
) -> dict[str, pd.DataFrame]:
    """Compute rows added/removed in a step, storing only essential columns.

    Matches classical pipeline behavior: stores mutation_string_pe and minimal
    identifying columns, plus a reason column.
    """
    before_keys = set(before_df["mutation_string_pe"])
    after_keys = set(after_df["mutation_string_pe"])

    new_keys = after_keys - before_keys
    removed_keys = before_keys - after_keys

    # Select only essential columns (like classical pipeline)
    essential_cols = [
        c
        for c in (
            "mutation_string_pe",
            "mutation_string",
            "motif_length",
            "inn_count",
            "chain_count",
            "inns",
            "chain_type",
            "constant_chain_species",
            "gene",
            "species",
            "isotype",
        )
        if c in before_df.columns or c in after_df.columns
    ]

    added = after_df.query("mutation_string_pe.isin(@new_keys)")[essential_cols].copy()
    removed = before_df.query("mutation_string_pe.isin(@removed_keys)")[
        essential_cols
    ].copy()

    if not added.empty and add_reason:
        added["reason"] = add_reason
    if not removed.empty and remove_reason:
        removed["reason"] = remove_reason

    return {"added": added, "removed": removed}


# ---------------------------------------------------------------------------
# Orchestration helpers (moved from enumerate.py)
# ---------------------------------------------------------------------------


def resolve_super_motifs(chain_motif_df: pd.DataFrame) -> pd.DataFrame:
    """Resolve ``super_motifs`` to non-consumed ancestors.

    Walks the raw direct-parent lists in ``super_motifs`` up the DAG until
    reaching a non-consumed (``remove_motif=False``) ancestor, then collapses
    to the shortest unique set.  Call once after all motif table updates are
    complete (post cross-chain, post any flag changes).

    Parameters
    ----------
    chain_motif_df :
        Output of :func:`~analytics.motif_pipeline.enumerate.enumerate_chain_motifs`
        or :func:`update_chain_motif_df_with_cross_chain`.

    Returns
    -------
    Copy of *chain_motif_df* with ``super_motifs`` resolved.
    """
    df = chain_motif_df.copy().reset_index(drop=True)
    n_rows = len(df)

    pe_token_map: dict[str, set[str]] = {
        pe: set(t.strip() for t in pe.split("/") if t.strip())
        for pe in df["mutation_string_pe"]
    }
    parents_map: dict[str, list[str]] = (
        df[["mutation_string_pe", "super_motifs"]]
        .set_index("mutation_string_pe")
        .to_dict()["super_motifs"]
    )
    not_consumed: set[str] = set(df.loc[~df["remove_motif"], "mutation_string_pe"])
    resolved_cache: dict[str, list[str]] = {}

    resolved_super_motifs: list[list[str]] = []
    for _, row in _tqdm(
        df.iterrows(),
        total=n_rows,
        desc=f"Resolving super-motifs ({n_rows} motifs)",
        leave=False,
    ):
        candidates: list[str] = [
            new_p
            for p in row["super_motifs"]
            for new_p in _find_resolved_parent(
                p, resolved_cache, not_consumed, parents_map
            )
        ]
        resolved_super_motifs.append(_collapse_to_shortest(candidates, pe_token_map))

    df["super_motifs"] = pd.Series(resolved_super_motifs, index=df.index)
    return df


def expand_motifs_by_cross_chain_pairing(
    chain_motif_df: pd.DataFrame,
    multi_chain_inns: dict[str, dict[str, list[str]]],
    same_chain_type_groups: dict[str, list[tuple[str, ...]]],
    min_motif_size: int = 1,
    max_motif_size: int | None = None,
    cdb: "ConstantDB | None" = None,
    cn: "ConstantChainNumbering | None" = None,
    ind_mut_allele_map: "dict | None" = None,
    chain_id_region_allele_map: "object | None" = None,
    inn_chain_domain_map: "dict | None" = None,
) -> tuple[pd.DataFrame, list[tuple]]:
    """Derive cross-chain bispecific arm and anchor motifs.

    For each multi-chain INN, takes the longest motif per chain and decomposes
    pairwise differences into arm-only (``bispecific_ind_branch``) and
    shared-anchor (``bispecific_anchor``) token sets.

    When ``cdb``, ``cn``, ``ind_mut_allele_map``, and
    ``chain_id_region_allele_map`` are all provided, a germline-reversion
    check is applied to every arm-exclusive token — see
    :func:`~analytics.motifs.enumerate._is_germline_reversion`.  Tokens
    rescued by the check are recorded in ``rescued_germline_pair`` on the arm
    rows and also emitted as ``ind_chain_full_motif`` rows
    (``rescued ∪ anchor``).

    Parameters
    ----------
    chain_motif_df :
        Output of :func:`~analytics.motif_pipeline.enumerate.enumerate_chain_motifs`.
        Must have ``unique_inn_chains`` and ``c_allele`` columns.
    multi_chain_inns :
        ``{inn: {chain_type: [chain_ids]}}`` from :func:`~analytics._helpers.find_multi_chain_inns`.
    same_chain_type_groups :
        ``{inn: [tuple_of_chain_ids]}`` derived from
        ``ds.constant_chain_motifs_by_inn_region``.
    cdb :
        Optional :class:`~models.inn_structure.constant_validator.ConstantDB`
        for germline-reversion check.  ``None`` disables the check.
    cn :
        Optional :class:`~models.numbering.core.ConstantChainNumbering`
        for germline-reversion check.  ``None`` disables the check.
    ind_mut_allele_map :
        Output of :func:`~analytics.motifs.allele_mapping._infer_chain_id_mut_to_allele_map`.
        Required (together with ``cdb``/``cn``) to enable the check.
    chain_id_region_allele_map :
        Output of :func:`~analytics.motifs.allele_mapping._infer_chain_id_region_to_allele_map`.
        Required (together with ``cdb``/``cn``) to enable the check.
    inn_chain_domain_map :
        Output of :func:`~analytics.motifs.allele_mapping._infer_inn_chain_domain_map`.
        Required (together with ``cdb``/``cn``) to enable the check —
        restricts germline-reversion comparisons to chains, within the
        same INN, sharing the token's sub_domain category (V-to-V or
        CK/CL-to-CK/CL only).

    Returns
    -------
    cross_chain_df : pd.DataFrame
        One row per unique cross-chain motif with columns:
        ``mutation_string_pe``, ``tokens``, ``type``, ``inn``, ``chain``,
        ``source="cross_chain_pairing"``, and optionally
        ``rescued_germline_pair``.
    skipped : list[tuple]
        ``(inn, chain_type, chains)`` entries skipped because the chain
        combination was not found in *same_chain_type_groups*.
    """
    pe_token_map: dict[str, set[str]] = {
        pe: set(t.strip() for t in pe.split("/") if t.strip())
        for pe in chain_motif_df["mutation_string_pe"]
    }
    inn_chain_to_motifs = _gen_inn_chain_mut_string_map(chain_motif_df)

    new_motif_info, skipped = _extract_multi_chain_overlaps(
        multi_chain_inns=multi_chain_inns,
        same_chain_type_groups=same_chain_type_groups,
        inn_chain_to_motifs=inn_chain_to_motifs,
        pe_token_map=pe_token_map,
        ind_mut_allele_map=ind_mut_allele_map,
        chain_id_region_allele_map=chain_id_region_allele_map,
        inn_chain_domain_map=inn_chain_domain_map,
        cdb=cdb,
        cn=cn,
    )

    min_size = min_motif_size or 1
    max_size = max_motif_size or float("inf")

    new_motif_info = [
        item
        for item in new_motif_info
        if min_size <= len(item["mutation_string_pe"].split("/")) <= max_size
    ]

    if not new_motif_info:
        return (
            pd.DataFrame(
                columns=[
                    "mutation_string_pe",
                    "tokens",
                    "type",
                    "inn",
                    "chain",
                    "source",
                ]
            ),
            skipped,
        )

    cross_chain_df = pd.DataFrame(new_motif_info)
    cross_chain_df["source"] = "cross_chain_pairing"

    # If a mutation_string_pe is classified as "bispecific_anchor" under any
    # row, force every row for that same mutation_string_pe to that type too
    # — an anchor (literal shared-token match) takes precedence over any
    # other classification (e.g. bispecific_ind_branch) the same string may
    # have picked up from a different chain pairing.
    pe_has_anchor = cross_chain_df.groupby("mutation_string_pe")["type"].transform(
        lambda types: (types == "bispecific_anchor").any()
    )
    cross_chain_df.loc[
        pe_has_anchor & (cross_chain_df["type"] != "bispecific_anchor"), "type"
    ] = "bispecific_anchor"

    type_counts_per_pe = cross_chain_df.groupby("mutation_string_pe")["type"].nunique()
    if (type_counts_per_pe > 1).any():
        conflicting = sorted(type_counts_per_pe[type_counts_per_pe > 1].index)
        raise AssertionError(
            "cross_chain_df contains mutation_string_pe value(s) classified under "
            "more than one 'type': " + ", ".join(conflicting)
        )

    cross_chain_df = cross_chain_df.drop_duplicates(
        subset=["inn", "chain", "mutation_string_pe"]
    ).reset_index(drop=True)

    return cross_chain_df, skipped


def update_chain_motif_df_with_cross_chain(
    chain_motif_df: pd.DataFrame,
    cross_chain_df: pd.DataFrame,
) -> pd.DataFrame:
    """Merge cross-chain motif rows into *chain_motif_df*.

    - **Existing motifs** in *cross_chain_df* have ``remove_motif`` reset to
      ``False``, ``motif_selection_reason`` cleared, and ``motif_detail`` +
      ``source`` updated.
    - **New motifs** are appended as full rows, inheriting group-by columns
      from a representative row of the same INN.
    - Deduplicates on ``(mutation_string_pe, remove_motif, source,
      chain_type, constant_chain_species)`` — using whichever columns exist.

    Parameters
    ----------
    chain_motif_df :
        Output of :func:`~analytics.motif_pipeline.enumerate.enumerate_chain_motifs`.
    cross_chain_df :
        Output of :func:`expand_motifs_by_cross_chain_pairing`.

    Returns
    -------
    pd.DataFrame
        Sorted and deduplicated motif table with cross-chain rows integrated.
    """
    if cross_chain_df.empty:
        return chain_motif_df.copy()

    pe_to_type: dict[str, str] = {}
    pe_to_carriers: dict[str, dict[str, set[str]]] = defaultdict(
        lambda: defaultdict(set)
    )
    for _, record in cross_chain_df.iterrows():
        pe = record["mutation_string_pe"]
        pe_to_type[pe] = record["type"]
        for chain_id in record["chain"].split(","):
            pe_to_carriers[pe][record["inn"]].add(chain_id.strip())
    chain_motif_df = chain_motif_df.set_index("mutation_string_pe")
    cross_chain_df = cross_chain_df.reset_index().set_index("mutation_string_pe")

    if cross_chain_df.index.difference(chain_motif_df.index).any():
        raise AssertionError(
            "Cross-chain dataframe contains mutation strings not present in the chain motif dataframe: "
            + ", ".join(cross_chain_df.index.difference(chain_motif_df.index))
        )

    # Bring cross-chain type in via index-aligned join — no positional misalignment
    chain_motif_df = chain_motif_df.join(
        cross_chain_df[["type"]].rename(columns={"type": "_cross_type"}),
        how="left",
    )

    # Only update rows that were filtered out AND now appear in cross-chain.
    # Compute both masks from the ORIGINAL remove_motif values up front —
    # otherwise flipping remove_motif for update_mask rows would make them
    # also match already_retained_mask below, double-applying the reason.
    _orig_remove_motif = chain_motif_df["remove_motif"].copy()
    update_mask = _orig_remove_motif & chain_motif_df["_cross_type"].notna()
    already_retained_mask = ~_orig_remove_motif & chain_motif_df["_cross_type"].notna()

    chain_motif_df.loc[update_mask, "remove_motif"] = False
    chain_motif_df.loc[update_mask, "type"] = chain_motif_df.loc[
        update_mask, "_cross_type"
    ]
    chain_motif_df.loc[update_mask, "motif_selection_reason"] = chain_motif_df.loc[
        update_mask, "_cross_type"
    ].map(
        {
            "bispecific_ind_branch": "bispecific independent arm",
            "bispecific_ind_branch_not_fc": "bispecific independent arm (no Fc pair)",
            "bispecific_anchor": "bispecific shared anchor",
            "ind_chain_full_motif": "bispecific chain full motif",
        }
    )

    chain_motif_df.loc[update_mask, "motif_detail"] = chain_motif_df.loc[
        update_mask, "_cross_type"
    ].map(
        {
            "bispecific_ind_branch": "cross_chain_integration - bispecific independent arm",
            "bispecific_ind_branch_not_fc": (
                "cross_chain_integration - bispecific independent arm "
                "(paired chain has no Fc region to compare against)"
            ),
            "bispecific_anchor": "cross_chain_integration - bispecific shared anchor",
            "ind_chain_full_motif": "cross_chain_integration - bispecific chain full motif",
        }
    )

    # Rows that were ALREADY retained (remove_motif == False) for some other
    # reason (e.g. recurring across INNs) but are *also* flagged in
    # cross_chain_df — motif_selection_reason is overwritten with the
    # bispecific reason (same as the rescued-row case above), but
    # motif_detail is appended (";"-joined) rather than overwritten, so the
    # original detail explaining why the motif was retained isn't lost.
    _cross_reason_map = {
        "bispecific_ind_branch": "bispecific independent arm",
        "bispecific_ind_branch_not_fc": "bispecific independent arm (no Fc pair)",
        "bispecific_anchor": "bispecific shared anchor",
        "ind_chain_full_motif": "bispecific chain full motif",
    }
    _cross_detail_map = {
        "bispecific_ind_branch": "cross_chain_integration - bispecific independent arm",
        "bispecific_ind_branch_not_fc": (
            "cross_chain_integration - bispecific independent arm "
            "(paired chain has no Fc region to compare against)"
        ),
        "bispecific_anchor": "cross_chain_integration - bispecific shared anchor",
        "ind_chain_full_motif": "cross_chain_integration - bispecific chain full motif",
    }
    chain_motif_df.loc[already_retained_mask, "motif_selection_reason"] = (
        chain_motif_df.loc[already_retained_mask, "_cross_type"].map(_cross_reason_map)
    )
    _append_detail = chain_motif_df.loc[already_retained_mask, "_cross_type"].map(
        _cross_detail_map
    )
    chain_motif_df.loc[already_retained_mask, "motif_detail"] = (
        chain_motif_df.loc[already_retained_mask, "motif_detail"].fillna("")
        + chain_motif_df.loc[already_retained_mask, "motif_detail"]
        .notna()
        .map({True: "; ", False: ""})
        + _append_detail
    )

    chain_motif_df = chain_motif_df.drop(columns=["_cross_type"]).reset_index()

    # pe_token_map: dict[str, set[str]] = {
    #     pe: set(t.strip() for t in pe.split("/") if t.strip())
    #     for pe in df["mutation_string_pe"]
    # }
    # token_frozenset_to_pe: dict[frozenset, str] = {
    #     frozenset(tokens): pe for pe, tokens in pe_token_map.items()
    # }
    # all_tokens_in_universe: set[str] = set().union(*pe_token_map.values())

    # def _direct_super_motifs(pe: str) -> list[str]:
    #     child_fset = frozenset(pe_token_map[pe])
    #     return sorted(
    #         token_frozenset_to_pe[child_fset | {tok}]
    #         for tok in all_tokens_in_universe - child_fset
    #         if (child_fset | {tok}) in token_frozenset_to_pe
    #     )

    # existing_pe_set: set[str] = set(chain_motif_df["mutation_string_pe"])
    # inn_to_representative_row: dict[str, dict] = {}
    # for _, row in chain_motif_df.iterrows():
    #     for inn in row.get("unique_inns") or []:
    #         if inn not in inn_to_representative_row:
    #             inn_to_representative_row[inn] = row.to_dict()

    # new_rows: list[dict] = []
    # _log.info(f"  Processing {len(cross_chain_df)} cross-chain records...")
    # for _, record in _tqdm(
    #     cross_chain_df.iterrows(),
    #     total=len(cross_chain_df),
    #     desc="  Processing cross-chain records",
    #     leave=False,
    # ):
    #     pe = record["mutation_string_pe"]
    #     if pe in existing_pe_set:
    #         continue
    #     motif_type = record["type"]
    #     carriers = pe_to_carriers[pe]
    #     unique_inn_chains = {
    #         inn: sorted(chains) for inn, chains in sorted(carriers.items())
    #     }
    #     unique_inns = frozenset(unique_inn_chains.keys())
    #     tokens = set(t.strip() for t in pe.split("/") if t.strip())
    #     pe_token_map[pe] = tokens
    #     token_frozenset_to_pe[frozenset(tokens)] = pe
    #     all_tokens_in_universe |= tokens

    #     base = inn_to_representative_row.get(next(iter(unique_inns)), {})
    #     row_dict = {col: base.get(col) for col in chain_motif_df.columns}
    #     row_dict.update(
    #         {
    #             "mutation_string": " / ".join(sorted(tokens)),
    #             "mutation_string_pe": pe,
    #             "motif_length": len(tokens),
    #             "inn_count": len(unique_inns),
    #             "chain_count": sum(
    #                 len(v) for v in unique_inn_chains.values()
    #             ),
    #             "unique_inns": unique_inns,
    #             "unique_inn_chains": unique_inn_chains,
    #             "unique_inn_chain_counts": {
    #                 inn: len(c) for inn, c in unique_inn_chains.items()
    #             },
    #             "remove_motif": False,
    #             "motif_selection_reason": None,
    #             "motif_detail": (
    #                 "bispecific independent arm"
    #                 if motif_type == "bispecific_ind_branch"
    #                 else "bispecific shared anchor"
    #             ),
    #             "super_motifs": [],
    #             "motif_token_set": frozenset(tokens),
    #             "source": record["source"],
    #         }
    #     )
    #     new_rows.append(row_dict)

    # if new_rows:
    #     new_df = pd.DataFrame(new_rows, columns=df.columns)
    #     new_df["super_motifs"] = new_df["mutation_string_pe"].map(
    #         _direct_super_motifs
    #     )
    #     df = pd.concat([df, new_df], ignore_index=True)

    return chain_motif_df.sort_values(
        ["motif_length", "inn_count"], ascending=[False, False]
    ).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Typed pipeline step functions (imported from other modules)
# ---------------------------------------------------------------------------


def step_add_effector_annotation(
    step_in: ChainMotifStep,
    effector_df: Optional[pd.DataFrame] = None,
    join_cols: Optional[list[str]] = None,
    annotation_cols: list[str] = None,
) -> ChainMotifStepResult:
    """Add Fc effector group and sub_classification annotations via left-join.

    Parameters
    ----------
    step_in :
        ChainMotifStep with motif_df to annotate.
    effector_df :
        Effector annotation DataFrame (e.g., from
        :func:`models.fc_effector_annotation.load_fc_effector_annotations`).
        If ``None``, returns motif_df unchanged.
    join_cols :
        Columns to join on (e.g., ``["mutation_string_pe"]``).
        If ``None``, defaults to ``["mutation_string_pe"]``.
    annotation_cols :
        Columns to keep from effector_df. Required — must be a non-empty list.

    Returns
    -------
    ChainMotifStepResult
        motif_df with effector columns added (``group``, ``sub_classification``, etc.).
    """
    from inn_analysis_ops.analytics.constant_chain_analysis import (
        _left_join_effector,
        classify_motif_annotation_by_sub_motifs,
    )

    df = step_in.motif_df.copy().reset_index(drop=True)

    if annotation_cols is None or len(annotation_cols) == 0:
        raise ValueError(
            "annotation_cols is required and must be a non-empty list of column names to keep from effector_df."
        )

    if effector_df is None or effector_df.empty:
        _log.warning("No effector_df provided — annotation skipped")
        return ChainMotifStepResult(
            motif_df=df,
            step_name="add_effector_annotation",
            rows_added=0,
            rows_dropped=0,
            detail="No effector_df provided — annotation skipped",
        )

    if join_cols is None:
        join_cols = ["mutation_string_pe"]

    before_cols = set(df.columns)
    df = _left_join_effector(
        df,
        effector_df,
        join_cols=join_cols,
        annotation_cols=annotation_cols,
    )
    if "_merge" not in df.columns:
        raise AssertionError(
            "Indicator column not included on df.merge operation. Please insure indicator is set to True"
        )

    after_cols = set(df.columns) - {"_merge"}
    added_cols = after_cols - before_cols

    # set seeds that had no annotation in the effector map to unknown
    df.loc[(df.is_seed) & (df._merge == "left_only"), annotation_cols] = "Unknown"

    # evaluate compounding effector of non seed
    # motifs  by trying to match them with the effector annotation
    # of sub motifs
    default_classifier = "sub_classification"
    if default_classifier not in annotation_cols:
        check = (
            effector_df[annotation_cols].map(lambda x: x.lower() == "multiple").any()
        )
        check_no_multiple = list(check[check == False].index)
        if len(check_no_multiple) == 1:
            main_classifer_col = check_no_multiple[0]
        else:
            main_classifer_col = annotation_cols[0]
    else:
        main_classifer_col = default_classifier

    check_for_sub_motif = (~df.is_seed) & (df._merge == "left_only")
    inferred_annotated_mutations = classify_motif_annotation_by_sub_motifs(
        df.loc[check_for_sub_motif, :],
        effector_df=effector_df,
        main_classifer_col=main_classifer_col,
    )

    df.loc[inferred_annotated_mutations.index, annotation_cols] = (
        inferred_annotated_mutations[annotation_cols]
    )

    return ChainMotifStepResult(
        motif_df=df,
        step_name="add_effector_annotation",
        rows_added=0,
        rows_dropped=0,
        detail=f"Effector annotation joined; {len(added_cols)} columns added: {', '.join(sorted(added_cols))}",
    )


def _add_gene_and_species_columns(
    motif_df: pd.DataFrame,
    individual_mutation_profile: pd.DataFrame,
    user_fields: list[str] | None = None,
    restrict_gene_assignment_to_region: list[str] | None = None,
) -> pd.DataFrame:
    """
    Expand motif_df rows by all (gene, isotype) combos from
    individual_mutation_profile.
    """
    if user_fields is None:
        field_keys = FIELDS_FOR_NOMENCLATURE
    else:
        field_keys = user_fields
    inn_mut_map = _create_inn_map_of_mutation_gene_info(
        individual_mutation_profile, field_keys
    )

    inn_chain_mut_map_by_fc_only = _create_gene_map_for_inn_chains_by_fc_only(
        individual_mutation_profile, field_keys
    )

    # create a backup object just incase mutations are only in the hinge/chs region
    backup_inn_mapping = (
        individual_mutation_profile.reset_index()[
            [
                "inn",
                "chain_id",
                "region_classification",
                "region_iteration",
                "gene",
                "gene_species",
                "allele",
            ]
        ]
        .value_counts(dropna=False)
        .reset_index()
        .groupby(by=["inn", "chain_id"])
    )

    def _get_backup_info(failed_assignments):
        updated_assignments = []
        ignore = ["HINGE-REGION", "HINGE", "CHS"]
        for inn, chain_id, _ in failed_assignments:
            if (inn, chain_id) not in backup_inn_mapping.groups:
                raise AssertionError((inn, chain_id))
            grp_data = backup_inn_mapping.get_group((inn, chain_id))
            if grp_data["region_classification"].isin(ignore).all():
                inn_hits = [
                    tuple(row[field] for field in field_keys)
                    for _, row in grp_data.iterrows()
                ]
            else:
                inn_hits = [
                    tuple(row[field] for field in field_keys)
                    for _, row in grp_data.iterrows()
                    if row["region_classification"] not in ignore
                ]
            updated_assignments.extend(inn_hits)
        return sorted(set(updated_assignments))

    tqdm.pandas()  # enables .progress_apply

    motif_df = motif_df.explode("unique_inn_chain_combos")
    motif_df["inn_count"] = 1
    motif_df["chain_count"] = 1

    motif_df["simple_gene_annotation"] = motif_df["unique_inn_chain_combos"].isin(
        inn_chain_mut_map_by_fc_only.keys()
    )

    group1 = motif_df[motif_df["simple_gene_annotation"]]
    group2 = motif_df[~motif_df["simple_gene_annotation"]]

    motif_df_annotation_raw = pd.DataFrame.from_records(
        group2.progress_apply(
            lambda row: _get_genes_associated_with_motif(
                row["mutation_string_pe"],
                row["unique_inn_chains"],
                inn_mut_map,
                field_keys,
            ),
            axis=1,
        ).values,
        index=group2["mutation_string_pe"],
    )

    motif_df_annotation_raw.loc[
        motif_df_annotation_raw["gene_assignment_failure"].apply(len) > 0,
        "motif_gene_annotation",
    ] = motif_df_annotation_raw.loc[
        motif_df_annotation_raw["gene_assignment_failure"].apply(len) > 0,
        "gene_assignment_failure",
    ].apply(_get_backup_info)
    motif_df_annotation_raw = motif_df_annotation_raw.explode("motif_gene_annotation")

    motif_df_annotation = pd.DataFrame(
        list(motif_df_annotation_raw["motif_gene_annotation"].values),
        columns=field_keys,
        index=motif_df_annotation_raw.index,
    )
    motif_df_annotation[
        [
            "motif_gene_annotation",
            "gene_assignment_failure",
            "core_failure",
        ]
    ] = motif_df_annotation_raw[
        [
            "motif_gene_annotation",
            "gene_assignment_failure",
            "core_failure",
        ]
    ]

    # debug here...
    # chk = motif_df_annotation[motif_df_annotation["gene"].isnull()]
    # if chk.shape[0]:
    #     print(chk.sort_index())
    #     print("stokpe", chk)

    # return motif_df_annotation

    motif_df_annotation = motif_df_annotation.join(
        motif_df[
            [
                "mutation_string_pe",
                "mutation_string",
                "motif_length",
                "remove_motif",
                "motif_selection_reason",
                "motif_detail",
                "super_motifs",
                "motif_token_set",
            ]
        ].set_index("mutation_string_pe"),
        how="left",
    )

    return motif_df_annotation.reset_index()


def _build_cluster_ref_tokens_from_motif_df(
    motif_df: pd.DataFrame,
    group_by_stratify: tuple[str, ...],
    inn_id_to_inn_map: dict[str, str],
) -> dict[tuple[str, ...], dict[tuple[str, str], frozenset[str]]]:
    """Build cluster reference tokens from an enumeration-derived motif DataFrame.

    For each stratification group, maps every ``(inn, chain_id)`` pair to the
    union of PE token sets from every motif row in which that chain appears.

    Should be called with ``enumeration_df`` (post-gene-assignment, pre-filtering)
    so the pool covers all chains for every gene group regardless of which motif
    rows survived downstream filtering.  This preserves gene scoping —
    ``enumeration_df`` is already stratified so each chain only contributes
    tokens attributed to its gene group — while ensuring no motif ends up with
    a zero inn_count due to its chains having been removed from earlier pipeline
    snapshots.

    Parameters
    ----------
    motif_df:
        ``enumeration_df`` after gene/species annotation — must have ``tokens``
        (frozenset of PE token strings) and ``unique_inn_chains`` (set of
        chain_id strings) columns, plus the stratification columns.
    group_by_stratify:
        Stratification columns matching those used during enumeration
        (e.g. ``("gene", "gene_species")``).
    inn_id_to_inn_map:
        ``{inn_id: full_inn_name}`` — resolves chain_id → INN name via
        ``chain_id.split("_")[0].replace("INN", "").strip()``.

    Returns
    -------
    dict mapping ``group_key → {(inn, chain_id): frozenset[PE_token]}``
    """

    def _chain_id_to_inn(chain_id: str) -> str:
        inn_id = chain_id.split("_")[0].replace("INN", "").strip()
        return inn_id_to_inn_map[inn_id]

    def _build_ref(
        sdf: pd.DataFrame,
    ) -> dict[tuple[str, str], frozenset[str]]:
        assert "mutation_string_pe" in sdf.columns, (
            "_build_cluster_ref_tokens_from_motif_df: 'mutation_string_pe' column missing"
        )
        assert "unique_inn_chains" in sdf.columns, (
            "_build_cluster_ref_tokens_from_motif_df: 'unique_inn_chains' column missing"
        )
        ref: dict[tuple[str, str], frozenset[str]] = {}
        for _, row in sdf.iterrows():
            tokens: frozenset = frozenset(
                t.strip()
                for t in str(row["mutation_string_pe"]).split("/")
                if t.strip()
            )
            uic = row["unique_inn_chains"]
            assert isinstance(uic, (set, frozenset, list)), (
                f"unique_inn_chains must be a set/list of chain_ids, got {type(uic)} "
                f"for motif {row['mutation_string_pe']!r}"
            )
            for chain_id in uic:
                inn = _chain_id_to_inn(str(chain_id))
                key = (inn, str(chain_id))
                ref[key] = ref.get(key, frozenset()) | tokens
        return ref

    cluster_db_ref_tokens: dict[
        tuple[str, ...], dict[tuple[str, str], frozenset[str]]
    ] = {}

    group_cols = [c for c in group_by_stratify if c in motif_df.columns]

    if not group_cols:
        cluster_db_ref_tokens[()] = _build_ref(motif_df)
    else:
        for grp_key, sdf in motif_df.groupby(group_cols, sort=False, dropna=False):
            grp = grp_key if isinstance(grp_key, tuple) else (grp_key,)
            cluster_db_ref_tokens[grp] = _build_ref(sdf)

    return cluster_db_ref_tokens


def _add_sub_motifs_column(motif_df: pd.DataFrame) -> pd.DataFrame:
    """Add ``sub_motifs`` column — list of PE strings for all proper sub-combinations.

    For each row, generates every proper non-empty subset of its ``tokens``
    frozenset using :func:`~analytics.motifs.stat_functions.generate_submotifs`
    and checks which subsets have a corresponding row in *motif_df*.  The
    result is stored as a list of ``mutation_string_pe`` values, sorted by
    token-set size descending (largest sub-motifs first).

    Rows with fewer than 2 tokens always get an empty list.
    """
    assert "tokens" in motif_df.columns, (
        "_add_sub_motifs_column: 'tokens' column missing from motif_df"
    )
    assert "mutation_string_pe" in motif_df.columns, (
        "_add_sub_motifs_column: 'mutation_string_pe' column missing from motif_df"
    )
    # Build token-set → pe lookup for all rows
    tokens_to_pe: dict[frozenset, str] = {
        row["tokens"]: row["mutation_string_pe"]
        for _, row in motif_df.iterrows()
        if isinstance(row["tokens"], frozenset)
    }

    def _sub_motifs_for_row(toks: frozenset) -> list[str]:
        if not isinstance(toks, frozenset) or len(toks) < 2:
            return []
        # generate_submotifs yields (sign, subset) — we only need the subsets
        found = [
            tokens_to_pe[sub_ts]
            for _, sub_ts in generate_submotifs(toks)
            if sub_ts in tokens_to_pe
        ]
        # deduplicate preserving first occurrence (largest sub-motifs first
        # because generate_submotifs yields size-1 subsets before size-2 etc,
        # so we reverse to get largest first)
        seen: set[str] = set()
        result: list[str] = []
        for pe in reversed(found):
            if pe not in seen:
                seen.add(pe)
                result.append(pe)
        return result

    df = motif_df.copy()
    df["sub_motifs"] = df["tokens"].apply(_sub_motifs_for_row)

    df["novel_motif_mutations"] = df[["tokens", "sub_motifs"]].apply(
        lambda x: (
            x["tokens"]
            - set(
                [
                    indiv_mut
                    for sub_mut in x["sub_motifs"]
                    for indiv_mut in sub_mut.split("/")
                ]
            )
        ),
        axis=1,
    )
    return df


def _get_group_inn_counts(df: pd.DataFrame, groups: list[str]) -> dict[tuple, int]:
    def _group_counts(sdf: pd.DataFrame) -> int:
        all_stratum_inns = []
        _unique_checks = 10000
        for idx, i in enumerate(sdf["inns"].apply(lambda x: list(x))):
            all_stratum_inns.extend(i)
            if idx % _unique_checks == 0:
                all_stratum_inns = sorted(set(all_stratum_inns))
        all_stratum_inns = sorted(set(all_stratum_inns))
        return len(all_stratum_inns)

    return df.groupby(by=list(groups), dropna=False).apply(_group_counts).to_dict()


def _chain_to_inn_name(chain: str, inn_id_to_inn_map: dict[str, str]) -> str:
    return inn_id_to_inn_map[chain.split("_")[0].replace("INN", "").strip()]


def unique_inn_chain_pairs(
    grp_members: list[list[str]],
    inn_id_to_inn_map: dict[str, str],
) -> set[tuple[str, str]]:

    return {
        (
            _chain_to_inn_name(chain, inn_id_to_inn_map),
            chain,
        )
        for chain_groups in grp_members
        for chain in chain_groups
    }


# ---------------------------------------------------------------------------
# Public pipeline orchestrator
# ---------------------------------------------------------------------------


def build_chain_motif_index(
    inn_dataset: INNAnalysisDataset,
    multi_chain_inns: dict[str, dict[str, list[str]]],
    same_chain_type_groups: dict[str, list[tuple[str, ...]]],
    group_by: list[str] | None = None,
    min_chain_count: int = 2,
    min_motif_size: int = 2,
    max_motif_size: int | None = None,
    max_consecutive_deletions: int | None = 5,
    mutations_to_ignore: dict[str, set[str]] | None = None,
    min_lift: float = 1.0,
    max_pvalue: float = 0.05,
    allowed_motifs_by_select: list[str] | None = None,
    cohesion_frac_threshold: float | None = 1.0,
    marginals_table_user: Optional[pd.DataFrame] = None,
    inn_count_invariant_mode: str = "warn",
    debug_stop_at_step: int | None = None,
    max_cluster_filter_iteration: int | None = 0,
    hard_stat_filters: (list[Literal["pvalue", "lift", "cohesion"]] | None) = None,
    additional_user_set_filter_terms: (dict[str, tuple[str, float]] | None) = None,
    cache_object: ChainMotifIndexResult | None = None,
    forced_manual_motif_strings: list[str] | None = None,
    cdb: "ConstantDB | None" = None,
    cn: "ConstantChainNumbering | None" = None,
) -> ChainMotifIndexResult:
    """Run the full chain-level motif pipeline.

    Steps
    -----
    1. ``enumerate_chain_motifs`` — enumerate all sub-motifs from per-chain EU strings.
    2. ``_add_gene_and_species_columns`` — derive gene/species/isotype for motif rows.
    3. ``expand_motifs_by_cross_chain_pairing`` — derive bispecific arm/anchor rows.
    4. ``update_chain_motif_df_with_cross_chain`` — merge cross-chain rows.
    5. ``resolve_super_motifs`` — walk parent DAG to non-consumed ancestors.
    6. ``step_filter_remove_motif`` — drop ``remove_motif=True`` rows.
    7. ``step_restore_other_gene_hits`` — recover same-signature motifs across gene groups.
    8. ``step_cluster_motifs`` — cluster by token-subset membership.

    10. ``step_subtract_inns`` — final INN subtraction (bispecific rows exempt).
    11. ``step_add_statistics`` — Möbius stats, independence baseline, motif_score.
    13. ``step_apply_final_config_filter`` — enforce min_lift/max_pvalue/cohesion.
    14. **Deduplicate** on (mutation_string_pe, gene, species, isotype).

    Parameters
    ----------
    chain_rows :
        ``INNAnalysisDataset`` or ``Sequence[ChainEnumerationRow]``.
    multi_chain_inns :
        ``{inn: {chain_type: [chain_ids]}}`` from :func:`~analytics._helpers.find_multi_chain_inns`.
    same_chain_type_groups :
        Either a dict ``{inn: [tuple_of_chain_ids]}`` OR a DataFrame
        (e.g., ``ds.constant_chain_motifs_by_inn_region``). If a DataFrame,
        it is automatically converted to the dict format.
    group_by :
        Stratification fields for enumeration.
    min_chain_count, min_motif_size, max_motif_size, mutations_to_ignore :
        Enumeration parameters.
    max_consecutive_deletions :
        Maximum allowed run length of consecutive deletion tokens per
        contiguous EU-position run within a chain profile. Deletion tokens in
        over-threshold runs are skipped from enumeration, while non-deletion
        tokens in the same profile are retained. ``None`` disables this safeguard.
    min_lift, max_pvalue :
        Statistical significance thresholds.
    cohesion_frac_threshold :
        Minimum cohesion threshold for final filtering (uses
        ``min_frac_pair_in_full`` when available). ``None`` disables cohesion filter.
    marginals_table :
        Optional artifact-fraction lookup.
    original_motif_df :
        Pre-enumerated table for other-gene-hit recovery.  If ``None``, recovery
        is skipped.
    individual_mutation_profile :
        Optional per-mutation table; when omitted and ``chain_rows`` is an
        ``INNAnalysisDataset``, uses ``chain_rows.individual_mutation_profile``.
        Used to assign motif-level gene/isotype combinations before effector join.
    debug_stop_at_step :
        (DEBUG) If set to a step number (1-15), return early after that step
        with just the motif_df (for debugging). Default is None (run full pipeline).

    Returns
    -------
    :class:`ChainMotifIndexResult`
    """

    def _maybe_debug_checkpoint(step_num: int, mpr: ChainMotifIndexResult):
        """Return early if debug_stop_at_step is set and matches current step."""
        if debug_stop_at_step is not None and step_num >= debug_stop_at_step:
            _log.warning(f"DEBUG: Returning at step {step_num}")
            return mpr
        return None

    # ── Convert DataFrame to dict if needed ────────────────────────────────
    if isinstance(same_chain_type_groups, pd.DataFrame):
        _log.info("Converting same_chain_type_groups DataFrame to dict format...")
        df = same_chain_type_groups
        groups_dict: dict[str, list[tuple[str, ...]]] = {}

        # Group by inn and chain_type, collect unique chain_id tuples
        for inn in df["inn"].unique():
            inn_data = df[df["inn"] == inn]
            chain_tuples = []
            for chain_type in inn_data["chain_type"].unique():
                type_data = inn_data[inn_data["chain_type"] == chain_type]
                chain_ids = tuple(sorted(type_data["chain_id"].unique()))
                if chain_ids and chain_ids not in chain_tuples:
                    chain_tuples.append(chain_ids)
            groups_dict[inn] = chain_tuples

        same_chain_type_groups = groups_dict
        _log.info(
            f"  ✓ Converted to {len(same_chain_type_groups)} INNs with chain groups"
        )

    # motif pipeline result tracking
    _mpr = ChainMotifIndexResult(
        step_changes={},
        filters={
            "min_chain_count": min_chain_count,
            "min_motif_size": min_motif_size,
            "max_motif_size": max_motif_size,
            "min_lift": min_lift,
            "max_pvalue": max_pvalue,
            "cohesion_frac_threshold": cohesion_frac_threshold,
        },
    )

    clustering_stat_hard_filters: dict[str, tuple[str, float]] = {}
    if hard_stat_filters is not None:
        mapping_term_to_col: dict[str, tuple[str, str, float]] = {
            "pvalue": (C.pvalue_vs_indep, "max", max_pvalue),
            "lift": (C.lift_vs_indep, "min", min_lift),
            "cohesion": (
                C.pairwise_cohesion,
                "min",
                cohesion_frac_threshold or np.inf,
            ),
        }
        for filter_term in hard_stat_filters:
            if filter_term not in mapping_term_to_col:
                raise ValueError(f"Unknown hard stat filter term: {filter_term}")

            col_name, cutoff_type, threshold = mapping_term_to_col[filter_term]
            clustering_stat_hard_filters[col_name] = (
                cutoff_type,
                threshold,
            )
    if additional_user_set_filter_terms:
        clustering_stat_hard_filters.update(deepcopy(additional_user_set_filter_terms))

    t_start = time.time()

    _log.info("=" * 80)
    _log.info("build_chain_motif_index: STARTING chain-level motif pipeline")
    _log.info("  Filters: %s", _mpr.filters)
    _log.info("=" * 80)

    if group_by is None:
        group_by = ["mutation_string_pe"]

    group_by_stratify = [grp for grp in group_by if grp != "mutation_string_pe"]

    # Initialize step counter and cross_chain_df/skipped for debug checkpoints
    current_step = 0

    if cache_object is not None:
        _mpr.marginals_table = cache_object.marginals_table
    elif marginals_table_user is None:
        _mpr.marginals_table = build_marginals_table(
            inn_summary=inn_dataset.individual_mutation_profile,  # full mutation profile
            position_col="EU",
            stratify_by=group_by_stratify or None,
        )
    else:
        _mpr.marginals_table = marginals_table_user.copy()

    db_sequences_mutation_info: list[ChainEnumerationRowAnnotated] = [
        seq_info
        for seq_info in tqdm(
            chain_collection_to_enumeration_rows(
                inn_dataset.full_length_chain_collection
            ),
            total=len(inn_dataset.full_length_chain_collection.records),
            desc="Enumerating chains",
        )
        if seq_info.total_constant_mutations > 0
    ]

    # Build the allele-lookup structures needed for the cross-chain
    # germline-reversion check (see analytics.motifs.allele_mapping):
    #   ind_mut_allele_map           — {(chain_id, EU_pos): mutation_allele_mapper}
    #   chain_id_region_allele_map   — {(chain_id, region): (allele, species)}
    #   inn_chain_domain_map         — {inn_id: {"CK"|"CL"|"V": [chain_id, ...]}}
    _ind_mut_allele_map = None
    _chain_id_region_allele_map = None
    _inn_chain_domain_map = None
    if cdb is not None and cn is not None:
        from inn_analysis_ops.analytics.motifs.allele_mapping import (
            _infer_chain_id_mut_to_allele_map,
            _infer_chain_id_region_to_allele_map,
            _infer_inn_chain_domain_map,
        )

        _ind_mut_allele_map = _infer_chain_id_mut_to_allele_map(inn_dataset)
        _chain_id_region_allele_map = _infer_chain_id_region_to_allele_map(inn_dataset)
        _inn_chain_domain_map = _infer_inn_chain_domain_map(inn_dataset)

    chain_clustering_iteration = None
    inn_id_to_inn_map = (
        inn_dataset.full_length_chain_collection.to_dataframe()
        .groupby(by=["inn_id"])["inn"]
        .first()
        .to_dict()
    )
    inn_to_inn_id_map = (
        inn_dataset.full_length_chain_collection.to_dataframe()
        .groupby(by=["inn"])["inn_id"]
        .first()
        .to_dict()
    )
    # same_chain_type_groups may be keyed either by full "inn" name (legacy
    # find_multi_chain_inns / constant_chain_motifs_by_inn_region path) or
    # already by inn_id (find_multi_chain_inns_by_subdomain, which groups by
    # sub_domain and returns inn_id keys directly). Only remap keys that are
    # actually full inn names — remapping an inn_id key would raise a
    # spurious KeyError since inn_to_inn_id_map is keyed by full inn names.
    same_chain_type_groups.update(
        {
            inn_to_inn_id_map[inn]: value
            for inn, value in same_chain_type_groups.items()
            if inn in inn_to_inn_id_map
        }
    )

    try:
        motifs_removed_at_clustering = []

        t0 = time.time()
        if cache_object is None:
            _log.info("[Step 1/15] enumerate_chain_motifs...")
            _mpr.enumeration_df = enumerate_chain_motifs(
                db_sequences_mutation_info,
                inn_dataset.individual_mutation_profile,
                min_motif_size=max(
                    2, min_motif_size
                ),  # always initialliy filter non signeltones
                max_motif_size=max_motif_size,
                max_consecutive_deletions=max_consecutive_deletions,
                mutations_to_ignore=mutations_to_ignore,
                forced_manual_motif_strings=forced_manual_motif_strings,
            )

            _log.info(
                "  ✓ %d motifs enumerated (%.2fs)",
                len(_mpr.enumeration_df),
                time.time() - t0,
            )

            _log.info(
                "  ✓ gene/species/isotype columns added (unique genes: %s)",
                sorted(_mpr.enumeration_df["gene"].dropna().unique())[:10],
            )

            # group by and collapses motif string info
            _mpr.enumeration_df = _collapse_motif_df_results(
                _mpr.enumeration_df, group_by
            ).reset_index()

            t0 = time.time()
            _log.info("[Step 2/15] expand_motifs_by_cross_chain_pairing...")
            current_step = 2
            assert (
                _mpr.enumeration_df[["mutation_string_pe"] + group_by_stratify]
                .value_counts()
                .max()
                == 1
            )

            _mpr.cross_chain_df, _mpr.skipped_cross_chain = (
                expand_motifs_by_cross_chain_pairing(
                    _mpr.enumeration_df,
                    multi_chain_inns,
                    same_chain_type_groups,
                    1,
                    max_motif_size,
                    cdb=cdb,
                    cn=cn,
                    ind_mut_allele_map=_ind_mut_allele_map,
                    chain_id_region_allele_map=_chain_id_region_allele_map,
                    inn_chain_domain_map=_inn_chain_domain_map,
                )
            )

            _mpr._enumeration_collapsed_df = _collapse_motif_df_results(
                update_chain_motif_df_with_cross_chain(
                    _mpr.enumeration_df, _mpr.cross_chain_df
                ),
                group_by,
            ).reset_index()

        else:
            _log.info("Loading from cache")
            _mpr.enumeration_df = cache_object.enumeration_df.copy()
            _mpr.other_dfs["enumeration_df_orig"] = cache_object.other_dfs.get(
                "enumeration_df_orig", pd.DataFrame()
            ).copy()
            _mpr.cross_chain_df = cache_object.cross_chain_df.copy()
            _mpr.skipped_cross_chain = deepcopy(cache_object.skipped_cross_chain)
            _mpr._enumeration_collapsed_df = (
                cache_object._enumeration_collapsed_df.copy()
            )

        _mpr.inn_chains_per_group = (
            _mpr.enumeration_df.groupby(by=list(group_by_stratify))["unique_inn_chains"]
            .apply(lambda x: unique_inn_chain_pairs(x, inn_id_to_inn_map))
            .to_dict()
        )

        group_inn_counts = _get_group_inn_counts(
            _mpr.enumeration_df, list(group_by_stratify)
        )
        _log.info(
            "  ✓ %d cross-chain rows derived, %d skipped (%.2fs)",
            len(_mpr.cross_chain_df),
            len(_mpr.skipped_cross_chain),
            time.time() - t0,
        )
        if _checkpoint := _maybe_debug_checkpoint(current_step, _mpr):
            return _checkpoint

        t0 = time.time()
        _log.info("[Step 3/15] update_chain_motif_df_with_cross_chain...")
        current_step = 3

        _before = _mpr.enumeration_df.copy()
        _mpr.motif_df = _mpr._enumeration_collapsed_df

        _mpr.step_changes["update_cross_chain"] = _compute_step_delta(
            _before,
            _mpr.motif_df,
            add_reason="cross_chain_pairing",
            remove_reason="dropped_during_cross_chain_update",
        )
        _log.info(
            "  ✓ %d rows after cross-chain merge (%.2fs)",
            len(_mpr.motif_df),
            time.time() - t0,
        )
        if _checkpoint := _maybe_debug_checkpoint(current_step, _mpr):
            return _checkpoint

        t0 = time.time()
        _log.info("[Step 4/15] resolve_super_motifs...")
        current_step = 4

        _mpr.other_dfs["motif_df_pre_resolve_super_motifs"] = _mpr.motif_df.copy()
        _mpr.motif_df = resolve_super_motifs(_mpr.motif_df)
        _mpr.other_dfs["motif_df_post_resolve_super_motifs"] = _mpr.motif_df.copy()
        _log.info("  ✓ super-motif DAG resolved (%.2fs)", time.time() - t0)
        if _checkpoint := _maybe_debug_checkpoint(current_step, _mpr):
            return _checkpoint
        original_motif_before_filtering = _mpr.motif_df.copy()

        t0 = time.time()
        _log.info(
            "[Step 5/15] filter_remove_motif (min_chain_count=%d)...",
            min_chain_count,
        )
        current_step = 5

        # Apply min_chain_count flagging before hard filter
        if min_chain_count > 1:
            rare_mask = _mpr.motif_df["chain_count"] < min_chain_count
            _mpr.motif_df.loc[
                rare_mask & ~_mpr.motif_df["remove_motif"], "remove_motif"
            ] = True
            _mpr.motif_df.loc[
                rare_mask & _mpr.motif_df["motif_selection_reason"].isna(),
                "motif_selection_reason",
            ] = "low_count"

        _before = _mpr.motif_df.copy()
        filter_result, removed_rows = step_filter_remove_motif(
            ChainMotifStep(motif_df=_mpr.motif_df, step_name="filter_remove_motif")
        )
        _mpr.motif_df = filter_result.motif_df
        _mpr.other_dfs["motif_df_post_filter_remove_motif"] = _mpr.motif_df.copy()
        _mpr.step_changes["filter_remove_motif"] = _compute_step_delta(
            _before,
            _mpr.motif_df,
            add_reason="",
            remove_reason=filter_result.detail,
        )
        _log.info(
            "  ✓ %d rows kept, %d dropped (%.2fs)",
            len(_mpr.motif_df),
            filter_result.rows_dropped,
            time.time() - t0,
        )
        _log.info(
            "  ⚠️  Steps 6-15 operate ONLY on the filtered subset (%d rows)",
            len(_mpr.motif_df),
        )
        current_step = 6
        if original_motif_before_filtering.shape[0]:
            t0 = time.time()
            _log.info("[Step 6/15] restore_other_gene_hits...")

            _before = _mpr.motif_df.copy()
            restore_result = step_restore_other_gene_hits(
                _RestoreOtherGeneHitsIn(
                    motif_df=_mpr.motif_df,
                    original_motif_df=original_motif_before_filtering,
                    min_count=min_chain_count,
                    step_name="restore_other_gene_hits",
                )
            )
            _mpr.motif_df = restore_result.motif_df
            _mpr.other_dfs["motif_df_post_filter_restore_other_gene_hits"] = (
                _mpr.motif_df.copy()
            )
            _mpr.step_changes["restore_other_gene_hits"] = _compute_step_delta(
                _before,
                _mpr.motif_df,
                add_reason="other_gene_hit",
                remove_reason="",
            )
            _log.info(
                "  ✓ %d rows recovered (%.2fs)",
                restore_result.recovered_rows,
                time.time() - t0,
            )
        if _checkpoint := _maybe_debug_checkpoint(current_step, _mpr):
            return _checkpoint

        _before_clustering = _mpr.motif_df.copy()
        _mpr.motif_df_per_cluster_iteration = []

        # Build cluster reference token map from the full enumeration_df
        # (post-gene-assignment, pre-filtering) so every (inn, chain) pair is
        # present for its gene group.  Using motif_df_before_clustering would
        # miss chains whose motif rows were removed by earlier pipeline steps,
        # causing zero inn_count errors.  Gene scoping is preserved because
        # enumeration_df is already stratified by gene group.
        _mpr.cluster_db_ref_tokens = _build_cluster_ref_tokens_from_motif_df(
            _mpr.enumeration_df,
            tuple(group_by_stratify),
            inn_id_to_inn_map,
        )
        current_iteration = 0

        # do a final confirmation of whats been removed
        # some rows may have been borught back
        removed_rows = removed_rows[
            ~removed_rows["mutation_string_pe"].isin(
                _mpr.motif_df["mutation_string_pe"]
            )
        ]
        motifs_removed_at_clustering.append(removed_rows)
        _mpr.other_dfs["motif_df_before_clustering"] = _mpr.motif_df.copy()
        while True:
            t0 = time.time()
            _log.info("[Step 7/15] cluster_motifs...")
            current_step = 7
            _before = _mpr.motif_df.copy()

            chain_clustering_iteration = ChainIterativeClusters(
                name=f"clustering_iteration_{current_iteration}",
                iteration=current_iteration,
            )
            chain_clustering_iteration.init_df = _before

            # create clusters
            cluster_result, removed = step_build_cluster_info(
                ChainMotifStep(
                    motif_df=_mpr.motif_df,
                    group_cols=group_by_stratify,
                    step_name="cluster_motifs",
                    allowed_motifs_by_select=allowed_motifs_by_select,
                ),
                _mpr.cluster_db_ref_tokens,
            )

            if removed is not None:
                removed["_removal_step"] = "building clusters (adj_count=0)"
                removed["_iteration"] = current_iteration
                motifs_removed_at_clustering.append(removed)
            chain_clustering_iteration.cluster_df = cluster_result.motif_df

            # add sub_motifs column before statistics so downstream tools have
            # explicit sub-combination lists rather than recomputing on the fly
            motif_df_with_sub = _add_sub_motifs_column(cluster_result.motif_df)

            # add statitisics based on clsuters
            stats_result = step_add_statistics(
                ChainMotifStep(
                    motif_df=motif_df_with_sub,
                    step_name="add_statistics",
                    group_cols=group_by_stratify,
                ),
                group_inn_counts,
                marginals_table=_mpr.marginals_table,
                min_lift_pass_condition=min_lift,
                max_pvalue_pass_condition=max_pvalue,
            )

            filtered_mutation_strings = list(
                set(cluster_result.motif_df["mutation_string_pe"])
                - set(stats_result.motif_df["mutation_string_pe"])
            )

            removed_found = cluster_result.motif_df.query(
                "mutation_string_pe in @filtered_mutation_strings"
            )

            if removed_found.shape[0]:
                removed_found["_iteration"] = current_iteration
                removed_found["_removal_step"] = "stat_calculation"
                motifs_removed_at_clustering.append(removed_found)
            chain_clustering_iteration.stats_df = stats_result.motif_df

            _mpr.motif_df = stats_result.motif_df

            shape_init = _mpr.motif_df.shape[0]

            if clustering_stat_hard_filters:
                filter_result = step_filter_by_statistics(
                    ChainMotifStep(
                        motif_df=_mpr.motif_df,
                        step_name="filter_by_statistics",
                    ),
                    filters=clustering_stat_hard_filters,
                    allowed_motifs_by_select=allowed_motifs_by_select,
                )

                filtered_mutation_strings = list(
                    set(_mpr.motif_df["mutation_string_pe"])
                    - set(filter_result.motif_df["mutation_string_pe"])
                )
                removed_found = _mpr.motif_df.query(
                    "mutation_string_pe in @filtered_mutation_strings"
                )
                if removed_found.shape[0]:
                    removed_found["_iteration"] = current_iteration
                    removed_found["_removal_step"] = "filter_by_statistics"
                    motifs_removed_at_clustering.append(removed_found)

                _mpr.motif_df = filter_result.motif_df
            chain_clustering_iteration.output_df = _mpr.motif_df

            shape_after = _mpr.motif_df.shape[0]
            _log.info(
                "  ✓ %s (before: %d, after: %d)",
                "exit loop",
                shape_init,
                shape_after,
            )
            _mpr.motif_df_per_cluster_iteration.append(
                (_before, _mpr.motif_df.copy(), chain_clustering_iteration)
            )

            if shape_init == shape_after:
                break

            current_iteration += 1
            if (current_iteration is not None) and (
                current_iteration >= max_cluster_filter_iteration
            ):
                # break early since we do not need to repeat,
                break

        # reclualate clusters one last time after
        # data was filtered by previous iterations
        post_iteration_cluster_result, removed = step_build_cluster_info(
            ChainMotifStep(
                motif_df=_mpr.motif_df,
                group_cols=group_by_stratify,
                step_name="cluster_motifs",
                allowed_motifs_by_select=allowed_motifs_by_select,
            ),
            _mpr.cluster_db_ref_tokens,
        )
        if removed is not None:
            removed["_removal_step"] = (
                "post_loop_iteration building clusters (adj_count=0)"
            )
            removed["_iteration"] = -1
            motifs_removed_at_clustering.append(removed)

        post_iteration_motif_df_with_sub = _add_sub_motifs_column(
            post_iteration_cluster_result.motif_df
        )

        post_iteration_stats_result = step_add_statistics(
            ChainMotifStep(
                motif_df=post_iteration_motif_df_with_sub,
                step_name="add_statistics",
                group_cols=group_by_stratify,
            ),
            group_inn_counts,
            marginals_table=_mpr.marginals_table,
            min_lift_pass_condition=min_lift,
            max_pvalue_pass_condition=max_pvalue,
        )
        filtered_mutation_strings = list(
            set(post_iteration_cluster_result.motif_df["mutation_string_pe"])
            - set(post_iteration_stats_result.motif_df["mutation_string_pe"])
        )

        removed_found = post_iteration_cluster_result.motif_df.query(
            "mutation_string_pe in @filtered_mutation_strings"
        )

        if removed_found.shape[0]:
            removed_found["_iteration"] = -1
            removed_found["_removal_step"] = "post_loop_iteration stat_calculation"
            motifs_removed_at_clustering.append(removed_found)

        _mpr.motif_df = post_iteration_stats_result.motif_df
        _mpr.other_dfs["motif_df_after_clustering"] = _mpr.motif_df.copy()
        _post_loop_cluster_iter = ChainIterativeClusters(
            name="final_clustering",
            iteration=-1,
        )
        _post_loop_cluster_iter.init_df = _before_clustering
        _post_loop_cluster_iter.cluster_df = post_iteration_cluster_result.motif_df
        _post_loop_cluster_iter.stats_df = post_iteration_stats_result.motif_df
        _post_loop_cluster_iter.output_df = _mpr.motif_df
        _mpr.motif_df_per_cluster_iteration.append(
            (
                _before_clustering,
                _mpr.motif_df.copy(),
                _post_loop_cluster_iter,
            )
        )

        # # Deduplicate on (mutation_string_pe, gene, species, isotype)
        # t0 = time.time()
        # _log.info(
        #     "[Step 14/15] Deduplicating on (mutation_string_pe, gene, species, isotype)..."
        # )
        # current_step = 14
        # if all(
        #     col in motif_df.columns
        #     for col in ["mutation_string_pe", "gene", "species", "isotype"]
        # ):
        #     motif_df_before = len(motif_df)
        #     motif_df = motif_df.drop_duplicates(
        #         subset=["mutation_string_pe", "gene", "species", "isotype"],
        #         keep="first",
        #     ).reset_index(drop=True)
        #     motif_df_after = len(motif_df)
        #     _log.info(
        #         "  ✓ Removed %d duplicate (mutation_string_pe, gene, species, isotype) rows; %d remaining (%.2fs)",
        #         motif_df_before - motif_df_after,
        #         motif_df_after,
        #         time.time() - t0,
        #     )
        # else:
        #     _log.warning(
        #         "  ⚠️ Skipping deduplication: missing columns. Available: %s",
        #         list(motif_df.columns),
        #     )
        # if _checkpoint := _maybe_debug_checkpoint(current_step):
        #     return _checkpoint

        # Final sort and motif_id assignment
        _log.info("[Step 15/15] final_sort_and_motif_id...")
        current_step = 15
        # Sort by: is_seed (seed motifs first), inn_count (highest first),
        # motif_length (shortest first), mutation_string_pe (alphabetical)
        _mpr.motif_df = _mpr.motif_df.sort_values(
            by=[
                "is_seed",
                "inn_count",
                "motif_length",
                "mutation_string_pe",
            ],
            ascending=[False, False, True, True],
        ).reset_index(drop=True)
        _mpr.motif_df["motif_id"] = _mpr.motif_df.index

        # # Post-step: recompute support counts on the final retained motif table.
        # # This guarantees inn_count/adjusted_count reflect the final motif list
        # # (including final seed/cluster outcomes), not an earlier intermediate table.
        # t0 = time.time()
        # _log.info("[Post-Step] recompute_final_inn_and_adjusted_counts...")
        # motif_df = step_subtract_inns(
        #     _SubtractInnsIn(
        #         motif_df=motif_df,
        #         step_name="recompute_final_inn_and_adjusted_counts",
        #         skip_bispecific=True,
        #     )
        # ).motif_df
        if "inns" in _mpr.motif_df.columns:
            _mpr.motif_df["inn_count"] = _mpr.motif_df["inns"].map(
                lambda s: len(s) if isinstance(s, (set, frozenset)) else 0
            )
        if "adjusted_inns" in _mpr.motif_df.columns:
            _mpr.motif_df["adjusted_count"] = _mpr.motif_df["adjusted_inns"].map(
                lambda s: len(s) if isinstance(s, set) else 0
            )
        _over_mask = (
            (_mpr.motif_df["adjusted_count"] > _mpr.motif_df["inn_count"])
            if {"adjusted_count", "inn_count"} <= set(_mpr.motif_df.columns)
            else pd.Series(False, index=_mpr.motif_df.index)
        )
        if _over_mask.any():
            _bad = _mpr.motif_df.loc[
                _over_mask,
                [
                    c
                    for c in (
                        "mutation_string_pe",
                        "gene",
                        "species",
                        "inn_count",
                        "adjusted_count",
                    )
                    if c in _mpr.motif_df.columns
                ],
            ].head(10)
            _msg = (
                "Invariant violation: adjusted_count > inn_count after final recompute. "
                f"Rows violating invariant: {int(_over_mask.sum())}. "
                f"Examples:\n{_bad.to_string(index=False)}"
            )
            if inn_count_invariant_mode == "assert":
                raise ValueError(_msg)
            else:
                _log.warning("⚠️  %s", _msg)
        _log.info(
            "  ✓ Final support counts recomputed for %d motifs (%.2fs)",
            len(_mpr.motif_df),
            time.time() - t0,
        )
        if _checkpoint := _maybe_debug_checkpoint(current_step, _mpr):
            return _checkpoint

        t_total = time.time() - t_start
        _log.info("=" * 80)
        _log.info("✓ build_chain_motif_index: COMPLETE")
        _log.info("  Final motif count: %d", len(_mpr.motif_df))
        _log.info("  Total time: %.2fs", t_total)
        _log.info("=" * 80)
    except Exception as err:
        _log.error("WE have reached an  unexpected exception")
        _log.exception(err)
        if len(motifs_removed_at_clustering) > 0:
            motifs_removed_at_clustering_dfs = pd.concat(motifs_removed_at_clustering)
            _mpr.removed_post_clustering = motifs_removed_at_clustering_dfs
        return _mpr

    if len(motifs_removed_at_clustering) > 0:
        motifs_removed_at_clustering_dfs = pd.concat(motifs_removed_at_clustering)
        _mpr.removed_post_clustering = motifs_removed_at_clustering_dfs

    _mpr.motif_df = _mpr.motif_df.sort_values(
        by=[
            "is_seed",
            "inn_count",
            "motif_length",
            "mutation_string_pe",
        ],
        ascending=[False, False, True, True],
    ).reset_index(drop=True)
    return _mpr
