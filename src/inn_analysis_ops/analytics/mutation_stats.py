"""Mutation-statistics helpers for inn_summary DataFrames.

Functions
---------
build_mutation_freq(inn_summary, gene)
    Per-EU-position mutation frequency for one gene.

_build_mutation_residues(inn_summary, mutation_types)
    Flatten inn_summary into a per-(gene, EU, residue) count table
    with occurrence counts and INN lists.
"""

from __future__ import annotations

from typing import List

import pandas as pd

from ._helpers import _build_mutation_string_pe, _split_eu_pos_col


def build_positional_mutation_freq(
    inn_summary: pd.DataFrame,
    region_classification_groups: dict[str, list[str]],
    column_maps: dict[str, str] | None = None,
    additional_groups: list[str] | None = None,
) -> pd.DataFrame:
    """Return per-EU-position mutation frequency for *gene*.

    Returns a DataFrame with columns: ``position`` (int) and ``frequency``
      (0-1).
    Insertion-slot positions (e.g. ``'230.i1'``) are dropped.
    """
    column_mapper = {
        "gene": "gene",
        "region": "region_classification",
        "position": "EU",
        "inn": "inn",
        "species": "gene_species",
    }
    if column_maps is not None:
        column_mapper.update(column_maps)

    flat = inn_summary.reset_index().drop(
        ["index"], axis=1, errors="ignore"
    )

    for field, col_name in column_mapper.items():
        if col_name not in inn_summary:
            raise ValueError(
                f"Expected column for {field} -> {col_name} -> not found in "
                "inn_summary. Manually define column name through "
                "'column_maps'"
            )

    unique_regions = flat[column_mapper["region"]].unique()
    if set(unique_regions) - {
        val
        for group in region_classification_groups.values()
        for val in group
    }:
        raise ValueError(
            "inn_summary contains regions not present "
            "in region_classification_groups."
            f"Unique regions in df is {unique_regions}"
            f"User defined set classifiers: {region_classification_groups}"
        )
    region_to_group: dict[str, str] = {}
    for group, members in region_classification_groups.items():
        for member in members:
            if member in region_to_group:
                raise ValueError(
                    f"Region {member} is assigned to multiple groups: "
                    f"{region_to_group[member]} and {group}"
                )
            region_to_group[member] = group

    df, (pos_col, ins_pos_col) = _split_eu_pos_col(
        flat.copy(), column_mapper["position"]
    )
    df["position_numbering_group"] = df[column_mapper["region"]].map(
        region_to_group
    )

    inns_mutated = df.groupby(by="position_numbering_group", dropna=False)[
        column_mapper["inn"]
    ].unique()

    df_collapsed = (
        df.groupby(
            by=[
                pos_col,
                ins_pos_col,
                column_mapper["region"],  # raw region_classification
                "position_numbering_group",
                column_mapper["gene"],
                column_mapper["species"],
            ]
            + (additional_groups or []),
            dropna=False,
        )[column_mapper["inn"]]
        .unique()
        .reset_index()
    )
    df_collapsed[column_mapper["inn"]] = df_collapsed[
        column_mapper["inn"]
    ].apply(lambda x: sorted(set(x)))
    df_collapsed["inn_count"] = df_collapsed[column_mapper["inn"]].apply(
        len
    )

    # Uppercase classifier columns
    for _uc_col in [
        "position_numbering_group",
        column_mapper["gene"],
        column_mapper["species"],
    ]:
        if _uc_col in df_collapsed.columns:
            df_collapsed[_uc_col] = df_collapsed[_uc_col].str.upper()

    chk = (
        df_collapsed.groupby(by="position_numbering_group")[
            column_mapper["inn"]
        ]
        .sum()
        .apply(lambda x: len(sorted(set(x))))
    ).to_dict()
    for name, inns in inns_mutated.to_dict().items():
        unique_vals = len(sorted(set(inns)))
        if unique_vals != chk.get(name, 0.0):
            # After uppercasing, re-check with uppercase key
            if unique_vals != chk.get(
                name.upper() if isinstance(name, str) else name, 0
            ):
                raise ValueError(
                    f"Mismatch in unique INNs for position_numbering_group {name}: "
                    f"{unique_vals} vs {chk.get(name, 0)}"
                )

    _result = df_collapsed.assign(
        eu_position_label=lambda d: d.apply(
            lambda row: (
                f"{int(row[pos_col])}.{row[ins_pos_col]}"
                if pd.notna(row[ins_pos_col])
                and str(row[ins_pos_col]).strip()
                else str(int(row[pos_col]))
            ),
            axis=1,
        )
    ).sort_values(by=["inn_count"], ascending=False)
    # Promote eu_position_label to first column
    cols = ["eu_position_label"] + [
        c for c in _result.columns if c != "eu_position_label"
    ]
    return _result[cols]


def build_mutation_residues(
    inn_summary: pd.DataFrame,
    group_level: list[str],
    mutation_types: List[str] | None = None,
    prefered_col_order: list[str] | None = None,
) -> pd.DataFrame:
    """Flatten inn_summary into a per-(gene, EU, residue) count table.

    Parameters
    ----------
    inn_summary:
        The long-form mutation DataFrame produced by
        ``INNStructureCollection.collapse_to_vh_vl_binder_info()`` (or the
        equivalent inn_summary sheet from the analysis workbook).
    mutation_types:
        Which ``mutation_type`` values to include.
        Defaults to ``["substitution"]``.

    Returns
    -------
    DataFrame with one row per unique (gene, allele, gene_species, EU, residue,
    mutation_type) combination.  Key columns:

    ``residue``
        The mutated amino-acid (single letter).
    ``germline_seq``
        The germline amino-acid at that position.
    ``count``
        Number of INNs carrying this exact mutation.
    ``inn``
        Comma-joined list of INN names that carry the mutation.
    ``mutation_string_pe``
        Concise ``<germline><EU><residue>`` label (e.g. ``"S239D"``).
    """
    if mutation_types is None:
        mutation_types = ["substitution"]

    if prefered_col_order is not None:
        col_order = prefered_col_order
    else:
        col_order = [
            "gene_species",
            "gene",
            "allele",
            "unique_inn_count",
            "total_occurrences",
            "mutation_type",
            "region",
            "germline_seq",
            "residue",
            "EU",
            "KABAT",
            "EXON",
            "IMGT",
            "region",
            "inn",
        ]

    df_gene = inn_summary.reset_index()
    df_gene["gene"] = df_gene["gene"].str.upper()

    mutation_residues = (
        df_gene.dropna(subset=["mutated_seq"])
        .query("mutation_type.isin(@mutation_types)")
        .groupby(
            [
                "EU",
                "KABAT",
                "EXON",
                "IMGT",
                "mutated_seq",
                "germline_seq",
                "region",
                "mutation_type",
            ]
            + group_level,
            dropna=False,
        )
        .agg({"inn": list, "inn_id": "size"})
        .reset_index()
        .rename(
            columns={
                "inn_id": "total_occurrences",
                "mutated_seq": "residue",
            }
        )
    )

    mutation_residues["mutation_string_pe"] = mutation_residues.apply(
        _build_mutation_string_pe, axis=1
    )
    mutation_residues["inn"] = mutation_residues["inn"].apply(
        lambda x: sorted(set(x))
    )
    mutation_residues["unique_inn_count"] = mutation_residues["inn"].apply(
        len
    )
    _imgt = pd.to_numeric(mutation_residues["IMGT"], errors="coerce")
    mutation_residues["IMGT"] = pd.array(
        [pd.NA if pd.isna(v) else int(round(v)) for v in _imgt],
        dtype="Int64",
    )
    mutation_residues = mutation_residues.reset_index()
    skip_idx = ["mutation_string_pe"] + group_level
    col_order_dedup = []
    for c in col_order:
        if (
            c in skip_idx
        ):  # skip mutation_string_pe AND group_level (they become index)
            continue
        if c not in col_order_dedup:
            col_order_dedup.append(c)

    other_columns = [
        c
        for c in mutation_residues.columns
        if c not in col_order_dedup and c not in skip_idx
    ]

    result = mutation_residues.set_index(skip_idx).sort_values(
        by="unique_inn_count", ascending=False
    )
    # Only select columns that are actually present in the DataFrame
    available = set(result.columns)
    final_cols = [
        c for c in col_order_dedup + other_columns if c in available
    ]
    return result[final_cols]


def mutation_repetitions_per_inn(
    inn_summary: pd.DataFrame,
    group_level: list[str],
    mutation_types: list[str] | None = None,
    prefered_col_order: list[str] | None = None,
) -> pd.DataFrame:

    if mutation_types is None:
        mutation_types = ["substitution"]

    if prefered_col_order is not None:
        col_order = prefered_col_order
    else:
        col_order = [
            "gene_species",
            "gene",
            "allele",
            "occurrences_per_inn",
            "mutation_type",
            "region",
            "germline_seq",
            "residue",
            "EU",
            "KABAT",
            "EXON",
            "IMGT",
            "region",
        ]

    df_gene = inn_summary.reset_index()
    df_gene["gene"] = df_gene["gene"].str.upper()

    mutation_residues = (
        df_gene.dropna(subset=["mutated_seq"])
        .query("mutation_type.isin(@mutation_types)")
        .groupby(
            [
                "EU",
                "KABAT",
                "EXON",
                "IMGT",
                "mutated_seq",
                "germline_seq",
                "region",
                "inn",
                "mutation_type",
            ]
            + group_level,
            dropna=False,
        )
        .agg({"inn_id": "size"})
        .reset_index()
        .rename(
            columns={
                "inn_id": "occurrences_per_inn",
                "mutated_seq": "residue",
            }
        )
    )

    mutation_residues["mutation_string_pe"] = mutation_residues.apply(
        _build_mutation_string_pe, axis=1
    )

    _imgt = pd.to_numeric(mutation_residues["IMGT"], errors="coerce")
    mutation_residues["IMGT"] = pd.array(
        [pd.NA if pd.isna(v) else int(round(v)) for v in _imgt],
        dtype="Int64",
    )

    mutation_residues = mutation_residues.reset_index()

    skip_idx = ["inn", "mutation_string_pe"] + group_level
    # Collect ordered columns that are NOT in the index
    col_order_dedup = []
    for c in col_order:
        if c in skip_idx:
            continue
        if c not in col_order_dedup:
            col_order_dedup.append(c)

    other_columns = [
        c
        for c in mutation_residues.columns
        if c not in col_order_dedup and c not in skip_idx
    ]

    result = mutation_residues.set_index(skip_idx).sort_index()
    available = set(result.columns)
    final_cols = [
        c for c in col_order_dedup + other_columns if c in available
    ]
    return result[final_cols]


# ---------------------------------------------------------------------------
# Population N table — correct denominator per gene × species
# ---------------------------------------------------------------------------


def build_gene_species_n_table(
    motifs_df: "pd.DataFrame",
    group_by: str = "isotype",
) -> "pd.DataFrame":
    """Build the correct denominator (N) per isotype × species for frequency stats.

    Groups by ``isotype`` (default) rather than raw ``gene`` to avoid double-
    counting INNs that carry mutations in multiple genes within the same isotype
    class (e.g. IGHG1 + IGHG4 → both map to "IGHG", counted once per INN).

    Parameters
    ----------
    motifs_df:
        ``ds.constant_chain_motifs_by_inn_region`` — the loader has already
        applied uppercase gene names, isotype, and species_main columns.
    group_by:
        Column to group by alongside ``gene_species``.  Default ``"isotype"``
        (first 4 chars of gene, e.g. ``"IGHG"``, ``"IGKC"``, ``"IGK/IGL"``).
        Pass ``"gene"`` to get per-gene counts (note: causes double-counting
        for INNs with mutations in multiple genes of the same class).

    Returns
    -------
    pd.DataFrame
        Index: (isotype [or gene], gene_species).
        Columns:
            n_with_mutation — unique INNs with ≥1 mutation in this
                              isotype × species group (no double-counting
                              within the same isotype class).
        Sorted descending by n_with_mutation.

    Examples
    --------
    ::

        n_table = build_gene_species_n_table(ds.constant_chain_motifs_by_inn_region)
        # n_with_mutation for IGHG / human = unique INNs with any IgG mutation
        n_table.loc[("IGHG", "human")]

        # Join onto per-motif stats to compute frequency
        per_motif = build_motif_stats_per_motif(
            build_motif_stats_per_inn(ds.constant_chain_motifs_by_inn_region)
        )
        per_motif = per_motif.reset_index().merge(
            n_table.reset_index(),
            on=["isotype", "gene_species"],
            how="left",
        )
        per_motif["frequency"] = per_motif["inn_count"] / per_motif["n_with_mutation"]
    """
    if group_by not in motifs_df.columns:
        raise ValueError(
            f"Column '{group_by}' not found in motifs_df. "
            f"Available: {list(motifs_df.columns)}"
        )

    n_mut = (
        motifs_df.groupby([group_by, "gene_species"], dropna=False)["inn"]
        .nunique()
        .rename("n_with_mutation")
        .reset_index()
        .set_index([group_by, "gene_species"])
    )

    return n_mut.sort_values("n_with_mutation", ascending=False)


# ---------------------------------------------------------------------------
# Motif-level aggregation helpers
# ---------------------------------------------------------------------------

_MOTIF_LIST_COLS = ("EU", "mutation_type", "germline_seq", "mutated_seq")


def build_motif_stats_per_inn(
    motifs_df: pd.DataFrame,
    extra_groupby_keys: list[str] | None = None,
) -> pd.DataFrame:
    """Aggregate ``constant_chain_motifs_by_inn_region`` to one row per
    (INN × region_classification × chain_type_h_or_l × motif × isotype
    [× any extra keys]).

    Parameters
    ----------
    motifs_df:
        ``ds.constant_chain_motifs_by_inn_region`` — all postprocessing
        (list parsing, num_mutations cast, chain_type_h_or_l, isotype) is
        already applied by the loader; no manual pre-processing needed.
    extra_groupby_keys:
        Additional column names to include in the groupby, e.g.
        ``["gene_species"]``.  Any key not present in *motifs_df* is
        silently ignored.

    Returns
    -------
    pd.DataFrame
        Index: (inn, region_classification, chain_type_h_or_l,
                constant_chain_mutational_motif, isotype[, extra_keys…]).
        Columns:
            chain_id, chain_type, gene, gene_species — comma-joined sets
            EU, mutation_type, germline_seq, mutated_seq — comma-joined (first)
            num_mutations — first occurrence
            occurrences_per_inn — number of chain-region rows for this INN/motif
    """
    base_keys = [
        "inn",
        "chain_type_h_or_l",
        "constant_chain_mutational_motif",
        "isotype",
    ]

    groupby_keys = base_keys + (extra_groupby_keys or [])
    aggregators = {
        "chain_id": lambda x: ",".join(sorted(set(x))),
        "gene": lambda x: ",".join(sorted(set(x))),
        "gene_species": lambda x: ",".join(sorted(set(x))),
        "EU": "first",
        "mutation_type": "first",
        "germline_seq": "first",
        "mutated_seq": "first",
        "num_mutations": "first",
        "inn": lambda x: len(x),
    }

    aggregators.update(
        {"region_classification": lambda x: ",".join(sorted(set(x)))}
    )
    per_inn = (
        motifs_df.groupby(groupby_keys, dropna=False)
        .agg(aggregators)
        .sort_index()
        .rename(columns={"inn": "occurrences_per_inn"})
    )

    # Flatten list-valued columns to comma-joined strings
    for col in _MOTIF_LIST_COLS:
        if col in per_inn.columns:
            per_inn[col] = per_inn[col].apply(
                lambda v: (
                    ",".join(str(y) for y in v)
                    if isinstance(v, list)
                    else v
                )
            )

    return per_inn


def build_motif_stats_per_motif(
    motifs_df: pd.DataFrame,
    extra_groupby_keys: list[str] | None = None,
) -> pd.DataFrame:
    """Collapse ``build_motif_stats_per_inn`` output to one row per
    (motif × region_classification × chain_type_h_or_l × isotype
    [× extra_keys…]) across all INNs.

    Parameters
    ----------
    motif_stats_per_inn:
        Output of :func:`build_motif_stats_per_inn`.
    extra_groupby_keys:
        Additional index-level names to retain when collapsing to motif level,
        e.g. ``["gene_species"]``.  Must already be present in the index of
        *motif_stats_per_inn*.  Any key not found is silently ignored.

    Returns
    -------
    pd.DataFrame
        Index: (constant_chain_mutational_motif, region_classification,
                chain_type_h_or_l, isotype[, extra_keys…]).
        Columns: all per-inn columns (first occurrence) plus
            inn         — comma-joined INN names carrying this motif
            total_count — sum of occurrences_per_inn across all INNs
            inn_count   — number of distinct INNs carrying this motif
        Sorted descending by inn_count.
    """

    motif_stats_per_inn = build_motif_stats_per_inn(
        motifs_df, extra_groupby_keys
    )

    col_names = motif_stats_per_inn.columns.tolist()

    base_motif_keys = [
        "constant_chain_mutational_motif",
        "chain_type_h_or_l",
        "isotype",
    ]
    motif_level_keys = sorted(
        set(base_motif_keys + (extra_groupby_keys or []))
    )

    per_motif = (
        motif_stats_per_inn.reset_index(["inn"])
        .groupby(level=motif_level_keys, dropna=False)
        .agg(
            {
                **{col: "first" for col in col_names},
                "inn": list,
                "occurrences_per_inn": "sum",
            }
        )
        .rename(columns={"occurrences_per_inn": "total_count"})
    )
    per_motif["inn_count"] = per_motif["inn"].apply(len)
    per_motif["inn"] = per_motif["inn"].apply(lambda x: ",".join(x))

    return per_motif.sort_values("inn_count", ascending=False)
