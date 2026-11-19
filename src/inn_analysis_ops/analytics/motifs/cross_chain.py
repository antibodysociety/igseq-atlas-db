"""Cross-chain motif pairing lookup for bubble-chart bracket connectors.

Determines which motifs in a plotted motif table should be drawn as
cross-chain-paired (dashed-line bracket connectors in
``plot_motif_cluster_bubbles``), using ``result.cross_chain_pairing_motifs``
(long-format table: one row per ``(inn, mutation_string_pe, chain_id)``,
restricted to ``bispecific_ind_branch`` rows — see
:func:`analytics.constant_chain_analysis._build_cross_chain_pairing_motifs`).

Pipeline (mirrors the reference notebook logic exactly):

1. :func:`assign_motif_chain_ids` — for each *motif_df* row, flatten its
   ``unique_inn_chains`` dict into a plain list of chain_ids
   (``motif_chain``).
2. :func:`assign_paired_ref_ids` — for each row whose ``mutation_string_pe``
   is a known cross-chain arm motif, keep only the ``motif_chain`` entries
   that are also present in ``cross_chain_pairing_motifs["chain_id"]``
   (``paired_ref_id``).
3. :func:`collapse_super_string_motifs` — within one INN's group of
   arm-paired rows, drop any motif whose token set is a subset of a
   longer motif already kept (keeps only "maximal" motifs per INN).
4. :func:`build_cross_chain_groups` — group the collapsed per-INN motif
   tuples, count occurrences across INNs, and assign each unique
   multi-motif tuple a stable ``idx`` string (``":"``-joined sorted
   motifs) — this ``idx`` is the shared identifier used to link rows in
   the plotted report table and draw dashed cross-chain connectors
   (row-to-row, seed-to-row, seed-to-seed).
5. :func:`explode_cross_chain_groups` / :func:`build_cross_chain_pair_map`
   — split each group's ``idx`` back into individual motifs and explode,
   producing a ``motif -> idx`` join table.  Merge a report/plotted
   table's ``motif`` column against this table's ``motif`` column to
   attach the group ``idx``; rows sharing the same ``idx`` are
   cross-chain paired.
"""

from __future__ import annotations

import logging

import pandas as pd

_log = logging.getLogger(__name__)


def assign_motif_chain_ids(
    motif_df: pd.DataFrame,
    unique_inn_chains_col: str = "unique_inn_chains",
) -> pd.DataFrame:
    """Flatten ``unique_inn_chains`` (dict of inn -> chain_ids) into a flat list.

    Parameters
    ----------
    motif_df :
        Plotted motif table.  Must have *unique_inn_chains_col*.
    unique_inn_chains_col :
        Column name override.

    Returns
    -------
    pd.DataFrame
        Copy of *motif_df* with an added ``motif_chain`` column
        (``list[str]`` or ``None``).
    """
    out = motif_df.copy()
    out["motif_chain"] = out[unique_inn_chains_col].apply(
        lambda x: (
            [z for _, chains in x.items() for z in chains]
            if not pd.isnull(x)
            else None
        )
    )
    return out


def assign_paired_ref_ids(
    motif_df: pd.DataFrame,
    cross_chain_pairing_motifs: pd.DataFrame,
    mutation_col: str = "mutation_string_pe",
    motif_chain_col: str = "motif_chain",
    chain_id_col: str = "chain_id",
) -> pd.DataFrame:
    """Restrict each row's ``motif_chain`` to chain_ids known to be cross-chain arms.

    Parameters
    ----------
    motif_df :
        Plotted motif table.  Must have *mutation_col* and *motif_chain_col*
        (see :func:`assign_motif_chain_ids`).
    cross_chain_pairing_motifs :
        ``result.cross_chain_pairing_motifs``.  Must have *chain_id_col*
        and *mutation_col*.
    mutation_col, motif_chain_col, chain_id_col :
        Column name overrides.

    Returns
    -------
    pd.DataFrame
        Copy of *motif_df* with an added ``paired_ref_id`` column
        (sorted ``list[str]`` of matching chain_ids, or ``None`` if the
        row's motif isn't a known cross-chain arm or has no
        ``motif_chain``).
    """
    out = motif_df.copy()
    if (
        not isinstance(cross_chain_pairing_motifs, pd.DataFrame)
        or cross_chain_pairing_motifs.empty
    ):
        out["paired_ref_id"] = None
        return out

    check_against = list(cross_chain_pairing_motifs[chain_id_col].values)
    motifs_check_against = list(cross_chain_pairing_motifs[mutation_col].values)

    def _paired_ref_id(row: pd.Series):
        motif_chain = row[motif_chain_col]
        if not isinstance(motif_chain, list) or row[mutation_col] not in motifs_check_against:
            return None
        return sorted(set(z for z in motif_chain if z in check_against))

    out["paired_ref_id"] = out[[mutation_col, motif_chain_col]].apply(
        _paired_ref_id, axis=1
    )
    return out


def collapse_super_string_motifs(
    group: pd.DataFrame,
    mutation_col: str = "mutation_string_pe",
) -> tuple[str, ...]:
    """Within one INN's group, keep only "maximal" motifs (drop token subsets).

    Sorts by descending token count and keeps a motif only if its token
    set is not a subset of an already-kept (longer or equal) motif's
    token set.

    Parameters
    ----------
    group :
        Rows for a single INN (already exploded/filtered to those with a
        non-null ``paired_ref_id``).  Must have *mutation_col*.
    mutation_col :
        Column name override.

    Returns
    -------
    tuple[str, ...]
        Sorted tuple of the surviving (maximal) ``mutation_string_pe``
        values for this INN.
    """
    group = group.copy()
    group["mut_len"] = group[mutation_col].apply(lambda x: x.count("/"))
    group = group.sort_values(by="mut_len", ascending=False)

    collapsed: dict[str, set[str]] = {}
    for mtf in group[mutation_col]:
        mtf_set = set(mtf.split("/"))
        ref = None
        for cluster, cluster_set in collapsed.items():
            if mtf_set.issubset(cluster_set):
                ref = cluster
                break
        if ref is None:
            collapsed[mtf] = mtf_set

    return tuple(sorted(collapsed.keys()))


def build_cross_chain_groups(
    motif_df: pd.DataFrame,
    cross_chain_pairing_motifs: pd.DataFrame,
    mutation_col: str = "mutation_string_pe",
    unique_inn_chains_col: str = "unique_inn_chains",
    chain_id_col: str = "chain_id",
) -> pd.DataFrame:
    """Derive the cross-chain motif groups (and their shared ``idx``) from *motif_df*.

    Combines :func:`assign_motif_chain_ids`, :func:`assign_paired_ref_ids`,
    and :func:`collapse_super_string_motifs` end-to-end, then counts how
    many INNs produced each unique multi-motif tuple and assigns each one
    a stable, ``":"``-joined ``idx`` string.

    Parameters
    ----------
    motif_df :
        Plotted motif table.  Must have *mutation_col* and
        *unique_inn_chains_col*.
    cross_chain_pairing_motifs :
        ``result.cross_chain_pairing_motifs``.
    mutation_col, unique_inn_chains_col, chain_id_col :
        Column name overrides.

    Returns
    -------
    pd.DataFrame
        Indexed by the multi-motif tuple, with columns ``count`` (number
        of INNs producing that exact tuple) and ``idx`` (``":"``-joined
        sorted motif strings — the shared identifier to use for linking
        rows in the report table).  Only tuples with 2+ motifs are kept
        (single-motif "groups" can't be paired).
    """
    annotated = assign_motif_chain_ids(motif_df, unique_inn_chains_col)
    annotated = assign_paired_ref_ids(
        annotated,
        cross_chain_pairing_motifs,
        mutation_col=mutation_col,
        chain_id_col=chain_id_col,
    )

    motif_chain_paired = annotated.explode("paired_ref_id").dropna(
        subset="paired_ref_id"
    )
    if motif_chain_paired.empty:
        return pd.DataFrame(columns=["count", "idx"])

    motif_chain_paired = motif_chain_paired.copy()
    motif_chain_paired["inn_id"] = motif_chain_paired["paired_ref_id"].apply(
        lambda x: x.split("_")[0]
    )

    pairs = motif_chain_paired.groupby(by="inn_id").apply(
        lambda g: collapse_super_string_motifs(g, mutation_col=mutation_col)
    )
    pairs = pairs[pairs.apply(len) > 1]

    if pairs.empty:
        return pd.DataFrame(columns=["count", "idx"])

    cross_chains = pd.DataFrame(pairs.value_counts())
    cross_chains["idx"] = list(cross_chains.index.map(lambda x: ":".join(x)))

    _log.info(
        "build_cross_chain_groups: %d unique cross-chain motif group(s) found "
        "across %d INN(s) with paired arm motifs.",
        len(cross_chains),
        motif_chain_paired["inn_id"].nunique(),
    )

    return cross_chains


def explode_cross_chain_groups(
    cross_chain_groups: pd.DataFrame,
) -> pd.DataFrame:
    """Explode ``build_cross_chain_groups`` output into one row per motif.

    Splits each group's ``idx`` (``":"``-joined motif strings) back apart
    and explodes it, so every individual motif gets its own row carrying
    the ``idx`` of the group it belongs to.  This ``idx`` is the join key
    to use against a report/plotted table's ``motif`` column — rows
    sharing the same ``idx`` are cross-chain paired.

    Parameters
    ----------
    cross_chain_groups :
        Output of :func:`build_cross_chain_groups` — indexed by motif
        tuple, with an ``idx`` column.

    Returns
    -------
    pd.DataFrame
        Columns: ``motif`` (str), ``idx`` (str), ``count`` (int, carried
        over from *cross_chain_groups*).  One row per ``(idx, motif)``
        pair.
    """
    if cross_chain_groups.empty or "idx" not in cross_chain_groups.columns:
        return pd.DataFrame(columns=["motif", "idx", "count"])

    out = cross_chain_groups.copy()
    out["motif"] = out["idx"].apply(lambda x: x.split(":"))
    out = out.explode("motif").reset_index(drop=True)
    return out[["motif", "idx", "count"]]


def build_cross_chain_pair_map(
    motif_df: pd.DataFrame,
    cross_chain_pairing_motifs: pd.DataFrame,
    mutation_col: str = "mutation_string_pe",
    unique_inn_chains_col: str = "unique_inn_chains",
    chain_id_col: str = "chain_id",
) -> pd.DataFrame:
    """Build the ``motif -> idx`` join table for report-table cross-chain linking.

    Combines :func:`build_cross_chain_groups` and
    :func:`explode_cross_chain_groups`: every motif that belongs to a
    cross-chain group gets a row with that group's shared ``idx`` — join
    a report/plotted table's ``motif`` column onto this table's ``motif``
    column to attach the group ``idx``.  Rows (including seeds, combo
    rows, etc.) that end up sharing the same ``idx`` are cross-chain
    paired and should be linked by a dashed connector.

    Parameters
    ----------
    motif_df, cross_chain_pairing_motifs, mutation_col,
    unique_inn_chains_col, chain_id_col :
        Passed through to :func:`build_cross_chain_groups`.

    Returns
    -------
    pd.DataFrame
        Columns: ``motif``, ``idx``, ``count`` — see
        :func:`explode_cross_chain_groups`.
    """
    groups = build_cross_chain_groups(
        motif_df,
        cross_chain_pairing_motifs,
        mutation_col=mutation_col,
        unique_inn_chains_col=unique_inn_chains_col,
        chain_id_col=chain_id_col,
    )
    return explode_cross_chain_groups(groups)
