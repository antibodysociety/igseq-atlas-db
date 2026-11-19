"""INN subtraction helpers for the motif pipeline.

Functions
---------
_add_raw_sub_motifs         Add ``raw_sub_motifs`` column to a motif DataFrame.
_subtract_inns_inline       Run INN subtraction in-place and emit adjusted_inns.
recompute_adjusted_counts   Public helper — recompute adjusted_count after
                            filtering a motif DataFrame (e.g. by inn_count).
"""

from __future__ import annotations

from typing import Optional

import pandas as pd

from .audit import _audit_set


def _add_raw_sub_motifs(df: pd.DataFrame) -> pd.DataFrame:
    """Add ``raw_sub_motifs`` column — unpruned list of all size-≥-2 sub-motifs.

    For each row, scans all other rows in *df* for frozensets that are strict
    subsets of this row's ``motif``.  Unlike ``sub_motifs`` (which is pruned to
    exclude consumed motifs), this column preserves the full ancestry chain and
    is used for INN subtraction.
    """
    motif_set = set(df["motif"])
    motif_to_str = dict(zip(df["motif"], df["mutation_string"]))

    def _raw(m: frozenset) -> list[str]:
        return [
            motif_to_str[s]
            for s in motif_set
            if len(s) >= 2 and s < m  # strict subset
        ]

    df = df.copy()
    df["raw_sub_motifs"] = df["motif"].apply(_raw)
    return df


def _subtract_inns_inline(
    df: pd.DataFrame,
    ancestry_col: str = "raw_sub_motifs",
    id_col: str = "motif_id",
) -> pd.DataFrame:
    """Run INN subtraction in-place on *df* using *ancestry_col* for ancestry.

    Identical logic to :func:`subtract_parent_inns` but uses ``raw_sub_motifs``
    (unpruned) rather than ``sub_motifs`` (pruned), ensuring consumed sub-motifs
    are correctly attributed.  Returns a copy with added
    ``adjusted_inns``, ``adjusted_count``, ``adjusted_freq`` columns and
    ``is_fully_described_by_larger_motif = (adjusted_count == 0)``.
    Rows with ``adjusted_count == 0`` are **not** dropped here; call
    ``pruned_df = df[df["adjusted_count"] > 0]`` to prune.
    """
    _GROUP_COLS = ["gene", "chain_type", "gene_species"]
    df = (
        df.copy()
        .sort_values("num_mutations", ascending=False)
        .reset_index(drop=True)
    )

    if not df[id_col].is_unique:
        raise ValueError(
            f"_subtract_inns_inline: '{id_col}' contains duplicate values."
        )

    group_cols = [c for c in _GROUP_COLS if c in df.columns]

    def _gkey(row) -> tuple:
        return tuple(row[c] for c in group_cols)

    mutation_col = "mutation_string"
    str_to_id: dict[tuple, int] = {}
    id_to_ms: dict[int, str] = {}
    for _, row in df.iterrows():
        str_to_id[(row[mutation_col], _gkey(row))] = row[id_col]
        id_to_ms[row[id_col]] = row.get(mutation_col, str(row[id_col]))

    current: dict[int, set] = {
        row[id_col]: set(row["inns"]) for _, row in df.iterrows()
    }
    # Audit: initial INN assignments
    for mid, inns in current.items():
        ms = id_to_ms.get(mid, str(mid))
        _audit_set(
            inns,
            "_subtract_inns_inline",
            ms,
            "ASSIGNED",
            "initial INN set from collapsed data",
        )

    for _, row in df.iterrows():
        mid = row[id_col]
        gkey = _gkey(row)
        substrings = row[ancestry_col]
        if not substrings or not isinstance(substrings, list):
            continue
        for substr in substrings:
            parent_id = str_to_id.get((substr, gkey))
            if parent_id is None or parent_id not in current:
                continue
            subtracted = current[parent_id] & current[mid]
            current[parent_id] = current[parent_id] - current[mid]
            # Audit: INN subtraction
            if subtracted:
                parent_ms = id_to_ms.get(parent_id, str(parent_id))
                child_ms = id_to_ms.get(mid, str(mid))
                _audit_set(
                    subtracted,
                    "_subtract_inns_inline",
                    parent_ms,
                    "SUBTRACTED",
                    f"absorbed by sub-motif '{child_ms}'",
                )
                _audit_set(
                    subtracted,
                    "_subtract_inns_inline",
                    child_ms,
                    "RESPONSIBLE_FOR_SUBTRACTION",
                    f"caused removal from parent '{parent_ms}'",
                )

    adj_inns = df[id_col].map(current)
    df["adjusted_inns"] = adj_inns
    df["adjusted_count"] = adj_inns.apply(
        lambda s: len(s) if isinstance(s, set) else 0
    )
    df["adjusted_freq"] = df.apply(
        lambda r: (
            round(r["adjusted_count"] / r["inn_total"], 5)
            if r.get("inn_total") and r["inn_total"] > 0
            else None
        ),
        axis=1,
    )
    df["is_fully_described_by_larger_motif"] = df["adjusted_count"] == 0
    return df


def recompute_adjusted_counts(
    df: pd.DataFrame,
    destination_cols: tuple[str, str] = ("adjusted_count", "adjusted_inns"),
    prev_col_prefix: str | None = None,
    mutation_col: str = "mutation_string_pe",
) -> pd.DataFrame:
    """Recompute adjusted INN counts after filtering a motif DataFrame.

    When rows are dropped (e.g. ``df.query("inn_count >= @n")``), the
    ``adjusted_count`` values become stale: sub-motifs may reclaim INNs that
    were previously consumed by now-absent super-motifs.  This function
    rebuilds ``raw_sub_motifs`` from scratch against the surviving rows and
    re-runs INN subtraction to produce correct counts.

    Parameters
    ----------
    df:
        Filtered motif DataFrame.  Must contain ``inns`` (set of INN strings)
        and ``mutation_col``.  The ``motif`` (frozenset) and ``num_mutations``
        columns are reconstructed from ``mutation_col`` if absent.
    destination_cols:
        ``(count_col, inns_col)`` tuple naming the output columns for the
        recomputed count and INN set.  Defaults to
        ``("adjusted_count", "adjusted_inns")`` (overwrites in-place).  Set to
        different names (e.g. ``("adjusted_count_trimmed", "adjusted_inns_trimmed")``)
        to write alongside the originals without clobbering them.
    prev_col_prefix:
        If given, the pre-recomputation ``adjusted_count`` and ``adjusted_inns``
        are snapshotted under ``"{prev_col_prefix}_adjusted_count"`` and
        ``"{prev_col_prefix}_adjusted_inns"`` respectively before overwriting.
        Useful for comparing old vs new values, e.g. ``prev_col_prefix="pre"``.
    mutation_col:
        Column containing the human-readable mutation string used as the motif
        key.  Defaults to ``"mutation_string_pe"`` (the name used in the merged
        plotting table); pass ``"mutation_string"`` when working directly with
        ``motif_df_clustered_post_filter``.

    Returns
    -------
    pd.DataFrame
        Copy of *df* with refreshed ``adjusted_inns``, *destination_cols*,
        ``adjusted_freq``, and ``is_fully_described_by_larger_motif`` columns.
        Singletons (motif size == 1) are exempt from subtraction and always
        receive ``adjusted_count == inn_count``.
    """
    dest_count_col, dest_inns_col = destination_cols
    df = df.reset_index(drop=True).copy()

    # ── Preserve pre-recomputation values if requested ────────────────────────
    if prev_col_prefix is not None:
        if "adjusted_count" in df.columns:
            df[f"{prev_col_prefix}_adjusted_count"] = df["adjusted_count"]
        if "adjusted_inns" in df.columns:
            df[f"{prev_col_prefix}_adjusted_inns"] = df["adjusted_inns"]

    # ── Reconstruct motif frozenset if absent (e.g. after merge/rename) ───────
    if "motif" not in df.columns:
        df["motif"] = df[mutation_col].apply(
            lambda s: frozenset(t.strip() for t in str(s).split("/") if t.strip())
        )

    # ── Reconstruct num_mutations if absent ───────────────────────────────────
    if "num_mutations" not in df.columns:
        size_col = next(
            (c for c in ("motif_length", "size") if c in df.columns), None
        )
        if size_col is not None:
            df["num_mutations"] = df[size_col]
        else:
            df["num_mutations"] = df["motif"].apply(len)

    # ── Normalise the mutation key to "mutation_string" for _subtract_inns_inline
    _tmp_rename = mutation_col != "mutation_string" and "mutation_string" not in df.columns
    if _tmp_rename:
        df = df.rename(columns={mutation_col: "mutation_string"})

    # ── Fresh motif_id keyed on the trimmed index ─────────────────────────────
    df["motif_id"] = df.index

    # ── Split singletons out — they don't participate in subtraction ──────────
    singleton_mask = df["num_mutations"] <= 1
    motif_rows = df[~singleton_mask].copy()
    single_rows = df[singleton_mask].copy()

    # ── Recompute ancestry and subtraction on multi-mutation motifs ───────────
    if not motif_rows.empty:
        motif_rows = _add_raw_sub_motifs(motif_rows)
        motif_rows = _subtract_inns_inline(motif_rows, ancestry_col="raw_sub_motifs")

    # ── Singletons: adjusted = inn_count minus INNs absorbed by surviving composites ──
    # Without this step, singletons keep their full inn_count even when composites
    # that contain them already explain (and have claimed) a subset of their INNs.
    if not single_rows.empty:
        # Build a set of all INNs that surviving composites claimed after subtraction
        _absorbed_by_composites: dict[str, set] = {}  # singleton mutation token → absorbed INN set
        if not motif_rows.empty and "adjusted_inns" in motif_rows.columns:
            for _, crow in motif_rows.iterrows():
                cadj = crow.get("adjusted_inns")
                if not isinstance(cadj, set) or not cadj:
                    continue
                # Each token of this composite claims those INNs from the singleton
                ms_str = crow.get("mutation_string", crow.get(mutation_col, ""))
                for token in (t.strip() for t in str(ms_str).split("/") if t.strip()):
                    _absorbed_by_composites.setdefault(token, set()).update(cadj)

        def _singleton_adjusted(row) -> tuple[set, int]:
            ms_str = row.get("mutation_string", row.get(mutation_col, ""))
            token = str(ms_str).strip()
            raw_inns = row.get("inns")
            if not isinstance(raw_inns, (set, frozenset)):
                raw_inns = set()
            absorbed = _absorbed_by_composites.get(token, set())
            adj_inns = set(raw_inns) - absorbed
            return adj_inns, len(adj_inns)

        _adj = single_rows.apply(_singleton_adjusted, axis=1)
        single_rows["adjusted_inns"] = _adj.apply(lambda t: t[0])
        single_rows["adjusted_count"] = _adj.apply(lambda t: t[1])
        single_rows["adjusted_freq"] = single_rows.apply(
            lambda r: (
                round(r["adjusted_count"] / r["inn_total"], 5)
                if r.get("inn_total") and r["inn_total"] > 0
                else None
            ),
            axis=1,
        )
        single_rows["is_fully_described_by_larger_motif"] = single_rows["adjusted_count"] == 0

    df = pd.concat([motif_rows, single_rows]).sort_index()

    # ── Write to destination columns (rename from internal names if needed) ───
    if dest_count_col != "adjusted_count":
        df[dest_count_col] = df["adjusted_count"]
    if dest_inns_col != "adjusted_inns":
        df[dest_inns_col] = df["adjusted_inns"]

    # ── Undo temporary rename ─────────────────────────────────────────────────
    if _tmp_rename:
        df = df.rename(columns={"mutation_string": mutation_col})

    return df
