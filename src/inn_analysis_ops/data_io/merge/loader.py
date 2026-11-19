"""
Loader for the consolidated INN analysis Excel export (``inn_analysis_updated.xlsx``).

Reads all worksheets in a single call and returns an :class:`INNAnalysisDataset`
that exposes each sheet as a named attribute.  Where a Pydantic model exists for the
underlying data (``kept_inn``, ``removed_inn``, ``inn_ab_sequence_info``,
``IMGT Info Per Chain``, ``IndividualMutationProfile``) the rows are reconstructed
into the matching collection objects so the data is available in both DataFrame form
and as typed Pydantic collections.  Analytics-only sheets (constant-chain pivot,
motif tables) are exposed as plain DataFrames with appropriate post-processing.

Usage
-----
::

    from inn_analysis_ops.data_io.merge.loader import load_inn_analysis, build_mutation_fixer

    ds = load_inn_analysis("notebooks/db_sources/inn_analysis_updated.xlsx")
    print(ds.stats)

    # DataFrame view — for plotting
    ds.kept_inn.head()
    ds.individual_mutation_profile.reset_index().query('gene_species == "human"')

    # Pydantic collections — for programmatic access
    ds.inn_collection.records[0].isotype
    ds.mutation_collection.to_dataframe()

Optionally apply insertion-site germline correction.
Build a fixer from a pre-constructed ``ConstantDB`` and
``ConstantChainNumbering``::

    from inn_analysis_ops.data_io.merge.loader import build_mutation_fixer

    fixer = build_mutation_fixer(cdb, cn)
    ds = load_inn_analysis(source, mutation_fixer=fixer)
"""

from __future__ import annotations

import ast
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import pandas as pd

from inn_analysis_ops.models.enum_wrapper import StrEnumBase
from inn_analysis_ops.models.inn_structure.analytics import (
    IndividualMutationCollection,
    IndividualMutationInfo,
    ObservedFullLengthChain,
    ObservedFullLengthCollection,
)
from inn_analysis_ops.models.inn_structure.wt_lookup import (
    get_germline_at_insertion_pos,
)
from .models import (
    INN_COLUMN_SOURCES,
    SEQ_COLUMN_SOURCES,
    ColumnSource,
    MergedINNCollection,
    MergedINNRecord,
    MergedSequenceCollection,
    MergedSequenceRecord,
)

# ---------------------------------------------------------------------------
# Internal constants — shared with the motif post-processing helpers
# ---------------------------------------------------------------------------

_LIST_COLS = frozenset({"EU", "mutation_type", "germline_seq", "mutated_seq"})
_HEAVY_REGIONS = frozenset({"VH", "CH1", "CH2", "CH3", "FC", "CH4", "CHS", "HINGE"})

_CONSTANT_CHAINS_INDEX_COLS = list(range(8))
_MUTATION_PROFILE_INDEX_COLS = list(range(11))


# ---------------------------------------------------------------------------
# Mutation issue taxonomy and fix registry
# ---------------------------------------------------------------------------


class MutationPatternIssue(StrEnumBase):
    """Known data-quality issues in insertion rows of ``IndividualMutationProfile``."""

    DOUBLED_MUTATED_SEQ = "doubled_mutated_seq"
    """Insertion ``mutated_seq`` is the correct sequence repeated twice
    (e.g. ``"FSFS"`` instead of ``"FS"``).  Fixed by :func:`fix_repeat_errors`."""

    MISSING_GERMLINE_SEQ = "missing_germline_seq"
    """``germline_seq`` is absent or ``"-"`` for an insertion row.  Fixed by
    re-deriving it from the germline reference at the anchor position."""

    DUPLICATE_ROWS = "duplicate_rows"
    """The same mutation appears more than once for the same INN / position key.
    Fixed by keeping the first occurrence."""


# ---------------------------------------------------------------------------
# Insertion-correction helpers
# ---------------------------------------------------------------------------


def fix_repeat_errors(seq: str) -> str:
    """Collapse a doubled insertion string to its true half.

    During the original parsing pipeline, insertion ``mutated_seq`` strings
    were sometimes stored as the correct value repeated twice
    (e.g. ``"FSFS"`` instead of ``"FS"``).  This function detects and
    corrects that duplication for even-length sequences.
    """
    if len(seq) % 2 != 0:
        return seq
    half = len(seq) // 2
    s1, s2 = seq[:half], seq[half:]
    if s1 == s2:
        return s1
    return seq


def _fix_doubled_mutated_seq(df: pd.DataFrame, **_) -> pd.DataFrame:
    """Apply :func:`fix_repeat_errors` to all insertion rows."""
    is_ins = df["mutation_type"] == "insertion"
    df.loc[is_ins, "mutated_seq"] = df.loc[is_ins, "mutated_seq"].map(fix_repeat_errors)
    return df


def _fix_missing_germline_seq(df: pd.DataFrame, *, cdb, cn, **_) -> pd.DataFrame:
    """Re-derive ``germline_seq`` for any insertion row where it is empty or NaN."""
    df = df.copy()
    df["germline_seq"] = df["germline_seq"].fillna("").replace("-", "")
    is_ins = df["mutation_type"] == "insertion"
    needs_fix = is_ins & (df["germline_seq"].str.strip() == "")

    if needs_fix.any():
        df.loc[needs_fix, "germline_seq"] = (
            df.loc[needs_fix]
            .reset_index()
            .apply(
                lambda x: get_germline_at_insertion_pos(
                    cdb, cn, x["allele"], x["region"], x["EU"]
                ),
                axis=1,
            )
            .values
        )

    still_empty = is_ins & (df["germline_seq"].fillna("").str.strip() == "")
    assert not still_empty.any(), (
        f"_fix_missing_germline_seq: {still_empty.sum()} insertion row(s) still have "
        f"an empty germline_seq after fix. Check cdb/cn coverage for these positions:\n"
        f"{df.loc[still_empty].reset_index()[['allele', 'region', 'EU']].to_string()}"
    )

    return df


def _fix_duplicate_rows(df: pd.DataFrame, **_) -> pd.DataFrame:
    """De-duplicate by mutation key, keeping the first occurrence."""
    idx_names = df.index.names
    dedup_keys = list(idx_names) + [
        "IMGT",
        "EU",
        "EXON",
        "KABAT",
        "mutation_type",
        "germline_seq",
        "mutated_seq",
    ]
    return (
        df.reset_index()
        .groupby(by=dedup_keys, as_index=False, dropna=False)
        .first()
        .set_index(idx_names)
    )


def _detect_doubled_mutated_seq(df: pd.DataFrame, **_) -> None:
    is_ins = df["mutation_type"] == "insertion"
    doubled = df.loc[is_ins, "mutated_seq"].apply(
        lambda s: (
            isinstance(s, str)
            and len(s) % 2 == 0
            and s[: len(s) // 2] == s[len(s) // 2 :]
        )
    )
    if doubled.any():
        import warnings

        warnings.warn(
            f"MutationPatternIssue.DOUBLED_MUTATED_SEQ: {doubled.sum()} insertion row(s) "
            f"have a doubled mutated_seq (e.g. {df.loc[is_ins][doubled]['mutated_seq'].iloc[0]!r}). "
            f"Add MutationPatternIssue.DOUBLED_MUTATED_SEQ to the fix list to correct.",
            stacklevel=4,
        )


def _detect_missing_germline_seq(df: pd.DataFrame, **_) -> None:
    missing = df["germline_seq"].fillna("").apply(lambda s: not s or s == "-")
    if missing.any():
        import warnings

        warnings.warn(
            f"MutationPatternIssue.MISSING_GERMLINE_SEQ: {missing.sum()} insertion row(s) "
            f"have a missing/empty germline_seq. "
            f"Add MutationPatternIssue.MISSING_GERMLINE_SEQ to the fix list to correct.",
            stacklevel=4,
        )


def _detect_duplicate_rows(df: pd.DataFrame, **_) -> None:
    idx_names = df.index.names
    dedup_keys = list(idx_names) + [
        "IMGT",
        "EU",
        "EXON",
        "KABAT",
        "mutation_type",
        "germline_seq",
        "mutated_seq",
    ]
    flat = df.reset_index()
    dups = flat.duplicated(subset=dedup_keys, keep=False)
    if dups.any():
        import warnings

        warnings.warn(
            f"MutationPatternIssue.DUPLICATE_ROWS: {dups.sum()} row(s) are duplicates by "
            f"mutation key. Add MutationPatternIssue.DUPLICATE_ROWS to the fix list to correct.",
            stacklevel=4,
        )


# Maps each MutationPatternIssue → (fix_fn, detect_fn)
_ISSUE_REGISTRY: dict[
    MutationPatternIssue,
    tuple[Callable, Callable],
] = {
    MutationPatternIssue.DOUBLED_MUTATED_SEQ: (
        _fix_doubled_mutated_seq,
        _detect_doubled_mutated_seq,
    ),
    MutationPatternIssue.MISSING_GERMLINE_SEQ: (
        _fix_missing_germline_seq,
        _detect_missing_germline_seq,
    ),
    MutationPatternIssue.DUPLICATE_ROWS: (
        _fix_duplicate_rows,
        _detect_duplicate_rows,
    ),
}


def build_mutation_fixer(
    cdb,
    cn,
    *,
    fix: list[MutationPatternIssue],
) -> Callable[[pd.DataFrame], pd.DataFrame]:
    """Return a ``mutation_fixer`` callable for the given list of issues.

    Parameters
    ----------
    cdb : ConstantDB — required when MISSING_GERMLINE_SEQ is in *fix*.
    cn  : ConstantChainNumbering — required when MISSING_GERMLINE_SEQ is in *fix*.
    fix : list[MutationPatternIssue]
        Ordered list of issues to correct.  Empty list → detect-only mode.
    """
    _to_fix: list[MutationPatternIssue] = list(fix)
    _to_fix_set = frozenset(_to_fix)
    _kwargs = {"cdb": cdb, "cn": cn}

    def _apply(inn_summary: pd.DataFrame) -> pd.DataFrame:
        df = inn_summary.copy()
        for issue in _to_fix:
            fix_fn, _ = _ISSUE_REGISTRY[issue]
            df = fix_fn(df, **_kwargs)
        for issue, (_, detect_fn) in _ISSUE_REGISTRY.items():
            if issue not in _to_fix_set:
                detect_fn(df, **_kwargs)
        return df

    return _apply


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _nan_to_none(record: dict) -> dict:
    """Replace float NaN and numpy NaN with None in a flat dict."""
    result = {}
    for k, v in record.items():
        if isinstance(v, float) and np.isnan(v):
            result[k] = None
        else:
            result[k] = v
    return result


def _df_to_records(df: pd.DataFrame) -> list[dict]:
    """Convert a DataFrame to a list of NaN-free dicts."""
    return [_nan_to_none(r) for r in df.to_dict("records")]


def _build_collection(
    records: list[dict],
    record_cls,
    collection_cls,
    column_sources: list[dict] | None = None,
) -> object:
    """Reconstruct a Pydantic collection from a list of row dicts.

    Rows are projected to only the fields declared on *record_cls* and
    constructed with ``model_construct`` (no runtime re-validation — data
    originates from our own validated pipeline).  Unknown columns from Excel
    (e.g. computed ``min_year``) are silently dropped.

    Args:
        records: NaN-free list of dicts (from :func:`_df_to_records`).
        record_cls: The Pydantic *BaseModel* subclass for each row.
        collection_cls: The collection *BaseModel* that wraps ``records``.
        column_sources: Raw dicts for ``ColumnSource`` objects (optional).

    Returns:
        An instance of *collection_cls*.
    """
    known = set(record_cls.model_fields.keys())
    # inn_id is declared as str (via InnId alias) but model_construct bypasses
    # validators, so coerce int → str here to silence Pydantic serialization warnings.
    _has_inn_id = "inn_id" in known

    def _coerce_row(row: dict) -> dict:
        filtered = {k: v for k, v in row.items() if k in known}
        if (
            _has_inn_id
            and "inn_id" in filtered
            and not isinstance(filtered["inn_id"], str)
        ):
            filtered["inn_id"] = (
                str(int(float(filtered["inn_id"])))
                if filtered["inn_id"] == filtered["inn_id"]
                else filtered["inn_id"]
            )
        return filtered

    model_records = [record_cls.model_construct(**_coerce_row(row)) for row in records]
    kwargs: dict = {"records": model_records}
    if column_sources is not None:
        kwargs["column_sources"] = [
            ColumnSource.model_construct(**cs) for cs in column_sources
        ]
    return collection_cls.model_construct(**kwargs)


# Columns that should always be uppercase across every sheet
_UPPERCASE_COLS = ("gene", "species", "allele", "gene_species", "isotype")


def _uppercase_categorical(df: pd.DataFrame) -> pd.DataFrame:
    """Uppercase gene, species, allele, gene_species, and isotype columns in-place."""
    for col in _UPPERCASE_COLS:
        if col in df.columns:
            df[col] = df[col].apply(
                lambda x: str(x).upper() if pd.notna(x) and x != "" else x
            )
    return df


_SEQ_COLS = frozenset({"germline_seq", "mutated_seq"})


def _apply_mutation_profile_postprocessing(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """Post-process the ``IndividualMutationProfile`` sheet after loading.

    Fills ``germline_seq`` and ``mutated_seq`` NaN → ``""`` so that downstream
    string formatting (e.g. ``mutation_string_pe`` construction) does not
    produce literal ``"nan"`` prefixes/suffixes.  The columns may be regular
    columns or MultiIndex levels — both are handled.
    """
    df = df.copy()

    # ── Fix regular columns ───────────────────────────────────────────────
    for col in _SEQ_COLS:
        if col in df.columns:
            df[col] = df[col].fillna("").astype(str)

    # ── Fix MultiIndex levels (columns promoted to index on load) ─────────
    if isinstance(df.index, pd.MultiIndex):
        idx_names = list(df.index.names)
        seq_in_idx = [n for n in idx_names if n in _SEQ_COLS]
        if seq_in_idx:
            idx_df = df.index.to_frame(index=False)
            for col in seq_in_idx:
                idx_df[col] = idx_df[col].fillna("").astype(str)
            df.index = pd.MultiIndex.from_frame(idx_df[idx_names])

    return df


def _apply_motif_postprocessing(df: pd.DataFrame) -> pd.DataFrame:
    """Post-process a ``ConstantChainMotifs*`` sheet in-place (copy returned):

    * Parse list-valued columns (EU, mutation_type, germline_seq, mutated_seq).
    * Cast ``num_mutations`` to int.
    * Add ``chain_type_h_or_l`` ("H"/"L") from ``region_classification``.
    """
    df = df.copy()
    for col in _LIST_COLS:
        if col in df.columns:
            df[col] = df[col].apply(
                lambda v: ast.literal_eval(v) if isinstance(v, str) else v
            )
    if "num_mutations" in df.columns:
        df["num_mutations"] = df["num_mutations"].astype(int)
    if "region_classification" in df.columns and "chain_type_h_or_l" not in df.columns:
        df["chain_type_h_or_l"] = df["region_classification"].apply(
            lambda x: "H" if str(x).upper() in _HEAVY_REGIONS else "L"
        )
    return _uppercase_categorical(df)


def _apply_year_postprocessing(df: pd.DataFrame) -> pd.DataFrame:
    """Backfill year columns from IMGT sources; derive ``min_year``."""
    df = df.copy()
    if {"year_proposed", "year_proposed_imgt"}.issubset(df.columns):
        df["year_proposed"] = df[["year_proposed", "year_proposed_imgt"]].apply(
            lambda x: (
                x["year_proposed_imgt"]
                if pd.isna(x["year_proposed"])
                else x["year_proposed"]
            ),
            axis=1,
        )
    if {"year_recommended", "year_recommended_imgt"}.issubset(df.columns):
        df["year_recommended"] = df[
            ["year_recommended", "year_recommended_imgt"]
        ].apply(
            lambda x: (
                x["year_recommended_imgt"]
                if pd.isna(x["year_recommended"])
                else x["year_recommended"]
            ),
            axis=1,
        )
    if {"year_received", "year_proposed", "year_recommended"}.issubset(df.columns):
        df["min_year"] = df[
            ["year_received", "year_proposed", "year_recommended"]
        ].apply(
            lambda x: (
                x["year_received"] if not pd.isna(x["year_received"]) else np.nanmin(x)
            ),
            axis=1,
        )
    return df


def _inn_set(df: pd.DataFrame) -> set[str]:
    """Extract the unique INN name set from a DataFrame index or columns."""
    for candidate in ("inn", "inn_name"):
        if candidate in df.index.names:
            return set(df.index.get_level_values(candidate).dropna())
        if candidate in df.columns:
            return set(df[candidate].dropna())
    return set()


# ---------------------------------------------------------------------------
# Stats dataclass
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class INNAnalysisStats:
    """Summary statistics inferred from the loaded dataset.

    All counts reflect *unique* INN names.
    """

    total_inns: int
    """Sum of kept and excluded INNs."""

    inns_kept: int
    """INNs retained for analysis (``kept_inn`` sheet)."""

    inns_excluded: int
    """INNs excluded from analysis (``removed_inn`` sheet)."""

    inns_with_constant_chain_sequence: int
    """Kept INNs with at least one constant chain sequence (``INNConstantChains``)."""

    inns_with_mutations: int
    """Kept INNs carrying ≥1 mutation in a constant chain."""

    inns_without_mutations: int
    """Kept INNs with no detected mutations in any constant chain."""

    def __str__(self) -> str:
        return (
            "INNAnalysisStats\n"
            f"  Total INNs                        : {self.total_inns}\n"
            f"  INNs kept (analysis)              : {self.inns_kept}\n"
            f"  INNs excluded                     : {self.inns_excluded}\n"
            f"  INNs with constant chain sequence : {self.inns_with_constant_chain_sequence}\n"
            f"  INNs with ≥1 mutation             : {self.inns_with_mutations}\n"
            f"  INNs without mutations            : {self.inns_without_mutations}"
        )

    __repr__ = __str__


def _compute_stats(ds: "INNAnalysisDataset") -> INNAnalysisStats:
    inns_kept = _inn_set(ds.kept_inn)
    inns_excluded = _inn_set(ds.removed_inn)
    inns_with_mutations = _inn_set(ds.individual_mutation_profile)
    inns_with_cc = _inn_set(ds.constant_chains)
    return INNAnalysisStats(
        total_inns=len(inns_kept) + len(inns_excluded),
        inns_kept=len(inns_kept),
        inns_excluded=len(inns_excluded),
        inns_with_constant_chain_sequence=len(inns_with_cc),
        inns_with_mutations=len(inns_with_mutations),
        inns_without_mutations=len(inns_kept - inns_with_mutations),
    )


# ---------------------------------------------------------------------------
# INN Sets — four mutually exclusive partitions
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class INNSets:
    """Four mutually exclusive, exhaustive partitions of all known INNs.

    Attributes
    ----------
    excluded : frozenset[str]
        INNs removed from analysis (no usable sequence information).
    no_cc_seq : frozenset[str]
        Kept INNs that have no constant chain sequence.
    cc_no_mutation : frozenset[str]
        Kept INNs with constant chain sequence but zero mutations.
    cc_with_mutation : frozenset[str]
        Kept INNs carrying ≥1 mutation in a constant chain domain.
    """

    excluded: frozenset
    no_cc_seq: frozenset
    cc_no_mutation: frozenset
    cc_with_mutation: frozenset

    def __post_init__(self):
        # Verify mutual exclusivity and exhaustiveness
        sets = [
            self.excluded,
            self.no_cc_seq,
            self.cc_no_mutation,
            self.cc_with_mutation,
        ]
        all_inns = set().union(*sets)
        total = sum(len(s) for s in sets)
        assert total == len(all_inns), (
            f"INN sets overlap: union size {len(all_inns)} != sum of sizes {total}"
        )

    def __str__(self) -> str:
        return (
            "INNSets (mutually exclusive partitions)\n"
            f"  excluded         : {len(self.excluded):>5}\n"
            f"  no_cc_seq        : {len(self.no_cc_seq):>5}\n"
            f"  cc_no_mutation   : {len(self.cc_no_mutation):>5}\n"
            f"  cc_with_mutation : {len(self.cc_with_mutation):>5}\n"
            f"  ─────────────────────\n"
            f"  total            : {len(self.excluded) + len(self.no_cc_seq) + len(self.cc_no_mutation) + len(self.cc_with_mutation):>5}"
        )

    __repr__ = __str__


def _compute_inn_sets(ds: "INNAnalysisDataset") -> INNSets:
    inns_kept = _inn_set(ds.kept_inn)
    inns_excluded = _inn_set(ds.removed_inn)
    inns_with_cc = _inn_set(ds.constant_chains) & inns_kept
    inns_with_mut = _inn_set(ds.individual_mutation_profile) & inns_kept
    return INNSets(
        excluded=frozenset(inns_excluded),
        no_cc_seq=frozenset(inns_kept - inns_with_cc),
        cc_no_mutation=frozenset(inns_with_cc - inns_with_mut),
        cc_with_mutation=frozenset(inns_with_mut),
    )


# ---------------------------------------------------------------------------
# Dataset container
# ---------------------------------------------------------------------------


@dataclass
class INNAnalysisDataset:
    """All worksheets of ``inn_analysis_updated.xlsx`` in a single container.

    Primary interface (DataFrames)
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    Attributes that mirror the ExcelWriter sheet names:

    kept_inn
        INN-level metadata for retained antibodies (sheet ``kept_inn``).
        Indexed by ``inn_name``.  Year columns backfilled from IMGT data;
        ``min_year`` derived.
    removed_inn
        INN-level metadata for excluded antibodies (sheet ``removed_inn``).
    inn_ab_sequence_info
        VH / VL / constant chain sequence info per binding arm
        (sheet ``inn_ab_sequence_info``).
    imgt_info_per_chain
        Per IMGT chain-ID full-length chain summary
        (sheet ``IMGT Info Per Chain``).
    constant_chains
        Constant-chain sequence pivot; multi-level column header and
        multi-level row index (sheet ``INNConstantChains``).
    individual_mutation_profile
        Per-mutation table with 11-level row index.  Optionally post-
        processed by a ``mutation_fixer`` callable
        (sheet ``IndividualMutationProfile``).
    constant_chain_motifs_by_inn_region
        Concatenated motif strings per INN × region.  ``EU``,
        ``mutation_type``, ``germline_seq``, ``mutated_seq`` parsed to
        Python lists; ``chain_type_h_or_l`` and ``isotype`` (first 4 chars
        of ``gene``, e.g. ``"ighg"`` / ``"igkc"``) added
        (sheet ``ConstantChainMotifsByINNRegion``).
    unique_constant_chain_motifs
        De-duplicated motif patterns per INN/gene/region
        (sheet ``UniqueConstantChainMotifsByINNRegion``).
    constant_chain_mutation_motifs
        Cross-INN motif catalogue with occurrence counts; list columns
        parsed; ``chain_type_h_or_l`` added
        (sheet ``Constant ChainMutationMotifs``).
    inn_schema
        Column-level schema for the INN table (sheet ``inn_schema``).
    sequence_schema
        Column-level schema for the sequence table
        (sheet ``sequence_schema``).

    Pydantic-backed collections
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~
    For sheets with a matching model the data is also available as a
    typed collection.  These are constructed with ``model_construct``
    (no re-validation; data originates from the validated pipeline) so
    they are suitable for programmatic access but should not be used to
    assert that records are fully valid according to the model.

    inn_collection
        :class:`~data_io.merge.models.MergedINNCollection` from ``kept_inn``.
    removed_inn_collection
        :class:`~data_io.merge.models.MergedINNCollection` from ``removed_inn``.
    sequence_collection
        :class:`~data_io.merge.models.MergedSequenceCollection` from
        ``inn_ab_sequence_info``.
    full_length_chain_collection
        :class:`~models.inn_structure.analytics.ObservedFullLengthCollection`
        from ``IMGT Info Per Chain``.
    mutation_collection
        :class:`~models.inn_structure.analytics.IndividualMutationCollection`
        from ``IndividualMutationProfile``.

    Stats
    ~~~~~
    stats
        :class:`INNAnalysisStats` — computed lazily on first access.
    """

    # ── Primary DataFrame attributes ──────────────────────────────────────
    kept_inn: pd.DataFrame
    removed_inn: pd.DataFrame
    inn_ab_sequence_info: pd.DataFrame
    imgt_info_per_chain: pd.DataFrame
    constant_chains: pd.DataFrame
    individual_mutation_profile: pd.DataFrame
    constant_chain_motifs_by_inn_region: pd.DataFrame
    unique_constant_chain_motifs: pd.DataFrame
    constant_chain_mutation_motifs: pd.DataFrame
    inn_schema: pd.DataFrame
    sequence_schema: pd.DataFrame

    # ── Pydantic-backed collections ───────────────────────────────────────
    inn_collection: MergedINNCollection
    removed_inn_collection: MergedINNCollection
    sequence_collection: MergedSequenceCollection
    full_length_chain_collection: ObservedFullLengthCollection
    mutation_collection: IndividualMutationCollection

    # ── Lazy stats ────────────────────────────────────────────────────────
    _stats: Optional[INNAnalysisStats] = field(default=None, repr=False, compare=False)
    _inn_sets: Optional["INNSets"] = field(default=None, repr=False, compare=False)

    @property
    def stats(self) -> INNAnalysisStats:
        """Summary statistics (lazy — computed on first access)."""
        if self._stats is None:
            self._stats = _compute_stats(self)
        return self._stats

    @property
    def inn_sets(self) -> "INNSets":
        """Four mutually exclusive, exhaustive INN name sets (lazy).

        The sets partition all known INNs::

            excluded ∪ no_cc_seq ∪ cc_no_mutation ∪ cc_with_mutation
              == all INNs (excluded + kept)

        and their pairwise intersections are empty.

        Returns
        -------
        INNSets
            ``excluded``         – INNs not retained for analysis.
            ``no_cc_seq``        – Kept INNs with no constant chain sequence.
            ``cc_no_mutation``   – Kept INNs with CC seq but no mutations.
            ``cc_with_mutation`` – Kept INNs with ≥1 CC mutation.
        """
        if self._inn_sets is None:
            self._inn_sets = _compute_inn_sets(self)
        return self._inn_sets

    def __repr__(self) -> str:
        s = self.stats
        # DataFrame table summary: name → (rows, cols)
        _df_tables: list[tuple[str, pd.DataFrame]] = [
            ("kept_inn", self.kept_inn),
            ("removed_inn", self.removed_inn),
            ("inn_ab_sequence_info", self.inn_ab_sequence_info),
            ("imgt_info_per_chain", self.imgt_info_per_chain),
            ("constant_chains", self.constant_chains),
            (
                "individual_mutation_profile",
                self.individual_mutation_profile,
            ),
            (
                "constant_chain_motifs_by_inn_region",
                self.constant_chain_motifs_by_inn_region,
            ),
            (
                "unique_constant_chain_motifs",
                self.unique_constant_chain_motifs,
            ),
            (
                "constant_chain_mutation_motifs",
                self.constant_chain_mutation_motifs,
            ),
            ("inn_schema", self.inn_schema),
            ("sequence_schema", self.sequence_schema),
        ]
        col_w = max(len(name) for name, _ in _df_tables)
        table_lines = "\n".join(
            f"    {name:<{col_w}}  {df.shape[0]:>5} rows × {df.shape[1]:>3} cols"
            for name, df in _df_tables
        )
        return (
            "INNAnalysisDataset\n"
            "  Stats\n"
            f"    Total INNs                        : {s.total_inns}\n"
            f"    INNs kept                         : {s.inns_kept}\n"
            f"    INNs excluded                     : {s.inns_excluded}\n"
            f"    INNs with constant chain sequence : {s.inns_with_constant_chain_sequence}\n"
            f"    INNs with ≥1 mutation             : {s.inns_with_mutations}\n"
            f"    INNs without mutations            : {s.inns_without_mutations}\n"
            "  Tables\n"
            f"{table_lines}"
        )


# ---------------------------------------------------------------------------
# Public factory
# ---------------------------------------------------------------------------


def load_inn_analysis(
    path: str | Path,
    *,
    mutation_fixer: list[MutationPatternIssue] | None = None,
    cdb=None,
    cn=None,
) -> INNAnalysisDataset:
    """Load all worksheets of ``inn_analysis_updated.xlsx`` into a single typed object.

    Parameters
    ----------
    path:
        Path to ``inn_analysis_updated.xlsx``.
    mutation_fixer : list[MutationPatternIssue], optional
        Ordered list of issues to correct in ``IndividualMutationProfile``.
        Issues are applied in the order given; unfixed issues are reported as
        warnings::

            ds = load_inn_analysis(source, mutation_fixer=[
                MutationPatternIssue.DOUBLED_MUTATED_SEQ,
                MutationPatternIssue.DUPLICATE_ROWS,
            ])

    cdb, cn :
        Required when :attr:`~MutationPatternIssue.MISSING_GERMLINE_SEQ` is
        included in *mutation_fixer*.

    Returns
    -------
    INNAnalysisDataset
    """
    _requires_deps = {MutationPatternIssue.MISSING_GERMLINE_SEQ}
    if mutation_fixer is not None:
        if callable(mutation_fixer):
            raise TypeError(
                "load_inn_analysis: mutation_fixer must be a list[MutationPatternIssue], "
                "not a callable. Pass the enum list directly:\n"
                "  ds = load_inn_analysis(source, mutation_fixer=[\n"
                "      MutationPatternIssue.DOUBLED_MUTATED_SEQ,\n"
                "      MutationPatternIssue.DUPLICATE_ROWS,\n"
                "  ])"
            )
        _needs_deps = [i for i in mutation_fixer if i in _requires_deps]
        if _needs_deps and (cdb is None or cn is None):
            raise ValueError(
                f"load_inn_analysis: {_needs_deps} requires cdb and cn — "
                f"pass them as keyword arguments."
            )
        _fixer = build_mutation_fixer(cdb, cn, fix=list(mutation_fixer))
    else:
        _fixer = None
    _resolved_fixer = _fixer
    path = Path(path)
    available = set(pd.ExcelFile(path).sheet_names)

    def _read(sheet: str, **kw) -> pd.DataFrame:
        header = kw.pop("header", 0)
        index_col = kw.pop("index_col", None)
        df = pd.read_excel(path, sheet_name=sheet, header=header, **kw)
        if index_col is None:
            return df
        idx_positions = [index_col] if isinstance(index_col, int) else list(index_col)
        is_multi_header = isinstance(header, (list, tuple))
        index_names = None
        if is_multi_header:
            # A saved MultiIndex-row + MultiIndex-header sheet has one extra
            # row (right after the header) holding the index level names.
            # pandas only strips this row automatically when index_col is
            # passed directly to read_excel — which we deliberately avoid,
            # since that path also makes pandas unconditionally forward-fill
            # blank index cells (see pandas.io.excel._base._parse_sheet),
            # corrupting any row whose actual index value is NaN by
            # inheriting the value from the row above. Strip the name row
            # manually and set the index ourselves instead.
            data_cols = [c for i, c in enumerate(df.columns) if i not in idx_positions]
            first_row = df.iloc[0]
            if data_cols and first_row[data_cols].isna().all():
                index_names = [first_row.iloc[i] for i in idx_positions]
                df = df.iloc[1:].reset_index(drop=True)
        idx_cols = [df.columns[i] for i in idx_positions]
        df = df.set_index(idx_cols)
        if index_names is not None:
            df.index = df.index.set_names(index_names)
        return df

    # ------------------------------------------------------------------
    # INN metadata
    # ------------------------------------------------------------------
    kept_inn_raw = _read("kept_inn", header=0, index_col=0)
    kept_inn_df = _apply_year_postprocessing(kept_inn_raw)

    removed_inn_df = _read("removed_inn", header=0)

    # ------------------------------------------------------------------
    # Sequence info — accept legacy sheet name too
    # ------------------------------------------------------------------
    seq_sheet = (
        "inn_ab_sequence_info"
        if "inn_ab_sequence_info" in available
        else "kept_sequences"
    )
    inn_ab_seq_df = _read(seq_sheet, header=0)

    # ------------------------------------------------------------------
    # Per-chain full-length summary
    # ------------------------------------------------------------------
    imgt_per_chain_df = _read("IMGT Info Per Chain", header=0)

    # ------------------------------------------------------------------
    # Constant chain pivot (multi-level header + index — no Pydantic model)
    # ------------------------------------------------------------------
    constant_chains_df = _read(
        "INNConstantChains",
        header=[0, 1],
        index_col=_CONSTANT_CHAINS_INDEX_COLS,
    )

    # ------------------------------------------------------------------
    # Individual mutation profile (multi-level index)
    # ------------------------------------------------------------------
    mut_profile_raw = _read(
        "IndividualMutationProfile",
        header=0,
        index_col=_MUTATION_PROFILE_INDEX_COLS,
    )
    mut_profile_df = (
        _resolved_fixer(mut_profile_raw)
        if _resolved_fixer is not None
        else mut_profile_raw
    )
    mut_profile_df = _apply_mutation_profile_postprocessing(mut_profile_df)

    # ------------------------------------------------------------------
    # Motif tables
    # ------------------------------------------------------------------
    motifs_sheet = (
        "ConstantChainMotifsByINNRegion"
        if "ConstantChainMotifsByINNRegion" in available
        else "ConstantChainMotifsByRegion"
    )
    motifs_by_region_df = _apply_motif_postprocessing(_read(motifs_sheet, header=0))
    if "gene" in motifs_by_region_df.columns:

        def _to_isotype(x):
            if pd.isna(x):
                return x
            s = str(x)[:4].lower()
            if s in ("igkc", "iglc", "igk", "igl"):
                return "IGK/IGL"
            return str(x)[:4].upper()

        motifs_by_region_df["isotype"] = (
            motifs_by_region_df["gene"].apply(_to_isotype).str.upper()
        )
        motifs_by_region_df["gene"] = motifs_by_region_df["gene"].apply(
            lambda x: str(x).upper() if pd.notna(x) else x
        )
    if "gene_species" in motifs_by_region_df.columns:
        motifs_by_region_df["species_main"] = motifs_by_region_df["gene_species"].apply(
            lambda x: (
                str(x).lower()
                if pd.notna(x) and str(x).lower() in ("human", "mouse")
                else "other"
            )
        )

    unique_motifs_sheet = (
        "UniqueConstantChainMotifsByINNRegion"
        if "UniqueConstantChainMotifsByINNRegion" in available
        else (
            "UniqueChainMotifsByINNRegion"
            if "UniqueChainMotifsByINNRegion" in available
            else None
        )
    )
    unique_motifs_df = (
        _read(unique_motifs_sheet, header=0)
        if unique_motifs_sheet is not None
        else pd.DataFrame()
    )

    motif_catalogue_df = _apply_motif_postprocessing(
        _read("Constant ChainMutationMotifs", header=0)
    )

    # ------------------------------------------------------------------
    # Schema reference
    # ------------------------------------------------------------------
    inn_schema_df = _read("inn_schema", header=0)
    sequence_schema_df = _read("sequence_schema", header=0)

    # ------------------------------------------------------------------
    # Reconstruct Pydantic collections
    # ------------------------------------------------------------------

    # MergedINNCollection from kept_inn (reset index so inn_name is a column)
    inn_collection = _build_collection(
        _df_to_records(kept_inn_df.reset_index()),
        MergedINNRecord,
        MergedINNCollection,
        column_sources=INN_COLUMN_SOURCES,
    )

    # MergedINNCollection from removed_inn
    removed_inn_collection = _build_collection(
        _df_to_records(removed_inn_df),
        MergedINNRecord,
        MergedINNCollection,
        column_sources=INN_COLUMN_SOURCES,
    )

    # MergedSequenceCollection from inn_ab_sequence_info
    sequence_collection = _build_collection(
        _df_to_records(inn_ab_seq_df),
        MergedSequenceRecord,
        MergedSequenceCollection,
        column_sources=SEQ_COLUMN_SOURCES,
    )

    # ObservedFullLengthCollection from IMGT Info Per Chain
    full_length_chain_collection = _build_collection(
        _df_to_records(imgt_per_chain_df),
        ObservedFullLengthChain,
        ObservedFullLengthCollection,
    )

    # IndividualMutationCollection from IndividualMutationProfile
    # Reset index so all 11 index levels are available as regular columns
    mut_records_df = mut_profile_df.reset_index()
    mutation_collection = _build_collection(
        _df_to_records(mut_records_df),
        IndividualMutationInfo,
        IndividualMutationCollection,
    )

    return INNAnalysisDataset(
        # DataFrames
        kept_inn=kept_inn_df,
        removed_inn=removed_inn_df,
        inn_ab_sequence_info=inn_ab_seq_df,
        imgt_info_per_chain=imgt_per_chain_df,
        constant_chains=constant_chains_df,
        individual_mutation_profile=mut_profile_df,
        constant_chain_motifs_by_inn_region=motifs_by_region_df,
        unique_constant_chain_motifs=unique_motifs_df,
        constant_chain_mutation_motifs=motif_catalogue_df,
        inn_schema=inn_schema_df,
        sequence_schema=sequence_schema_df,
        # Pydantic collections
        inn_collection=inn_collection,
        removed_inn_collection=removed_inn_collection,
        sequence_collection=sequence_collection,
        full_length_chain_collection=full_length_chain_collection,
        mutation_collection=mutation_collection,
    )
