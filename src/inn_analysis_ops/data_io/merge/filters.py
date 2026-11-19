"""Filtering logic for the merged INN database."""

from typing import NamedTuple, Set

import pandas as pd


class FilterResult(NamedTuple):
    kept: pd.DataFrame
    removed: (
        pd.DataFrame
    )  # includes 'removal_reason' column (first matching rule)


# Ordered removal rules — first match sets removal_reason.
_RULES: list[tuple[str, callable]] = [
    (
        "no_inn_number_in_therasabdab",
        lambda df: df["inn_number"].isna()
        & df["sources"].str.contains("therasabdab", na=False),
    ),
    (
        "no_inn_number_not_in_therasabdab",
        lambda df: df["inn_number"].isna()
        & ~df["sources"].str.contains("therasabdab", na=False),
    ),
    # ("no_binders_in_either_source", lambda df: df["n_binders_therasabdab"].isna() & df["n_binders_imgt"].isna()),
    # ("no_imgt_binders", lambda df: df["n_binders_imgt"].isna()),
]


def filter_inn_df(df: pd.DataFrame) -> FilterResult:
    """Split merged INN dataframe into kept/removed.

    Rules are checked in order; the first matching rule sets 'removal_reason'.
    """
    flags = pd.concat({name: fn(df) for name, fn in _RULES}, axis=1)
    remove_mask = flags.any(axis=1)

    removed = df[remove_mask].copy()
    removed["removal_reason"] = flags[remove_mask].apply(
        lambda row: next(name for name in flags.columns if row[name]),
        axis=1,
    )

    return FilterResult(kept=df[~remove_mask].copy(), removed=removed)


def filter_by_sequence_availability(
    kept: pd.DataFrame,
    removed: pd.DataFrame,
    seq_inn_names: Set[str],
) -> FilterResult:
    """Move INNs from kept to removed when no sequence was found for them.

    This is a second-pass filter applied after sequences have been built
    (i.e. after ``merged.sequences.to_dataframe()``).  INNs that passed
    ``filter_inn_df`` but have no row in the sequence table — neither a
    VH/VL arm nor a constant-chain-only row — are moved to removed.

    Removal reason:
        ``no_sequence_found``        — INN has no sequence data from any source
                                       and its receptor_type is an antibody-like format
        ``no_sequence_non_antibody`` — INN has no sequence data and receptor_type
                                       indicates a non-antibody format (FPIA, etc.)

    Args:
        kept: ``kept`` DataFrame from a previous ``filter_inn_df`` call.
        removed: ``removed`` DataFrame from the same call.
        seq_inn_names: Set of inn_name values that have at least one sequence row
                       (e.g. ``set(kept_seq["inn_name"])``).

    Returns:
        A new FilterResult with the no-sequence INNs moved to removed.
    """
    _NON_ANTIBODY_RECEPTOR_TYPES = {
        "FPIA",   # Fusion Protein IgG-like: Antigen binding
        "FPIG",   # Fusion Protein IgG-like
        "FPIE",   # Fusion Protein IgG-like: Enzyme
        "FPIN",   # Fusion Protein IgG-like: Non-antibody binding
        "FPII",   # Fusion Protein IgG-like: Inhibitor
        "RPI",    # Receptor-based protein inhibitor (non-Ig)
    }

    no_seq_mask = ~kept["inn_name"].isin(seq_inn_names)
    no_seq = kept[no_seq_mask].copy()

    if no_seq.empty:
        return FilterResult(kept=kept.copy(), removed=removed.copy())

    def _reason(row) -> str:
        rt = str(row.get("receptor_type") or "").strip().upper()
        if rt in _NON_ANTIBODY_RECEPTOR_TYPES:
            return f"no_sequence_non_antibody (receptor_type={rt})"
        return "no_sequence_found"

    no_seq["removal_reason"] = no_seq.apply(_reason, axis=1)

    new_kept = kept[~no_seq_mask].copy()
    new_removed = pd.concat([removed, no_seq], ignore_index=True)

    return FilterResult(kept=new_kept, removed=new_removed)
