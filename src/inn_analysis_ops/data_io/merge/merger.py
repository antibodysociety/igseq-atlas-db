"""Merger logic: consolidates TheraSAbDab, IMGT mAb-DB, mAbSolve, and IMGT 2D Structure."""

import logging
import re
from difflib import get_close_matches
from pathlib import Path
from string import ascii_lowercase
from typing import Dict, Optional

import pandas as pd
from inn_analysis_ops.models.inn_structure import INNStructureCollection
from inn_analysis_ops.data_io.therasabdab import build_inn_table, build_sequence_table

from .models import (
    INN_COLUMN_SOURCES,
    SEQ_COLUMN_SOURCES,
    ColumnSource,
    MergedDatabase,
    MergedINNCollection,
    MergedINNRecord,
    MergedSequenceCollection,
    MergedSequenceRecord,
)

logger = logging.getLogger(__name__)


# ── Helpers ────────────────────────────────────────────────────────────


def _normalize(name: str) -> str:
    return str(name).strip().lower()


def _inn_mol_name(name: str) -> str:
    norm = _normalize(name)
    return norm.split()[0] if norm else ""


def _safe(val) -> Optional[str]:
    if pd.isna(val):
        return None
    s = str(val).strip()
    if not s:
        return None
    parts = re.split(r"[,;/|]+", s)
    clean = [
        p.strip() for p in parts if p.strip().lower() not in ("na", "n/a", "none", "")
    ]
    return ", ".join(clean) if clean else None


def _seq_clean(val) -> Optional[str]:
    if pd.isna(val):
        return None
    s = str(val).strip()
    return None if s.lower() in ("na", "n/a", "") else s


def _gapless(seq: Optional[str]) -> Optional[str]:
    """Strip alignment gaps and return uppercase sequence."""
    if not seq:
        return None
    return seq.upper().replace("-", "").replace(".", "")


def _seq_resolve(t_seq: Optional[str], i_seq: Optional[str]) -> tuple:
    """Compare two sequences and return (consensus, t_mismatch, i_mismatch, is_mismatch, is_truncated).

    - identical / one missing → no mismatch, no truncation
    - one is substring of the other → truncated (not a genuine mismatch)
    - genuinely different    → mismatch
    """
    gt = _gapless(t_seq)
    gi = _gapless(i_seq)
    if not gt or not gi:
        return t_seq or i_seq, None, None, False, False
    if gt == gi:
        return t_seq, None, None, False, False
    if gt in gi:  # therasabdab is shorter substring of imgt
        return i_seq, gt, None, False, True
    if gi in gt:  # imgt is shorter substring of therasabdab
        return t_seq, None, gi, False, True
    return None, gt, gi, True, False  # genuinely different


def _seqs_match(a: Optional[str], b: Optional[str]) -> bool:
    """Kept for compatibility — use _seq_resolve for merge logic."""
    _, _, _, is_mismatch, _ = _seq_resolve(a, b)
    return not is_mismatch


def _fuzzy_resolve(name: str, pool: set[str]) -> Optional[str]:
    tokens = re.split(r"[\s\-]+", name)
    hits = {t for t in tokens if t in pool}
    return hits.pop() if len(hits) == 1 else None


def _find_longer_alias(canonical: str, pool: set[str]) -> Optional[str]:
    hits = {n for n in pool if canonical in re.split(r"[\s\-]+", n) and n != canonical}
    return hits.pop() if len(hits) == 1 else None


def _build_fuzzy_maps(
    imgt_names: set[str], other_names: set[str]
) -> tuple[dict[str, str], dict[str, str]]:
    fuzzy_map: dict[str, str] = {}
    reverse_map: dict[str, str] = {}
    unmatched = imgt_names - other_names
    for name in unmatched:
        hit = _fuzzy_resolve(name, other_names)
        if hit and hit not in reverse_map:
            fuzzy_map[name] = hit
            reverse_map[hit] = name
    remaining = unmatched - set(fuzzy_map)
    for canonical in other_names - imgt_names:
        if canonical in reverse_map:
            continue
        hit = _find_longer_alias(canonical, remaining)
        if hit and hit not in fuzzy_map:
            fuzzy_map[hit] = canonical
            reverse_map[canonical] = hit
    return fuzzy_map, reverse_map


def _fuzzy_match_single(
    name: str, pool: set[str], cutoff: float = 0.92
) -> Optional[str]:
    if not name or not pool:
        return None
    hits = get_close_matches(name, list(pool), n=1, cutoff=cutoff)
    return hits[0] if hits else None


def _token_hit_in_pool(name: str, pool: set[str]) -> Optional[str]:
    if not name or not pool:
        return None
    tokens = [t for t in re.split(r"[^a-z0-9]+", _normalize(name)) if t]
    hits = sorted({tok for tok in tokens if tok in pool})
    return hits[0] if len(hits) == 1 else None


def _extract_imgt_years(
    raw_val, hardcoded_year_map: dict[str, int]
) -> tuple[Optional[str], list[str]]:
    if pd.isna(raw_val):
        return None, []
    text = str(raw_val).strip()
    if not text:
        return None, []

    years = re.findall(r"\((\d{4})\)", text)
    unique_years = list(dict.fromkeys(years))
    year_value = ";".join(unique_years) if unique_years else None

    text_no_year_paren = re.sub(r"\(\d{4}\)", "", text)
    bare_numbers = re.findall(r"\b\d+\b", text_no_year_paren)
    if bare_numbers and not unique_years:
        resolved_years: list[str] = []
        unresolved_numbers: list[str] = []
        for num in list(dict.fromkeys(bare_numbers)):
            if num in hardcoded_year_map:
                resolved_years.append(str(hardcoded_year_map[num]))
            else:
                unresolved_numbers.append(num)
        if resolved_years:
            year_value = ";".join(list(dict.fromkeys(resolved_years)))
        return year_value, unresolved_numbers
    return year_value, []


# ── Field lists ────────────────────────────────────────────────────────

_THERA_META_FIELDS = [
    "target",
    "format",
    "isotype",
    "highest_clinical_trial",
    "estimated_status",
    "has_100_si_struc",
    "has_99_si_struc",
    "has_95_98_si_struc",
    "pdb_100",
    "pdb_99",
    "pdb_95_98",
    "dev_technology",
    "companies",
    "year_proposed",
    "year_recommended",
    "year_received",
    "conditions_approved",
    "conditions_active",
    "conditions_discontinued",
    "notes",
]

_IMGT_FIELDS = {
    "inn_number": "inn_number",
    "imgt_id": "imgt_id",
    "structure_2d": "structure_2d",
    "structure_3d": "structure_3d",
    "receptor_type": "receptor_type",
    "specificity": "specificity",
    "species": "species",
    "conjugated": "conjugated",
    "origin_species": "origin_species",
    "origin_clone": "origin_clone",
    "expression_system": "expression_system",
    "mechanism_of_action": "mechanism_of_action",
    "clinical_indication": "clinical_indication",
    "clinical_domain": "clinical_domain",
    "clinical_trials": "clinical_trials",
    "engineered_variant": "engineered_variant",
    "development_status": "development_status",
    "biosimilars": "biosimilars",
}

_MABSOLVE_BOOL = {
    "heterodimerization",
    "site_specific_conj",
    "stabilized",
    "c_term_truncation",
    "other_fc_mod",
}
_MABSOLVE_FIELDS = {
    "format": "mabsolve_format",
    "effector_function": "effector_function",
    "half_life": "half_life",
    "heterodimerization": "heterodimerization",
    "site_specific_conj": "site_specific_conj",
    "stabilized": "stabilized",
    "c_term_truncation": "c_term_truncation",
    "other": "other_fc_mod",
}

# Fallback year mapping for malformed IMGT lists without parenthesized years.
PROP_LIST_HARDCODE: dict[str, int] = {"134": 2026}
REC_LIST_HARDCODE: dict[str, int] = {}


# ── IMGT 2D arm mapping ────────────────────────────────────────────────


def _prepare_imgt_df(imgt_df: pd.DataFrame) -> pd.DataFrame:
    out = imgt_df.copy()
    if "inn" not in out.columns:
        out["inn"] = ""
    out["inn"] = out["inn"].fillna("").astype(str).str.strip()
    out["inn_norm"] = out["inn"].apply(_normalize)
    out["inn_mol_name"] = out["inn_norm"].apply(_inn_mol_name)

    # Keep duplicate normalized IMGT names as distinct merge rows.
    # First occurrence keeps inn_norm; later occurrences get _dupN suffix.
    dup_idx = out.groupby("inn_norm", sort=False).cumcount()
    out["inn_merge_name"] = out["inn_norm"]
    non_empty = out["inn_norm"] != ""
    dup_mask = non_empty & (dup_idx > 0)
    out.loc[dup_mask, "inn_merge_name"] = (
        out.loc[dup_mask, "inn_norm"] + "_dup" + dup_idx[dup_mask].astype(str)
    )
    return out


def _inn_number_map(imgt_df: pd.DataFrame) -> dict[str, dict[str, str]]:
    m: dict[str, dict[str, str]] = {}
    for _, row in imgt_df.iterrows():
        inn = _normalize(row.get("inn_merge_name", row.get("inn", "")))
        inn_mol = _normalize(row.get("inn_mol_name", "")) or _inn_mol_name(
            row.get("inn", "")
        )
        num = row.get("inn_number")
        if inn and pd.notna(num):
            m[str(int(float(num)))] = {"inn_name": inn, "inn_mol_name": inn_mol}
    return m


def _map_arm_df(
    imgt_2d_df: pd.DataFrame, inn_num_map: dict[str, dict[str, str]]
) -> dict[str, pd.DataFrame]:
    """Map inn_id → inn_name and return {inn_name: sorted arm rows}.

    Accepts three formats:
      - arm_slot column  → already collapsed (from collapse_to_vh_vl_binder_info)
      - chain_type column → raw per-region flat (from to_flat_dataframe)
      - neither           → legacy summary CSV (no per-arm sequences; returns {})
    """
    if "arm_slot" in imgt_2d_df.columns:
        arm_df = imgt_2d_df
    elif "chain_type" in imgt_2d_df.columns:
        arm_df = INNStructureCollection.from_flat_dataframe(
            imgt_2d_df
        ).collapse_to_vh_vl_binder_info()
    else:
        return {}
    result = {}
    for inn_id, group in arm_df.groupby("inn_id", sort=False):
        try:
            inn_meta = inn_num_map.get(str(int(float(inn_id))))
        except (ValueError, TypeError):
            continue
        if inn_meta:
            mapped = group.sort_values("arm_slot").reset_index(drop=True).copy()
            mapped["inn_name"] = inn_meta["inn_name"]
            mapped["inn_mol_name"] = inn_meta["inn_mol_name"]
            result[inn_meta["inn_name"]] = mapped
    return result


def update_imgt_df_using_merged(
    imgt_df: pd.DataFrame, merged: MergedDatabase
) -> pd.DataFrame:
    """Return IMGT dataframe with merge-aligned naming columns.

    Adds/updates:
      - inn_orig_name: original IMGT inn string
      - inn_name: merge key used by DatabaseMerger (includes _dupN for duplicate normalized names)
      - inn_mol_name: molecule-level key, including token-based flips captured in merged.fuzzy_aliases
    """
    out = imgt_df.copy()
    if "inn" in out.columns:
        out["inn_orig_name"] = out["inn"].fillna("").astype(str).str.strip()
    else:
        out["inn_orig_name"] = ""

    prepared = _prepare_imgt_df(out)
    out["inn_name"] = prepared["inn_merge_name"]
    out["inn_mol_name"] = prepared["inn_mol_name"]

    # IMGT-level mol-name flips are stored with plain keys (no "|source" suffix).
    imgt_mol_flips = {k: v for k, v in merged.fuzzy_aliases.items() if "|" not in k}
    if imgt_mol_flips:
        out["inn_mol_name"] = (
            out["inn_name"].map(imgt_mol_flips).fillna(out["inn_mol_name"])
        )

    return out


# ── DatabaseMerger ─────────────────────────────────────────────────────


class DatabaseMerger:
    update_imgt_df_using_merged = staticmethod(update_imgt_df_using_merged)

    @classmethod
    def from_csvs(
        cls,
        therasabdab_path: str | Path,
        imgt_mabdb_path: str | Path,
        mabsolve_path: str | Path,
        imgt_2d_path: str | Path,
    ) -> MergedDatabase:
        """Load from CSVs and merge."""

        thera_df = pd.read_csv(therasabdab_path, index_col=0)
        imgt_df = _prepare_imgt_df(pd.read_csv(imgt_mabdb_path, index_col=0))
        mabsolve_df = pd.read_csv(mabsolve_path)
        imgt_2d_df = pd.read_csv(imgt_2d_path)
        thera_inn_df = build_inn_table(thera_df).to_dataframe()
        thera_seq_df = build_sequence_table(thera_df).to_dataframe()
        imgt_arms = _map_arm_df(imgt_2d_df, _inn_number_map(imgt_df))
        return cls._merge(thera_inn_df, thera_seq_df, imgt_df, mabsolve_df, imgt_arms)

    @classmethod
    def from_models(
        cls,
        thera_df: pd.DataFrame,
        imgt_df: pd.DataFrame,
        mabsolve_df: pd.DataFrame,
        imgt_2d_collection,  # INNStructureCollection
    ) -> MergedDatabase:
        """Load from models and merge."""

        thera_inn_df = build_inn_table(thera_df).to_dataframe()
        thera_seq_df = build_sequence_table(thera_df).to_dataframe()
        arm_df = imgt_2d_collection.collapse_to_vh_vl_binder_info()
        imgt_df = _prepare_imgt_df(imgt_df)
        imgt_arms = _map_arm_df(arm_df, _inn_number_map(imgt_df))
        return cls._merge(thera_inn_df, thera_seq_df, imgt_df, mabsolve_df, imgt_arms)

    @classmethod
    def from_collapsed(
        cls,
        thera_inn_df: pd.DataFrame,
        thera_seq_df: pd.DataFrame,
        imgt_df: pd.DataFrame,
        mabsolve_df: pd.DataFrame,
        imgt_2d_arm_df: pd.DataFrame,
    ) -> MergedDatabase:
        """Explicit entry point: pass pre-collapsed tables directly."""
        imgt_df = _prepare_imgt_df(imgt_df)
        imgt_arms = _map_arm_df(imgt_2d_arm_df, _inn_number_map(imgt_df))
        return cls._merge(thera_inn_df, thera_seq_df, imgt_df, mabsolve_df, imgt_arms)

    @classmethod
    def _merge(
        cls,
        thera_inn_df: pd.DataFrame,
        thera_seq_df: pd.DataFrame,
        imgt_df: pd.DataFrame,
        mabsolve_df: pd.DataFrame,
        imgt_arms_by_inn: dict[str, pd.DataFrame],
    ) -> MergedDatabase:
        """Merge databases.

        Populates MergedDatabase with:
          - fuzzy_aliases: {inn|source: matched_mol_name} — fuzzy mol-name fallbacks used for joins
          - inn_id_consolidations: {inn_name: [inn_ids]} — multiple INN IDs per merged record
          - excluded_imgt: reserved merge-time exclusion tracking
          - reverse_map: {matched_mol_name: inn|source} for fuzzy fallbacks
        """

        # ── Index sources ──────────────────────────────────────────────
        thera_inn_idx = {
            _normalize(row.get("inn_name", "")): row
            for _, row in thera_inn_df.iterrows()
            if _normalize(row.get("inn_name", ""))
        }
        thera_inn_by_mol = {
            _inn_mol_name(row.get("inn_name", "")): row
            for _, row in thera_inn_df.iterrows()
            if _inn_mol_name(row.get("inn_name", ""))
        }
        thera_seq_by_inn = {
            _normalize(name): grp.sort_values("molecule_slot").reset_index(drop=True)
            for name, grp in thera_seq_df.groupby("inn_name")
            if _normalize(name)
        }
        thera_seq_by_mol = {
            _inn_mol_name(name): grp.sort_values("molecule_slot").reset_index(drop=True)
            for name, grp in thera_seq_df.groupby("inn_name")
            if _inn_mol_name(name)
        }
        imgt_idx = {
            _normalize(row.get("inn_merge_name", row.get("inn", ""))): row
            for _, row in imgt_df.iterrows()
            if _normalize(row.get("inn_merge_name", row.get("inn", "")))
        }
        mabsolve_idx = {
            _normalize(row.get("inn", "")): row
            for _, row in mabsolve_df.iterrows()
            if _normalize(row.get("inn", ""))
        }
        mabsolve_by_mol = {
            _inn_mol_name(row.get("inn", "")): row
            for _, row in mabsolve_df.iterrows()
            if _inn_mol_name(row.get("inn", ""))
        }

        fuzzy_map: dict[str, str] = {}
        reverse_map: dict[str, str] = {}
        excluded_imgt_dict: dict[str, dict[str, str]] = {}
        imgt_year_parse_issues: dict[str, set[str]] = {
            "prop_list": set(),
            "rec_list": set(),
        }

        imgt_inn_names = set(imgt_idx) | set(imgt_arms_by_inn)
        mol_pool = set(thera_inn_by_mol) | set(mabsolve_by_mol)
        imgt_mol_names: set[str] = set()
        for imgt_name in imgt_inn_names:
            row = imgt_idx.get(imgt_name)
            full_name = row.get("inn", imgt_name) if row is not None else imgt_name
            mol = _inn_mol_name(full_name)
            if mol not in mol_pool:
                token_hit = _token_hit_in_pool(full_name, mol_pool)
                if token_hit:
                    mol = token_hit
            if mol:
                imgt_mol_names.add(mol)
        all_inns = sorted(set(thera_inn_idx) | imgt_inn_names | set(mabsolve_idx))

        inn_records: list[MergedINNRecord] = []
        seq_records: list[MergedSequenceRecord] = []
        inn_id_consolidations: dict[
            str, list[str]
        ] = {}  # canonical_inn → [inn_ids from different sources]

        for inn in all_inns:
            src_list: list[str] = []
            kw: dict = {"inn_name": inn}
            imgt_row = imgt_idx.get(inn)
            full_inn_name = imgt_row.get("inn", inn) if imgt_row is not None else inn
            imgt_mol = _inn_mol_name(full_inn_name)
            kw["inn_mol_name"] = imgt_mol or _inn_mol_name(inn)
            imgt_arm_df = imgt_arms_by_inn.get(inn)

            # Prevent duplicate external-only rows when this name (or its mol-name root)
            # is already represented by IMGT entries that are enriched many-to-one.
            if imgt_row is None and imgt_arm_df is None:
                candidate_mol = kw["inn_mol_name"]
                if inn in imgt_inn_names or (
                    candidate_mol and candidate_mol in imgt_mol_names
                ):
                    continue
            # Track INN IDs for consolidation reporting
            inn_ids_for_record: list[str] = []

            # ── IMGT mAb-DB ────────────────────────────────────────────
            if imgt_row is not None or imgt_arm_df is not None:
                src_list.append("imgt")
            if imgt_row is not None:
                for csv_col, field in _IMGT_FIELDS.items():
                    if csv_col in imgt_row.index:
                        kw[field] = _safe(imgt_row[csv_col])
                if pd.notna(imgt_row.get("inn_number")):
                    inn_ids_for_record.append(("imgt", str(imgt_row.get("inn_number"))))
                year_prop, prop_missing_nums = _extract_imgt_years(
                    imgt_row.get("prop_list"), PROP_LIST_HARDCODE
                )
                year_rec, rec_missing_nums = _extract_imgt_years(
                    imgt_row.get("rec_list"), REC_LIST_HARDCODE
                )
                kw["year_proposed_imgt"] = year_prop
                kw["year_recommended_imgt"] = year_rec
                if prop_missing_nums or rec_missing_nums:
                    if prop_missing_nums:
                        raw_prop = str(imgt_row.get("prop_list")).strip()
                        if raw_prop:
                            imgt_year_parse_issues["prop_list"].add(raw_prop)
                    if rec_missing_nums:
                        raw_rec = str(imgt_row.get("rec_list")).strip()
                        if raw_rec:
                            imgt_year_parse_issues["rec_list"].add(raw_rec)
            if imgt_arm_df is not None and len(imgt_arm_df):
                kw["n_binders_imgt"] = len(imgt_arm_df)
                kw["imgt_2d_chain_count"] = len(imgt_arm_df)
                kw["imgt_2d_heavy_count"] = int(imgt_arm_df["vh"].notna().sum())
                kw["imgt_2d_light_count"] = int(imgt_arm_df["vl"].notna().sum())
                if "inn_id" in imgt_arm_df.columns:
                    inn_id_set: set[str] = set()
                    for raw_inn_id in imgt_arm_df["inn_id"].dropna().tolist():
                        raw_text = str(raw_inn_id).strip()
                        if not raw_text:
                            continue
                        try:
                            inn_id_set.add(str(int(float(raw_text))))
                        except ValueError:
                            inn_id_set.add(raw_text)
                    inn_ids = sorted(inn_id_set)
                    if inn_ids:
                        kw["imgt_2d_inn_id"] = ";".join(inn_ids)

            # Final mol-name normalization layer:
            # for prefixed INN names (e.g., radio-label prefixes), try token-level
            # resolution against known mol-name pools and update inn_mol_name.
            mol_pool = set(thera_inn_by_mol) | set(mabsolve_by_mol)
            if (
                imgt_row is not None or imgt_arm_df is not None
            ) and imgt_mol not in mol_pool:
                token_hit = _token_hit_in_pool(full_inn_name, mol_pool)
                if token_hit and token_hit != imgt_mol:
                    imgt_mol = token_hit
                    kw["inn_mol_name"] = token_hit
                    fuzzy_map[inn] = token_hit
                    reverse_map.setdefault(token_hit, inn)

            # ── TheraSAbDab (exact -> mol -> fuzzy mol) ────────────────
            thera_row = thera_inn_idx.get(inn)
            thera_seq_key = inn if thera_row is not None else None
            if thera_row is None and imgt_mol:
                thera_row = thera_inn_by_mol.get(imgt_mol)
                if thera_row is not None:
                    thera_seq_key = _normalize(thera_row.get("inn_name", ""))
            if thera_row is None and imgt_mol:
                fuzzy_hit = _fuzzy_match_single(imgt_mol, set(thera_inn_by_mol))
                if fuzzy_hit:
                    thera_row = thera_inn_by_mol[fuzzy_hit]
                    thera_seq_key = _normalize(thera_row.get("inn_name", ""))
                    issue_key = f"{inn}|therasabdab"
                    fuzzy_map[issue_key] = fuzzy_hit
                    reverse_map.setdefault(fuzzy_hit, issue_key)
            if thera_row is not None:
                src_list.append("therasabdab")
                for f in _THERA_META_FIELDS:
                    if f in thera_row.index:
                        kw[f] = _safe(thera_row[f])
                n = thera_row.get("n_binders")
                kw["n_binders_therasabdab"] = int(n) if pd.notna(n) else None
                if pd.notna(thera_row.get("inn_number")):
                    inn_ids_for_record.append(
                        ("therasabdab", str(thera_row.get("inn_number")))
                    )

            # ── mAbSolve (exact -> mol -> fuzzy mol) ───────────────────
            mab_row = mabsolve_idx.get(inn)
            if mab_row is None and imgt_mol:
                mab_row = mabsolve_by_mol.get(imgt_mol)
            if mab_row is None and imgt_mol:
                fuzzy_hit = _fuzzy_match_single(imgt_mol, set(mabsolve_by_mol))
                if fuzzy_hit:
                    mab_row = mabsolve_by_mol[fuzzy_hit]
                    issue_key = f"{inn}|mabsolve"
                    fuzzy_map[issue_key] = fuzzy_hit
                    reverse_map.setdefault(fuzzy_hit, issue_key)

            if mab_row is not None:
                src_list.append("mabsolve")
                for csv_col, field in _MABSOLVE_FIELDS.items():
                    if csv_col in mab_row.index:
                        val = mab_row[csv_col]
                        if field in _MABSOLVE_BOOL:
                            kw[field] = bool(val) if not pd.isna(val) else None
                        else:
                            kw[field] = _safe(val)
                if pd.notna(mab_row.get("inn_number")):
                    inn_ids_for_record.append(
                        ("mabsolve", str(mab_row.get("inn_number")))
                    )

            # ── Track INN ID consolidations ─────────────────────────────
            unique_inn_ids = list(set(inn_id for _, inn_id in inn_ids_for_record))
            if len(unique_inn_ids) > 1:
                inn_id_consolidations[inn] = unique_inn_ids

            # ── n_binders mismatch ─────────────────────────────────────
            t_count = kw.get("n_binders_therasabdab")
            i_count = kw.get("n_binders_imgt")
            if t_count is not None and i_count is not None:
                kw["n_binders_mismatch"] = t_count != i_count

            kw["sources"] = ",".join(src_list)
            inn_records.append(MergedINNRecord(**kw))

            # ── Sequence records ───────────────────────────────────────
            if thera_seq_key:
                thera_seq = thera_seq_by_inn.get(thera_seq_key)
            elif imgt_mol:
                thera_seq = thera_seq_by_mol.get(imgt_mol)
            else:
                thera_seq = None

            # Sort both sides by gapless VH so arm pairs align by heavy chain,
            # not arbitrary source ordering (matters for bispecifics).
            # VH is the primary arm identity; VL sort order may differ across sources.
            def _sort_by_seq(df: pd.DataFrame) -> pd.DataFrame:
                key = df["vh"].apply(lambda v: _gapless(_seq_clean(v)) or "")
                return df.iloc[key.argsort()].reset_index(drop=True)

            if thera_seq is not None and len(thera_seq) > 1:
                thera_seq = _sort_by_seq(thera_seq)
            if imgt_arm_df is not None and len(imgt_arm_df) > 1:
                imgt_arm_df = _sort_by_seq(imgt_arm_df)
            n_slots = max(
                len(thera_seq) if thera_seq is not None else 0,
                len(imgt_arm_df) if imgt_arm_df is not None else 0,
            )
            for slot_idx in range(n_slots):
                label = ascii_lowercase[slot_idx] if slot_idx < 26 else str(slot_idx)
                t = (
                    thera_seq.iloc[slot_idx]
                    if thera_seq is not None and slot_idx < len(thera_seq)
                    else None
                )
                i = (
                    imgt_arm_df.iloc[slot_idx]
                    if imgt_arm_df is not None and slot_idx < len(imgt_arm_df)
                    else None
                )

                t_vh = _seq_clean(t["vh"]) if t is not None else None
                t_vl = _seq_clean(t["vl"]) if t is not None else None
                i_vh = _seq_clean(i.get("vh")) if i is not None else None
                i_vl = _seq_clean(i.get("vl")) if i is not None else None

                def _nan_none(val):
                    return (
                        None
                        if val is None or (isinstance(val, float) and pd.isna(val))
                        else str(val)
                    )

                vh, t_vh_mm, i_vh_mm, vh_mismatch, vh_truncated = _seq_resolve(
                    t_vh, i_vh
                )
                vl, t_vl_mm, i_vl_mm, vl_mismatch, vl_truncated = _seq_resolve(
                    t_vl, i_vl
                )
                mismatch = vh_mismatch or vl_mismatch
                truncated = vh_truncated or vl_truncated
                arm_src = (["therasabdab"] if t is not None else []) + (
                    ["imgt"] if i is not None else []
                )

                seq_records.append(
                    MergedSequenceRecord(
                        inn_name=inn,
                        inn_mol_name=kw.get("inn_mol_name"),
                        inn_number=kw.get("inn_number"),
                        molecule_slot=label,
                        sources=",".join(arm_src),
                        vh=vh,
                        vl=vl,
                        sequence_mismatch=mismatch,
                        sequence_truncated=truncated,
                        therasabdab_vh_mismatch=t_vh_mm,
                        therasabdab_vl_mismatch=t_vl_mm,
                        imgt_vh_mismatch=i_vh_mm,
                        imgt_vl_mismatch=i_vl_mm,
                        imgt_vh_v_gene=_nan_none(i.get("vh_v_gene"))
                        if i is not None
                        else None,
                        imgt_vh_j_gene=_nan_none(i.get("vh_j_gene"))
                        if i is not None
                        else None,
                        imgt_vl_v_gene=_nan_none(i.get("vl_v_gene"))
                        if i is not None
                        else None,
                        imgt_vl_j_gene=_nan_none(i.get("vl_j_gene"))
                        if i is not None
                        else None,
                        imgt_vh_total_mutations=(
                            int(i["vh_total_mutations"])
                            if i is not None and pd.notna(i.get("vh_total_mutations"))
                            else None
                        ),
                        imgt_vh_mutations=_nan_none(i.get("vh_mutations"))
                        if i is not None
                        else None,
                        imgt_vl_total_mutations=(
                            int(i["vl_total_mutations"])
                            if i is not None and pd.notna(i.get("vl_total_mutations"))
                            else None
                        ),
                        imgt_vl_mutations=_nan_none(i.get("vl_mutations"))
                        if i is not None
                        else None,
                        imgt_ch1=_nan_none(i.get("ch1")) if i is not None else None,
                        imgt_ch1_gene=_nan_none(i.get("ch1_gene"))
                        if i is not None
                        else None,
                        imgt_ch1_total_mutations=(
                            int(i["ch1_total_mutations"])
                            if i is not None and pd.notna(i.get("ch1_total_mutations"))
                            else None
                        ),
                        imgt_ch2=_nan_none(i.get("ch2")) if i is not None else None,
                        imgt_ch2_gene=_nan_none(i.get("ch2_gene"))
                        if i is not None
                        else None,
                        imgt_ch2_total_mutations=(
                            int(i["ch2_total_mutations"])
                            if i is not None and pd.notna(i.get("ch2_total_mutations"))
                            else None
                        ),
                        imgt_ch3=_nan_none(i.get("ch3")) if i is not None else None,
                        imgt_ch3_gene=_nan_none(i.get("ch3_gene"))
                        if i is not None
                        else None,
                        imgt_ch3_total_mutations=(
                            int(i["ch3_total_mutations"])
                            if i is not None and pd.notna(i.get("ch3_total_mutations"))
                            else None
                        ),
                        imgt_chs=_nan_none(i.get("chs")) if i is not None else None,
                        imgt_chs_gene=_nan_none(i.get("chs_gene"))
                        if i is not None
                        else None,
                        imgt_chs_total_mutations=(
                            int(i["chs_total_mutations"])
                            if i is not None and pd.notna(i.get("chs_total_mutations"))
                            else None
                        ),
                        imgt_tcr_alpha=_nan_none(i.get("tcr_alpha"))
                        if i is not None
                        else None,
                        imgt_tcr_alpha_gene=_nan_none(i.get("tcr_alpha_gene"))
                        if i is not None
                        else None,
                        imgt_tcr_alpha_total_mutations=(
                            int(i["tcr_alpha_total_mutations"])
                            if i is not None
                            and pd.notna(i.get("tcr_alpha_total_mutations"))
                            else None
                        ),
                        imgt_tcr_beta=_nan_none(i.get("tcr_beta"))
                        if i is not None
                        else None,
                        imgt_tcr_beta_gene=_nan_none(i.get("tcr_beta_gene"))
                        if i is not None
                        else None,
                        imgt_tcr_beta_total_mutations=(
                            int(i["tcr_beta_total_mutations"])
                            if i is not None
                            and pd.notna(i.get("tcr_beta_total_mutations"))
                            else None
                        ),
                        imgt_hinge_region=_nan_none(i.get("hinge_region"))
                        if i is not None
                        else None,
                        imgt_hinge_region_total_mutations=(
                            int(i["hinge_region_total_mutations"])
                            if i is not None
                            and pd.notna(i.get("hinge_region_total_mutations"))
                            else None
                        ),
                        imgt_cl=_nan_none(i.get("cl")) if i is not None else None,
                        imgt_cl_type=_nan_none(i.get("cl_type"))
                        if i is not None
                        else None,
                        imgt_cl_total_mutations=(
                            int(i["cl_total_mutations"])
                            if i is not None and pd.notna(i.get("cl_total_mutations"))
                            else None
                        ),
                        imgt_ch_mutations=_nan_none(i.get("ch_mutations"))
                        if i is not None
                        else None,
                        imgt_ch_mutations_imgt=_nan_none(i.get("ch_mutations_imgt"))
                        if i is not None
                        else None,
                        imgt_ch_mutations_eu=_nan_none(i.get("ch_mutations_eu"))
                        if i is not None
                        else None,
                        imgt_ch_mutations_kabat=_nan_none(i.get("ch_mutations_kabat"))
                        if i is not None
                        else None,
                        imgt_cl_mutations=_nan_none(i.get("cl_mutations"))
                        if i is not None
                        else None,
                        imgt_cl_mutations_imgt=_nan_none(i.get("cl_mutations_imgt"))
                        if i is not None
                        else None,
                        imgt_cl_mutations_eu=_nan_none(i.get("cl_mutations_eu"))
                        if i is not None
                        else None,
                        imgt_cl_mutations_kabat=_nan_none(i.get("cl_mutations_kabat"))
                        if i is not None
                        else None,
                        imgt_full_length_heavy=_nan_none(i.get("full_length_heavy"))
                        if i is not None
                        else None,
                        imgt_full_length_light=_nan_none(i.get("full_length_light"))
                        if i is not None
                        else None,
                        imgt_linker=_nan_none(i.get("linker"))
                        if i is not None
                        else None,
                        imgt_other_annotations=_nan_none(
                            i.get("other_annotations_skipped")
                        )
                        if i is not None
                        else None,
                        imgt_vh_chain_id=_nan_none(i.get("vh_chain_id"))
                        if i is not None
                        else None,
                        imgt_vl_chain_id=_nan_none(i.get("vl_chain_id"))
                        if i is not None
                        else None,
                        imgt_cl_chain_id=_nan_none(i.get("cl_chain_id"))
                        if i is not None
                        else None,
                        imgt_ch_detail_order_idxs=_nan_none(
                            i.get("ch_chain_detail_order_idxs")
                        )
                        if i is not None
                        else None,
                        imgt_cl_detail_order_idxs=_nan_none(
                            i.get("cl_chain_detail_order_idxs")
                        )
                        if i is not None
                        else None,
                        target=_nan_none(t["target"]) if t is not None else None,
                        row_sources=_nan_none(t.get("row_sources"))
                        if t is not None
                        else None,
                    )
                )

        return MergedDatabase(
            inn=MergedINNCollection(
                records=inn_records,
                column_sources=[ColumnSource(**cs) for cs in INN_COLUMN_SOURCES],
            ),
            sequences=MergedSequenceCollection(
                records=seq_records,
                column_sources=[ColumnSource(**cs) for cs in SEQ_COLUMN_SOURCES],
            ),
            fuzzy_aliases=fuzzy_map,
            inn_id_consolidations=inn_id_consolidations,
            excluded_imgt=excluded_imgt_dict,
            reverse_map=reverse_map,
            imgt_year_parse_issues={
                "prop_list": sorted(imgt_year_parse_issues["prop_list"]),
                "rec_list": sorted(imgt_year_parse_issues["rec_list"]),
            },
        )


# ---------------------------------------------------------------------------
# Internal: backfill INN binder counts from collection for no-arm INNs
# ---------------------------------------------------------------------------


def _backfill_inn_binder_counts(
    df: pd.DataFrame,
    collection,  # INNStructureCollection
    inn_id_to_name: Dict[str, str],
) -> pd.DataFrame:
    """Backfill n_binders_imgt and related columns for INNs where the arm
    pipeline produced no arm rows (e.g. Fc-fusions with no VH/VL).

    Called by MergedINNCollection.to_dataframe(include_sequences_without_fv=True).

    Rules:
    - All INNs in the collection must already be present in df (asserted).
    - Only rows where n_binders_imgt is NaN or 0 are candidates for update.
    - For those INNs, chain counts are derived from INNChainDetail objects:
        n_binders_imgt       ← number of distinct chain_ids with constant domains
        imgt_2d_chain_count  ← same as n_binders_imgt (one row = one polypeptide)
        imgt_2d_heavy_count  ← chain_ids where chain_type == 'h'
        imgt_2d_light_count  ← chain_ids where chain_type == 'l'
        imgt_2d_inn_id       ← inn_id (unchanged, already present)
    - Updated rows are marked _binders_updated=True.
    """
    df = df.copy()
    df["_binders_updated"] = False

    # Assert all collection INNs are already in df
    df_inn_names: set = set(df["inn_name"].dropna())
    for inn_result in collection.results:
        inn_name = inn_id_to_name.get(inn_result.inn_id) or inn_id_to_name.get(
            str(int(float(inn_result.inn_id))), None
        )
        if inn_name is not None:
            assert inn_name in df_inn_names, (
                f"_backfill_inn_binder_counts: INN '{inn_name}' (id={inn_result.inn_id}) "
                f"is in collection but not in the INN DataFrame. "
                f"This function only updates existing rows, never adds new ones."
            )

    # Candidates: rows where n_binders_imgt is NaN or 0
    needs_update = df["n_binders_imgt"].isna() | (df["n_binders_imgt"] == 0)
    candidate_names: set = set(df.loc[needs_update, "inn_name"].dropna())

    if not candidate_names:
        return df

    # Build updates from collection
    updates: Dict[str, dict] = {}
    for inn_result in collection.results:
        inn_name = inn_id_to_name.get(inn_result.inn_id) or inn_id_to_name.get(
            str(int(float(inn_result.inn_id))), None
        )
        if inn_name not in candidate_names:
            continue

        chains_by_id: dict = {}
        for d in inn_result.chains:
            if d.region_annotation not in {"Full Length", "N.D", "LINKER"}:
                chains_by_id.setdefault(d.chain_id, []).append(d)

        if not chains_by_id:
            continue

        heavy_count = sum(
            1 for ds in chains_by_id.values() if any(d.chain_type == "h" for d in ds)
        )
        light_count = sum(
            1 for ds in chains_by_id.values() if any(d.chain_type == "l" for d in ds)
        )
        n_chains = len(chains_by_id)

        updates[inn_name] = {
            "n_binders_imgt": n_chains,
            "imgt_2d_chain_count": n_chains,
            "imgt_2d_heavy_count": heavy_count,
            "imgt_2d_light_count": light_count,
        }

    if not updates:
        return df

    for inn_name, vals in updates.items():
        mask = df["inn_name"] == inn_name
        for col, val in vals.items():
            df.loc[mask, col] = val
        df.loc[mask, "_binders_updated"] = True

    updated_names = sorted(updates)
    print(
        f"include_sequences_without_fv (INN): updated n_binders_imgt for "
        f"{len(updated_names)} INN(s) — " + ", ".join(updated_names)
    )
    return df


# ---------------------------------------------------------------------------
# Internal: supplement sequence DataFrame with Fc-fusion / no-VH entries
# ---------------------------------------------------------------------------

# region_annotation → sequence column in the merged sequences schema
_DOMAIN_TO_SEQ_COL: Dict[str, str] = {
    "CH1": "imgt_ch1",
    "CH2": "imgt_ch2",
    "CH3": "imgt_ch3",
    "CHS": "imgt_chs",
    "C-ALPHA": "imgt_tcr_alpha",
    "C-BETA-2": "imgt_tcr_beta",
    "HINGE-REGION": "imgt_hinge_region",
}
_DOMAIN_TO_GENE_COL: Dict[str, str] = {
    "CH1": "imgt_ch1_gene",
    "CH2": "imgt_ch2_gene",
    "CH3": "imgt_ch3_gene",
    "CHS": "imgt_chs_gene",
    "C-ALPHA": "imgt_tcr_alpha_gene",
    "C-BETA-2": "imgt_tcr_beta_gene",
}
_DOMAIN_TO_MUT_COL: Dict[str, str] = {
    "CH1": "imgt_ch1_total_mutations",
    "CH2": "imgt_ch2_total_mutations",
    "CH3": "imgt_ch3_total_mutations",
    "CHS": "imgt_chs_total_mutations",
    "C-ALPHA": "imgt_tcr_alpha_total_mutations",
    "C-BETA-2": "imgt_tcr_beta_total_mutations",
    "HINGE-REGION": "imgt_hinge_region_total_mutations",
}
_CL_ANNOTATIONS = {"C-KAPPA", "C-LAMBDA"}


def _supplement_from_collection(
    base_df: pd.DataFrame,
    collection,  # INNStructureCollection
    inn_id_to_name: Dict[str, str],
) -> pd.DataFrame:
    """Append rows for INNs in collection that are absent from base_df (no VH/VL arm).

    Called by MergedSequenceCollection.to_dataframe(include_sequences_without_fv=True).

    What IS populated per appended row (from INNChainDetail):
        inn_name / inn_number / molecule_slot / sources
        imgt_full_length_heavy/_light    ← Full Length region sequence (gapless)
        imgt_ch1 / imgt_ch2 / imgt_ch3  ← per-domain sequence
        imgt_ch1/ch2/ch3_gene            ← realigned_gene or gene_name
        imgt_ch1/ch2/ch3_total_mutations ← len(final_mutations) per domain
        imgt_hinge_region / imgt_hinge_region_total_mutations
        imgt_cl / imgt_cl_type / imgt_cl_total_mutations
        imgt_ch_detail_order_idxs / imgt_cl_detail_order_idxs

    What is NOT populated (unavailable for Fc-fusions without a variable domain):
        vh, vl, imgt_vh/vl_* columns
        imgt_ch_mutations_imgt/eu/kabat  (numbered mutations need arm-pipeline context)
        therasabdab_* / target / row_sources
        sequence_mismatch / sequence_truncated
    """
    covered_inns: set = set(base_df["inn_name"].dropna())
    supplemented: list[dict] = []

    for inn_result in collection.results:
        inn_name = inn_id_to_name.get(inn_result.inn_id) or inn_id_to_name.get(
            str(int(float(inn_result.inn_id))), None
        )
        if inn_name is None or inn_name in covered_inns:
            continue

        chains_by_id: dict = {}
        for d in inn_result.chains:
            chains_by_id.setdefault(d.chain_id, []).append(d)

        heavy_cid = next(
            (
                cid
                for cid, ds in chains_by_id.items()
                if any(d.chain_type == "h" for d in ds)
            ),
            None,
        )
        light_cid = next(
            (
                cid
                for cid, ds in chains_by_id.items()
                if any(d.chain_type == "l" for d in ds)
            ),
            None,
        )

        row: dict = {
            "inn_name": inn_name,
            "inn_number": inn_result.inn_id,
            "molecule_slot": "a",
            "sources": "imgt",
        }

        # Full-length sequences (gapless)
        for cid, col in [
            (heavy_cid, "imgt_full_length_heavy"),
            (light_cid, "imgt_full_length_light"),
        ]:
            if cid is None:
                continue
            fl = next(
                (d for d in chains_by_id[cid] if d.region_annotation == "Full Length"),
                None,
            )
            if fl and fl.sequence:
                row[col] = fl.sequence.replace(".", "")

        # Per-domain data from the heavy chain (fall back to light if no heavy)
        primary_cid = heavy_cid or light_cid
        if primary_cid:
            ch_idxs, cl_idxs = [], []
            for d in chains_by_id[primary_cid]:
                ann = d.region_annotation
                if ann in _DOMAIN_TO_SEQ_COL:
                    row.setdefault(_DOMAIN_TO_SEQ_COL[ann], d.sequence)
                    gene = d.realigned_gene or d.gene_name
                    if gene and ann in _DOMAIN_TO_GENE_COL:
                        row.setdefault(_DOMAIN_TO_GENE_COL[ann], gene)
                    n_muts = len(d.final_mutations)
                    if n_muts:
                        row.setdefault(_DOMAIN_TO_MUT_COL[ann], n_muts)
                    ch_idxs.append(d.detail_order_idx)
                elif ann in _CL_ANNOTATIONS:
                    if "imgt_cl" not in row:
                        row["imgt_cl"] = d.sequence
                        row["imgt_cl_type"] = "kappa" if ann == "C-KAPPA" else "lambda"
                        n_muts = len(d.final_mutations)
                        if n_muts:
                            row["imgt_cl_total_mutations"] = n_muts
                    cl_idxs.append(d.detail_order_idx)
            if ch_idxs:
                row["imgt_ch_detail_order_idxs"] = ";".join(sorted(ch_idxs))
            if cl_idxs:
                row["imgt_cl_detail_order_idxs"] = ";".join(sorted(cl_idxs))

        supplemented.append(row)

    if not supplemented:
        base_df = base_df.copy()
        base_df["_supplemented"] = False
        return base_df

    supplemented_df = pd.DataFrame(supplemented)
    for col in base_df.columns:
        if col not in supplemented_df.columns:
            supplemented_df[col] = None
    supplemented_df = supplemented_df[base_df.columns]

    # Assert no overlap between newly added INNs and original DataFrame
    added_names = set(r["inn_name"] for r in supplemented)
    already_present = added_names & covered_inns
    assert not already_present, (
        f"_supplement_from_collection: the following INN(s) were added but are "
        f"already present in the original DataFrame: {already_present}"
    )

    base_df = base_df.copy()
    base_df["_supplemented"] = False
    supplemented_df["_supplemented"] = True

    print(
        f"include_sequences_without_fv: added {len(supplemented)} row(s) — "
        + ", ".join(sorted(added_names))
    )

    return pd.concat([base_df, supplemented_df], ignore_index=True)
