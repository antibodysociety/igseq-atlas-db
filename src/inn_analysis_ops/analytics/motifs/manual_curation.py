"""Utilities for building localeffector motif PE tables and processing mutation dataframes.

This module replaces ad-hoc notebook logic used to generate
``effector_motif_pe.csv`` from
``motif_df_summary_Categorised_labels.csv``.

It also provides utilities for enriching mutation dataframes with gene/allele information
from germline references.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, Iterable, Optional

import pandas as pd
from inn_analysis_ops.analytics._helpers import (
    format_motif_label,
    parse_motif_label_pe,
    _parse_mutation_pe_into_parts,
)


def _parse_mutation_pe_into_parts_with_error_filter(
    mutation_string_pe: str,
) -> tuple[int, str, str, str]:
    try:
        parsed = _parse_mutation_pe_into_parts(mutation_string_pe)
    except AttributeError:
        print(f"ERROR CANNOT PARSE {mutation_string_pe}")
        return (100000, "Z", "Z", "Z")
    return parsed


def _load_imgt_motif_curation(filename):
    def _parse_ins(row):
        def _parse_token(token, ins_str, eu_pos_imgt_row):
            if "+" not in token:
                return token
            prefix = token.split("+")[0]
            if prefix not in eu_pos_imgt_row:
                raise AssertionError(f"{eu_pos_imgt_row} not found for ins in {token}")
            return f"{prefix}^{ins_str}"

        if "+" not in row["mutation_string_pe"]:
            return row["mutation_string_pe"]
        if "ins" not in row["variant_label"]:
            return row["mutation_string_pe"]
        assert "ins^" in row["variant_label"], row["variant_label"]
        assert row["variant_label"].count("ins^") == 1, row["variant_label"]
        ins_pos = row["variant_label"].split("ins^")[1].split(" ")[0]
        new_string = "/".join(
            [
                _parse_token(token.strip(" "), ins_pos, row["eu_imgt_positions"])
                for token in row["mutation_string_pe"].split("/")
            ]
        )
        return new_string

    imgt_biotechnology_manual_motif_curation = pd.read_csv(filename)
    imgt_biotechnology_manual_motif_curation["mutation_string"] = (
        imgt_biotechnology_manual_motif_curation["mutation_string_pe"].map(
            parse_motif_label_pe
        )
    )
    imgt_biotechnology_manual_motif_curation["mutation_string_pe"] = (
        imgt_biotechnology_manual_motif_curation["mutation_string"]
        .map(format_motif_label)
        .apply(lambda x: x.replace(" ", ""))
    )
    imgt_biotechnology_manual_motif_curation["motif_size"] = (
        imgt_biotechnology_manual_motif_curation["mutation_string_pe"].map(
            lambda x: len(x.split("/"))
        )
    )

    imgt_biotechnology_manual_motif_curation["mutation_string_pe_pre_reparse"] = (
        imgt_biotechnology_manual_motif_curation["mutation_string_pe"].copy()
    )
    imgt_biotechnology_manual_motif_curation["mutation_string_pe"] = (
        imgt_biotechnology_manual_motif_curation.fillna("").apply(_parse_ins, axis=1)
    )

    imgt_biotechnology_manual_motif_curation["mutation_string_pre_reparse"] = (
        imgt_biotechnology_manual_motif_curation["mutation_string"].copy()
    )
    imgt_biotechnology_manual_motif_curation["mutation_string"] = (
        imgt_biotechnology_manual_motif_curation["mutation_string_pe"].apply(
            lambda x: (
                parse_motif_label_pe(x)
                .replace("->", ">")
                .replace(":", "")
                .replace(" ", "")
                .replace("/", ",")
                .replace("del", "-")
            )
        )
    )

    # make sure its in order
    imgt_biotechnology_manual_motif_curation["mutation_string_pe"] = (
        imgt_biotechnology_manual_motif_curation["mutation_string_pe"].apply(
            lambda x: "/".join(
                sorted(
                    [tkn.replace("del", "-") for tkn in x.split("/")],
                    key=_parse_mutation_pe_into_parts_with_error_filter,
                )
            )
        )
    )

    return imgt_biotechnology_manual_motif_curation


EFF_CATEGORY_COLUMNS: tuple[str, ...] = (
    "Mut_category",
    "Mut_category_2",
    "Mut_category_3",
)

BASE_EXPORT_COLUMNS: tuple[str, ...] = (
    "mutation_string_pe",
    "count",
    "freq_motif",
    "pvalue",
    "lift",
    "Mut_category",
    "Mut_category_2",
    "Mut_category_3",
    "mutation_string",
)

EXPORT_COLUMNS_WITH_GROUP: tuple[str, ...] = BASE_EXPORT_COLUMNS + (
    "total_calls",
    "group",
    "sub_classification",
)

_UNKNOWN_TOKENS: set[str] = {"?", "unknown"}
_NONE_VALUES: set[str] = {"", "nan", "none", "null"}


def _display_label(value: str) -> str:
    txt = str(value).strip().replace("_", " ")
    if not txt:
        return txt
    return " ".join(part.capitalize() for part in txt.split())


def normalize_eff_category(value: object) -> Optional[str]:
    """Normalize category labels into canonical text tokens."""
    raw = "" if pd.isna(value) else str(value)
    if not raw or raw.lower() in _NONE_VALUES:
        return None
    cleaned = raw.replace("*IgG4", "").replace("Mixed", "").strip()

    return "unknown" if cleaned.lower() in _UNKNOWN_TOKENS else cleaned


def mutation_to_pe_token(mutation: str) -> str | None:
    """Convert one mutation token like ``216:E->-`` to ``E216-``.

    Insertions represented with ``+`` (e.g., ``221:+``) are converted to
    ``.221+``.
    """
    if pd.isna(mutation):
        return None
    token = str(mutation).strip()
    if not token:
        return None
    if "+" in token:
        pos = token.split(":")[0].strip()
        return f".{pos}+"
    pos, rest = token.split(":")
    wt, mut = rest.split("->")
    return f"{wt}{pos}{mut}".replace(" ", "")


def mutation_string_to_pe(mutation_string: str) -> str:
    """Convert a multi-mutation string to PE slash notation."""
    pieces = re.split(r"[;/]", str(mutation_string))
    pe_tokens = [
        mutation_to_pe_token(piece.strip())
        for piece in pieces
        if piece and str(piece).strip()
    ]
    return "/".join(tok.lstrip(".") for tok in pe_tokens if tok)


def normalize_mutation_string(mutation_string: str) -> str:
    """Normalize mutation separators to the expected `` / `` format."""
    return str(mutation_string).replace(";", " /").replace("  ", " ").strip()


def _sort_mut_class_order(label: str) -> int:
    """Sort key for class labels."""
    preferred_order = [
        "effector",
        "silence",
        "structure",
        "halflife",
        "other",
        "unknown",
    ]

    try:
        return preferred_order.index(label.lower())
    except ValueError:
        return len(preferred_order)


def classify_group(categories: Iterable[str]) -> tuple[int, str, str]:
    """Return ``(total_calls, group, sub_classification)``.

    Rules:
    - Unknown-like values (unknown, ?, none, null, blank) become ``unknown``.
    - If only ``other`` + ``unknown`` are present, collapse to ``unknown``.
    - Single unique category => that category as group.
    - Multiple distinct informative categories => ``multiple`` group.
    """

    cleaned = [normalize_eff_category(c) for c in categories]
    cleaned_lower = [c.lower() for c in cleaned if c]
    if not cleaned_lower:
        # no mapping of information to this label
        return 0, _display_label("unknown"), _display_label("unknown")
    # informative = [
    #     c
    #     for c in cleaned_lower
    #     if c.lower() not in ["unknown", "undefined"]
    # ]
    informative = cleaned_lower
    total_calls = len(informative)
    ordered_unique = list(dict.fromkeys(informative))
    if not ordered_unique:
        return 0, _display_label("unknown"), _display_label("unknown")
    if set(ordered_unique).issubset({"other"}):
        return (
            total_calls,
            _display_label("other"),
            _display_label("other"),
        )

    if len(ordered_unique) == 1:
        cat = ordered_unique[0]
        disp = _display_label(cat)
        return total_calls, disp, disp

    return (
        total_calls,
        _display_label("multiple"),
        ",".join(
            sorted(
                [_display_label(c) for c in ordered_unique],
                key=lambda x: _sort_mut_class_order(x),
            )
        ),
    )


def build_effector_single_motifs_dataframe(
    source: str | Path | pd.DataFrame, melt_data: bool = True
) -> pd.DataFrame:
    """Build normalized single mutation dataframe ready for PE export."""
    effector_single_mut_df = pd.read_excel(source).dropna(subset=["mutation_string_pe"])
    effector_single_mut_df["wt"] = effector_single_mut_df["mutation_string_pe"].apply(
        lambda x: x[0]
    )
    effector_single_mut_df["mut"] = effector_single_mut_df["mutation_string_pe"].apply(
        lambda x: (
            x[-1]
            if "+" not in x and "(" not in x
            else "+"
            if "+" in x
            else x.split("(")[0][-1]
        )
    )
    if melt_data:
        effector_single_mut_df = effector_single_mut_df.melt(
            id_vars=["position", "mutation_string_pe", "wt", "mut"],
            value_vars=[
                "Effector",
                "Halflife",
                "Other",
                "Silence",
                "Structure",
                "Unknown",
            ],
            var_name="group",
            value_name="count",
        ).dropna(subset=["count"])
        effector_single_mut_df["sub_classification_grp"] = effector_single_mut_df[
            "group"
        ]

        effector_single_mut_df["sub_classification"] = effector_single_mut_df.groupby(
            by="mutation_string_pe"
        )["sub_classification_grp"].transform(
            lambda x: ",".join(sorted(x, key=_sort_mut_class_order))
        )
        effector_single_mut_df = (
            effector_single_mut_df.groupby(by="mutation_string_pe", as_index=False)
            .first()
            .drop("sub_classification_grp", axis=1)
        )
    effector_single_mut_df["total_calls"] = effector_single_mut_df[
        "sub_classification"
    ].apply(lambda x: len(x.split(",")))
    effector_single_mut_df["group"] = (
        effector_single_mut_df["group"]
        .str.lower()
        .replace("undefined", "unknown")
        .str.capitalize()
    )
    effector_single_mut_df.loc[
        effector_single_mut_df["position"].isna(), "position"
    ] = effector_single_mut_df.loc[
        effector_single_mut_df["position"].isna(),
        "mutation_string_pe",
    ].apply(
        lambda x: float(re.search(r"\d+", x).group(0)) if re.search(r"\d+", x) else None
    )
    return effector_single_mut_df


def build_effector_motif_pe_dataframe(
    source: str | Path | pd.DataFrame,
) -> pd.DataFrame:
    """Build normalized  motif dataframe ready for PE export."""
    df = pd.read_csv(source) if isinstance(source, (str, Path)) else source.copy()

    df.loc[df["mutation_string"].isnull(), "mutation_string"] = df.loc[
        df["mutation_string"].isnull(), "mutation_string_pe"
    ].map(parse_motif_label_pe)

    # delete bad rwols
    bad_rows = df[df["mutation_string"].map(lambda x: ":" not in x)]
    print(
        f"Deleting {len(bad_rows)} rows with bad mutation_string: {bad_rows['mutation_string'].tolist()}"
    )
    df = df.drop(bad_rows.index)
    if "mutation_string" not in df.columns:
        raise ValueError("Input data must contain 'mutation_string'.")

    out = df.copy()
    out["mutation_string"] = out["mutation_string"].map(normalize_mutation_string)
    out["mutation_string_pe"] = out["mutation_string"].map(mutation_string_to_pe)

    found_columns = [col for col in EFF_CATEGORY_COLUMNS if col in out.columns]
    if not found_columns:
        raise AssertionError(
            f"No category columns found in the input data: {out.columns.tolist()}. "
            f"Expected {EFF_CATEGORY_COLUMNS}"
        )
    for col in found_columns:
        out[col] = out[col].map(normalize_eff_category)

    group_info = out.loc[:, found_columns].apply(
        lambda row: classify_group(row.values.tolist()), axis=1
    )
    out["total_calls"] = group_info.map(lambda t: t[0])
    out["group"] = group_info.map(lambda t: t[1])
    out["sub_classification"] = group_info.map(lambda t: t[2])

    return out


def update_effector_motif_pe_dataframe_with_single_calls(
    motif_df: pd.DataFrame,
    single_mut_df: pd.DataFrame,
) -> pd.DataFrame:
    motif_df["size"] = motif_df["mutation_string_pe"].apply(lambda x: len(x.split("/")))
    assert motif_df["size"].isnull().sum() == 0
    assert motif_df["size"].min() >= 1
    motif_df_multiple = motif_df.query("size>1").copy()
    motif_df_single = (
        motif_df.query(
            "(size==1)&(mutation_string_pe in @single_mut_df['mutation_string_pe'])"
        )
        .copy()
        .drop(["group", "sub_classification"], axis=1)
    )
    motif_df_single = motif_df_single.merge(
        single_mut_df[
            [
                "mutation_string_pe",
                "group",
                "sub_classification",
                "gene",
                "allele",
                "gene_species",
            ]
        ],
        on="mutation_string_pe",
        how="outer",
    )

    return pd.concat([motif_df_multiple, motif_df_single], axis=0, ignore_index=True)


def report_effector_coverage(
    motif_df: pd.DataFrame,
    effector_df: pd.DataFrame,
    motif_col: str = "mutation_string_pe",
    type_col: str = "sub_classification",
) -> pd.DataFrame:
    """Report effector type coverage for motifs absent from *effector_df*.

    For each motif in *motif_df* not already in *effector_df*, decomposes it
    into sub-motifs and individual mutations, then reports what can and cannot
    be classified.

    Returns
    -------
    pd.DataFrame with one row per input motif and columns:
        ``mutation_string``              — original mutation string (if present)
        ``mutation_string_pe``           — PE key
        ``inn_count``                    — INN count (if present)
        ``inns``                         — INN list (if present)
        ``submotif_classifications``     — dict {submotif_pe: type} for every
            multi-mutation sub-motif found in effector_df
        ``ind_mutation_classifications`` — dict {mutation_pe: type} for every
            single mutation found in effector_df
        ``unknown_ind_mutations``        — list of individual mutations not
            found in effector_df
        ``unknown_submotifs``            — list of composite sub-motifs (≥2)
            not found in effector_df
    """
    from itertools import combinations as _combs

    if motif_col not in motif_df.columns:
        raise ValueError(f"motif_df missing column '{motif_col}'")
    if motif_col not in effector_df.columns or type_col not in effector_df.columns:
        raise ValueError(
            f"effector_df missing column(s): "
            f"{[c for c in [motif_col, type_col] if c not in effector_df.columns]}"
        )

    _eff_lookup: dict[str, str] = {}
    for _, row in effector_df.dropna(subset=[motif_col]).iterrows():
        key = str(row[motif_col]).replace(" ", "").lower()
        _eff_lookup[key] = str(row[type_col]).strip()

    already_in_effector = set(_eff_lookup.keys())
    missing_motifs = motif_df[
        ~motif_df[motif_col]
        .map(lambda x: str(x).replace(" ", "").lower())
        .isin(already_in_effector)
    ]

    # Passthrough columns — include if present in motif_df
    _passthrough = [
        "mutation_string",
        "num_mutations",
        "inn_count",
        "inns",
    ]
    _have = [c for c in _passthrough if c in motif_df.columns]

    records = []
    for _, row in missing_motifs.iterrows():
        pe = str(row[motif_col])
        tokens = [t.strip() for t in pe.split("/") if t.strip()]

        submotif_classifications: dict[str, str] = {}
        ind_mutation_classifications: dict[str, str] = {}
        unknown_ind_mutations: list[str] = []
        unknown_submotifs: list[str] = []

        if len(tokens) > 1:
            for size in range(len(tokens), 1, -1):
                for combo in _combs(tokens, size):
                    sub_key = "/".join(combo).replace(" ", "").lower()
                    if sub_key in _eff_lookup:
                        submotif_classifications[sub_key] = _eff_lookup[sub_key]
                    else:
                        unknown_submotifs.append(sub_key)

        for tok in tokens:
            ind_key = tok.replace(" ", "").lower()
            if ind_key in _eff_lookup:
                ind_mutation_classifications[ind_key] = _eff_lookup[ind_key]
            else:
                unknown_ind_mutations.append(ind_key)

        record: dict = {c: row.get(c) for c in _have}
        record[motif_col] = pe
        record["submotif_classifications"] = submotif_classifications
        record["ind_mutation_classifications"] = ind_mutation_classifications
        record["unknown_ind_mutations"] = unknown_ind_mutations
        record["unknown_submotifs"] = unknown_submotifs
        records.append(record)

    out_cols = _have + [
        motif_col,
        "submotif_classifications",
        "ind_mutation_classifications",
        "unknown_ind_mutations",
        "unknown_submotifs",
    ]
    return pd.DataFrame(records, columns=out_cols)


def debug_gene_allele_lookup(
    mutation_string_pe: str,
    inn_summary: pd.DataFrame,
    result_df: Optional[pd.DataFrame] = None,
) -> None:
    """Print a step-by-step trace of why a motif fails to match in
    :func:`add_gene_allele_info`.

    Runs the lookup in both modes:
    - ``join_on_mutated_seq=True``  (default) — matches (germline, EU, mutated)
    - ``join_on_mutated_seq=False`` — matches (germline, EU) only, ignores mutated residue

    Parameters
    ----------
    mutation_string_pe : str
        The PE motif string, e.g. ``"E233-/F234P/L235V/G236A/M428L/N434S"``.
    inn_summary : pd.DataFrame
        Raw INN summary dataframe.
    result_df : pd.DataFrame, optional
        If provided, also checks superstring propagation.
    """
    unique_germline_pos_combo = build_unique_germline_pos_combo(inn_summary)
    idx_names = unique_germline_pos_combo.index.names
    _ref_flat = unique_germline_pos_combo.reset_index()
    tokens = [t.strip().upper() for t in mutation_string_pe.split("/") if t.strip()]

    for mode, use_mut in [
        ("join_on_mutated_seq=True", True),
        ("join_on_mutated_seq=False", False),
    ]:
        print(f"\n{'═' * 60}")
        print(f"Mode: {mode}")
        print(f"{'═' * 60}")

        # Build lookup
        _tok_lookup: dict[str, set] = {}
        for _, ref_row in _ref_flat.iterrows():
            wt = str(ref_row[idx_names[0]]).upper()
            pos = str(ref_row[idx_names[1]])
            mut = str(ref_row[idx_names[2]]).upper()
            key = f"{wt}{pos}{mut}" if use_mut else f"{wt}{pos}"
            combo = (
                str(ref_row.get("gene", "")),
                str(ref_row.get("allele", "")),
                str(ref_row.get("gene_species", "")),
            )
            _tok_lookup.setdefault(key, set()).add(combo)

        running_intersection: Optional[set] = None
        for tok in tokens:
            query_key = (
                tok
                if use_mut
                else tok[0] + "".join(c for c in tok[1:] if c.isdigit() or c == ".")
            )
            # For position-only mode strip trailing mutation letter/symbol
            if not use_mut:
                import re as _re

                _m = _re.match(r"([A-Za-z.])([0-9]+(?:\.[A-Za-z0-9]+)?)", tok)
                query_key = f"{_m.group(1).upper()}{_m.group(2)}" if _m else tok
            combos = _tok_lookup.get(query_key, set())
            status = f"✓ {len(combos)} combo(s)" if combos else "✗ NOT FOUND"
            sample = sorted(combos)[:2] if combos else []
            print(f"  {tok:20s} key='{query_key}'  {status}  {sample}")
            if running_intersection is None:
                running_intersection = set(combos)
            else:
                before = len(running_intersection)
                running_intersection &= combos
                if len(running_intersection) < before:
                    print(
                        f"    ↳ intersection dropped from {before} → {len(running_intersection)}"
                    )
        print(f"  Intersection: {len(running_intersection or set())} combo(s)")
        if running_intersection:
            for c in sorted(running_intersection):
                print(f"    gene={c[0]}, allele={c[1]}, gene_species={c[2]}")
        else:
            # Show majority-vote result
            from collections import Counter as _Ctr

            _vote: _Ctr = _Ctr()
            for tok in tokens:
                query_key = (
                    tok
                    if use_mut
                    else (
                        lambda t: (
                            lambda m: f"{m.group(1).upper()}{m.group(2)}" if m else t
                        )(
                            __import__("re").match(
                                r"([A-Za-z.])([0-9]+(?:\.[A-Za-z0-9]+)?)", t
                            )
                        )
                    )(tok)
                )
                for combo in _tok_lookup.get(query_key, set()):
                    _vote[combo] += 1
            if _vote:
                _max_v = max(_vote.values())
                print(
                    f"  Majority-vote fallback (max {_max_v}/{len(tokens)} tokens covered):"
                )
                for combo, v in sorted(_vote.items(), key=lambda x: -x[1]):
                    if v == _max_v:
                        print(
                            f"    gene={combo[0]}, allele={combo[1]}, gene_species={combo[2]}  ({v} tokens)"
                        )

    # Superstring check
    print(f"\n{'═' * 60}")
    print("Step 4b: superstring propagation")
    print(f"{'═' * 60}")
    if result_df is None:
        print("  (Pass result_df= to check)")
        return
    if "mutation_string_pe" not in result_df.columns:
        print("  result_df has no 'mutation_string_pe' column")
        return
    _meta_cols = [
        c for c in ("gene", "allele", "gene_species") if c in result_df.columns
    ]
    _ntoks = frozenset(t.upper() for t in mutation_string_pe.split("/") if t.strip())
    _filled = result_df.dropna(subset=_meta_cols) if _meta_cols else result_df
    _super_matches = []
    for _, _fr in _filled.iterrows():
        _fpe = str(_fr["mutation_string_pe"]).replace(" ", "").upper()
        _ftoks = frozenset(t for t in _fpe.split("/") if t)
        if _ntoks <= _ftoks and _fpe != mutation_string_pe.replace(" ", "").upper():
            _super_matches.append(
                (_fpe, tuple(str(_fr.get(c, "")) for c in _meta_cols))
            )
    if not _super_matches:
        print("  No filled superstring rows found.")
    else:
        print(f"  Found {len(_super_matches)} superstring row(s):")
        for _fpe, _combo in sorted(_super_matches):
            print(f"    {_fpe}  →  {dict(zip(_meta_cols, _combo))}")

    print(f"Final result: {len(running_intersection or set())} matching combo(s)")
    if running_intersection:
        for c in sorted(running_intersection):
            print(f"  gene={c[0]}, allele={c[1]}, gene_species={c[2]}")


def write_effector_motif_pe_csv(
    source_csv: str | Path,
    output_csv: str | Path,
) -> pd.DataFrame:
    """Generate and write the canonical PE table CSV."""
    built = build_effector_motif_pe_dataframe(source_csv)
    available_cols = [c for c in EXPORT_COLUMNS_WITH_GROUP if c in built.columns]
    built.loc[:, available_cols].to_csv(output_csv, index=False)
    return built


def build_unique_germline_pos_combo(
    inn_summary: pd.DataFrame,
) -> pd.DataFrame:
    """Build the germline position reference table from INN summary data.

    For each (germline_seq, EU, mutated_seq) combination, selects the
    most-frequent (gene, allele, gene_species) assignment.

    Parameters
    ----------
    inn_summary : pd.DataFrame
        INN summary dataframe.  Must contain columns:
        ``gene``, ``allele``, ``gene_species``, ``EU``,
        ``germline_seq``, ``mutated_seq``.

    Returns
    -------
    pd.DataFrame
        DataFrame indexed by (germline_seq, EU, mutated_seq) with columns
        ``gene``, ``allele``, ``gene_species``, ``count``.
    """
    required = {
        "gene",
        "allele",
        "gene_species",
        "EU",
        "germline_seq",
        "mutated_seq",
    }
    missing = required - set(inn_summary.reset_index().columns)
    if missing:
        raise ValueError(f"inn_summary missing columns: {sorted(missing)}")
    return (
        inn_summary.reset_index()[
            [
                "gene",
                "allele",
                "gene_species",
                "EU",
                "germline_seq",
                "mutated_seq",
            ]
        ]
        .drop_duplicates()
        .set_index(["germline_seq", "EU", "mutated_seq"])
        .sort_index()
    )


# Gene/allele enrichment utilities

SPECIES_NAME_TO_CODE: dict[str, str] = {
    "Homo sapiens": "human",
    "Mus musculus": "mouse",
    "Felis catus": "Felis catus",
    "Canis lupus familiares": "dog",
}


def fill_gene_allele_from_imgt_columns(
    df: pd.DataFrame,
    species_mapping: Optional[Dict[str, str]] = None,
) -> pd.DataFrame:
    """Fill missing gene/allele/gene_species from IMGT annotation columns.

    Uses the ``Species`` and ``IMGT gene name`` columns that are typically
    present on single-mutation tables from IMGT data, applying them as
    fallbacks for any row where ``gene_species``, ``gene``, or ``allele``
    are NaN.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe.  Columns ``gene_species``, ``gene``, ``allele``,
        ``Species``, and ``IMGT gene name`` are used when present.
    species_mapping : dict, optional
        NCBI species name → code.  Defaults to ``SPECIES_NAME_TO_CODE``.

    Returns
    -------
    pd.DataFrame
        Copy of *df* with ``gene_species``, ``gene``, ``allele`` filled in
        where possible.
    """
    if species_mapping is None:
        species_mapping = SPECIES_NAME_TO_CODE

    result_df = df.copy()

    # Ensure columns exist
    for col in ("gene_species", "gene", "allele"):
        if col not in result_df.columns:
            result_df[col] = None

    # Fill missing gene_species from Species column
    if "Species" in result_df.columns:
        null_gs = result_df["gene_species"].isnull()
        if null_gs.any():
            result_df.loc[null_gs, "gene_species"] = result_df.loc[
                null_gs, "Species"
            ].map(lambda x: species_mapping.get(x) if not pd.isnull(x) else None)

        # Backfill gene_species where mapped Species disagrees with current value
        _mapped = result_df["Species"].map(
            lambda x: species_mapping.get(x) if not pd.isnull(x) else None
        )
        _mismatch = _mapped.notna() & (_mapped != result_df["gene_species"])
        if _mismatch.any():
            result_df.loc[_mismatch, "gene_species"] = _mapped[_mismatch]

    # Fill missing gene from IMGT gene name
    if "IMGT gene name" in result_df.columns:
        null_gene = result_df["gene"].isnull()
        if null_gene.any():
            result_df.loc[null_gene, "gene"] = result_df.loc[
                null_gene, "IMGT gene name"
            ].map(lambda x: x.lower() if not pd.isnull(x) else x)

        # Correct mismatches between IMGT gene name and gene
        _imgt_lower = result_df["IMGT gene name"].map(
            lambda x: x.lower() if not pd.isnull(x) else None
        )
        _gene_mismatch = _imgt_lower.notna() & (result_df["gene"] != _imgt_lower)
        if _gene_mismatch.any():
            result_df.loc[_gene_mismatch, "gene"] = _imgt_lower[_gene_mismatch]

        # Fill missing allele from IMGT gene name + "*01"
        null_allele = result_df["allele"].isnull()
        if null_allele.any():
            result_df.loc[null_allele, "allele"] = result_df.loc[
                null_allele, "IMGT gene name"
            ].map(lambda x: f"{x.lower()}*01" if not pd.isnull(x) else x)

    return result_df


def add_gene_allele_info(
    motif_df: pd.DataFrame,
    inn_summary: Optional[pd.DataFrame] = None,
    unique_germline_pos_combo: Optional[pd.DataFrame] = None,
    join_on_mutated_seq: bool = True,
    species_mapping: Optional[Dict[str, str]] = None,
) -> pd.DataFrame:
    """Add gene, allele, and gene_species to a motif dataframe by matching each
    motif's constituent mutations against germline position data.

    Pass either ``inn_summary`` (raw INN summary — the reference table is
    built automatically via :func:`build_unique_germline_pos_combo`) or a
    pre-built ``unique_germline_pos_combo``.  Exactly one must be provided.

    For each row where gene/allele/gene_species are missing, splits
    ``mutation_string_pe`` on ``/`` into individual mutations, and finds all
    ``(gene, gene_species)`` subclasses present in the reference for ALL of
    those mutations (allele is intentionally ignored when matching subclass
    coverage, since different positions along the same chain are often
    genotyped against slightly different alleles of the same gene). For each
    surviving ``(gene, gene_species)``, ``allele`` is filled with the
    comma-joined union of every allele observed for that subclass across the
    covering tokens. The motif df is exploded on those ``(gene, gene_species)``
    options. Asserts the result is unique on the
    (mutation_string_pe, gene, allele, gene_species) key after exploding.

    Parameters
    ----------
    motif_df : pd.DataFrame
        Input motif dataframe.  Must contain ``mutation_string_pe``.
    unique_germline_pos_combo : pd.DataFrame
        Germline position reference data.  Index must be a MultiIndex of
        ``(germline_seq, eu_position)`` when ``join_on_mutated_seq=False``, or
        ``(germline_seq, eu_position, mutated_seq)`` when
        ``join_on_mutated_seq=True``.
        Must contain columns: ``gene_species``, ``allele``, ``gene``.
    join_on_mutated_seq : bool
        When True (default), each mutation token ``"WTPOS->MUT"`` is matched
        on ``(germline_seq=WT, eu_position=POS, mutated_seq=MUT)``.
        When False, matched on ``(germline_seq=WT, eu_position=POS)`` only.
    species_mapping : dict, optional
        Mapping from NCBI species names to standardised codes.
        Defaults to ``SPECIES_NAME_TO_CODE``.

    Returns
    -------
    pd.DataFrame
        Motif dataframe with ``gene``, ``allele``, and ``gene_species`` filled
        for previously-empty rows.  Rows with multiple matching combinations
        are exploded (one row per combination).  Rows that already had all
        three columns populated are returned unchanged.
    """
    if species_mapping is None:
        species_mapping = SPECIES_NAME_TO_CODE

    if "mutation_string_pe" not in motif_df.columns:
        raise ValueError("motif_df must contain 'mutation_string_pe'")

    if inn_summary is not None and unique_germline_pos_combo is not None:
        raise ValueError(
            "Pass either inn_summary or unique_germline_pos_combo, not both."
        )
    if inn_summary is None and unique_germline_pos_combo is None:
        raise ValueError(
            "One of inn_summary or unique_germline_pos_combo must be provided."
        )
    if inn_summary is not None:
        unique_germline_pos_combo = build_unique_germline_pos_combo(inn_summary)

    required_ref_cols = {"gene_species", "allele", "gene"}
    missing_ref = required_ref_cols - set(unique_germline_pos_combo.columns)
    if missing_ref:
        raise ValueError(
            f"unique_germline_pos_combo missing columns: {sorted(missing_ref)}"
        )

    # ── 0. Pre-fill from IMGT annotation columns (Species / IMGT gene name) ──
    # This runs first so any row that can be resolved via IMGT metadata is
    # already populated before the germline-lookup + explode steps.
    motif_df = fill_gene_allele_from_imgt_columns(motif_df, species_mapping)

    # ── 1. Split rows: already-filled vs needs enrichment ────────────────────
    _needs = (
        (
            motif_df["gene"].isna()
            | motif_df["allele"].isna()
            | motif_df["gene_species"].isna()
        )
        if all(c in motif_df.columns for c in ("gene", "allele", "gene_species"))
        else pd.Series(True, index=motif_df.index)
    )

    already_filled = motif_df[~_needs].copy()
    to_enrich = motif_df[_needs].copy()

    if to_enrich.empty:
        return motif_df.copy()

    # ── 2. Build lookup from unique_germline_pos_combo ───────────────────────
    # Reconstruct the compact PE token from index levels (germline_seq, EU,
    # mutated_seq) by simple concatenation — e.g. ("L", "234", "A") → "L234A",
    # ("K", "447", "-") → "K447-".  No parsing of input tokens required.
    if unique_germline_pos_combo is None:
        raise ValueError(
            "unique_germline_pos_combo is required. "
            "Build it with build_unique_germline_pos_combo(inn_summary) first."
        )

    _ref_flat = unique_germline_pos_combo.reset_index()
    idx_names = unique_germline_pos_combo.index.names  # (germline_seq, EU, mutated_seq)

    # Compact key: wt + position + mutated (all as strings, uppercased)
    def _make_key(row) -> str:
        wt = str(row[idx_names[0]]).upper()
        pos = str(row[idx_names[1]])
        mut = str(row[idx_names[2]]).upper()
        return f"{wt}{pos}{mut}"

    # {compact_pe_token: set of (gene, allele, gene_species)}
    _meta_cols = [
        c for c in ("gene", "allele", "gene_species") if c in _ref_flat.columns
    ]
    _has_allele_col = "allele" in _meta_cols
    _tok_lookup: dict[str, list] = {}
    for _, ref_row in _ref_flat.iterrows():
        key = _make_key(ref_row)
        combo = tuple(str(ref_row.get(c, "")) for c in _meta_cols)
        _tok_lookup.setdefault(key, set()).add(combo)

    # (gene, gene_species) position(s) within a combo tuple, for matching
    # subclass identity independently of the allele call.
    _gs_cols = [c for c in ("gene", "gene_species") if c in _meta_cols]
    _gs_idx = [_meta_cols.index(c) for c in _gs_cols]
    _allele_idx = _meta_cols.index("allele") if _has_allele_col else None

    def _gs_key(combo: tuple) -> tuple:
        return tuple(combo[i] for i in _gs_idx)

    # ── 3. For each row to enrich, find combos covering ALL mutations ─────────
    # Primary: intersect on (gene, gene_species) only — a motif is attributed
    # to a subclass if EVERY token has at least one observation under that
    # subclass, regardless of which specific allele each individual position
    # was called against (different positions along the same chain are often
    # genotyped to slightly different alleles of the same gene/subclass).
    # The allele is then resolved separately: it's the union of all alleles
    # observed for that (gene, gene_species) across the covering tokens.
    # Fallback: if the (gene, gene_species) intersection is empty, find the
    # subclass that covers the MOST tokens (majority vote). This handles
    # cross-subclass engineered antibodies where some mutations are
    # germline-state in one subclass and therefore absent from the reference.
    enriched_records = []
    for _, row in to_enrich.iterrows():
        pe = str(row["mutation_string_pe"])
        tokens = [t.strip().upper() for t in pe.split("/") if t.strip()]

        # Per-token combos, keyed by (gene, gene_species) -> set of combos
        _per_tok_gs: list[dict[tuple, set]] = []
        for tok in tokens:
            _gs_map: dict[tuple, set] = {}
            for combo in _tok_lookup.get(tok, set()):
                _gs_map.setdefault(_gs_key(combo), set()).add(combo)
            _per_tok_gs.append(_gs_map)

        # Intersect on (gene, gene_species) across all tokens.
        valid_gs: Optional[set] = None
        for _gs_map in _per_tok_gs:
            _gs_keys = set(_gs_map.keys())
            valid_gs = _gs_keys if valid_gs is None else (valid_gs & _gs_keys)

        if not valid_gs:
            # Majority-vote fallback: score each (gene, gene_species) by how
            # many tokens it covers.
            from collections import Counter as _Counter

            _vote: _Counter = _Counter()
            for _gs_map in _per_tok_gs:
                for gs_key in _gs_map:
                    _vote[gs_key] += 1
            if _vote:
                _max_votes = max(_vote.values())
                valid_gs = {gs for gs, v in _vote.items() if v == _max_votes}

        if not valid_gs:
            enriched_records.append(row.to_dict())
            continue

        # For each surviving (gene, gene_species), resolve allele as the
        # union of alleles observed for it across all covering tokens.
        valid_combos = set()
        for gs_key in valid_gs:
            if not _has_allele_col:
                # No allele column to resolve — combo is just the gs_key
                # re-expanded back into meta-col order (allele absent).
                combo = [""] * len(_meta_cols)
                for i, v in zip(_gs_idx, gs_key):
                    combo[i] = v
                valid_combos.add(tuple(combo))
                continue

            _alleles: set = set()
            for _gs_map in _per_tok_gs:
                for combo in _gs_map.get(gs_key, ()):
                    _alleles.add(combo[_allele_idx])
            _allele_val = ",".join(sorted(a for a in _alleles if a))
            combo = [""] * len(_meta_cols)
            for i, v in zip(_gs_idx, gs_key):
                combo[i] = v
            combo[_allele_idx] = _allele_val
            valid_combos.add(tuple(combo))

        # Explode: one record per matching (gene, gene_species) combo
        for combo in sorted(valid_combos):
            rec = row.to_dict()
            for col, val in zip(_meta_cols, combo):
                rec[col] = val
            enriched_records.append(rec)

    enriched_df = pd.DataFrame(enriched_records, columns=to_enrich.columns)

    # ── 4. Combine with already-filled rows ───────────────────────────────────
    result = pd.concat([already_filled, enriched_df], ignore_index=True)

    # ── 4b. Superstring propagation ───────────────────────────────────────────
    # For any row still missing gene/allele/gene_species, find rows in the
    # result whose mutation_string_pe is a SUPERSTRING (contains all tokens
    # of the null row as a subset).  Copy all observed (gene, allele,
    # gene_species) combos from those superstring rows and explode.
    if "mutation_string_pe" in result.columns:
        _still_null = (
            (
                result["gene"].isna()
                | result["allele"].isna()
                | result["gene_species"].isna()
            )
            if all(c in result.columns for c in ("gene", "allele", "gene_species"))
            else pd.Series(False, index=result.index)
        )

        if _still_null.any():
            _filled = result[~_still_null].copy()
            _null_rows = result[_still_null].copy()

            # Build {pe_key: set of (gene, allele, gene_species)} from filled rows
            _filled_combos: dict[str, set] = {}
            for _, _fr in _filled.iterrows():
                _fpe = str(_fr["mutation_string_pe"]).replace(" ", "").upper()
                _combo = tuple(str(_fr.get(c, "")) for c in _meta_cols)
                _filled_combos.setdefault(_fpe, set()).add(_combo)

            # For each null row, find filled rows whose tokens are a superset
            superstring_records = []
            unresolved_records = []
            for _, _nr in _null_rows.iterrows():
                _npe = str(_nr["mutation_string_pe"]).replace(" ", "").upper()
                _ntoks = frozenset(t for t in _npe.split("/") if t)
                # Find all filled PEs whose token set contains our tokens
                valid_super_combos: set = set()
                for _fpe, _combos in _filled_combos.items():
                    _ftoks = frozenset(t for t in _fpe.split("/") if t)
                    if _ntoks <= _ftoks:  # our tokens ⊆ filled row's tokens
                        valid_super_combos |= _combos
                # Filter out empty/nan combos
                valid_super_combos = {
                    c
                    for c in valid_super_combos
                    if any(v and v not in ("nan", "None", "") for v in c)
                }
                if not valid_super_combos:
                    unresolved_records.append(_nr.to_dict())
                    continue
                for combo in sorted(valid_super_combos):
                    rec = _nr.to_dict()
                    for col, val in zip(_meta_cols, combo):
                        rec[col] = val
                    superstring_records.append(rec)

            _super_df = pd.DataFrame(
                superstring_records + unresolved_records,
                columns=result.columns,
            )
            result = pd.concat([_filled, _super_df], ignore_index=True)

    # ── 5. Assert uniqueness on (mutation_string_pe, gene, allele, gene_species) ─
    _key_cols = [
        c
        for c in ["mutation_string_pe", "gene", "allele", "gene_species"]
        if c in result.columns
    ]
    dupes = result.duplicated(subset=_key_cols, keep=False)
    if dupes.any():
        _bad_rows = result.loc[dupes, _key_cols].sort_values(_key_cols)
        raise AssertionError(
            f"Result is not unique on {_key_cols}. "
            f"{dupes.sum()} duplicate rows found. "
            "Check that unique_germline_pos_combo has no duplicate (gene/allele/species) "
            "entries for the same mutation position.\n"
            f"Duplicate rows:\n{_bad_rows.to_string()}"
        )

    return result
