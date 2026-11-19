import re
from typing import Literal

import pandas as pd


def _canonical_str(toks: list[str]) -> str:
    """Canonical ' / '-joined representation of a token list (sorted)."""
    return " / ".join(sorted(toks))


def _parse_tokens(mutation_string: str) -> list[str]:
    """Split a mutation_string into a sorted list of token strings."""
    return sorted(
        t.strip() for t in re.split(r"\s*/\s*", mutation_string.strip()) if t.strip()
    )


def _parse_eu_pos(eu_pos: str) -> tuple[int, str | None]:
    """
    Ensure EU position is an integer but do not lose context
    of IMGT positions that are not present in original
    EU nomenclature (i.e. 305.i1)
    """
    if pd.isnull(eu_pos) or eu_pos is None or not eu_pos:
        raise AssertionError(f"Expected EU pos to be numeric or numeric.ins: {eu_pos}")
    pos = str(eu_pos).split(".")
    return int(pos[0]), pos[1] if len(pos) > 1 else None


def _parse_mutation_pe_into_parts(
    mutation_string_pe: str,
) -> tuple[int, str, str, str]:
    """Parse a PE mutation string into ``(pos, pos_suffix, germline, mut)``.

    ``mut`` captures the entire mutated-residue tail losslessly, including any
    embedded ``^`` insertion marker, so combined substitution+insertion tokens
    (e.g. ``"T225L^SS"`` — germline T substituted to L *and* SS inserted) are
    preserved rather than silently truncated at the first non-letter char.

    Examples
    --------
    ``"L234A"``     → ``(234, "", "L", "A")``           plain substitution
    ``"G237-"``     → ``(237, "", "G", "-")``            deletion
    ``"L230.i1^A"`` → ``(230, "i1", "L", "^A")``         pure insertion
    ``"T225L^SS"``  → ``(225, "", "T", "L^SS")``         substitution + insertion
    """
    numeric = r"([A-Z])(\d+(?:\.i\d+)?)([A-Z\^\-\+]+)"
    germline, pos_string, mut = re.match(
        numeric, mutation_string_pe, flags=re.IGNORECASE
    ).groups()
    pos, pos_suffix = _parse_eu_pos(pos_string)

    return (pos, pos_suffix or "", germline, mut)


def _split_eu_pos_col(
    df: pd.DataFrame, eu_col: str
) -> tuple[pd.DataFrame, tuple[str, str]]:
    eu_parsed = df[eu_col].apply(_parse_eu_pos)
    df[f"{eu_col}_pos"], df[f"{eu_col}_ins_pos"] = zip(*eu_parsed)
    return df, (f"{eu_col}_pos", f"{eu_col}_ins_pos")


def _build_mutation_string_pe(row: pd.Series) -> str:
    """Format a single mutation row as a compact PE string: ``{germline}{eu}{residue}``.

    The formula is the same for all mutation types. The caller is responsible
    for supplying the correct ``residue`` value:

    - Substitution / deletion: residue is the single mutated AA or ``"-"``.
    - Insertion (unspecified): residue must be ``"+"``.
    - Insertion (specific):    residue must start with ``"^"`` (e.g. ``"^A"``).
    - Insertion + substitution (combined): residue is ``"<AA>^<AA...>"`` —
      the position's germline AA is substituted (e.g. to ``"L"``) *and* a
      specific residue sequence is inserted right after (e.g. ``"L^SS"``
      means substituted to L, then SS inserted).

    Raises
    ------
    ValueError
        If ``germline_seq``, ``EU``, or ``residue`` is missing/empty, or if
        ``mutation_type == "insertion"`` but ``residue`` does not match one
        of ``"+"``, ``"^<AA>..."``, or ``"<AA>^<AA>..."``.
    """
    g = row.get("germline_seq")
    eu = row.get("EU")
    mut = row.get("residue")

    if pd.isna(g) or str(g).strip() == "":
        raise ValueError(
            f"_build_mutation_string_pe: germline_seq is missing for row: {dict(row)}"
        )
    if pd.isna(eu) or str(eu).strip() == "":
        raise ValueError(
            f"_build_mutation_string_pe: EU is missing for row: {dict(row)}"
        )
    if pd.isna(mut) or str(mut).strip() == "":
        raise ValueError(
            f"_build_mutation_string_pe: residue is missing for row: {dict(row)}"
        )

    mut_str = str(mut).strip()
    if row.get("mutation_type") == "insertion":
        _valid_insertion = (
            mut_str == "+"
            or re.fullmatch(r"\^[A-Za-z]+", mut_str)
            or re.fullmatch(r"[A-Za-z]\^[A-Za-z]+", mut_str)
        )
        if not _valid_insertion:
            raise ValueError(
                f"_build_mutation_string_pe: insertion residue must be '+', "
                f"'^<AA>' (e.g. '^A'), or '<AA>^<AA>' for a combined "
                f"substitution+insertion (e.g. 'L^SS'), got {mut_str!r}. "
                f"Row: {dict(row)}"
            )

    return f"{g}{eu}{mut_str}"


def full_length_chain_eu_to_token(raw_token: str) -> str:
    """Convert one ``constant_chain_mutations_EU`` token to motif format.

    ``constant_chain_mutations_EU`` in ``ds.full_length_chain_collection``
    uses compact notation. Formats observed:

    - ``"234L>A"``        substitution/deletion  → ``"234:L->A"``
    - ``"107.i4G>-"``     insertion at gap pos   → ``"107.i4:G->-"``
    - ``"193.i3V>^TV"``   insertion with ``^``   → ``"193.i3:V->^TV"``

    The position may be ``digits`` or ``digits.suffix`` (e.g. ``230.i1``).
    The mutated residue may start with ``^`` (insertion notation).

    Raises
    ------
    ValueError
        If the token cannot be parsed.

    Examples
    --------
    >>> full_length_chain_eu_to_token("234L>A")
    '234:L->A'
    >>> full_length_chain_eu_to_token("107.i4G>-")
    '107.i4:G->-'
    >>> full_length_chain_eu_to_token("193.i3V>^TV")
    '193.i3:V->^TV'
    """
    raw = raw_token.strip()
    if not raw:
        return raw
    m = re.fullmatch(r"(\d+(?:\.[A-Za-z0-9]+)?)([A-Za-z\-]+)>([\^A-Za-z\-+]+)", raw)
    if m is None:
        raise ValueError(
            f"full_length_chain_eu_to_token: cannot parse token '{raw}'. "
            f"Expected formats: '234L>A', '107.i4G>-', '193.i3V>^TV' "
            f"(position + germline + '>' + mutated)."
        )
    pos, germline, mutated = m.group(1), m.group(2), m.group(3)
    return f"{pos}:{germline}->{mutated}"


def eu_mutations_to_tokens(eu_str: str) -> frozenset:
    """Convert a ``constant_chain_mutations_EU`` string to motif-format tokens.

    ``full_length_chain_collection`` stores mutations as comma-separated
    compact tokens: ``"147K>E, 213K>E, 234L>A, 235L>A"``.

    Each token is converted via :func:`full_length_chain_eu_to_token` to the
    motif pipeline format (e.g. ``"147:K->E"``).

    Empty or NaN input returns an empty frozenset (germline chain).

    Examples
    --------
    >>> eu_mutations_to_tokens("234L>A, 235L>A, 366T>W")
    frozenset({'234:L->A', '235:L->A', '366:T->W'})
    """
    if eu_str is None or (isinstance(eu_str, float) and pd.isna(eu_str)):
        return frozenset()
    s = str(eu_str).strip()
    if not s:
        return frozenset()
    tokens: set[str] = set()
    for raw in s.split(","):
        raw = raw.strip()
        if raw:
            tokens.add(full_length_chain_eu_to_token(raw))
    return frozenset(tokens)


def parse_motif_label_pe(label_pe: str) -> str:
    """Convert compact PE motif string to colon notation.

    Examples
    --------
    ``"L234A"``              → ``"234:L->A"``
    ``"L234A / L235A"``      → ``"234:L->A / 235:L->A"``
    ``"G236-"``              → ``"236:G->-"``
    ``".236+"``              → ``"236:+"``
    ``"230.i1+"``            → ``"230.i1:+"``
    ``"L230.i1^A"``          → ``"230.i1:L->^A"``
    ``"C220S/L230.i1^A"``    → ``"220:C->S / 230.i1:L->^A"``
    ``"T225L^SS"``           → ``"225:T->L^SS"``  (substitution + insertion)

    Notes
    -----
    If ambiguous patterns like ``K/R214T`` or extended deletion forms like
    ``G107Adel`` are present, this function skips conversion for the entire
    input string and returns ``label_pe`` unchanged.
    """
    # Preserve sentinel/special labels and non-mutation strings.
    _SENTINELS = {
        "Other motifs",
        "No mutations",
        "Other mutations",
        "Other Mutations",
    }
    if label_pe in _SENTINELS:
        return label_pe

    if not isinstance(label_pe, str):
        raise ValueError(
            f"Label PE format error: expected string, got {type(label_pe).__name__}"
        )

    text = label_pe.strip()
    if not text:
        return text

    # Strip parenthetical annotations globally and accept separators as either
    # slash or whitespace, since some records replace "/" with spaces.
    text = re.sub(r"\s*\([^)]*\)", "", text).strip()
    tokens = [
        t.strip(" .") for t in re.split(r"(?:\s*/\s*|\s+)", text) if t.strip(" .")
    ]
    parsed: list[str] = []

    i = 0
    while i < len(tokens):
        tok = tokens[i]
        # Skip ambiguous notations like "K/R214T" that are split into
        # adjacent tokens "K" and "R214T" by slash tokenization.
        if re.fullmatch(r"[A-Za-z]", tok) and i + 1 < len(tokens):
            nxt = tokens[i + 1]
            if re.fullmatch(r"[A-Za-z][0-9]+(?:\.[A-Za-z0-9]+)?[A-Za-z\^+\-]+", nxt):
                return label_pe

        # Skip extended deletion annotations like "G107Adel".
        if re.fullmatch(
            r"[A-Za-z][0-9]+(?:\.[A-Za-z0-9]+)?\^?[A-Za-z]*del",
            tok,
            flags=re.IGNORECASE,
        ):
            return label_pe

        # Insertion shorthand: "230.i1+" (no germline prefix) -> "230.i1:+"
        # Only applies when token starts with a digit — if it starts with a letter
        # (germline prefix) the regex below handles it as "L230.i1+" -> "230.i1:L->+".
        if tok.endswith("+") and ":" not in tok and re.match(r"[0-9]", tok):
            pos = tok[:-1].strip()
            if not pos:
                raise ValueError(
                    f"Label PE format error: expected 'WTPOSMUT' or 'POS+' token, got '{tok}' in '{label_pe}'"
                )
            parsed.append(f"{pos}:+")
            i += 1
            continue

        # Substitution/deletion/insertion token:
        # "L234A"     -> "234:L->A"
        # "L230.i1^A" -> "230.i1:L->^A"    (specific insertion)
        # "L230.i1+"  -> "230.i1:L->+"     (unspecified insertion)
        # "G236-"     -> "236:G->-"
        # ".236+"     -> "236:+"
        # "T225L^SS"  -> "225:T->L^SS"     (substitution + specific insertion)
        m = re.fullmatch(
            r"([A-Za-z.])\s*([0-9]+(?:\.[A-Za-z0-9]+)?)\s*([A-Za-z\^+\-]+)",
            tok,
        )
        if m:
            wt, pos, mut = m.groups()
            if wt == "." and mut == "+":
                parsed.append(f"{pos}:+")
            elif wt == "." and mut == "-":
                parsed.append(f"{pos}:-")
            else:
                parsed.append(f"{pos}:{wt}->{mut}")
            i += 1
            continue

        raise ValueError(
            f"Label PE format error: expected 'WTPOSMUT' or 'POS+' token, got '{tok}' in '{label_pe}'"
        )

    return " / ".join(parsed)


def format_motif_label(label: str) -> str:
    """Convert colon-notation motif string to compact PE format.

    Examples
    --------
    ``"234:L->A"``              → ``"L234A"``
    ``"234:L->A / 235:L->A"``   → ``"L234A / L235A"``
    ``"234.i1:L->A"``           → ``"L234.i1^A"``
    ``"230.i1:+"``              → ``"230.i1+"``
    """
    # Pass through sentinel / special-value strings that are not mutation strings
    _SENTINELS = {
        "Other motifs",
        "No mutations",
        "Other mutations",
        "Other Mutations",
    }
    if label in _SENTINELS or ":" not in label:
        return label
    mutations = label.split(" / ")
    formatted = []
    for mut in mutations:
        mut = mut.strip()
        if ":" not in mut:
            raise ValueError(
                f"Label format error: expected 'POS:WT->MUT' or 'POS:+', got '{mut}' in '{label}'"
            )
        raw_pos, mutation_part = mut.split(":", 1)
        raw_pos = raw_pos.strip()
        mutation_part = mutation_part.strip()
        # Normalise float-formatted integers: "234.0" → "234"
        # but leave insertion positions like "230.i1" unchanged
        try:
            f = float(raw_pos)
            pos = str(int(f)) if f == int(f) else raw_pos
        except ValueError:
            pos = raw_pos
        if mutation_part == "+":
            formatted.append(f"{pos}+")
        elif "->" in mutation_part:
            parts = mutation_part.split("->")
            if len(parts) != 2:
                raise ValueError(
                    f"Label format error: expected 'POS:WT->MUT', got '{mut}' in '{label}'"
                )
            # The ^ is part of the residue in the colon notation (e.g. "230.i1:L->^A").
            # No auto-injection — just concatenate germline + pos + mut as-is.
            formatted.append(f"{parts[0].strip()}{pos}{parts[1].strip()}")
        else:
            raise ValueError(
                f"Label format error: expected 'POS:WT->MUT' or 'POS:+', got '{mut}' in '{label}'"
            )
    return " / ".join(formatted)


def split_motif_tokens(motif: str, sort: bool = False) -> list[str]:
    """Split a motif string into normalized token strings.

    Parameters
    ----------
    motif : str
        Colon-notation mutation string, e.g. ``"234:L->A / 235:L->A"``.
    sort : bool
        When ``True``, return tokens in sorted order (by string value).
        Defaults to ``False`` (preserves original order).

    Returns
    -------
    list[str]
    """
    if not isinstance(motif, str):
        return []
    tokens = [t.strip() for t in re.split(r"\s*/\s*", motif.strip()) if t.strip()]
    return sorted(tokens) if sort else tokens


def motif_tokens_set(motif: str) -> frozenset[str]:
    """Return canonical token set for a motif string."""
    return frozenset(split_motif_tokens(motif))


def tokens_to_mutation_string(
    toks,
    sort: bool = True,
) -> str:
    """Convert an iterable of ``(pos, change)`` tuples to a mutation string.

    Produces the canonical ``"pos:change / ..."`` format used throughout the
    motif pipeline.  By default tokens are sorted by ``(str(pos), change)`` so
    the output is position-ordered regardless of iteration order.

    Parameters
    ----------
    toks :
        Iterable of ``(pos, change)`` 2-tuples where *pos* is an ``int`` or
        insertion-slot ``str`` (e.g. ``"230.i1"``) and *change* is a string
        like ``"L->A"``.
    sort : bool
        Sort tokens by ``(str(pos), change)`` before joining.  Default ``True``.

    Returns
    -------
    str
        ``" / "``-joined string, e.g. ``"234:L->A / 235:L->A"``.

    Examples
    --------
    >>> tokens_to_mutation_string([(234, "L->A"), (235, "L->A")])
    '234:L->A / 235:L->A'
    >>> tokens_to_mutation_string([(235, "L->A"), (234, "L->A")], sort=True)
    '234:L->A / 235:L->A'
    """
    items = list(toks)
    if sort:
        items = sorted(items, key=lambda t: (str(t[0]), t[1]))
    return " / ".join(f"{pos}:{change}" for pos, change in items)


def find_best_subset_motif(
    candidate_tokens: frozenset,
    token_dict: "dict[str, frozenset]",
) -> "str | None":
    """Return the mutation string whose token set is the largest subset of *candidate_tokens*.

    Searches *token_dict* (a ``{mutation_string: frozenset}`` map) for the
    entry whose token set is a subset of *candidate_tokens* and has the most
    tokens.  Returns ``None`` when no entry qualifies.

    Used by the bispecific branch functions to identify the closest existing
    motif that acts as the common stem for two bispecific arms.

    Parameters
    ----------
    candidate_tokens : frozenset
        Token set to find the best subset for (e.g. the intersection of two
        bispecific arm token sets).
    token_dict : dict[str, frozenset]
        Mapping of ``mutation_string → token_frozenset`` for all motifs in
        scope.

    Returns
    -------
    str or None
        The ``mutation_string`` of the best-matching entry, or ``None`` if
        no entry's token set is a subset of *candidate_tokens*.

    Examples
    --------
    >>> td = {"A": frozenset({"x", "y"}), "B": frozenset({"x"})}
    >>> find_best_subset_motif(frozenset({"x", "y", "z"}), td)
    'A'
    """
    best_ms: "str | None" = None
    best_size = -1
    for ms, ts in token_dict.items():
        if ts <= candidate_tokens and len(ts) > best_size:
            best_size = len(ts)
            best_ms = ms
    return best_ms


def build_token_ignore_set(mutations_to_ignore: dict) -> frozenset:
    """Flatten ``mutations_to_ignore_for_motif_ops`` to ``(mut_type, pos_prefix)`` pairs.

    Region classification (HINGE, FC, …) is unavailable on aggregated EU strings,
    so all positions across all regions are collapsed into a single ignore set.
    This is conservative: a deletion at position 216 is ignored regardless of region.

    Parameters
    ----------
    mutations_to_ignore :
        ``MotifIndexConfig.mutations_to_ignore_for_motif_ops`` —
        ``{mutation_type: {region: [positions]}}``.

    Returns
    -------
    frozenset of ``(mutation_type_str, position_prefix_str)`` pairs.
    """
    result: set[tuple[str, str]] = set()
    for mut_type, region_map in mutations_to_ignore.items():
        for _region, positions in region_map.items():
            for pos in positions:
                result.add((str(mut_type), str(pos)))
    return frozenset(result)


def token_matches_ignore_set(token: str, ignore_set: frozenset) -> bool:
    """Return ``True`` if *token* matches any rule in *ignore_set*.

    Mutation type is inferred from the ``->`` change part of the token:

    - alt ``"-"``          → deletion
    - alt starts ``"^"`` or is ``"+"`` → insertion
    - otherwise            → substitution

    Insertion-slot positions use prefix matching: rule ``("insertion", "230.i")``
    matches ``"230.i1"``, ``"230.i2"``, etc.

    Parameters
    ----------
    token :
        Colon-notation token string, e.g. ``"234:L->A"`` or ``"230.i1:L->+"``.
    ignore_set :
        Output of :func:`build_token_ignore_set`.
    """
    if not ignore_set:
        return False
    token_parts = token.split(":", 1)
    if len(token_parts) != 2:
        raise ValueError(
            f"token_matches_ignore_set: malformed token '{token}' — expected 'POS:CHANGE'."
        )
    pos_str, change = token_parts[0].strip(), token_parts[1].strip()
    if "->" in change:
        alt = change.split("->", 1)[1].strip()
        if alt == "-":
            inferred_mut_type = "deletion"
        elif alt.startswith("^") or alt == "+":
            inferred_mut_type = "insertion"
        else:
            inferred_mut_type = "substitution"
    elif change == "+":
        inferred_mut_type = "insertion"
    else:
        inferred_mut_type = "substitution"

    for ignore_mut_type, ignore_pos_prefix in ignore_set:
        if ignore_mut_type != inferred_mut_type:
            continue
        if ".i" in ignore_pos_prefix:
            if pos_str.startswith(ignore_pos_prefix):
                return True
        elif pos_str == ignore_pos_prefix:
            return True
    return False


def token_signature_without_germline(token: str) -> str:
    """Drop germline residue from ``POS:WT->MUT`` token to ``POS->MUT``."""
    t = str(token).strip()
    if ":" in t and "->" in t:
        pos, change = t.split(":", 1)
        rhs = change.split("->", 1)[1].strip()
        return f"{pos.strip()}->{rhs}"
    return t


def motif_signature_without_germline(motif: str) -> frozenset[str]:
    """Return token signature by ``POS->MUT`` (ignoring germline residue)."""
    return frozenset(
        token_signature_without_germline(tok) for tok in split_motif_tokens(motif)
    )


def motif_is_subset(child_motif: str, parent_motif: str) -> bool:
    """True when all child tokens exist in parent motif."""
    return motif_tokens_set(child_motif).issubset(motif_tokens_set(parent_motif))


def find_multi_chain_inns(
    chain_df: pd.DataFrame,
    inn_col: str = "inn",
    chain_type_col: str = "chain_type",
    chain_id_col: str = "chain_id",
    chain_mutation_col: str = "constant_chain_mutations_EU",
    multi_chain_inn_key: Literal["by_inn", "by_inn_id"] = "by_inn_id",
) -> dict[str, dict[str, list[str]]]:
    """Return INNs that have ≥2 chains of the same chain_type, with their chain lists.

    Only chains with a non-null ``chain_mutation_col`` are considered — germline
    chains carry no motif signal and are excluded from multi-chain detection.

    Parameters
    ----------
    chain_df :
        ``ds.full_length_chain_collection.to_dataframe()`` — one row per
        ``(inn, chain_id)``.
    inn_col, chain_type_col, chain_id_col :
        Column names (defaults ``"inn"``, ``"chain_type"``, ``"chain_id"``).
    chain_mutation_col :
        Column holding EU mutation strings; rows where this is null/empty are
        excluded (default ``"constant_chain_mutations_EU"``).
    c_allele :
        Column holding the constant-gene allele (default ``"c_allele"``).
        Reserved for future allele-aware multi-chain grouping.

    Returns
    -------
    dict[str, dict[str, list[str]]]
        ``{inn: {chain_type: [sorted chain_ids]}}`` — only chain_types with
        ≥2 mutated chains are included per INN. INNs with no qualifying
        chain_type are excluded entirely.

    Examples
    --------
    >>> find_multi_chain_inns(df)
    {'faricimab': {'HC': ['H1', 'H2']}}
    """
    if multi_chain_inn_key not in ["by_inn", "by_inn_id"]:
        raise ValueError(
            f"find_multi_chain_inns: invalid multi_chain_inn_key '{multi_chain_inn_key}'. "
            f'Must be one of ["by_inn", "by_inn_id"]'
        )
    for required_col in (
        inn_col,
        chain_type_col,
        chain_id_col,
        chain_mutation_col,
    ):
        if required_col not in chain_df.columns:
            raise ValueError(
                f"find_multi_chain_inns: required column '{required_col}' "
                f"not found. Available: {list(chain_df.columns)}"
            )

    mutated_df = chain_df[
        chain_df[chain_mutation_col].notna()
        & chain_df[chain_mutation_col].astype(str).str.strip().ne("")
    ]

    result_by_inn: dict[str, dict[str, list[str]]] = {}
    for (inn, chain_type), group in mutated_df.groupby(
        [inn_col, chain_type_col], dropna=False
    ):
        if len(group) > 1:
            chain_ids = group[chain_id_col].tolist()
            if multi_chain_inn_key == "by_inn":
                # ABATACEPT
                inn_key = inn
            else:
                # "INN 135531_H"
                inn_key = chain_ids[0].split("_")[0]
            result_by_inn.setdefault(inn_key, {})[chain_type] = sorted(chain_ids)
    return result_by_inn


def build_inn_lookup(
    scored_df: pd.DataFrame,
    mutation_col: str = "mutation_string",
    inns_col: str = "inns",
) -> dict[str, set]:
    """Build a ``mutation_string → frozenset-of-INNs`` dict from *scored_df*.

    Iterates over column arrays (not rows) so construction is O(n) with minimal
    Python overhead.  Build this **once** and pass it to
    :func:`debug_pairwise_inn_overlap` and :func:`audit_motif_cohesion` to
    avoid redundant scanning when calling either function multiple times.

    Parameters
    ----------
    scored_df:
        ``MotifIndexResult.scored_df`` or any DataFrame with *mutation_col*
        and *inns_col* columns.
    mutation_col, inns_col:
        Column names (defaults ``"mutation_string"``, ``"inns"``).

    Returns
    -------
    dict[str, set]
        Keys are canonical mutation strings; values are INN sets.
    """
    lookup: dict[str, set] = {}
    for ms, inns in zip(scored_df[mutation_col], scored_df[inns_col]):
        if not isinstance(ms, str):
            continue
        lookup[ms] = set(inns) if isinstance(inns, (set, frozenset, list)) else set()
    return lookup
