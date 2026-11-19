"""Shared gap/sequence utilities and DB index building."""

_GAP_CHARS = frozenset(".-")

_SPECIES_MAP = {
    "human": "Homo sapiens",
    "house mouse": "Mus musculus",
    "mouse": "Mus musculus",
    "mus musculus": "Mus musculus",
    "rabbit": "Oryctolagus cuniculus",
    "chicken": "Gallus gallus",
    "dog": "Canis lupus familiaris",
    "canis lupus familiaris": "Canis lupus familiaris",
    "domestic cat": "Felis catus",
    "felis catus": "Felis catus",
}


def _gapless(seq):
    return "".join(c for c in seq if c not in _GAP_CHARS)


def _gapless_end_pos(seq, n):
    count = 0
    for i, c in enumerate(seq):
        if c not in _GAP_CHARS:
            count += 1
            if count == n:
                return i + 1
    return len(seq)


def _gapless_to_imgt_pos(gapless_pos: int, ref_with_gaps: str) -> int:
    """Map 1-based gapless position to 1-based IMGT position (counting gap chars)."""
    count = 0
    for i, c in enumerate(ref_with_gaps):
        if c not in _GAP_CHARS:
            count += 1
            if count == gapless_pos:
                return i + 1
    return len(ref_with_gaps) + 1


def _leading_gap_count(seq):
    for i, c in enumerate(seq):
        if c not in _GAP_CHARS:
            return i
    return len(seq)


def _stitch_query(a_seq, b_seq, germline):
    """Concatenate a_seq + bridge_gaps + b_seq, using germline to find the inter-region gaps."""
    gl_b = _gapless(b_seq)
    if not gl_b:
        return a_seq
    gl_germ = _gapless(germline)
    # Anchor on where b starts in germline (use last occurrence — b is at the 3' end)
    b_in_germ = gl_germ.rfind(gl_b)
    if b_in_germ < 0:
        return a_seq.rstrip(".-") + b_seq.lstrip(".-")
    if b_in_germ == 0:
        bridge = germline[: _leading_gap_count(germline)]
    else:
        # gapped position of the last char before b_seq starts in germline
        pre_b = _gapless_end_pos(germline, b_in_germ)
        bridge = germline[pre_b : pre_b + _leading_gap_count(germline[pre_b:])]
    return a_seq.rstrip(".-") + bridge + b_seq.lstrip(".-")


def _filter_by_species(entries, species_str):
    prefix = _SPECIES_MAP.get(species_str.lower(), "")
    if not prefix:
        return [seq for _, seq in entries]
    return [seq for sp, seq in entries if sp.startswith(prefix)]


_COL_GENE = "IMGT gene and allele name"
_COL_REGION = "exon(s), region name(s), or extracted label(s)"
_COL_SEQ = "sequence"
_COL_SPECIES = "species"


_COL_PARTIAL = "partial (if it is)"


def _build_db_index(db):
    if _COL_PARTIAL in db.columns:
        db = db[
            db[_COL_PARTIAL].isna() | (db[_COL_PARTIAL].astype(str).str.strip() == "")
        ]
    db = db.rename(
        columns={
            _COL_GENE: "gene_allele",
            _COL_REGION: "region",
            _COL_SEQ: "sequence",
            _COL_SPECIES: "species",
        }
    )
    db = db[["gene_allele", "region", "species", "sequence"]].dropna(
        subset=["gene_allele", "region", "sequence"]
    )
    gene_idx = {}
    region_idx = {}
    for _, row in db.iterrows():
        key = (row["gene_allele"].strip().lower(), row["region"].strip())
        gene_idx.setdefault(key, []).append(
            (str(row["species"]).strip(), row["sequence"])
        )
        region_idx.setdefault(row["region"].strip(), []).append(row["sequence"])
    return gene_idx, region_idx


def _build_composite_seqs(gene_idx, region_idx, specs):
    for spec in specs.values():
        for sub_labels in spec.compound_build:
            synthetic_label = "+".join(sub_labels)
            genes_per_label = [
                {gene for gene, lbl in gene_idx if lbl == sl} for sl in sub_labels
            ]
            common_genes = (
                set.intersection(*genes_per_label) if genes_per_label else set()
            )
            for gene in common_genes:
                species_seqs = {}
                for lbl in sub_labels:
                    for sp, seq in gene_idx.get((gene, lbl), []):
                        species_seqs.setdefault(sp, []).append(seq)
                n = len(sub_labels)
                for sp, seqs in species_seqs.items():
                    if len(seqs) == n:
                        composite = "".join(seqs)
                        gene_idx.setdefault((gene, synthetic_label), []).append(
                            (sp, composite)
                        )
                        region_idx.setdefault(synthetic_label, []).append(composite)


_COMPOUND_COMPONENTS = {
    "CH3-CHS": ("CH3", "CHS"),
    "CH4-CHS": ("CH4", "CHS"),
    "H-CH2": ("H", "CH2"),
    "CH3+CHS": ("CH3", "CHS"),
    "CH4+CHS": ("CH4", "CHS"),
    "H+CH2": ("H", "CH2"),
}


def _split_by_gk_pattern(compound_seq: str, gene: str, label: str) -> tuple[str, str]:
    """Split compound at the last dot-run immediately before GK.

    Rule 1: find last gap-run where the following residues start with GK.
    Rule 2: gapless ends with GK (no gap before it) — split before the last 2 gapless chars.
    Raises ValueError if neither rule applies.
    """
    import re

    last_m = None
    for m in re.finditer(r"[.\-]+", compound_seq):
        rest = compound_seq[m.end() :]
        if _gapless(rest).startswith("GK"):
            last_m = m
    if last_m is not None:
        return compound_seq[: last_m.end()], compound_seq[last_m.end() :]

    gl = _gapless(compound_seq)
    if gl.endswith("GK"):
        count = 0
        target = len(gl) - 2
        for i, c in enumerate(compound_seq):
            if c not in _GAP_CHARS:
                if count == target:
                    return compound_seq[:i], compound_seq[i:]
                count += 1

    raise ValueError(
        f"{gene}/{label}: cannot find ..GK split point in {compound_seq!r}"
    )


def _parse_compound_split_points(gene_idx: dict) -> dict:
    """Pre-compute split refs for compound labels (CH3-CHS, CH4-CHS, H-CH2).

    For CH3-CHS / CH4-CHS: trailing dots stay on primary; extra_ref starts at GK (no leading dots).
    For H-CH2: first dot-run stays on extra (CH2) side; primary (H) has no trailing dots.
    Returns {(gene, compound_label_with_dash): {"primary_ref": str, "extra_ref": str}}.
    """
    import re

    split_points = {}
    for (gene, label), entries in gene_idx.items():
        if label not in _COMPOUND_COMPONENTS:
            continue
        primary_label, extra_label = _COMPOUND_COMPONENTS[label]
        normalized = label.replace("+", "-")

        primary_entries = gene_idx.get((gene, primary_label), [])
        extra_entries = gene_idx.get((gene, extra_label), [])

        if primary_entries and extra_entries:
            _, primary_seq = primary_entries[0]
            _, extra_seq = extra_entries[0]
            # Dots stay on primary side: strip any leading gaps from extra
            leading = len(extra_seq) - len(extra_seq.lstrip(".-"))
            primary_ref = primary_seq + extra_seq[:leading]
            extra_ref = extra_seq[leading:]
        elif extra_label == "CHS":
            # Compound-only entry: derive split from ..GK pattern
            _, compound_seq = entries[0]
            try:
                primary_ref, extra_ref = _split_by_gk_pattern(compound_seq, gene, label)
            except ValueError:
                continue  # skip entries that don't follow the ..GK rule
        else:
            # H-CH2 compound-only: split at first dot-run (dots belong to CH2)
            _, compound_seq = entries[0]
            m = re.search(r"[.\-]+", compound_seq)
            if not m:
                raise ValueError(f"{gene}/{label}: no dot-run found for H-CH2 split")
            primary_ref = compound_seq[: m.start()]
            extra_ref = compound_seq[m.start() :]

        if extra_label == "CHS" and not _gapless(extra_ref).startswith("GK"):
            raise ValueError(
                f"{gene}/{label}: CHS does not start with GK: {extra_ref!r}"
            )

        split_points[(gene, normalized)] = {
            "primary_ref": primary_ref,
            "extra_ref": extra_ref,
        }
    return split_points
