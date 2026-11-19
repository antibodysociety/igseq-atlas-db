"""Parsers for IMGT 2D structure detail pages.

Extracts chain sequences, mutations, and domain annotations from
the HTML tables on https://imgt.org/3Dstructure-DB/cgi/details.cgi?pdbcode={id}.
"""

import logging
import re
import warnings
from typing import Optional

from bs4 import Tag

from .chain_type import infer_chain_type
from .models import INNChainDetail, MutationRecord

# Regex patterns to classify gene segment type from allele name
_V_GENE_RE = re.compile(r"^IG[HKL]V", re.IGNORECASE)
_D_GENE_RE = re.compile(r"^IG[H]D", re.IGNORECASE)
_J_GENE_RE = re.compile(r"^IG[HKL]J", re.IGNORECASE)

logger = logging.getLogger(__name__)



def extract_data(td: Tag) -> dict:
    """Extract cleaned text and child elements from a <td>."""
    text = td.get_text(strip=True)
    text = text.replace("\u00a0", " ").replace("\n", " ").replace("\r", "").strip()
    return {
        "text": text,
        "children": [child for child in td.children if child.name],
    }


def parse_alignment_sequence(pre_object: Tag, strict_positions: bool = True) -> tuple[str, list[MutationRecord]]:
    """Parse an IMGT alignment <pre> block into sequence + mutations.

    Args:
        pre_object: BeautifulSoup <pre> tag containing <a> elements.
        strict_positions: If True, raise on non-consecutive positions.
            If False, warn and continue.
    """
    letters = pre_object.find_all("a")
    sequence = ""
    mutations: list[MutationRecord] = []
    number = 0

    for idx, letter in enumerate(letters):
        if not letter.has_attr("title"):
            raise ValueError(f"Letter {idx} has no title")

        position = int(letter["title"][:-1])
        letter_seq = letter["title"][-1]

        children = [child for child in letter.children if child.name]
        if len(children) > 1:
            raise ValueError(f"Letter {idx} has more than one child")

        if children:
            if not children[0].text:
                continue
            class_name = children[0].get("class", [])
            title_val = children[0].get("title", "")
            if "colordiffali" in class_name:
                if title_val != letter_seq:
                    mutations.append(
                        MutationRecord(
                            position=position,
                            mutated_seq=letter_seq,
                            germline_seq=title_val,
                        )
                    )
                # else: synonymous annotation (e.g. codon-level diff) — skip
            else:
                raise ValueError(f"Letter {idx}: unexpected child class={class_name}")
        elif not letter.text:
            continue

        if position != number + 1:
            msg = f"Letter {idx}: position {position}, expected {number + 1}"
            if strict_positions:
                raise ValueError(msg)
            warnings.warn(msg)
        number = position
        sequence += letter_seq

    return sequence, mutations


def extract_seq_inn_info(sequence: str, mutations: list[MutationRecord]) -> dict:
    """Reconstruct germline sequence from mutated sequence + mutations."""
    germline = list(sequence)
    for mut in mutations:
        pos = mut.position - 1  # 1-based → 0-based
        if sequence[pos] != mut.mutated_seq:
            raise ValueError(
                f"Mutation at pos {mut.position}: expected '{mut.mutated_seq}' " f"but found '{sequence[pos]}'"
            )
        germline[pos] = mut.germline_seq

    return {
        "sequence": sequence,
        "germline_seq": "".join(germline),
        "scraped_mutations": mutations,
        "total_mutations": len(mutations),
    }


def parse_imgt_inn_summary_table(
    table: Tag,
    chain_type: str,
    inn_id: str,
    strict_positions: bool = True,
) -> list[INNChainDetail]:
    """Parse a single chain table from the IMGT 2D structure detail page."""

    CORE_FEATURES = {"Chain ID", "Chain length", "IMGT chain description", "Chain sequence"}
    IGNORE_FEATURES = {"2D representation", "Remarks", "Contact analysis"}
    DOMAIN_GAP_ALIGN_PREFIX = "https://imgt.org/3Dstructure-DB/cgi/DomainGapAlign.cgi?"

    prefix_child_elements: Optional[int] = None
    sections: list[dict] = []
    core: dict = {}

    for idx, tr in enumerate(table.find_all("tr")):
        first_col = tr.find("td")
        if not first_col:
            continue
        first_col_name = first_col.get_text(strip=True)

        titre = tr.find("td", class_=f"titre_{chain_type}")
        data_td = tr.find("td", class_=f"data_{chain_type}")

        if prefix_child_elements == idx:
            prefix_child_elements = None

        if titre and prefix_child_elements is None:
            stop = int(titre.attrs.get("rowspan", 1))
            prefix_child_elements = idx + stop
            sections.append({"name": first_col_name})

        key = "domain_sequence" if not titre else extract_data(titre)["text"]

        if not data_td:
            continue

        # Capture DomainGapAlign URLs from any <a> in this row
        if sections:
            for a_tag in tr.find_all("a", href=True):
                href = a_tag["href"]
                if href.startswith(DOMAIN_GAP_ALIGN_PREFIX):
                    sections[-1]["domain_gap_align_url"] = href

        data = extract_data(data_td)

        if key in IGNORE_FEATURES:
            continue

        if key in ("domain_sequence", "Chain sequence"):
            value = parse_alignment_sequence(data["children"][0], strict_positions)
        else:
            value = data["text"]

        if key in CORE_FEATURES:
            if key in core:
                raise ValueError(f"Duplicate core feature: {key}")
            core[key] = value
        elif key == "IMGT gene and allele name":
            # V-DOMAIN sections have separate rows for V, D, J genes.
            # Classify by prefix and store into dedicated keys.
            if _V_GENE_RE.search(value):
                sections[-1]["v_gene"] = value
            elif _D_GENE_RE.search(value):
                sections[-1]["d_gene"] = value
            elif _J_GENE_RE.search(value):
                sections[-1]["j_gene"] = value
            # Always store latest as fallback for constant regions etc.
            sections[-1][key] = value
        else:
            sections[-1][key] = value

    sections = [s for s in sections if len(s) > 1]

    chain_type_inferred = infer_chain_type(core["IMGT chain description"], chain_type)

    results: list[INNChainDetail] = []
    for idx, s in enumerate(sections):
        if "IMGT domain description" not in s:
            seq_info = {"sequence": s.get(s["name"])}
            domain = s["name"]
            alleles = None
        else:
            domain = s.get("IMGT domain description", "")
            alleles = s.get("IMGT gene and allele name")
            v_gene = s.get("v_gene")
            d_gene = s.get("d_gene")
            j_gene = s.get("j_gene")
            seq_info = {}
            for seq_key in ("domain_sequence", "Chain sequence"):
                if seq_key in s:
                    seq, muts = s[seq_key]
                    seq_info = extract_seq_inn_info(seq, muts)

        results.append(
            INNChainDetail(
                inn_id=inn_id,
                chain_type=chain_type_inferred,
                chain_id=core["Chain ID"],
                chain_length=int(core["Chain length"]),
                chain_description=core["IMGT chain description"],
                region_annotation=domain.split("(")[0].strip(),
                region_annotation_description=domain,
                chain_feature_type=s["name"],
                sequence=seq_info.get("sequence"),
                germline_sequence=seq_info.get("germline_seq"),
                scraped_mutations=seq_info.get("scraped_mutations"),
                total_mutations=seq_info.get("total_mutations"),
                gene_name=alleles,
                v_gene=v_gene if "IMGT domain description" in s else None,
                d_gene=d_gene if "IMGT domain description" in s else None,
                j_gene=j_gene if "IMGT domain description" in s else None,
                detail_order_idx=f"{core['Chain ID']}.{idx}"
            )
        )

    # Full length entry
    full_seq, full_muts = core["Chain sequence"]
    full_info = extract_seq_inn_info(full_seq, full_muts)
    results.append(
        INNChainDetail(
            inn_id=inn_id,
            chain_type=chain_type_inferred,
            chain_id=core["Chain ID"],
            chain_length=int(core["Chain length"]),
            chain_description=core["IMGT chain description"],
            chain_feature_type="Full Length",
            region_annotation="Full Length",
            region_annotation_description=core["IMGT chain description"],
            sequence=full_info["sequence"],
            germline_sequence=full_info["germline_seq"],
            scraped_mutations=full_info["scraped_mutations"],
            total_mutations=full_info["total_mutations"],
            detail_order_idx=f"{core['Chain ID']}.full_chain"
        )
    )

    return results
