"""Convert mutation positions between numbering systems (IMGT, EU, KABAT, EXON)."""

from logging import getLogger
from pathlib import Path
from typing import Dict, List, Optional, Union

from ..mutation_models import MutationRecord

from .core import (
    ConstantChainNumbering,
    NumberingTypes,
    get_imgt_constant_numbering_map_table,
)

logger = getLogger(__name__)


def create_numbering_object(
    number_table_map_file: Path, regions: List[str] = ["VH", "IGL", "VH-HINGE", "IGK"]
) -> ConstantChainNumbering:
    if not Path(number_table_map_file).exists():
        raise AssertionError(f"file: {number_table_map_file} does not exist")

    numbering = []
    for region in regions:
        numbering.append(
            get_imgt_constant_numbering_map_table(number_table_map_file, region)
        )
    cn = ConstantChainNumbering()

    cn.add_region_pos_map(numbering)
    return cn


def convert_mutation_positioning(
    region: str,
    mutation_list: Optional[List[MutationRecord]],
    numbering_types: List[NumberingTypes],
    cn: ConstantChainNumbering,
    region_repeat_count: Optional[int] = None,
) -> Optional[Dict[NumberingTypes, List[MutationRecord]]]:
    """Convert mutations from sequence positions to multiple numbering systems.

    Args:
        region: Constant chain region (e.g. "CH1", "HINGE-REGION").
        mutation_list: List of mutation dicts/objects with 'position' field.
        numbering_types: Which numbering systems to add (IMGT, EU, KABAT, EXON).
        cn: Loaded ConstantChainNumbering with region maps.

    Returns:
        Enhanced mutation list with additional numbering fields, or None.
    """
    if mutation_list is None:
        return None
    if region not in cn.regions:
        logger.warning(
            f"Region {region} not found in numbering maps; cannot convert mutation positions."
        )
        logger.warning(f"Available regions: {cn.regions}")
        return None

    cn_stacked_df = cn.filled_position_df[region].copy().dropna(how="all")

    cn_stacked_df_positions = cn_stacked_df.loc[
        [m.position for m in mutation_list]
    ].copy()

    mutations_by_numbering = {}
    for numbering_type in numbering_types:
        mutation_numeric_system = []
        if cn_stacked_df[numbering_type].isna().all():
            logger.warning(
                f"Numbering type {numbering_type} has no data for region {region}; skipping."
            )
            continue

        positions: List[str] = list(cn_stacked_df_positions[f"{numbering_type}"].values)
        if len(positions) != len(mutation_list):
            raise ValueError(
                f"Length mismatch: {len(positions)} positions vs {len(mutation_list)} mutations for region {region} and numbering {numbering_type}"
            )
        for mutation, pos in zip(mutation_list, positions):
            if hasattr(mutation, "mutation_type"):
                mutation_type = mutation.mutation_type
            elif mutation.mutated_seq is None or mutation.mutated_seq == "-":
                mutation_type = "deletion"
            elif (
                "+" in str(mutation.position)
                or "+" in mutation.mutated_seq
                or getattr(mutation, "insertion_str", None)
            ):
                mutation_type = "insertion"
            else:
                mutation_type = "substitution"

            # if ".i" in str(pos):
            #     if mutation.mutated_seq is None or mutation.mutated_seq == '-' or mutation_type == "deletion":
            #         continue  # likely a deletion with no specific mutated residue, so skip as its not really a deletion at the position

            #     if mutation_type != 'insertion':
            #         mutation_type = "insertion"
            #         pos = pos.split('.i')[0] # keep only the base position for insertions, and report it as an insertion

            mutation_numeric_system.append(
                MutationRecord(
                    position=pos,
                    germline_seq=mutation.germline_seq,
                    mutated_seq=mutation.mutated_seq,
                    mutation_type=mutation_type,
                    insertion_str=getattr(mutation, "insertion_str", None),
                    repeat_region_count=region_repeat_count,
                    mutation_region=region,
                    numbering_system=numbering_type.value,
                    original_position=int(mutation.position),
                )
            )
        mutations_by_numbering[numbering_type] = mutation_numeric_system

    # result = []
    # for mut in mutation_list:
    #     # Support both dict and Pydantic model
    #     m = mut if isinstance(mut, dict) else mut.model_dump()
    #     enriched = dict(m)
    #     for nt in numbering_types:
    #         enriched[str(nt)] = cn.seq_pos_to_ab_numbering_type(
    #             region, m["position"], nt
    #         )
    #     result.append(enriched)

    return mutations_by_numbering
    return mutations_by_numbering
    return mutations_by_numbering
    return mutations_by_numbering
