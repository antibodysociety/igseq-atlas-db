"""Shared mutation data models — no upstream dependencies."""
from typing import Optional, Union

from pydantic import BaseModel


class MutationRecord(BaseModel):
    """Single amino acid mutation vs germline."""

    position: Union[int, str, float]
    mutated_seq: str
    germline_seq: str
    mutation_type: str = "substitution"  # "substitution" | "insertion" | "deletion"
    insertion_str: Optional[str] = None
    mutation_region: Optional[str] = None  # e.g. "CDR1", "FR2", "CH2"
    repeat_region_count: Optional[int] = None
    numbering_system: Optional[str] = None  # e.g. "IMGT", "EU", "KABAT", "EXON"
    original_position: Optional[int] = None


class NumberedMutationRecord(BaseModel):
    """A mutation expressed in all supported numbering systems.

    ``position`` is the original IMGT sequence-indexed mutation.
    The remaining fields hold the same mutation re-expressed under
    each alternative numbering scheme (None if not available for the region).
    """

    position: MutationRecord
    imgt_position: Optional[MutationRecord] = None
    eu_position: Optional[MutationRecord] = None
    kabat_position: Optional[MutationRecord] = None
    exon_position: Optional[MutationRecord] = None
