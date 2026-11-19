"""Motif pipeline implementations.

classic : build_motif_index  — INN-level Möbius statistical pipeline.
chain   : build_chain_motif_index — chain-level bispecific-aware pipeline.
"""

from .chain import (
    build_chain_motif_index,
    step_add_statistics,
    step_build_cluster_info,
    step_filter_remove_motif,
    step_restore_other_gene_hits,
)
from .classic import (
    MotifIndexResult,
    build_motif_index,
    collapse_motifs_chains,
)

__all__ = [
    "build_chain_motif_index",
    "step_filter_remove_motif",
    "step_restore_other_gene_hits",
    "step_add_statistics",
    "step_build_cluster_info",
    "build_motif_index",
    "collapse_motifs_chains",
    "MotifIndexResult",
]
