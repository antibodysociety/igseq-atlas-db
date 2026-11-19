"""Analytics package — pure data-analysis modules with no plotting dependency."""

from ._helpers import build_inn_lookup
from .co_mutation_analytics import (  # noqa: F401
    MutEvent,
    build_comutation_hierarchy,
    build_inn_mutation_timeline,
    build_marginals_table,
    build_motif_timeline_table,
    build_motif_timeline_table_from_counts,
    cluster_motifs_by_substring,
    co_df_to_export_table,
    compute_co_mutation_frequencies,
    motif_yearly_counts,
    remove_intra_pairs,
    select_top_seeds_and_filter,
)
from .co_mutation_analytics import _process_constant_chain_no_mut  # noqa: F401
from .debug import (  # noqa: F401
    DebugMotifResult,
    INNDiscrepancyDetail,
    MotifMismatchDetail,
    audit_motif_cohesion,
    compute_motif_ancestry,
    debug_cluster_overlaps,
    debug_motif_inn_counts,
    debug_pairwise_inn_overlap,
    decompose_leaky_motif,
    investigate_inn_count_mismatches,
    trace_discrepancy_pipeline,
)
from .constant_chain_analysis import (  # noqa: F401
    CoMutationConfig,
    ConstantChainAnalysisConfig,
    ConstantChainAnalysisResult,
    EffectorAnnotationConfig,
    MotifIndexConfig,
    build_constant_chain_analysis,
    export_constant_chain_analysis,
)
from .motifs import (
    ChainMotifIndexResult,
    build_chain_motif_index,
)  # noqa: F401
from .mutation_stats import (  # noqa: F401
    build_gene_species_n_table,
    build_motif_stats_per_inn,
    build_motif_stats_per_motif,
    build_mutation_residues,
    build_positional_mutation_freq,
    mutation_repetitions_per_inn,
)
