"""Chain- and INN-level motif analysis package.

Sub-modules
-----------
audit                  INN change audit log (InnAuditLog, start/stop helpers).
bispecific             Bispecific branch detection. ``expand_bispecific_branches``
                       and ``resolve_bispecific_branches_dep`` are deprecated —
                       bispecific detection now runs natively inside
                       ``build_chain_motif_index`` via
                       ``expand_motifs_by_cross_chain_pairing`` (step 2/15).
clustering             Seed identification, INN attribution, statistics, cluster info.
exploded               Shared motif→seed-group expansion helpers.
_dep_cluster_resolution  (deprecated) Cluster overlap resolution — superseded by clustering.py.
enumerate              Chain-level motif enumeration (enumerate_chain_motifs).
subtraction            INN subtraction helpers.
pipelines/             Pipeline implementations:
  chain                build_chain_motif_index — new chain-level pipeline.
  classic              build_motif_index       — INN-level Möbius pipeline.
"""

from .audit import (
    InnAuditLog,
    get_inn_audit,
    start_inn_audit,
    stop_inn_audit,
)
from .bispecific import (
    _match_motif_to_chains,
    annotate_bispecific_lateral_sharing,
    expand_bispecific_branches,
    resolve_bispecific_branches_dep,
)
from .clustering import step_add_statistics, step_build_cluster_info
from .enumerate import (
    ChainEnumerationRow,
    ChainEnumerationRowAnnotated,
    annotate_constant_gene,
    chain_collection_to_enumeration_rows,
    enumerate_chain_motifs,
)
from .allele_mapping import (
    ChainID_EU_Key,
    ChainID_Region_Key,
    Region_Allele_Key,
    _infer_chain_id_mut_to_allele_map,
    _infer_chain_id_region_to_allele_map,
    _infer_inn_chain_domain_map,
    ChainIdRegionAlleleMapper,
    MutationAlleleMapper,
    find_compatible_chain_ids,
    find_multi_chain_inns_by_subdomain,
    chain_has_fc_region,
    region_to_subdomain,
)
from .models import (
    ChainMotifIndexResult,
    ChainMotifStep,
    ChainMotifStepResult,
    _AnnotateSeedsIn,
    _ComputeInnAttributionIn,
)
from .pipelines import (
    MotifIndexResult,
    build_chain_motif_index,
    build_motif_index,
    collapse_motifs_chains,
    step_filter_remove_motif,
    step_restore_other_gene_hits,
)
from .cross_chain import (
    assign_motif_chain_ids,
    assign_paired_ref_ids,
    build_cross_chain_groups,
    build_cross_chain_pair_map,
    collapse_super_string_motifs,
    explode_cross_chain_groups,
)
from .exploded import build_exploded_motif_df
from .subtraction import recompute_adjusted_counts
