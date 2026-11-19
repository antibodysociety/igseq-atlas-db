import json
from typing import Any, Optional

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, PrivateAttr


class ChainMotifStep(BaseModel):
    """Base schema for a single pipeline step input/output."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    motif_df: pd.DataFrame
    step_name: str = ""
    # Stratum columns: group_by columns minus mutation_string_pe.
    # All clustering/attribution/statistics steps operate independently within
    # each stratum defined by these columns.
    group_cols: list[str] = Field(default_factory=list)
    min_seed_size: int = 2
    allowed_motifs_by_select: list[str] | None = None


class ChainMotifStepResult(ChainMotifStep):
    """Output schema that records row-level deltas."""

    rows_added: int = 0
    rows_dropped: int = 0
    detail: str = ""


class _RestoreOtherGeneHitsIn(ChainMotifStep):
    original_motif_df: pd.DataFrame
    min_count: int = 1


class _RestoreOtherGeneHitsOut(ChainMotifStepResult):
    recovered_rows: int = 0


class _SubtractInnsIn(ChainMotifStep):
    """Subtract INNs absorbed by larger motifs, preserving bispecific rows."""

    skip_bispecific: bool = True


class _AnnotateSeedsIn(ChainMotifStep):
    """Input model for step_annotate_seeds."""

    min_seed_motif_size: int = 2


class _ComputeInnAttributionIn(ChainMotifStep):
    """Input model for step_compute_inn_attribution."""

    # Motif types to retain even when adjusted_count == 0.
    # All other zero-adjusted-count rows are dropped.
    retain_zero_adjusted_types: list[str] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Pipeline result container
# ---------------------------------------------------------------------------


class ChainIterativeClusters(BaseModel):
    """Container for iterative clustering results."""

    model_config = ConfigDict(arbitrary_types_allowed=True)
    name: str
    iteration: int
    _init_df: Optional[pd.DataFrame] = PrivateAttr(default=None)
    _cluster_df: Optional[pd.DataFrame] = PrivateAttr(default=None)
    _stats_df: Optional[pd.DataFrame] = PrivateAttr(default=None)
    _output_df: Optional[pd.DataFrame] = PrivateAttr(default=None)

    def __setattr__(self, name: str, value: Any) -> None:
        # Allow property setters to intercept before Pydantic's field check.
        cls = type(self)
        if isinstance(getattr(cls, name, None), property):
            getattr(cls, name).fset(self, value)
        else:
            super().__setattr__(name, value)

    @property
    def init_df(self) -> Optional[pd.DataFrame]:
        return self._init_df

    @init_df.setter
    def init_df(self, value: Optional[pd.DataFrame]) -> None:
        self._init_df = value

    @property
    def cluster_df(self) -> Optional[pd.DataFrame]:
        return self._cluster_df

    @cluster_df.setter
    def cluster_df(self, value: Optional[pd.DataFrame]) -> None:
        self._cluster_df = value

    @property
    def stats_df(self) -> Optional[pd.DataFrame]:
        return self._stats_df

    @stats_df.setter
    def stats_df(self, value: Optional[pd.DataFrame]) -> None:
        self._stats_df = value

    @property
    def output_df(self) -> Optional[pd.DataFrame]:
        return self._output_df

    @output_df.setter
    def output_df(self, value: Optional[pd.DataFrame]) -> None:
        self._output_df = value

    def changes_detected(self):
        start_rows_total = self.init_df.shape[0]
        starting_hits = set(self.init_df.mutation_string_pe)
        end_rows_total = self.output_df.shape[0]
        ending_hits = set(self.output_df.mutation_string_pe)
        clustering_filter = self.cluster_df.shape[0]
        clustering_hits_filter = set(self.cluster_df.mutation_string_pe)
        return {
            "init_rows": start_rows_total,
            "ending_hits": ",".join(list(ending_hits)),
            "rows_filtered": {
                "rows_removed_at_clustering": start_rows_total
                - clustering_filter,
                "rows_removed_at_stats_filter": clustering_filter
                - end_rows_total,
                "total_workflow_rows_removed": start_rows_total
                - end_rows_total,
            },
            "input_motifs": starting_hits,
            "post_cluster_motifs": clustering_hits_filter,
            "output_motifs": ending_hits,
            "motifs_removed": (
                starting_hits - clustering_hits_filter,
                clustering_hits_filter - ending_hits,
            ),
            "end_rows": end_rows_total,
            "clustering_filter": clustering_filter,
            "clustering_hits_filter": clustering_hits_filter,
        }

    def __repr__(self):
        stats = self.changes_detected()
        input_motifs = stats.pop("input_motifs")
        post_cluster_motifs = stats.pop("post_cluster_motifs")
        output_motifs = stats.pop("output_motifs")

        stats_report_keys = [
            "init_rows",
            "rows_filtered",
            "ending_hits",
            "end_rows",
            "clustering_filter",
        ]
        stat_summary = json.dumps(
            {k: stats[k] for k in stats_report_keys}, indent=4
        )
        return (
            "ChainIterativeClusters: \n"
            f"input_motifs={len(input_motifs)}, \n"
            f"post_cluster_motifs={len(post_cluster_motifs)}, \n"
            f"output_motifs={len(output_motifs)}\n====\n"
            f"{stat_summary}"
        )


class ChainMotifIndexResult(BaseModel):
    """Result of :func:`build_chain_motif_index`.

    Attributes
    ----------
    motif_df :
        Primary output — final motif table with all columns populated.
    step_changes :
        Per-step audit trail: ``{step_name: {"added": DataFrame, "removed": DataFrame}}``.
    filters :
        Configuration parameters used for this run.
    enumeration_df :
        Raw output of :func:`enumerate_chain_motifs` before any pipeline steps.
    cross_chain_df :
        Raw cross-chain rows produced by :func:`expand_motifs_by_cross_chain_pairing`.
    skipped_cross_chain :
        INN/chain-type combinations skipped during cross-chain pairing.
    marginals_table :
        Per-position marginal mutation frequencies used for statistical tests.
    cluster_db_ref_tokens :
        Reference tokens by stratum group for clustering iterations.
    removed_post_clustering :
        Rows removed during the final clustering iteration.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    _motif_df: Optional[pd.DataFrame] = PrivateAttr(default=None)
    _enumeration_df: Optional[pd.DataFrame] = PrivateAttr(default=None)
    _cross_chain_df: Optional[pd.DataFrame] = PrivateAttr(default=None)
    _skipped_cross_chain: Optional[list] = PrivateAttr(default=None)
    _marginals_table: Optional[pd.DataFrame] = PrivateAttr(default=None)
    _enumeration_collapsed_df: Optional[pd.DataFrame] = PrivateAttr(
        default=None
    )
    _cluster_db_ref_tokens: Optional[
        dict[tuple[str, ...], dict[tuple[str, str], frozenset[str]]]
    ] = PrivateAttr(default=None)
    _inn_chains_per_group: Optional[dict[tuple, list[tuple[str, str]]]] = (
        PrivateAttr(default=None)
    )
    other_dfs: dict = Field(default_factory=dict)
    _removed_post_clustering: Optional[pd.DataFrame] = PrivateAttr(
        default=None
    )
    _ind_mut_allele_map: Optional[dict] = PrivateAttr(default=None)
    _chain_id_region_allele_map: Optional[Any] = PrivateAttr(default=None)
    _inn_chain_domain_map: Optional[dict[str, dict[str, list[str]]]] = (
        PrivateAttr(default=None)
    )
    step_changes: dict[str, dict[str, pd.DataFrame]] = Field(
        default_factory=dict
    )
    filters: dict = Field(default_factory=dict)
    motif_df_per_cluster_iteration: list[
        tuple[
            pd.DataFrame | None,
            pd.DataFrame | None,
            ChainIterativeClusters,
        ]
    ] = Field(default_factory=list)

    def __setattr__(self, name: str, value: Any) -> None:
        cls = type(self)
        if isinstance(getattr(cls, name, None), property):
            getattr(cls, name).fset(self, value)
        else:
            super().__setattr__(name, value)

    @property
    def motif_df(self) -> Optional[pd.DataFrame]:
        """Get motif_df."""
        return self._motif_df

    @motif_df.setter
    def motif_df(self, value: Optional[pd.DataFrame]) -> None:
        """Set motif_df."""
        self._motif_df = value

    @property
    def enumeration_df(self) -> Optional[pd.DataFrame]:
        """Get enumeration_df."""
        return self._enumeration_df

    @enumeration_df.setter
    def enumeration_df(self, value: Optional[pd.DataFrame]) -> None:
        """Set enumeration_df."""
        self._enumeration_df = value

    @property
    def enumeration_collapsed_df(self) -> Optional[pd.DataFrame]:
        """Get enumeration_collapsed_df."""
        return self._enumeration_collapsed_df

    @enumeration_collapsed_df.setter
    def enumeration_collapsed_df(
        self, value: Optional[pd.DataFrame]
    ) -> None:
        """Set enumeration_collapsed_df."""
        self._enumeration_collapsed_df = value

    @property
    def inn_chains_per_group(
        self,
    ) -> dict[tuple[str, ...], list[tuple[str, str]]]:
        return self._inn_chains_per_group or {}

    @inn_chains_per_group.setter
    def inn_chains_per_group(
        self, value: dict[tuple[str, ...], list[tuple[str, str]]]
    ) -> None:
        self._inn_chains_per_group = value

    @property
    def cross_chain_df(self) -> Optional[pd.DataFrame]:
        """Get cross_chain_df."""
        return self._cross_chain_df

    @cross_chain_df.setter
    def cross_chain_df(self, value: Optional[pd.DataFrame]) -> None:
        """Set cross_chain_df."""
        self._cross_chain_df = value

    @property
    def skipped_cross_chain(self) -> Optional[list]:
        """Get skipped_cross_chain."""
        return self._skipped_cross_chain

    @skipped_cross_chain.setter
    def skipped_cross_chain(self, value: Optional[list]) -> None:
        """Set skipped_cross_chain."""
        self._skipped_cross_chain = value

    @property
    def marginals_table(self) -> Optional[pd.DataFrame]:
        """Get marginals_table."""
        return self._marginals_table

    @marginals_table.setter
    def marginals_table(self, value: Optional[pd.DataFrame]) -> None:
        """Set marginals_table."""
        self._marginals_table = value

    @property
    def cluster_db_ref_tokens(
        self,
    ) -> Optional[
        dict[tuple[str, ...], dict[tuple[str, str], frozenset[str]]]
    ]:
        """Get cluster_db_ref_tokens."""
        return self._cluster_db_ref_tokens

    @cluster_db_ref_tokens.setter
    def cluster_db_ref_tokens(
        self,
        value: Optional[
            dict[tuple[str, ...], dict[tuple[str, str], frozenset[str]]]
        ],
    ) -> None:
        """Set cluster_db_ref_tokens."""
        self._cluster_db_ref_tokens = value

    @property
    def removed_post_clustering(self) -> Optional[pd.DataFrame]:
        """Get removed_post_clustering."""
        return self._removed_post_clustering

    @removed_post_clustering.setter
    def removed_post_clustering(
        self, value: Optional[pd.DataFrame]
    ) -> None:
        """Set removed_post_clustering."""
        self._removed_post_clustering = value

    @property
    def ind_mut_allele_map(self) -> Optional[dict]:
        """Get ind_mut_allele_map — {(chain_id, EU_pos): MutationAlleleMapper}."""
        return self._ind_mut_allele_map

    @ind_mut_allele_map.setter
    def ind_mut_allele_map(self, value: Optional[dict]) -> None:
        """Set ind_mut_allele_map."""
        self._ind_mut_allele_map = value

    @property
    def chain_id_region_allele_map(self) -> Optional[Any]:
        """Get chain_id_region_allele_map — ChainIdRegionAlleleMapper."""
        return self._chain_id_region_allele_map

    @chain_id_region_allele_map.setter
    def chain_id_region_allele_map(self, value: Optional[Any]) -> None:
        """Set chain_id_region_allele_map."""
        self._chain_id_region_allele_map = value

    @property
    def inn_chain_domain_map(
        self,
    ) -> Optional[dict[str, dict[str, list[str]]]]:
        """Get inn_chain_domain_map — {inn_id: {"CK"|"CL"|"V": [chain_id, ...]}}."""
        return self._inn_chain_domain_map

    @inn_chain_domain_map.setter
    def inn_chain_domain_map(
        self, value: Optional[dict[str, dict[str, list[str]]]]
    ) -> None:
        """Set inn_chain_domain_map."""
        self._inn_chain_domain_map = value

    @staticmethod
    def _count_unique_inns(df: pd.DataFrame) -> int:
        if df.empty or "inns" not in df.columns:
            return 0
        return len(set().union(*df["inns"].dropna()))

    @staticmethod
    def _shape(df: pd.DataFrame) -> str:
        if not isinstance(df, pd.DataFrame):
            return "n/a"
        return f"{len(df)}x{len(df.columns)}"

    @staticmethod
    def _count_unique_mutation_string_pe(df: pd.DataFrame) -> int:
        """Count unique mutation_string_pe values."""
        if df.empty or "mutation_string_pe" not in df.columns:
            return 0
        return int(df["mutation_string_pe"].nunique())

    @staticmethod
    def _count_unique_seed_motifs(df: pd.DataFrame) -> int:
        """Count unique seed motifs (is_seed=True)."""
        if df.empty or "is_seed" not in df.columns:
            return 0
        return int((df["is_seed"] == True).sum())

    def __repr__(self) -> str:
        """Display pipeline execution summary with step-by-step changes."""
        # Table summary
        table_rows = [
            ("enumeration_df (raw)", self.enumeration_df),
            ("cross_chain_df (derived)", self.cross_chain_df),
            ("clustering iterations", self.motif_df_per_cluster_iteration),
            ("motif_df (final output)", self.motif_df),
        ]
        table_lines = [
            f"  {name}: shape={self._shape(df)}, "
            f"unique_inns={self._count_unique_inns(df)}, "
            f"unique_motifs={self._count_unique_mutation_string_pe(df)}, "
            f"seed_motifs={self._count_unique_seed_motifs(df)}, "
            f"total_cluster_iterations={len(self.motif_df_per_cluster_iteration)}, "
            for name, df in table_rows
            if isinstance(df, pd.DataFrame)
        ]

        # Step-by-step changes
        step_lines: list[str] = []
        for step_name, payload in (self.step_changes or {}).items():
            if not isinstance(payload, dict):
                continue
            added_df = payload.get("added")
            removed_df = payload.get("removed")
            added_rows = (
                len(added_df) if isinstance(added_df, pd.DataFrame) else 0
            )
            removed_rows = (
                len(removed_df)
                if isinstance(removed_df, pd.DataFrame)
                else 0
            )
            added_inns = (
                self._count_unique_inns(added_df)
                if isinstance(added_df, pd.DataFrame)
                else 0
            )
            removed_inns = (
                self._count_unique_inns(removed_df)
                if isinstance(removed_df, pd.DataFrame)
                else 0
            )
            step_lines.append(
                f"  {step_name}: added={added_rows} rows ({added_inns} INNs), "
                f"removed={removed_rows} rows ({removed_inns} INNs)"
            )

        # Filters summary
        filter_str = ", ".join(
            f"{k}={v}" for k, v in (self.filters or {}).items()
        )

        # Skipped cross-chain summary
        skipped_str = (
            f"  {len(self.skipped_cross_chain)} INN/chain-type combinations skipped"
            if self.skipped_cross_chain
            else "  No skipped combinations"
        )

        tables_block = (
            "\nTables:\n" + "\n".join(table_lines)
            if table_lines
            else "\nTables:\n  (none)"
        )
        steps_block = (
            "\nSteps:\n" + "\n".join(step_lines)
            if step_lines
            else "\nSteps:\n  (none)"
        )

        return (
            "ChainMotifIndexResult\n"
            + tables_block
            + steps_block
            + f"\nFilters: {filter_str}\n"
            + "Cross-chain:\n"
            + skipped_str
            + f"\n\nFinal output: {len(self.motif_df)} motifs"
        )
