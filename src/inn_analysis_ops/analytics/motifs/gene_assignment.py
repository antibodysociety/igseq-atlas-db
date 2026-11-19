import pandas as pd

FIELDS_FOR_NOMENCLATURE = [
    "gene",
    "gene_species",
    "allele",
    "inn",
    "chain_id",
]


def _create_gene_map_for_inn_chains_by_fc_only(
    individual_mutation_profile: pd.DataFrame,
    field_values_user: list[str] | None = None,
) -> dict[tuple[str, str], list[tuple[str, ...]]]:
    if field_values_user is None:
        fields = FIELDS_FOR_NOMENCLATURE
    else:
        fields = field_values_user

    unique_inn_chain_fc_by_iteration_annotation_map = (
        individual_mutation_profile.query("region_classification=='FC'")
        .reset_index()
        .groupby(by=["inn", "chain_id", "region_iteration"], dropna=False)
        .apply(
            lambda x: sorted(
                {
                    tuple(vals)
                    for vals in zip(*[x[field] for field in fields])
                }
            )
        )
        .reset_index()
        .sort_index()
    )
    unique_inn_chain_fc = (
        unique_inn_chain_fc_by_iteration_annotation_map.groupby(
            by=["inn", "chain_id"], dropna=False
        )
        .filter(lambda x: x[0].nunique(dropna=False) == 1)
        .drop(["region_iteration"], axis=1)
        .set_index(["inn", "chain_id"])
        .to_dict()[0]
    )
    return unique_inn_chain_fc


def _consolidate_duplicated_chains(
    chain_info: dict[str, tuple[str, str, str, str]],
) -> list[tuple[str, str, str, str]]:
    if len(chain_info) == 1:
        return [next(iter(chain_info.values()))]
    unique_matching = {}
    for chain_data in chain_info.values():
        matching_key = (chain_data[0], chain_data[1], chain_data[3])
        value_data = chain_data[2]
        if matching_key not in unique_matching:
            unique_matching[matching_key] = []
        unique_matching[matching_key].append(value_data)

    unique_matching = {
        key: ",".join(sorted(set(v))) for key, v in unique_matching.items()
    }

    unique_data = [
        (key[0], key[1], value, key[2])
        for key, value in unique_matching.items()
    ]
    return unique_data


def _create_inn_map_of_mutation_gene_info(
    individual_mutation_profile: pd.DataFrame,
    field_values_user: list[str] | None = None,
) -> tuple[
    dict[
        tuple[str, str, str],  # key
        dict[str, list[tuple[str, ...]]] | None,  # key
    ],
    dict[tuple[str, str, str], list[tuple[str, str, str, str]]],
]:
    if field_values_user is None:
        fields = FIELDS_FOR_NOMENCLATURE
    else:
        fields = field_values_user

    ignore_region = ["HINGE", "HINGE-REGION", "CHS"]
    mut_profile = individual_mutation_profile.reset_index()
    mut_profile["mutation_string_pe"] = mut_profile[
        ["germline_seq", "EU", "mutated_seq"]
    ].apply(
        lambda x: f"{x['germline_seq']}{x['EU']}{x['mutated_seq']}", axis=1
    )

    regions_per_inn_chain = mut_profile.groupby(
        by=["inn", "chain_id", "mutation_string_pe", "region_iteration"],
        dropna=False,
    ).apply(
        lambda x: sorted(
            set(
                zip(
                    x["region_classification"],
                    x["gene"],
                    x["allele"],
                    x["gene_species"],
                )
            )
        )
    )

    if regions_per_inn_chain.apply(lambda x: len(x) > 1).any():
        raise AssertionError(
            "Too may classifications per mutations"
            f"{regions_per_inn_chain[regions_per_inn_chain.apply(lambda x: len(x)>1)]}"
        )

    regions_per_inn_chain_dict = {}
    for (
        inn,
        chain,
        mutation,
        iteration,
    ), row_data in regions_per_inn_chain.items():
        if (inn, chain, mutation) not in regions_per_inn_chain_dict:
            regions_per_inn_chain_dict[(inn, chain, mutation)] = {}
        if pd.isnull(iteration):
            iteration = "NA"
        else:
            iteration = str(int(iteration))
        regions_per_inn_chain_dict[(inn, chain, mutation)][iteration] = (
            row_data[0]
        )
    regions_per_inn_chain_list = {
        (inn, chain, mutation): _consolidate_duplicated_chains(
            iteration_dict
        )
        for (
            inn,
            chain,
            mutation,
        ), iteration_dict in regions_per_inn_chain_dict.items()
    }

    mut_profile_collapsed_by_inn_mutation = mut_profile.groupby(
        by=["inn", "mutation_string_pe", "chain_id", "region_iteration"],
        dropna=False,
    ).agg(
        {
            "gene": list,
            "allele": list,
            "gene_species": list,
            "region": lambda x: sorted(set(x)),
        }
    )

    mapping_by_inn_mut = {}
    for (
        grp,
        grp_data,
    ) in mut_profile_collapsed_by_inn_mutation.reset_index().groupby(
        by=["inn", "chain_id", "mutation_string_pe"]
    ):
        mapping_by_inn_mut[grp] = {}
        for _, row in grp_data.iterrows():
            row.update({"inn": [grp[0]], "chain_id": [grp[1]]})
            region_idx = (
                "NA"
                if pd.isnull(row["region_iteration"])
                else str(int(row["region_iteration"]))
            )

            skip_analysis = False
            if "region" in row:
                assert len(row["region"]) == 1, row
                if row["region"][0] in ignore_region:
                    skip_analysis = True

            if skip_analysis:
                mapping_by_inn_mut[grp] = None
                continue
            if region_idx not in mapping_by_inn_mut[grp]:
                mapping_by_inn_mut[grp][region_idx] = []
            field_values = [row[f] for f in fields]
            lengths = {len(x) for x in field_values}
            assert len(lengths) == 1, (field_values, lengths)

            unique_val_set = list(zip(*field_values))
            mapping_by_inn_mut[grp][region_idx] = sorted(
                set(unique_val_set)
            )

    return mapping_by_inn_mut, regions_per_inn_chain_list


def _get_genes_associated_with_motif(
    motif: str,
    inn_chains: dict[str, list[str]],
    mapping_by_inn_mut: dict[
        tuple[str, str, str],  # key
        dict[str, list[tuple[str, ...]]] | None,  # key
    ],
    field_set: list[str] | None = None,
) -> dict:
    if field_set is None:
        fields = FIELDS_FOR_NOMENCLATURE
    else:
        fields = field_set

    all_field_data_per_mutation = []
    try:
        mutations = motif.split("/")
        failure = []
        for inn, chains in inn_chains.items():
            for chain in chains:
                mutations_mapping_result = [
                    mapping_by_inn_mut[(inn, chain, mut)]
                    for mut in mutations
                ]
                mutations_mapping_result = [
                    m for m in mutations_mapping_result if m is not None
                ]
                if len(mutations_mapping_result) == 0:
                    failure.append((inn, chain, mutations_mapping_result))
                    continue
                chain_common_to_all = mutations_mapping_result[0].keys()
                for region_map in mutations_mapping_result:
                    chain_common_to_all = (
                        chain_common_to_all & region_map.keys()
                    )
                if not chain_common_to_all:
                    failure.append((inn, chain, mutations_mapping_result))
                    continue

                for common_chain in chain_common_to_all:
                    for mutation in mutations_mapping_result:
                        all_field_data_per_mutation.extend(
                            mutation[common_chain]
                        )

    except Exception as err:
        return {
            "core_failure": err,
            "gene_assignment_failure": failure,
            "motif_gene_annotation": [],
        }

    unique_field_combis = {field: [] for field in fields}

    all_field_data_per_mutation = sorted(set(all_field_data_per_mutation))
    for motif_gene_info in all_field_data_per_mutation:
        for annotation, field in zip(motif_gene_info, fields):
            unique_field_combis[field].append(annotation)

    report_result = {
        # **{key: ','.join(val) for key, val in unique_field_combis.items()},
        "gene_assignment_failure": failure,
        "motif_gene_annotation": all_field_data_per_mutation,
        "core_failure": None,
    }
    return report_result
