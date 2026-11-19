import re
from pathlib import Path
from typing import List, Union

import numpy as np
import pandas as pd
from ..enum_wrapper import StrEnumBase


class NumberingTypes(StrEnumBase):
    IMGT = "IMGT"
    EU = "EU"
    KABAT = "KABAT"
    EXON = "EXON"


def get_imgt_constant_numbering_map_table(
    imgt_numbering_file: Path, sheet_name: str
) -> pd.DataFrame:
    raw_table = pd.read_excel(
        imgt_numbering_file,
        sheet_name=sheet_name,
        header=None,
    )

    raw_table = pd.DataFrame(
        raw_table.iloc[2:, 2:].values,
        index=pd.MultiIndex.from_arrays(
            raw_table.iloc[2:, :2].values.T,
            names=["Secondary Structure", "Position"],
        ),
        columns=pd.MultiIndex.from_arrays(
            raw_table.iloc[:2, 2:].values, names=["Region", "Position"]
        ),
    )
    raw_table = _fill_in_missing_numbering_rows(raw_table)
    return raw_table


def _fill_in_missing_numbering_rows(
    imgt_numbering_table: pd.DataFrame,
) -> pd.DataFrame:
    FILL_VALUE = "fme"

    unique_numbering = []

    prev_pos = 0

    for (
        structure,
        position,
    ), row in imgt_numbering_table.iterrows():
        if not pd.isnull(structure):
            structure = structure.strip().upper()
        for (subset, feature), pos in row.items():
            subset = subset.strip().upper()
            feature = feature.strip().upper()
            for p in range(prev_pos + 1, int(position)):
                unique_numbering.append((structure, p, subset, feature, "fme"))
            unique_numbering.append((structure, position, subset, feature, pos))

        prev_pos = int(position)

    numbering_table_filled = (
        (
            pd.DataFrame(
                unique_numbering,
                columns=[
                    "structure",
                    "position",
                    "subset",
                    "feature",
                    "pos",
                ],
            )
            .sort_values(by=["position"])
            .set_index(["position", "structure", "subset", "feature"])
            .unstack(level=["subset", "feature"])
        )
        .sort_index()
        .sort_index(axis=1)
    )

    new_values = []

    for idx, row in numbering_table_filled.iterrows():
        values = []
        for idx, (feature, val) in enumerate(row.items()):
            if val == FILL_VALUE:
                values.append(new_values[-1][idx] + 1)
            else:
                values.append(val)

        new_values.append(values)

    numbering_table_filled = pd.DataFrame(
        new_values,
        columns=pd.MultiIndex(
            levels=[
                numbering_table_filled.columns.levels[1],
                list(numbering_table_filled.columns.levels[2]),
            ],
            names=["region", "feature"],
            codes=[
                numbering_table_filled.columns.codes[1],
                numbering_table_filled.columns.codes[2],
            ],
        ),
        index=pd.MultiIndex(
            levels=[
                numbering_table_filled.index.levels[0],
                numbering_table_filled.index.levels[1],
            ],
            names=["position", "structure"],
            codes=[
                numbering_table_filled.index.codes[0],
                numbering_table_filled.index.codes[1],
            ],
        ),
    )

    regions = list(numbering_table_filled.columns.levels[0])
    structure_data = pd.concat(
        [
            pd.DataFrame(
                numbering_table_filled.index.get_level_values("structure").values,
                columns=["structure"],
                index=numbering_table_filled.index,
            )
            for i in range(len(regions))
        ],
        axis=1,
        keys=regions,
    )

    restructured_table = (
        pd.concat([numbering_table_filled, structure_data], axis=1)
        .sort_index(axis=1)
        .reset_index(level="structure", drop=True)
    )
    return restructured_table


def _fill_empty_series(arr):
    if arr.isnull().all():
        return arr

    if arr.name[1] == "structure":
        return arr
    max_val = float(re.findall(r"([0-9\.]+)", str(arr.dropna().iloc[-1]))[0])

    arr_fill = list(arr.bfill().fillna(int(max_val) + 1))
    arr_orig = list(arr)
    new_vals = []
    counter = 1
    for orig, filled in zip(arr_orig, arr_fill):
        if pd.isnull(orig):
            if isinstance(filled, str):
                filled = float(re.findall(r"([0-9]+)", filled)[0])
            filled = int(filled)
            new_vals.append(f"{filled - 1}.i{counter}")
            counter += 1
        else:
            try:
                val = float(orig) if int(orig) != float(orig) else int(orig)
                new_vals.append(val)
            except ValueError:
                new_vals.append(orig)

            counter = 1

    return pd.Series(new_vals, index=arr.index)


class ConstantChainNumbering:
    def __init__(self):
        self._position_df = pd.DataFrame()
        self._position_df_filled = None

    # @property
    # def stacked_by_region(self):
    #     df2 = self._position_df.copy()
    #     # df2.columns = pd.MultiIndex.from_tuples(list(df2.columns))
    #     df2.stack().stack().unstack(level="feature").sort_index().reorder_levels(
    #         ["region", "position", "structure"]
    #     ).sort_index()
    #     return df2

    @property
    def regions(self):
        return self._position_df.columns.get_level_values(0).unique()

    @property
    def filled_position_df(self):
        if self._position_df_filled is None:
            self._position_df_filled = self._position_df.apply(
                _fill_empty_series, axis=0
            ).copy()
            no_data = (
                self._position_df.drop(level=1, columns="structure", errors="ignore")
                .T.groupby(level=0)
                .apply(lambda x: x.isnull().all(axis=0))
                .T
            )
            for region in no_data.columns:
                self._position_df_filled.loc[no_data[region], (region, slice(None))] = (
                    np.nan
                )

        return self._position_df_filled

    def add_region_pos_map(
        self, new_maps: List[pd.DataFrame], validate_integrity: bool = True
    ):
        if isinstance(new_maps, pd.DataFrame):
            new_maps = [new_maps]

        tmp = pd.concat([self._position_df] + new_maps, axis=1)
        tmp = tmp.dropna(how="all", axis=0)
        if validate_integrity and tmp.columns.duplicated().any():
            raise ValueError("ERROR duplicated columns found with concatenating tables")

        self._position_df = tmp.sort_index().sort_index(axis=1)

    def __getitem__(self, region):
        return self._position_df[region.upper()].dropna(how="all", axis=0)

    def ab_numbering_feature_to_seq_pos(
        self,
        region: str,
        feature: NumberingTypes,
        feature_value: Union[int, float, str],
    ) -> int:
        subset = self[region]
        return subset.loc[subset[feature] == feature_value].index.get_level_values(
            "position"
        )[0]

    def ab_numbering_feature_to_feature(
        self,
        region: str,
        feature_source: NumberingTypes,
        feature_value: Union[int, float, str],
        feature_target: NumberingTypes,
    ):
        subset = self[region]
        return subset.loc[
            subset[feature_source] == feature_value, feature_target
        ].values[0]

    def seq_pos_to_ab_numbering_type(
        self, region: str, position: int, return_pos_type: NumberingTypes
    ) -> Union[str, int, float]:
        subset = self[region]
        res = subset.loc[position, f"{return_pos_type.upper()}"].values[0]
        if pd.isnull(res):
            return res

        try:
            res = float(res)
            try:
                return int(res)
            except ValueError:
                return res
        except ValueError:
            return str(res)
