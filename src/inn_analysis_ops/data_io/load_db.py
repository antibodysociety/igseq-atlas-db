import json
from inn_analysis_ops.models.inn_structure import INNStructureCollection


def load_from_json(json_file) -> INNStructureCollection:

    with open(json_file, "r") as f:
        collection_raw = json.load(f)

    collection = INNStructureCollection.model_validate(collection_raw)
    return collection
