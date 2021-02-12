"""
Mocks for JSON Schema
"""
from doctrans.tests.mocks.docstrings import docstring_header_and_return_str

config_schema = {
    "$id": "https://offscale.io/json.schema.json",
    "$schema": "http://json-schema.org/draft-07/schema#",
    "description": docstring_header_and_return_str,
    "type": "object",
    "properties": {
        "dataset_name": {
            "description": "name of dataset.",
            "type": "string",
            "default": "mnist",
        },
        "tfds_dir": {
            "description": "directory to look for models in.",
            "type": "string",
            "default": "~/tensorflow_datasets",
        },
        "K": {
            "description": "backend engine, e.g., `np` or `tf`.",
            "type": "string",
            "enum": ["np", "tf"],
            "default": "np",
        },
        "as_numpy": {
            "description": "Convert to numpy ndarrays.",
            "type": "boolean",
        },
        "data_loader_kwargs": {
            "description": "pass this as arguments to data_loader function",
            "type": "object",
        },
    },
    "required": ["dataset_name", "tfds_dir", "K"],
}


__all__ = ["config_schema"]
