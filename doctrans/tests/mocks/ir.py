"""
IR mocks
"""

method_complex_args_variety_ir = {
    "name": "call_cliff",
    "params": [
        {"doc": "name of dataset.", "name": "dataset_name"},
        {"doc": "Convert to numpy ndarrays", "name": "as_numpy"},
        {
            "doc": "backend engine, e.g., `np` or `tf`.",
            "name": "K",
            "typ": "Literal['np', 'tf']",
        },
        {
            "default": "~/tensorflow_datasets",
            "doc": "directory to look for models in.",
            "name": "tfds_dir",
        },
        {
            "default": "stdout",
            "doc": "IO object to write out to",
            "name": "writer",
        },
        {
            "doc": "additional keyword arguments",
            "name": "kwargs",
            "typ": "dict",
        },
    ],
    "returns": {
        "default": "K",
        "doc": "backend engine",
        "name": "return_type",
        "typ": "Literal['np', 'tf']",
    },
    "doc": "Call cliff",
    "type": "self",
}
