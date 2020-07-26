"""
Mocks for docstrings
"""

from copy import deepcopy

from doctrans.defaults_utils import remove_defaults_from_docstring_structure

docstring_structure = {
    'short_description': 'Acquire from the official tensorflow_datasets model '
                         'zoo, or the ophthalmology focussed ml-prepare '
                         'library',
    'long_description': '',
    'params': [
        {
            'default': 'mnist',
            'doc': 'name of dataset. Defaults to mnist',
            'name': 'dataset_name',
            'typ': 'str'
        },
        {
            'default': '~/tensorflow_datasets',
            'doc': 'directory to look for models in. Defaults to ~/tensorflow_datasets',
            'name': 'tfds_dir',
            'typ': 'Optional[str]'
        },
        {
            'default': 'np',
            'doc': 'backend engine, e.g., `np` or `tf`. Defaults to np',
            'name': 'K',
            'typ': 'Literal[\'np\', \'tf\']'
        },
        {
            'doc': 'Convert to numpy ndarrays',
            'name': 'as_numpy',
            'typ': 'Optional[bool]'
        },
        {
            'doc': 'pass this as arguments to data_loader function',
            'name': 'data_loader_kwargs',
            'typ': 'dict'
        }
    ],
    'returns': {
        'name': 'return_type',
        'doc': 'Train and tests dataset splits. Defaults to (np.empty(0), np.empty(0))',
        'default': '(np.empty(0), np.empty(0))',
        'typ': 'Union[Tuple[tf.data.Dataset, tf.data.Dataset], '
               'Tuple[np.ndarray, np.ndarray]]'
    }
}

docstring_str = """
Acquire from the official tensorflow_datasets model zoo, or the ophthalmology focussed ml-prepare library

:param dataset_name: name of dataset. Defaults to mnist
:type dataset_name: ```str```

:param tfds_dir: directory to look for models in. Defaults to ~/tensorflow_datasets
:type tfds_dir: ```Optional[str]```

:param K: backend engine, e.g., `np` or `tf`. Defaults to np
:type K: ```Literal['np', 'tf']```

:param as_numpy: Convert to numpy ndarrays
:type as_numpy: ```Optional[bool]```

:param data_loader_kwargs: pass this as arguments to data_loader function
:type data_loader_kwargs: ```**data_loader_kwargs```

:return: Train and tests dataset splits. Defaults to (np.empty(0), np.empty(0))
:rtype: ```Union[Tuple[tf.data.Dataset, tf.data.Dataset], Tuple[np.ndarray, np.ndarray]]```
"""

docstring_structure_no_default_doc = remove_defaults_from_docstring_structure(
    deepcopy(docstring_structure), emit_defaults=False
)

docstring_str_no_default_doc = """
Acquire from the official tensorflow_datasets model zoo, or the ophthalmology focussed ml-prepare library

:param dataset_name: name of dataset.
:type dataset_name: ```str```

:param tfds_dir: directory to look for models in.
:type tfds_dir: ```Optional[str]```

:param K: backend engine, e.g., `np` or `tf`.
:type K: ```Literal['np', 'tf']```

:param as_numpy: Convert to numpy ndarrays
:type as_numpy: ```Optional[bool]```

:param data_loader_kwargs: pass this as arguments to data_loader function
:type data_loader_kwargs: ```**data_loader_kwargs```

:return: Train and tests dataset splits.
:rtype: ```Union[Tuple[tf.data.Dataset, tf.data.Dataset], Tuple[np.ndarray, np.ndarray]]```
"""
