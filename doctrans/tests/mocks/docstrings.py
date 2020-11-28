"""
Mocks for docstrings
"""

from copy import deepcopy

from doctrans.defaults_utils import remove_defaults_from_intermediate_repr

intermediate_repr = {
    "name": None,
    "type": "static",
    "doc": "Acquire from the official tensorflow_datasets model "
    "zoo, or the ophthalmology focussed ml-prepare "
    "library",
    "params": [
        {
            "default": '"mnist"',
            "doc": 'name of dataset. Defaults to "mnist"',
            "name": "dataset_name",
            "typ": "str",
        },
        {
            "default": '"~/tensorflow_datasets"',
            "doc": 'directory to look for models in. Defaults to "~/tensorflow_datasets"',
            "name": "tfds_dir",
            "typ": "Optional[str]",
        },
        {
            "default": '"np"',
            "doc": 'backend engine, e.g., `np` or `tf`. Defaults to "np"',
            "name": "K",
            "typ": "Literal['np', 'tf']",
        },
        {
            "doc": "Convert to numpy ndarrays",
            "name": "as_numpy",
            "typ": "Optional[bool]",
        },
        {
            "doc": "pass this as arguments to data_loader function",
            "name": "data_loader_kwargs",
            "typ": "dict",
        },
    ],
    "returns": {
        "name": "return_type",
        "doc": "Train and tests dataset splits. Defaults to (np.empty(0), np.empty(0))",
        "default": "(np.empty(0), np.empty(0))",
        "typ": "Union[Tuple[tf.data.Dataset, tf.data.Dataset], "
        "Tuple[np.ndarray, np.ndarray]]",
    },
}

docstring_str = """
Acquire from the official tensorflow_datasets model zoo, or the ophthalmology focussed ml-prepare library

:param dataset_name: name of dataset. Defaults to "mnist"
:type dataset_name: ```str```

:param tfds_dir: directory to look for models in. Defaults to "~/tensorflow_datasets"
:type tfds_dir: ```Optional[str]```

:param K: backend engine, e.g., `np` or `tf`. Defaults to "np"
:type K: ```Literal['np', 'tf']```

:param as_numpy: Convert to numpy ndarrays
:type as_numpy: ```Optional[bool]```

:param data_loader_kwargs: pass this as arguments to data_loader function
:type data_loader_kwargs: ```**data_loader_kwargs```

:return: Train and tests dataset splits. Defaults to (np.empty(0), np.empty(0))
:rtype: ```Union[Tuple[tf.data.Dataset, tf.data.Dataset], Tuple[np.ndarray, np.ndarray]]```
"""

docstring_google_str = """
Acquire from the official tensorflow_datasets model zoo, or the ophthalmology focussed ml-prepare library

Args:
  dataset_name (str): name of dataset. Defaults to "mnist"
  tfds_dir (Optional[str]): directory to look for models in. Defaults to "~/tensorflow_datasets"
  K (Literal['np', 'tf']): backend engine, e.g., `np` or `tf`. Defaults to "np"
  as_numpy (Optional[bool]): Convert to numpy ndarrays
  data_loader_kwargs (dict): pass this as arguments to data_loader function

Returns:
  Union[Tuple[tf.data.Dataset, tf.data.Dataset], Tuple[np.ndarray, np.ndarray]]:
   Train and tests dataset splits. Defaults to (np.empty(0), np.empty(0))
"""


# https://github.com/tensorflow/tensorflow/blob/7ad2723/tensorflow/python/keras/losses.py#L845-L858
docstring_google_tf_squared_hinge_str = """Initializes `SquaredHinge` instance.
Args:
  reduction: (Optional) Type of `tf.keras.losses.Reduction` to apply to
    loss. Default value is `AUTO`. `AUTO` indicates that the reduction
    option will be determined by the usage context. For almost all cases
    this defaults to `SUM_OVER_BATCH_SIZE`. When used with
    `tf.distribute.Strategy`, outside of built-in training loops such as
    `tf.keras` `compile` and `fit`, using `AUTO` or `SUM_OVER_BATCH_SIZE`
    will raise an error. Please see this custom training [tutorial](
      https://www.tensorflow.org/tutorials/distribute/custom_training)
    for more details.
  name: Optional name for the op. Defaults to 'squared_hinge'.
"""

docstring_google_tf_squared_hinge_ir = {
    "doc": "Initializes `SquaredHinge` instance.",
    "name": None,
    "params": [
        {
            "default": "AUTO",
            "doc": "(Optional) Type of `tf.keras.losses.Reduction` "
            "to apply to\n"
            "    loss. Default value is `AUTO`. `AUTO` "
            "indicates that the reduction\n"
            "    option will be determined by the usage "
            "context. For almost all cases\n"
            "    this defaults to `SUM_OVER_BATCH_SIZE`. When "
            "used with\n"
            "    `tf.distribute.Strategy`, outside of "
            "built-in training loops such as\n"
            "    `tf.keras` `compile` and `fit`, using `AUTO` "
            "or `SUM_OVER_BATCH_SIZE`\n"
            "    will raise an error. Please see this custom "
            "training [tutorial](\n"
            "      "
            "https://www.tensorflow.org/tutorials/distribute/custom_training)\n"
            "    for more details.",
            "name": "reduction",
        },
        {
            "default": "'squared_hinge'",
            "doc": "Optional name for the op. Defaults to " "'squared_hinge'.",
            "name": "name",
        },
    ],
    "returns": None,
    "type": "static",
}

# https://github.com/tensorflow/tensorflow/blob/7ad2723/tensorflow/python/keras/optimizer_v2/adam.py#L35-L103
docstring_google_tf_adam_str = r"""Optimizer that implements the Adam algorithm.
Adam optimization is a stochastic gradient descent method that is based on
adaptive estimation of first-order and second-order moments.
According to
[Kingma et al., 2014](http://arxiv.org/abs/1412.6980),
the method is "*computationally
efficient, has little memory requirement, invariant to diagonal rescaling of
gradients, and is well suited for problems that are large in terms of
data/parameters*".
Args:
learning_rate: A `Tensor`, floating point value, or a schedule that is a
  `tf.keras.optimizers.schedules.LearningRateSchedule`, or a callable
  that takes no arguments and returns the actual value to use, The
  learning rate. Defaults to 0.001.
beta_1: A float value or a constant float tensor, or a callable
  that takes no arguments and returns the actual value to use. The
  exponential decay rate for the 1st moment estimates. Defaults to 0.9.
beta_2: A float value or a constant float tensor, or a callable
  that takes no arguments and returns the actual value to use, The
  exponential decay rate for the 2nd moment estimates. Defaults to 0.999.
epsilon: A small constant for numerical stability. This epsilon is
  "epsilon hat" in the Kingma and Ba paper (in the formula just before
  Section 2.1), not the epsilon in Algorithm 1 of the paper. Defaults to
  1e-7.
amsgrad: Boolean. Whether to apply AMSGrad variant of this algorithm from
  the paper "On the Convergence of Adam and beyond". Defaults to `False`.
name: Optional name for the operations created when applying gradients.
  Defaults to `"Adam"`.
**kwargs: Keyword arguments. Allowed to be one of
  `"clipnorm"` or `"clipvalue"`.
  `"clipnorm"` (float) clips gradients by norm; `"clipvalue"` (float) clips
  gradients by value.
Usage:
>>> opt = tf.keras.optimizers.Adam(learning_rate=0.1)
>>> var1 = tf.Variable(10.0)
>>> loss = lambda: (var1 ** 2)/2.0       # d(loss)/d(var1) == var1
>>> step_count = opt.minimize(loss, [var1]).numpy()
>>> # The first step is `-learning_rate*sign(grad)`
>>> var1.numpy()
9.9
Reference:
- [Kingma et al., 2014](http://arxiv.org/abs/1412.6980)
- [Reddi et al., 2018](
    https://openreview.net/pdf?id=ryQu7f-RZ) for `amsgrad`.
Notes:
The default value of 1e-7 for epsilon might not be a good default in
general. For example, when training an Inception network on ImageNet a
current good choice is 1.0 or 0.1. Note that since Adam uses the
formulation just before Section 2.1 of the Kingma and Ba paper rather than
the formulation in Algorithm 1, the "epsilon" referred to here is "epsilon
hat" in the paper.
The sparse implementation of this algorithm (used when the gradient is an
IndexedSlices object, typically because of `tf.gather` or an embedding
lookup in the forward pass) does apply momentum to variable slices even if
they were not used in the forward pass (meaning they have a gradient equal
to zero). Momentum decay (beta1) is also applied to the entire momentum
accumulator. This means that the sparse behavior is equivalent to the dense
behavior (in contrast to some momentum implementations which ignore momentum
unless a variable slice was actually used).
"""

docstring_google_tf_adam_ir = {
    "name": None,
    "type": "static",
    "doc": "Optimizer that implements the Adam algorithm."
    "Adam optimization is a stochastic gradient descent method that is based on"
    "adaptive estimation of first-order and second-order moments."
    "According to"
    "[Kingma et al., 2014](http://arxiv.org/abs/1412.6980),"
    'the method is "*computationally'
    "efficient, has little memory requirement, invariant to diagonal rescaling of"
    "gradients, and is well suited for problems that are large in terms of"
    "data/parameters*.",
    "params": [
        {
            "name": "learning_rate",
            "doc": "A `Tensor`, floating point value, or a schedule that is a"
            "`tf.keras.optimizers.schedules.LearningRateSchedule`, or a callable"
            "that takes no arguments and returns the actual value to use, The"
            "learning rate.",
            "default": 0.001,
            "typ": "float",
        },
        {
            "name": "beta_1",
            "doc": "A float value or a constant float tensor, or a callable"
            "that takes no arguments and returns the actual value to use. The"
            "exponential decay rate for the 1st moment estimates.",
            "default": 0.9,
            "typ": "float",
        },
        {
            "name": "beta_2",
            "doc": "A float value or a constant float tensor, or a callable"
            "that takes no arguments and returns the actual value to use, The"
            "exponential decay rate for the 2nd moment estimates.",
            "default": 0.999,
            "typ": "float",
        },
        {
            "name": "epsilon",
            "doc": "A small constant for numerical stability. This epsilon is"
            '"epsilon hat" in the Kingma and Ba paper (in the formula just before'
            "Section 2.1), not the epsilon in Algorithm 1 of the paper.",
            "default": 1e-7,
            "typ": "float",
        },
        {
            "name": "amsgrad",
            "doc": "Boolean. Whether to apply AMSGrad variant of this algorithm from"
            'the paper "On the Convergence of Adam and beyond".',
            "default": False,
            "typ": "bool",
        },
        {
            "name": "name",
            "doc": "Optional name for the operations created when applying gradients.",
            "default": "Adam",
            "typ": "str",
        },
        {
            "name": "kwargs",
            "doc": "Keyword arguments. Allowed to be one of"
            '`"clipnorm"` or `"clipvalue"`.'
            '`"clipnorm"` (float) clips gradients by norm; `"clipvalue"` (float) clips'
            "gradients by value.",
            "default": "{}",
            "typ": "dict",
        },
        {"name": "", "doc": "", "default": "", "typ": ""},
    ],
    "returns": None,
}

docstring_numpydoc_str = """
Acquire from the official tensorflow_datasets model zoo, or the ophthalmology focussed ml-prepare library

Parameters
----------
dataset_name : str
    name of dataset. Defaults to "mnist"
tfds_dir : Optional[str]
    directory to look for models in. Defaults to "~/tensorflow_datasets"
K : Literal['np', 'tf']
    backend engine, e.g., `np` or `tf`. Defaults to "np"
as_numpy : Optional[bool]
    Convert to numpy ndarrays
data_loader_kwargs : dict
    pass this as arguments to data_loader function

Returns
-------
Union[Tuple[tf.data.Dataset, tf.data.Dataset], Tuple[np.ndarray, np.ndarray]]
    Train and tests dataset splits. Defaults to (np.empty(0), np.empty(0))

"""

docstring_numpydoc_only_params_str = """
Parameters
----------
dataset_name : str
    name of dataset. Defaults to "mnist"
tfds_dir : Optional[str]
    directory to look for models in. Defaults to "~/tensorflow_datasets"
K : Literal['np', 'tf']
    backend engine, e.g., `np` or `tf`. Defaults to "np"
as_numpy : Optional[bool]
    Convert to numpy ndarrays
data_loader_kwargs : dict
    pass this as arguments to data_loader function
"""

docstring_numpydoc_only_returns_str = """
Returns
-------
Union[Tuple[tf.data.Dataset, tf.data.Dataset], Tuple[np.ndarray, np.ndarray]]
    Train and tests dataset splits. Defaults to (np.empty(0), np.empty(0))

"""

docstring_numpydoc_only_doc_str = """
Acquire from the official tensorflow_datasets model zoo, or the ophthalmology focussed ml-prepare library
"""

intermediate_repr_no_default_doc_or_prop = remove_defaults_from_intermediate_repr(
    deepcopy(intermediate_repr), emit_defaults=False
)

intermediate_repr_no_default_doc = {
    "name": None,
    "type": "static",
    "doc": "Acquire from the official tensorflow_datasets model "
    "zoo, or the ophthalmology focussed ml-prepare "
    "library",
    "params": [
        {
            "default": '"mnist"',
            "doc": "name of dataset.",
            "name": "dataset_name",
            "typ": "str",
        },
        {
            "default": '"~/tensorflow_datasets"',
            "doc": "directory to look for models in.",
            "name": "tfds_dir",
            "typ": "Optional[str]",
        },
        {
            "default": '"np"',
            "doc": "backend engine, e.g., `np` or `tf`.",
            "name": "K",
            "typ": "Literal['np', 'tf']",
        },
        {
            "doc": "Convert to numpy ndarrays",
            "name": "as_numpy",
            "typ": "Optional[bool]",
        },
        {
            "doc": "pass this as arguments to data_loader function",
            "name": "data_loader_kwargs",
            "typ": "dict",
        },
    ],
    "returns": {
        "default": "(np.empty(0), np.empty(0))",
        "doc": "Train and tests dataset splits.",
        "name": "return_type",
        "typ": "Union[Tuple[tf.data.Dataset, tf.data.Dataset], "
        "Tuple[np.ndarray, np.ndarray]]",
    },
}

intermediate_repr_extra_colons = {
    "name": None,
    "params": [
        {"doc": "Example: foo", "name": "dataset_name", "typ": "str"},
    ],
    "returns": None,
    "doc": "Some comment",
    "type": "static",
}

intermediate_repr_only_return_type = {
    "name": None,
    "type": "static",
    "doc": "Some comment",
    "params": [
        {
            "doc": "Example: foo",
            "name": "dataset_name",
        }
    ],
    "returns": {
        "doc": "Train and tests dataset splits.",
        "name": "return_type",
        "typ": "Union[Tuple[tf.data.Dataset, tf.data.Dataset], "
        "Tuple[np.ndarray, np.ndarray]]",
    },
}

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

docstring_str_extra_colons = """
Some comment

:param dataset_name: Example: foo
:type dataset_name: ```str```
"""

docstring_str_only_return_type = """
Some comment

:param dataset_name: Example: foo

:return: Train and tests dataset splits.
:rtype: ```Union[Tuple[tf.data.Dataset, tf.data.Dataset], Tuple[np.ndarray, np.ndarray]]```
"""

__all__ = [
    "docstring_google_str",
    "docstring_numpydoc_only_doc_str",
    "docstring_numpydoc_only_params_str",
    "docstring_numpydoc_only_returns_str",
    "docstring_numpydoc_str",
    "docstring_str",
    "docstring_str_extra_colons",
    "docstring_str_no_default_doc",
    "intermediate_repr",
    "intermediate_repr_no_default_doc",
    "intermediate_repr_no_default_doc_or_prop",
]
