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
            "default": "mnist",
            "doc": 'name of dataset. Defaults to "mnist"',
            "name": "dataset_name",
            "typ": "str",
        },
        {
            "default": "~/tensorflow_datasets",
            "doc": 'directory to look for models in. Defaults to "~/tensorflow_datasets"',
            "name": "tfds_dir",
            "typ": "Optional[str]",
        },
        {
            "default": "np",
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
            "typ": "Optional[dict]",
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
:type data_loader_kwargs: ```Optional[dict]```

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
  data_loader_kwargs (Optional[dict]): pass this as arguments to data_loader function

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
            "default": "squared_hinge",
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
    "doc": "Optimizer that implements the Adam algorithm.\n"
    "Adam optimization is a stochastic gradient descent method that is "
    "based on\n"
    "adaptive estimation of first-order and second-order moments.\n"
    "According to\n"
    "[Kingma et al., 2014](http://arxiv.org/abs/1412.6980),\n"
    'the method is "*computationally\n'
    "efficient, has little memory requirement, invariant to diagonal "
    "rescaling of\n"
    "gradients, and is well suited for problems that are large in terms "
    "of\n"
    'data/parameters*".\n'
    "\n"
    "\n"
    "Usage:\n"
    "    >>> opt = tf.keras.optimizers.Adam(learning_rate=0.1)\n"
    "    >>> var1 = tf.Variable(10.0)\n"
    "    >>> loss = lambda: (var1 ** 2)/2.0       # d(loss)/d(var1) == "
    "var1\n"
    "    >>> step_count = opt.minimize(loss, [var1]).numpy()\n"
    "    >>> # The first step is `-learning_rate*sign(grad)`\n"
    "    >>> var1.numpy()\n"
    "    9.9\n"
    "Reference:\n"
    "    - [Kingma et al., 2014](http://arxiv.org/abs/1412.6980)\n"
    "    - [Reddi et al., 2018](\n"
    "        https://openreview.net/pdf?id=ryQu7f-RZ) for `amsgrad`.\n"
    "Notes:\n"
    "    The default value of 1e-7 for epsilon might not be a good "
    "default in\n"
    "    general. For example, when training an Inception network on "
    "ImageNet a\n"
    "    current good choice is 1.0 or 0.1. Note that since Adam uses "
    "the\n"
    "    formulation just before Section 2.1 of the Kingma and Ba paper "
    "rather than\n"
    '    the formulation in Algorithm 1, the "epsilon" referred to here '
    'is "epsilon\n'
    '    hat" in the paper.\n'
    "    The sparse implementation of this algorithm (used when the "
    "gradient is an\n"
    "    IndexedSlices object, typically because of `tf.gather` or an "
    "embedding\n"
    "    lookup in the forward pass) does apply momentum to variable "
    "slices even if\n"
    "    they were not used in the forward pass (meaning they have a "
    "gradient equal\n"
    "    to zero). Momentum decay (beta1) is also applied to the entire "
    "momentum\n"
    "    accumulator. This means that the sparse behavior is equivalent "
    "to the dense\n"
    "    behavior (in contrast to some momentum implementations which "
    "ignore momentum\n"
    "    unless a variable slice was actually used).",
    "name": None,
    "params": [
        {
            "default": 0.001,
            "doc": "A `Tensor`, floating point value, or a schedule "
            "that is a\n"
            "  "
            "`tf.keras.optimizers.schedules.LearningRateSchedule`, "
            "or a callable\n"
            "  that takes no arguments and returns the actual "
            "value to use, The\n"
            "  learning rate. Defaults to 0.001.",
            "name": "learning_rate",
            "typ": "float",
        },
        {
            "default": 0.9,
            "doc": "A float value or a constant float tensor, or a "
            "callable\n"
            "  that takes no arguments and returns the actual "
            "value to use. The\n"
            "  exponential decay rate for the 1st moment "
            "estimates. Defaults to 0.9.",
            "name": "beta_1",
            "typ": "float",
        },
        {
            "default": 0.999,
            "doc": "A float value or a constant float tensor, or a "
            "callable\n"
            "  that takes no arguments and returns the actual "
            "value to use, The\n"
            "  exponential decay rate for the 2nd moment "
            "estimates. Defaults to 0.999.",
            "name": "beta_2",
            "typ": "float",
        },
        {
            "default": 1e-07,
            "doc": "A small constant for numerical stability. This "
            "epsilon is\n"
            '  "epsilon hat" in the Kingma and Ba paper (in '
            "the formula just before\n"
            "  Section 2.1), not the epsilon in Algorithm 1 "
            "of the paper. Defaults to\n"
            "  1e-7.",
            "name": "epsilon",
            "typ": "float",
        },
        {
            "default": False,
            "doc": "Boolean. Whether to apply AMSGrad variant of "
            "this algorithm from\n"
            '  the paper "On the Convergence of Adam and '
            'beyond". Defaults to `False`.',
            "name": "amsgrad",
            "typ": "bool",
        },
        {
            "default": "Adam",
            "doc": "Optional name for the operations created when "
            "applying gradients.\n"
            '  Defaults to `"Adam"`.',
            "name": "name",
            "typ": "str",
        },
        {
            "doc": "Keyword arguments. Allowed to be one of\n"
            '  `"clipnorm"` or `"clipvalue"`.\n'
            '  `"clipnorm"` (float) clips gradients by norm; '
            '`"clipvalue"` (float) clips\n'
            "  gradients by value.",
            "name": "kwargs",
            "typ": "Optional[dict]",
        },
    ],
    "returns": None,
    "type": "static",
}

# https://github.com/tensorflow/tensorflow/blob/7ad2723/tensorflow/python/keras/optimizer_v2/adadelta.py#L33-L77
docstring_google_tf_adadelta_str = """Optimizer that implements the Adadelta algorithm.

Adadelta optimization is a stochastic gradient descent method that is based on
adaptive learning rate per dimension to address two drawbacks:

- The continual decay of learning rates throughout training
- The need for a manually selected global learning rate

Adadelta is a more robust extension of Adagrad that adapts learning rates
based on a moving window of gradient updates, instead of accumulating all
past gradients. This way, Adadelta continues learning even when many updates
have been done. Compared to Adagrad, in the original version of Adadelta you
don't have to set an initial learning rate. In this version, initial
learning rate can be set, as in most other Keras optimizers.

According to section 4.3 ("Effective Learning rates"), near the end of
training step sizes converge to 1 which is effectively a high learning
rate which would cause divergence. This occurs only near the end of the
training as gradients and step sizes are small, and the epsilon constant
in the numerator and denominator dominate past gradients and parameter
updates which converge the learning rate to 1.

According to section 4.4("Speech Data"),where a large neural network with
4 hidden layers was trained on a corpus of US English data, ADADELTA was
used with 100 network replicas.The epsilon used is 1e-6 with rho=0.95
which converged faster than ADAGRAD, by the following construction:
def __init__(self, lr=1.0, rho=0.95, epsilon=1e-6, decay=0., **kwargs):

Args:
learning_rate: A `Tensor`, floating point value, or a schedule that is a
  `tf.keras.optimizers.schedules.LearningRateSchedule`. The learning rate.
  To match the exact form in the original paper use 1.0.
rho: A `Tensor` or a floating point value. The decay rate.
epsilon: A `Tensor` or a floating point value.  A constant epsilon used
         to better conditioning the grad update.
name: Optional name prefix for the operations created when applying
  gradients.  Defaults to `"Adadelta"`.
**kwargs: Keyword arguments. Allowed to be one of
  `"clipnorm"` or `"clipvalue"`.
  `"clipnorm"` (float) clips gradients by norm; `"clipvalue"` (float) clips
  gradients by value.

Reference:
- [Zeiler, 2012](http://arxiv.org/abs/1212.5701)
"""

docstring_google_tf_adadelta_ir = {
    "doc": "Optimizer that implements the Adadelta algorithm.\n"
    "\n"
    "Adadelta optimization is a stochastic gradient descent method that "
    "is based on\n"
    "adaptive learning rate per dimension to address two drawbacks:\n"
    "\n"
    "- The continual decay of learning rates throughout training\n"
    "- The need for a manually selected global learning rate\n"
    "\n"
    "Adadelta is a more robust extension of Adagrad that adapts "
    "learning rates\n"
    "based on a moving window of gradient updates, instead of "
    "accumulating all\n"
    "past gradients. This way, Adadelta continues learning even when "
    "many updates\n"
    "have been done. Compared to Adagrad, in the original version of "
    "Adadelta you\n"
    "don't have to set an initial learning rate. In this version, "
    "initial\n"
    "learning rate can be set, as in most other Keras optimizers.\n"
    "\n"
    'According to section 4.3 ("Effective Learning rates"), near the '
    "end of\n"
    "training step sizes converge to 1 which is effectively a high "
    "learning\n"
    "rate which would cause divergence. This occurs only near the end "
    "of the\n"
    "training as gradients and step sizes are small, and the epsilon "
    "constant\n"
    "in the numerator and denominator dominate past gradients and "
    "parameter\n"
    "updates which converge the learning rate to 1.\n"
    "\n"
    'According to section 4.4("Speech Data"),where a large neural '
    "network with\n"
    "4 hidden layers was trained on a corpus of US English data, "
    "ADADELTA was\n"
    "used with 100 network replicas.The epsilon used is 1e-6 with "
    "rho=0.95\n"
    "which converged faster than ADAGRAD, by the following "
    "construction:\n"
    "def __init__(self, lr=1.0, rho=0.95, epsilon=1e-6, decay=0., "
    "**kwargs):\n"
    "\n"
    "\n"
    "Reference:\n"
    "    - [Zeiler, 2012](http://arxiv.org/abs/1212.5701)",
    "name": None,
    "params": [
        {
            "doc": "A `Tensor`, floating point value, or a schedule "
            "that is a\n"
            "  "
            "`tf.keras.optimizers.schedules.LearningRateSchedule`. "
            "The learning rate.\n"
            "  To match the exact form in the original paper "
            "use 1.0.",
            "name": "learning_rate",
        },
        {
            "doc": "A `Tensor` or a floating point value. The decay " "rate.",
            "name": "rho",
        },
        {
            "doc": "A `Tensor` or a floating point value.  A "
            "constant epsilon used\n"
            "         to better conditioning the grad update.",
            "name": "epsilon",
        },
        {
            "default": "Adadelta",
            "doc": "Optional name prefix for the operations created "
            "when applying\n"
            '  gradients.  Defaults to `"Adadelta"`.',
            "name": "name",
            "typ": "str",
        },
        {
            "doc": "Keyword arguments. Allowed to be one of\n"
            '  `"clipnorm"` or `"clipvalue"`.\n'
            '  `"clipnorm"` (float) clips gradients by norm; '
            '`"clipvalue"` (float) clips\n'
            "  gradients by value.",
            "name": "kwargs",
            "typ": "Optional[dict]",
        },
    ],
    "returns": None,
    "type": "static",
}

# https://github.com/tensorflow/tensorflow/blob/7ad2723/tensorflow/python/keras/callbacks.py#L2649-L2699
docstring_google_tf_lambda_callback_str = """Callback for creating simple, custom callbacks on-the-fly.

This callback is constructed with anonymous functions that will be called
at the appropriate time. Note that the callbacks expects positional
arguments, as:

- `on_epoch_begin` and `on_epoch_end` expect two positional arguments:
`epoch`, `logs`
- `on_batch_begin` and `on_batch_end` expect two positional arguments:
`batch`, `logs`
- `on_train_begin` and `on_train_end` expect one positional argument:
`logs`

Args:
  on_epoch_begin: called at the beginning of every epoch.
  on_epoch_end: called at the end of every epoch.
  on_batch_begin: called at the beginning of every batch.
  on_batch_end: called at the end of every batch.
  on_train_begin: called at the beginning of model training.
  on_train_end: called at the end of model training.

Example:

```python
# Print the batch number at the beginning of every batch.
batch_print_callback = LambdaCallback(
  on_batch_begin=lambda batch,logs: print(batch))

# Stream the epoch loss to a file in JSON format. The file content
# is not well-formed JSON but rather has a JSON object per line.
import json
json_log = open('loss_log.json', mode='wt', buffering=1)
json_logging_callback = LambdaCallback(
  on_epoch_end=lambda epoch, logs: json_log.write(
      json.dumps({'epoch': epoch, 'loss': logs['loss']}) + '\n'),
  on_train_end=lambda logs: json_log.close()
)

# Terminate some processes after having finished model training.
processes = ...
cleanup_callback = LambdaCallback(
  on_train_end=lambda logs: [
      p.terminate() for p in processes if p.is_alive()])

model.fit(...,
        callbacks=[batch_print_callback,
                   json_logging_callback,
                   cleanup_callback])
```
"""

docstring_google_tf_lambda_callback_ir = {
    "doc": "Callback for creating simple, custom callbacks on-the-fly.\n"
    "\n"
    "This callback is constructed with anonymous functions that will be "
    "called\n"
    "at the appropriate time. Note that the callbacks expects "
    "positional\n"
    "arguments, as:\n"
    "\n"
    "- `on_epoch_begin` and `on_epoch_end` expect two positional "
    "arguments:\n"
    "`epoch`, `logs`\n"
    "- `on_batch_begin` and `on_batch_end` expect two positional "
    "arguments:\n"
    "`batch`, `logs`\n"
    "- `on_train_begin` and `on_train_end` expect one positional "
    "argument:\n"
    "`logs`"
    "\n"
    "\n"
    "\n"
    "Example:\n"
    "\n"
    "```python\n"
    "# Print the batch number at the beginning of every batch.\n"
    "batch_print_callback = LambdaCallback(\n"
    "  on_batch_begin=lambda batch,logs: print(batch))\n"
    "\n"
    "# Stream the epoch loss to a file in JSON format. The file "
    "content\n"
    "# is not well-formed JSON but rather has a JSON object per line.\n"
    "import json\n"
    "json_log = open('loss_log.json', mode='wt', buffering=1)\n"
    "json_logging_callback = LambdaCallback(\n"
    "  on_epoch_end=lambda epoch, logs: json_log.write(\n"
    "      json.dumps({'epoch': epoch, 'loss': logs['loss']}) + '\n"
    "'),\n"
    "  on_train_end=lambda logs: json_log.close()\n"
    ")\n"
    "\n"
    "# Terminate some processes after having finished model training.\n"
    "processes = ...\n"
    "cleanup_callback = LambdaCallback(\n"
    "  on_train_end=lambda logs: [\n"
    "      p.terminate() for p in processes if p.is_alive()])\n"
    "\n"
    "model.fit(...,\n"
    "        callbacks=[batch_print_callback,\n"
    "                   json_logging_callback,\n"
    "                   cleanup_callback])\n"
    "```",
    "name": None,
    "params": [
        {"doc": "called at the beginning of every epoch.", "name": "on_epoch_begin"},
        {"doc": "called at the end of every epoch.", "name": "on_epoch_end"},
        {"doc": "called at the beginning of every batch.", "name": "on_batch_begin"},
        {"doc": "called at the end of every batch.", "name": "on_batch_end"},
        {"doc": "called at the beginning of model training.", "name": "on_train_begin"},
        {"doc": "called at the end of model training.", "name": "on_train_end"},
    ],
    "returns": None,
    "type": "static",
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
data_loader_kwargs : Optional[dict]
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
data_loader_kwargs : Optional[dict]
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
            "default": "mnist",
            "doc": "name of dataset.",
            "name": "dataset_name",
            "typ": "str",
        },
        {
            "default": "~/tensorflow_datasets",
            "doc": "directory to look for models in.",
            "name": "tfds_dir",
            "typ": "Optional[str]",
        },
        {
            "default": "np",
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
            "typ": "Optional[dict]",
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
:type data_loader_kwargs: ```Optional[dict]```

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
    "docstring_google_tf_adadelta_ir",
    "docstring_google_tf_adadelta_str",
    "docstring_google_tf_adam_ir",
    "docstring_google_tf_adam_str",
    "docstring_google_tf_lambda_callback_ir",
    "docstring_google_tf_lambda_callback_str",
    "docstring_google_tf_squared_hinge_ir",
    "docstring_google_tf_squared_hinge_str",
    "docstring_numpydoc_only_doc_str",
    "docstring_numpydoc_only_params_str",
    "docstring_numpydoc_only_returns_str",
    "docstring_numpydoc_str",
    "docstring_str",
    "docstring_str_extra_colons",
    "docstring_str_no_default_doc",
    "docstring_str_only_return_type",
    "intermediate_repr",
    "intermediate_repr_extra_colons",
    "intermediate_repr_no_default_doc",
    "intermediate_repr_no_default_doc_or_prop",
    "intermediate_repr_only_return_type",
]
