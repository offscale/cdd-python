"""
Mocks for docstrings

Note: TensorFlow code is taken from `5a56eb1`; the same that tf 2.15.0 was released with on 14/11/2023.
"""

from itertools import chain
from textwrap import indent

from cdd.shared.pure_utils import identity, tab

docstring_header_no_nl_str: str = (
    "Acquire from the official tensorflow_datasets model zoo,"
    " or the ophthalmology focussed ml-prepare"
)
docstring_header_str: str = "{docstring_header_no_nl_str}\n".format(
    docstring_header_no_nl_str=docstring_header_no_nl_str
)

_docstring_header_and_return_str = (
    ":return: Train and tests dataset splits. Defaults to (np.empty(0), np.empty(0))",
    ":rtype: ```Union[Tuple[tf.data.Dataset, tf.data.Dataset], Tuple[np.ndarray, np.ndarray]]```",
)  # type: tuple[str, str]

docstring_header_and_return_str: str = "\n".join(
    (docstring_header_str, *_docstring_header_and_return_str)
)

docstring_header_and_return_no_nl_str: str = "\n".join(
    (docstring_header_no_nl_str, *_docstring_header_and_return_str)
)

docstring_header_and_return_two_nl_str: str = "\n".join(
    (docstring_header_no_nl_str, "\n", *_docstring_header_and_return_str)
)

docstring_extra_colons_str: str = """
Some comment

:param dataset_name: Example: foo
:type dataset_name: ```str```
"""

# ```py
# import ast
# import inspect
#
# import keras.losses
#
# ast.parse(inspect.getsource(keras.losses.SquaredHinge)).body[0].body[0].value.value.splitlines()
# ```
# ###############
# # Keras 3.0.1 #
# ###############
# https://github.com/keras-team/keras/blob/f889c1f/keras/losses/losses.py#L282-L297
docstring_google_keras_squared_hinge_no_args_doc_str: str = (
    "Initializes `SquaredHinge` instance."
)
docstring_google_keras_squared_hinge = (
    docstring_google_keras_squared_hinge_no_args_doc_str,
    "    Args:",
    "        reduction: Type of reduction to apply to the loss. In almost all cases",
    '            this should be `"sum_over_batch_size"`.',
    '            Supported options are `"sum"`, `"sum_over_batch_size"` or `None`.',
    "        name: Optional name for the loss instance.",
    "    ",
)  # type: tuple[str, ...]
docstring_google_keras_squared_hinge_str: str = "\n".join(
    docstring_google_keras_squared_hinge
)

docstring_google_keras_tensorboard_return_str: str = """
    Returns:
        True on success, or False if no summary was written because no default
        summary writer was available.
"""

# ```py
# import ast
# import inspect
#
# import keras.optimizers
# from keras.src.optimizers.base_optimizer import base_optimizer_keyword_args
#
# ast.parse(inspect.getsource(keras.optimizers.Adam)).body[0].body[0].value.value.replace(
#     "{{base_optimizer_keyword_args}}", base_optimizer_keyword_args).splitlines()
# ```
# ###############
# # Keras 3.0.1 #
# ###############
# https://github.com/keras-team/keras/blob/f889c1f/keras/optimizers/adam.py#L8-L40
docstring_google_keras_adam = (
    "Optimizer that implements the Adam algorithm.",
    "",
    "    Adam optimization is a stochastic gradient descent method that is based on",
    "    adaptive estimation of first-order and second-order moments.",
    "",
    "    According to",
    "    [Kingma et al., 2014](http://arxiv.org/abs/1412.6980),",
    '    the method is "*computationally',
    "    efficient, has little memory requirement, invariant to diagonal rescaling of",
    "    gradients, and is well suited for problems that are large in terms of",
    '    data/parameters*".',
    "",
    "    Args:",
    "        learning_rate: A float, a",
    "            `keras.optimizers.schedules.LearningRateSchedule` instance, or",
    "            a callable that takes no arguments and returns the actual value to",
    "            use. The learning rate. Defaults to `0.001`.",
    "        beta_1: A float value or a constant float tensor, or a callable",
    "            that takes no arguments and returns the actual value to use. The",
    "            exponential decay rate for the 1st moment estimates. Defaults to",
    "            `0.9`.",
    "        beta_2: A float value or a constant float tensor, or a callable",
    "            that takes no arguments and returns the actual value to use. The",
    "            exponential decay rate for the 2nd moment estimates. Defaults to",
    "            `0.999`.",
    "        epsilon: A small constant for numerical stability. This epsilon is",
    '            "epsilon hat" in the Kingma and Ba paper (in the formula just before',
    "            Section 2.1), not the epsilon in Algorithm 1 of the paper. Defaults",
    "            to `1e-7`.",
    "        amsgrad: Boolean. Whether to apply AMSGrad variant of this algorithm",
    '            from the paper "On the Convergence of Adam and beyond". Defaults',
    "            to `False`.",
    "        name: String. The name to use",
    "          for momentum accumulator weights created by",
    "          the optimizer.",
    "        weight_decay: Float. If set, weight decay is applied.",
    "        clipnorm: Float. If set, the gradient of each weight is individually",
    "          clipped so that its norm is no higher than this value.",
    "        clipvalue: Float. If set, the gradient of each weight is clipped to be",
    "          no higher than this value.",
    "        global_clipnorm: Float. If set, the gradient of all weights is clipped",
    "          so that their global norm is no higher than this value.",
    "        use_ema: Boolean, defaults to False. If True, exponential moving average",
    "          (EMA) is applied. EMA consists of computing an exponential moving",
    "          average of the weights of the model (as the weight values change after",
    "          each training batch), and periodically overwriting the weights with",
    "          their moving average.",
    "        ema_momentum: Float, defaults to 0.99. Only used if `use_ema=True`.",
    "          This is the momentum to use when computing",
    "          the EMA of the model's weights:",
    "          `new_average = ema_momentum * old_average + (1 - ema_momentum) *",
    "          current_variable_value`.",
    "        ema_overwrite_frequency: Int or None, defaults to None. Only used if",
    "          `use_ema=True`. Every `ema_overwrite_frequency` steps of iterations,",
    "          we overwrite the model variable by its moving average.",
    "          If None, the optimizer",
    "          does not overwrite model variables in the middle of training, and you",
    "          need to explicitly overwrite the variables at the end of training",
    "          by calling `optimizer.finalize_variable_values()`",
    "          (which updates the model",
    "          variables in-place). When using the built-in `fit()` training loop,",
    "          this happens automatically after the last epoch,",
    "          and you don't need to do anything.",
    "        loss_scale_factor: Float or `None`. If a float, the scale factor will",
    "          be multiplied the loss before computing gradients, and the inverse of",
    "          the scale factor will be multiplied by the gradients before updating",
    "          variables. Useful for preventing underflow during mixed precision",
    "          training. Alternately, `keras.optimizers.LossScaleOptimizer` will",
    "          automatically set a loss scale factor.",
    "",
    "    ",
)  # type: tuple[str, ...]
docstring_google_keras_adam_str: str = "\n".join(docstring_google_keras_adam)

# ```py
# import ast
# import inspect
#
# import keras.optimizers
# from keras.src.optimizers.base_optimizer import base_optimizer_keyword_args
#
# ast.parse(inspect.getsource(keras.optimizers.Adadelta)).body[0].body[0].value.value.replace(
#     "{{base_optimizer_keyword_args}}", base_optimizer_keyword_args).splitlines()
# ```
# ###############
# # Keras 3.0.1 #
# ###############
# https://github.com/keras-team/keras/blob/f889c1f/keras/optimizers/adadelta.py#L8-L37
docstring_google_keras_adadelta = (
    "Optimizer that implements the Adadelta algorithm.",
    "",
    "    Adadelta optimization is a stochastic gradient descent method that is based",
    "    on adaptive learning rate per dimension to address two drawbacks:",
    "",
    "    - The continual decay of learning rates throughout training.",
    "    - The need for a manually selected global learning rate.",
    "",
    "    Adadelta is a more robust extension of Adagrad that adapts learning rates",
    "    based on a moving window of gradient updates, instead of accumulating all",
    "    past gradients. This way, Adadelta continues learning even when many updates",
    "    have been done. Compared to Adagrad, in the original version of Adadelta you",
    "    don't have to set an initial learning rate. In this version, the initial",
    "    learning rate can be set, as in most other Keras optimizers.",
    "",
    "    Args:",
    "        learning_rate: A float, a",
    "            `keras.optimizers.schedules.LearningRateSchedule` instance, or",
    "            a callable that takes no arguments and returns the actual value to",
    "            use. The learning rate. Defaults to `0.001`. Note that `Adadelta`",
    "            tends to benefit from higher initial learning rate values compared",
    "            to other optimizers. To match the exact form in the original paper,",
    "            use 1.0.",
    "        rho: A floating point value. The decay rate. Defaults to `0.95`.",
    "        epsilon: Small floating point value for maintaining numerical stability.",
    "        name: String. The name to use",
    "          for momentum accumulator weights created by",
    "          the optimizer.",
    "        weight_decay: Float. If set, weight decay is applied.",
    "        clipnorm: Float. If set, the gradient of each weight is individually",
    "          clipped so that its norm is no higher than this value.",
    "        clipvalue: Float. If set, the gradient of each weight is clipped to be",
    "          no higher than this value.",
    "        global_clipnorm: Float. If set, the gradient of all weights is clipped",
    "          so that their global norm is no higher than this value.",
    "        use_ema: Boolean, defaults to False. If True, exponential moving average",
    "          (EMA) is applied. EMA consists of computing an exponential moving",
    "          average of the weights of the model (as the weight values change after",
    "          each training batch), and periodically overwriting the weights with",
    "          their moving average.",
    "        ema_momentum: Float, defaults to 0.99. Only used if `use_ema=True`.",
    "          This is the momentum to use when computing",
    "          the EMA of the model's weights:",
    "          `new_average = ema_momentum * old_average + (1 - ema_momentum) *",
    "          current_variable_value`.",
    "        ema_overwrite_frequency: Int or None, defaults to None. Only used if",
    "          `use_ema=True`. Every `ema_overwrite_frequency` steps of iterations,",
    "          we overwrite the model variable by its moving average.",
    "          If None, the optimizer",
    "          does not overwrite model variables in the middle of training, and you",
    "          need to explicitly overwrite the variables at the end of training",
    "          by calling `optimizer.finalize_variable_values()`",
    "          (which updates the model",
    "          variables in-place). When using the built-in `fit()` training loop,",
    "          this happens automatically after the last epoch,",
    "          and you don't need to do anything.",
    "        loss_scale_factor: Float or `None`. If a float, the scale factor will",
    "          be multiplied the loss before computing gradients, and the inverse of",
    "          the scale factor will be multiplied by the gradients before updating",
    "          variables. Useful for preventing underflow during mixed precision",
    "          training. Alternately, `keras.optimizers.LossScaleOptimizer` will",
    "          automatically set a loss scale factor.",
    "",
    "    Reference:",
    "",
    "    - [Zeiler, 2012](http://arxiv.org/abs/1212.5701)",
    "    ",
)  # type: tuple[str, ...]
docstring_google_keras_adadelta_str: str = "\n".join(docstring_google_keras_adadelta)

# ```py
# import ast
# import inspect
# import keras.callbacks
#
# ast.parse(inspect.getsource(keras.callbacks.LambdaCallback)).body[0].body[0].value.value.splitlines()
# ```
# ###############
# # Keras 3.0.1 #
# ###############
# https://github.com/keras-team/keras/blob/f889c1f/keras/callbacks/lambda_callback.py#L7-L61
docstring_google_keras_lambda_callback = (
    "Callback for creating simple, custom callbacks on-the-fly.",
    "",
    "    This callback is constructed with anonymous functions that will be called",
    "    at the appropriate time (during `Model.{fit | evaluate | predict}`).",
    "    Note that the callbacks expects positional arguments, as:",
    "",
    "    - `on_epoch_begin` and `on_epoch_end` expect two positional arguments:",
    "      `epoch`, `logs`",
    "    - `on_train_begin` and `on_train_end` expect one positional argument:",
    "      `logs`",
    "    - `on_train_batch_begin` and `on_train_batch_end` expect two positional",
    "      arguments: `batch`, `logs`",
    "    - See `Callback` class definition for the full list of functions and their",
    "      expected arguments.",
    "",
    "    Args:",
    "        on_epoch_begin: called at the beginning of every epoch.",
    "        on_epoch_end: called at the end of every epoch.",
    "        on_train_begin: called at the beginning of model training.",
    "        on_train_end: called at the end of model training.",
    "        on_train_batch_begin: called at the beginning of every train batch.",
    "        on_train_batch_end: called at the end of every train batch.",
    "        kwargs: Any function in `Callback` that you want to override by",
    "            passing `function_name=function`. For example,",
    "            `LambdaCallback(.., on_train_end=train_end_fn)`. The custom function",
    "            needs to have same arguments as the ones defined in `Callback`.",
    "",
    "    Example:",
    "",
    "    ```python",
    "    # Print the batch number at the beginning of every batch.",
    "    batch_print_callback = LambdaCallback(",
    "        on_train_batch_begin=lambda batch,logs: print(batch))",
    "",
    "    # Stream the epoch loss to a file in JSON format. The file content",
    "    # is not well-formed JSON but rather has a JSON object per line.",
    "    import json",
    "    json_log = open('loss_log.json', mode='wt', buffering=1)",
    "    json_logging_callback = LambdaCallback(",
    "        on_epoch_end=lambda epoch, logs: json_log.write(",
    "            json.dumps({'epoch': epoch, 'loss': logs['loss']}) + '",
    "'),",
    "        on_train_end=lambda logs: json_log.close()",
    "    )",
    "",
    "    # Terminate some processes after having finished model training.",
    "    processes = ...",
    "    cleanup_callback = LambdaCallback(",
    "        on_train_end=lambda logs: [",
    "            p.terminate() for p in processes if p.is_alive()])",
    "",
    "    model.fit(...,",
    "              callbacks=[batch_print_callback,",
    "                         json_logging_callback,",
    "                         cleanup_callback])",
    "    ```",
    "    ",
)
docstring_google_keras_lambda_callback_str: str = "\n".join(
    docstring_google_keras_lambda_callback
)

# ```py
# import ast
# import inspect
# from tensorflow.python.ops.losses.losses_impl import mean_squared_error
#
# ast.get_docstring(ast.parse(inspect.getsource(mean_squared_error)).body[0]).splitlines()
# ```
# #####################
# # TensorFlow 2.15.0 #
# #####################
# https://github.com/tensorflow/tensorflow/blob/5a56eb1/tensorflow/python/ops/losses/losses_impl.py#L627-L755
docstring_google_tf_mean_squared_error_header_tuple = (
    "Adds a Sum-of-Squares loss to the training procedure.",
    "",
    "  `weights` acts as a coefficient for the loss. If a scalar is provided, then",
    "  the loss is simply scaled by the given value. If `weights` is a tensor of size",
    "  `[batch_size]`, then the total loss for each sample of the batch is rescaled",
    "  by the corresponding element in the `weights` vector. If the shape of",
    "  `weights` matches the shape of `predictions`, then the loss of each",
    "  measurable element of `predictions` is scaled by the corresponding value of",
    "  `weights`.",
    "",
)
docstring_google_tf_mean_squared_error_args_tuple = (
    "  Args:",
    "    labels: The ground truth output tensor, same dimensions as 'predictions'.",
    "    predictions: The predicted outputs.",
    "    weights: Optional `Tensor` whose rank is either 0, or the same rank as",
    "      `labels`, and must be broadcastable to `labels` (i.e., all dimensions must",
    "      be either `1`, or the same as the corresponding `losses` dimension).",
    "    scope: The scope for the operations performed in computing the loss.",
    "    loss_collection: collection to which the loss will be added.",
    "    reduction: Type of reduction to apply to loss.",
    "",
    "  Returns:",
    "    Weighted loss float `Tensor`. If `reduction` is `NONE`, this has the same",
    "    shape as `labels`; otherwise, it is scalar.",
    "",
)
docstring_google_tf_mean_squared_error_footer_tuple = (
    "  Raises:",
    "    ValueError: If the shape of `predictions` doesn't match that of `labels` or",
    "      if the shape of `weights` is invalid.  Also if `labels` or `predictions`",
    "      is None.",
    "",
    "  @compatibility(TF2)",
    "",
    "  `tf.compat.v1.losses.mean_squared_error` is mostly compatible with eager",
    "  execution and `tf.function`. But, the `loss_collection` argument is",
    "  ignored when executing eagerly and no loss will be written to the loss",
    "  collections. You will need to either hold on to the return value manually",
    "  or rely on `tf.keras.Model` loss tracking.",
    "",
    "",
    "  To switch to native TF2 style, instantiate the",
    "   `tf.keras.losses.MeanSquaredError` class and call the object instead.",
    "",
    "",
    "  #### Structural Mapping to Native TF2",
    "",
    "  Before:",
    "",
    "  ```python",
    "  loss = tf.compat.v1.losses.mean_squared_error(",
    "    labels=labels,",
    "    predictions=predictions,",
    "    weights=weights,",
    "    reduction=reduction)",
    "  ```",
    "",
    "  After:",
    "",
    "  ```python",
    "  loss_fn = tf.keras.losses.MeanSquaredError(",
    "    reduction=reduction)",
    "  loss = loss_fn(",
    "    y_true=labels,",
    "    y_pred=predictions,",
    "    sample_weight=weights)",
    "  ```",
    "",
    "  #### How to Map Arguments",
    "",
    "  | TF1 Arg Name          | TF2 Arg Name     | Note                       |",
    "  | :-------------------- | :--------------- | :------------------------- |",
    "  | `labels`              | `y_true`         | In `__call__()` method     |",
    "  | `predictions`         | `y_pred`         | In `__call__()` method     |",
    "  | `weights`             | `sample_weight`  | In `__call__()` method.    |",
    "  : : : The shape requirements for `sample_weight` is different from      :",
    "  : : : `weights`. Please check the [argument definition][api_docs] for   :",
    "  : : : details.                                                          :",
    "  | `scope`               | Not supported    | -                          |",
    "  | `loss_collection`     | Not supported    | Losses should be tracked   |",
    "  : : : explicitly or with Keras APIs, for example, [add_loss][add_loss], :",
    "  : : : instead of via collections                                        :",
    "  | `reduction`           | `reduction`      | In constructor. Value of   |",
    "  : : : `tf.compat.v1.losses.Reduction.SUM_OVER_BATCH_SIZE`,              :",
    "  : : : `tf.compat.v1.losses.Reduction.SUM`,                              :",
    "  : : : `tf.compat.v1.losses.Reduction.NONE` in                           :",
    "  : : : `tf.compat.v1.losses.softmax_cross_entropy` correspond to         :",
    "  : : : `tf.keras.losses.Reduction.SUM_OVER_BATCH_SIZE`,                  :",
    "  : : : `tf.keras.losses.Reduction.SUM`,                                  :",
    "  : : : `tf.keras.losses.Reduction.NONE`, respectively. If you            :",
    "  : : : used other value for `reduction`, including the default value     :",
    "  : : :  `tf.compat.v1.losses.Reduction.SUM_BY_NONZERO_WEIGHTS`, there is :",
    "  : : : no directly corresponding value. Please modify the loss           :",
    "  : : : implementation manually.                                          :",
    "",
    "  [add_loss]:https://www.tensorflow.org/api_docs/python/tf/keras/layers/Layer#add_loss",
    "  [api_docs]:https://www.tensorflow.org/api_docs/python/tf/keras/losses/MeanSquaredError#__call__",
    "",
    "",
    "  #### Before & After Usage Example",
    "",
    "  Before:",
    "",
    "  >>> y_true = [1, 2, 3]",
    "  >>> y_pred = [1, 3, 5]",
    "  >>> weights = [0, 1, 0.25]",
    "  >>> # samples with zero-weight are excluded from calculation when `reduction`",
    "  >>> # argument is set to default value `Reduction.SUM_BY_NONZERO_WEIGHTS`",
    "  >>> tf.compat.v1.losses.mean_squared_error(",
    "  ...    labels=y_true,",
    "  ...    predictions=y_pred,",
    "  ...    weights=weights).numpy()",
    "  1.0",
    "",
    "  >>> tf.compat.v1.losses.mean_squared_error(",
    "  ...    labels=y_true,",
    "  ...    predictions=y_pred,",
    "  ...    weights=weights,",
    "  ...    reduction=tf.compat.v1.losses.Reduction.SUM_OVER_BATCH_SIZE).numpy()",
    "  0.66667",
    "",
    "  After:",
    "",
    "  >>> y_true = [[1.0], [2.0], [3.0]]",
    "  >>> y_pred = [[1.0], [3.0], [5.0]]",
    "  >>> weights = [1, 1, 0.25]",
    "  >>> mse = tf.keras.losses.MeanSquaredError(",
    "  ...    reduction=tf.keras.losses.Reduction.SUM_OVER_BATCH_SIZE)",
    "  >>> mse(y_true=y_true, y_pred=y_pred, sample_weight=weights).numpy()",
    "  0.66667",
    "",
    "  @end_compatibility",
    "  ",
)
docstring_google_tf_mean_squared_error_str: str = "\n".join(
    chain.from_iterable(
        (
            docstring_google_tf_mean_squared_error_header_tuple,
            docstring_google_tf_mean_squared_error_args_tuple,
            docstring_google_tf_mean_squared_error_footer_tuple,
        )
    )
)

docstring_google_pytorch_lbfgs = (
    "",
    "Implements L-BFGS algorithm, heavily inspired by `minFunc",
    "    <https://www.cs.ubc.ca/~schmidtm/Software/minFunc.html>`.",
    "",
    "    Args:",
    "        lr (float): learning rate (default: 1)",
    "        max_iter (int): maximal number of iterations per optimization step",
    "            (default: 20)",
    "        max_eval (int): maximal number of function evaluations per optimization",
    "            step (default: max_iter * 1.25).",
    "        tolerance_grad (float): termination tolerance on first order optimality",
    "            (default: 1e-5).",
    "        tolerance_change (float): termination tolerance on function",
    "            value/parameter changes (default: 1e-9).",
    "        history_size (int): update history size (default: 100).",
    "        line_search_fn (str): either 'strong_wolfe' or None (default: None).",
)
docstring_google_pytorch_lbfgs_str: str = "\n".join(docstring_google_pytorch_lbfgs)

docstring_google_str: str = (
    """{docstring_header_str}
Args:
  dataset_name (str): name of dataset. Defaults to "mnist"
  tfds_dir (str): directory to look for models in. Defaults to "~/tensorflow_datasets"
  K (Literal['np', 'tf']): backend engine, e.g., `np` or `tf`. Defaults to "np"
  as_numpy (Optional[bool]): Convert to numpy ndarrays
  data_loader_kwargs (Optional[dict]): pass this as arguments to data_loader function

Returns:
  Union[Tuple[tf.data.Dataset, tf.data.Dataset], Tuple[np.ndarray, np.ndarray]]:
   Train and tests dataset splits. Defaults to (np.empty(0), np.empty(0))
""".format(
        docstring_header_str=docstring_header_str
    )
)

docstring_no_default_doc_str: str = (
    """
{header_doc_str}
:param dataset_name: name of dataset.
:type dataset_name: ```str```

:param tfds_dir: directory to look for models in.
:type tfds_dir: ```str```

:param K: backend engine, e.g., `np` or `tf`.
:type K: ```Literal['np', 'tf']```

:param as_numpy: Convert to numpy ndarrays
:type as_numpy: ```Optional[bool]```

:param data_loader_kwargs: pass this as arguments to data_loader function
:type data_loader_kwargs: ```Optional[dict]```

:return: Train and tests dataset splits.
:rtype: ```Union[Tuple[tf.data.Dataset, tf.data.Dataset], Tuple[np.ndarray, np.ndarray]]```
""".format(
        header_doc_str=docstring_header_str
    )
)

docstring_no_default_doc_wrapped_str: str = docstring_no_default_doc_str.replace(
    " np.ndarray]]```", "\n{tab}np.ndarray]]```".format(tab=tab)
)

_docstring_no_default_str: str = """
{header_doc_str}\n
:param dataset_name: name of dataset.
:type dataset_name: ```str```

:param tfds_dir: directory to look for models in.
:type tfds_dir: ```str```

:param K: backend engine, e.g., `np` or `tf`.
:type K: ```Literal['np', 'tf']```

:param as_numpy: Convert to numpy ndarrays
:type as_numpy: ```Optional[bool]```

:param data_loader_kwargs: pass this as arguments to data_loader function
:type data_loader_kwargs: ```Optional[dict]```

:return: Train and tests dataset splits.
:rtype: ```Union[Tuple[tf.data.Dataset, tf.data.Dataset], Tuple[np.ndarray, np.ndarray]]```
"""

docstring_no_default_str: str = _docstring_no_default_str.format(
    header_doc_str=docstring_header_str
)

docstring_no_default_no_nl_str: str = _docstring_no_default_str.format(
    header_doc_str=docstring_header_no_nl_str
)

docstring_numpydoc_only_doc_str: str = (
    """
{header_doc_str}
""".format(
        header_doc_str=docstring_header_str
    )
)

docstring_numpydoc_only_params_str: str = """
Parameters
----------
dataset_name : str
    name of dataset. Defaults to "mnist"
tfds_dir : str
    directory to look for models in. Defaults to "~/tensorflow_datasets"
K : Literal['np', 'tf']
    backend engine, e.g., `np` or `tf`. Defaults to "np"
as_numpy : Optional[bool]
    Convert to numpy ndarrays
data_loader_kwargs : Optional[dict]
    pass this as arguments to data_loader function
"""

docstring_numpydoc_only_returns_str: str = """
Returns
-------
Union[Tuple[tf.data.Dataset, tf.data.Dataset], Tuple[np.ndarray, np.ndarray]]
    Train and tests dataset splits. Defaults to (np.empty(0), np.empty(0))
"""

docstring_numpydoc_str: str = (
    """{docstring_header_str}
Parameters
----------
dataset_name : str
    name of dataset. Defaults to "mnist"
tfds_dir : str
    directory to look for models in. Defaults to "~/tensorflow_datasets"
K : Literal['np', 'tf']
    backend engine, e.g., `np` or `tf`. Defaults to "np"
as_numpy : Optional[bool]
    Convert to numpy ndarrays
data_loader_kwargs : Optional[dict]
    pass this as arguments to data_loader function

{docstring_numpydoc_only_returns_str}""".format(
        docstring_header_str=docstring_header_str,
        docstring_numpydoc_only_returns_str=docstring_numpydoc_only_returns_str.lstrip(
            "\n"
        ),
    )
)

docstring_only_return_type_str: str = """
Some comment

:param dataset_name: Example: foo

:return: Train and tests dataset splits.
:rtype: ```Union[Tuple[tf.data.Dataset, tf.data.Dataset], Tuple[np.ndarray, np.ndarray]]```
"""

_docstring_str: str = """
{header_doc_str}\n
:param dataset_name: name of dataset. Defaults to "mnist"
:type dataset_name: ```str```

:param tfds_dir: directory to look for models in. Defaults to "~/tensorflow_datasets"
:type tfds_dir: ```str```

:param K: backend engine, e.g., `np` or `tf`. Defaults to "np"
:type K: ```Literal['np', 'tf']```

:param as_numpy: Convert to numpy ndarrays
:type as_numpy: ```Optional[bool]```

:param data_loader_kwargs: pass this as arguments to data_loader function
:type data_loader_kwargs: ```Optional[dict]```

:return: Train and tests dataset splits. Defaults to (np.empty(0), np.empty(0))
:rtype: ```Union[Tuple[tf.data.Dataset, tf.data.Dataset], Tuple[np.ndarray, np.ndarray]]```
"""

docstring_str: str = _docstring_str.format(header_doc_str=docstring_header_str)

docstring_no_nl_str: str = _docstring_str.format(
    header_doc_str=docstring_header_no_nl_str
)

docstring_no_nl_no_none_str: str = docstring_no_nl_str.replace(" Defaults to None", "")

docstring_wrapped_str: str = docstring_str.replace(
    " np.ndarray]]```", "\n{tab}np.ndarray]]```".format(tab=tab)
)

docstring_no_type_str: str = (
    """
{header_doc_str}

:param dataset_name: name of dataset. Defaults to "mnist"

:param tfds_dir: directory to look for models in. Defaults to "~/tensorflow_datasets"

:param K: backend engine, e.g., `np` or `tf`. Defaults to "np"

:param as_numpy: Convert to numpy ndarrays

:param data_loader_kwargs: pass this as arguments to data_loader function

:return: Train and tests dataset splits. Defaults to (np.empty(0), np.empty(0))
""".format(
        header_doc_str=docstring_header_str
    )
)

docstring_no_type_no_default_tpl_str: str = """
{header_doc_str}
:param dataset_name: name of dataset.

:param tfds_dir: directory to look for models in.

:param K: backend engine, e.g., `np` or `tf`.

:param as_numpy: Convert to numpy ndarrays

:param data_loader_kwargs: pass this as arguments to data_loader function

:return: Train and tests dataset splits.
"""

docstring_no_type_no_default_str: str = docstring_no_type_no_default_tpl_str.format(
    header_doc_str=docstring_header_str
)

docstring_repr_str: str = (
    indent(
        "\n".join(
            (
                "",
                "Emit a string representation of the current instance",
                "",
                ":return: String representation of instance",
                ":rtype: ```str```",
                "",
            )
        ),
        tab * 2,
        identity,
    )
    + tab * 2
)

docstring_create_from_attr_str: str = (
    indent(
        "\n".join(
            (
                "",
                "Construct an instance from an object with identical columns (as attributes) as this `class`/`Table`",
                "",
                ":return: A new instance made from the input object's attributes",
                ":rtype: ```self```",
                "",
            )
        ),
        tab * 2,
        identity,
    )
    + tab * 2
)

# docstring_repr_google_str = emit.docstring(parse.docstring(docstring_repr_str), docstring_format="google")
# docstring_repr_google_str = (
#     "\nConstruct an instance from an object with identical columns (as attributes) as this `class`/`Table`\n\n\n\n\n"
#     "Returns:\n"
#     "  self:\n"
#     "   A new instance made from the input object's attributes\n"
# )

docstring_create_from_attr_google_str: str = (
    "\nEmit a string representation of the current instance\n\n\n\n\n"
    "Returns:\n"
    "  str:\n"
    "   String representation of instance\n"
)

docstring_repr_google_str: str = (
    "\nEmit a string representation of the current instance\n\n\n\n\n"
    "Returns:\n"
    "  str:\n"
    "   String representation of instance\n"
)

docstring_reduction_v2_str: str = (
    "Types of loss reduction."
    "\n"
    "\n"
    "  Contains the following values:"
    "\n"
    "\n"
    "  * `AUTO`: Indicates that the reduction option will be determined by the usage\n"
    "     context. For almost all cases this defaults to `SUM_OVER_BATCH_SIZE`. When\n"
    "     used with `tf.distribute.Strategy`, outside of built-in training loops such\n"
    "     as `tf.keras` `compile` and `fit`, we expect reduction value to be\n"
    "     `SUM` or `NONE`. Using `AUTO` in that case will raise an error.\n"
    "  * `NONE`: No **additional** reduction is applied to the output of the wrapped\n"
    "     loss function. When non-scalar losses are returned to Keras functions like\n"
    "     `fit`/`evaluate`, the unreduced vector loss is passed to the optimizer\n"
    "     but the reported loss will be a scalar value.\n"
    "\n"
    "     Caution: **Verify the shape of the outputs when using** `Reduction.NONE`.\n"
    "     The builtin loss functions wrapped by the loss classes reduce\n"
    "     one dimension (`axis=-1`, or `axis` if specified by loss function).\n"
    "     `Reduction.NONE` just means that no **additional** reduction is applied by\n"
    "     the class wrapper. For categorical losses with an example input shape of\n"
    "     `[batch, W, H, n_classes]` the `n_classes` dimension is reduced. For\n"
    "     pointwise losses your must include a dummy axis so that `[batch, W, H, 1]`\n"
    "     is reduced to `[batch, W, H]`. Without the dummy axis `[batch, W, H]`\n"
    "     will be incorrectly reduced to `[batch, W]`.\n"
    "\n"
    "  * `SUM`: Scalar sum of weighted losses.\n"
    "  * `SUM_OVER_BATCH_SIZE`: Scalar `SUM` divided by number of elements in losses.\n"
    "     This reduction type is not supported when used with\n"
    "     `tf.distribute.Strategy` outside of built-in training loops like `tf.keras`\n"
    "     `compile`/`fit`.\n"
    "\n"
    "     You can implement 'SUM_OVER_BATCH_SIZE' using global batch size like:\n"
    "     ```\n"
    "     with strategy.scope():\n"
    "       loss_obj = tf.keras.losses.CategoricalCrossentropy(\n"
    "           reduction=tf.keras.losses.Reduction.NONE)\n"
    "       ....\n"
    "       loss = tf.reduce_sum(loss_obj(labels, predictions)) *\n"
    "           (1. / global_batch_size)\n"
    "     ```\n"
    "\n"
    "  Please see the [custom training guide](\n"
    "  https://www.tensorflow.org/tutorials/distribute/custom_training) for more\n"
    "  details on this.\n"
    "  "
)

docstring_keras_rmsprop_class_str = (
    "Optimizer that implements the RMSprop algorithm.\n"
    "\n"
    "  The gist of RMSprop is to:\n"
    "\n  - Maintain a moving (discounted) average of the square of gradients\n"
    "  - Divide the gradient by the root of this average\n"
    "\n"
    "  This implementation of RMSprop uses plain momentum, not Nesterov momentum.\n"
    "\n"
    "  The centered version additionally maintains a moving average of the\n"
    "  gradients, and uses that average to estimate the variance.\n"
    "\n"
    "  Args:\n"
    "    learning_rate: A `Tensor`, floating point value, or a schedule that is a\n"
    "      `tf.keras.optimizers.schedules.LearningRateSchedule`, or a callable\n"
    "      that takes no arguments and returns the actual value to use. The\n"
    "      learning rate. Defaults to 0.001.\n"
    "    rho: Discounting factor for the history/coming gradient. Defaults to 0.9.\n"
    "    momentum: A scalar or a scalar `Tensor`. Defaults to 0.0.\n"
    "    epsilon: A small constant for numerical stability. This epsilon is\n"
    '      "epsilon hat" in the Kingma and Ba paper (in the formula just before\n'
    "      Section 2.1), not the epsilon in Algorithm 1 of the paper. Defaults to\n"
    "      1e-7.\n"
    "    centered: Boolean. If `True`, gradients are normalized by the estimated\n"
    "      variance of the gradient; if False, by the uncentered second moment.\n"
    "      Setting this to `True` may help with training, but is slightly more\n"
    "      expensive in terms of computation and memory. Defaults to `False`.\n"
    "    name: Optional name prefix for the operations created when applying\n"
    '      gradients. Defaults to `"RMSprop"`.\n'
    "    **kwargs: Keyword arguments. Allowed to be one of\n"
    '      `"clipnorm"` or `"clipvalue"`.\n'
    '      `"clipnorm"` (float) clips gradients by norm; `"clipvalue"` (float) clips\n'
    "      gradients by value.\n"
    "\n"
    "  Note that in the dense implementation of this algorithm, variables and their\n"
    "  corresponding accumulators (momentum, gradient moving average, square\n"
    "  gradient moving average) will be updated even if the gradient is zero\n"
    "  (i.e. accumulators will decay, momentum will be applied). The sparse\n"
    "  implementation (used when the gradient is an `IndexedSlices` object,\n"
    "  typically because of `tf.gather` or an embedding lookup in the forward pass)\n"
    "  will not update variable slices or their accumulators unless those slices\n"
    '  were used in the forward pass (nor is there an "eventual" correction to\n'
    "  account for these omitted updates). This leads to more efficient updates for\n"
    "  large embedding lookup tables (where most of the slices are not accessed in\n"
    "  a particular graph execution), but differs from the published algorithm.\n"
    "\n"
    "  Usage:\n"
    "\n"
    "  >>> opt = tf.keras.optimizers.RMSprop(learning_rate=0.1)\n"
    "  >>> var1 = tf.Variable(10.0)\n"
    "  >>> loss = lambda: (var1 ** 2) / 2.0    # d(loss) / d(var1) = var1\n"
    "  >>> step_count = opt.minimize(loss, [var1]).numpy()\n"
    "  >>> var1.numpy()\n"
    "  9.683772\n"
    "\n"
    "  Reference:\n"
    "    - [Hinton, 2012](\n"
    "      http://www.cs.toronto.edu/~tijmen/csc321/slides/lecture_slides_lec6.pdf)\n"
    "  "
)

docstring_keras_rmsprop_method_str = (
    "Construct a new RMSprop optimizer.\n"
    "\n"
    "    Args:\n"
    "      learning_rate: A `Tensor`, floating point value, or a schedule that is a\n"
    "        `tf.keras.optimizers.schedules.LearningRateSchedule`, or a callable\n"
    "        that takes no arguments and returns the actual value to use. The\n"
    "        learning rate. Defaults to 0.001.\n"
    "      rho: Discounting factor for the history/coming gradient. Defaults to 0.9.\n"
    "      momentum: A scalar or a scalar `Tensor`. Defaults to 0.0.\n"
    "      epsilon: A small constant for numerical stability. This epsilon is\n"
    '        "epsilon hat" in the Kingma and Ba paper (in the formula just before\n'
    "        Section 2.1), not the epsilon in Algorithm 1 of the paper. Defaults to\n"
    "        1e-7.\n"
    "      centered: Boolean. If `True`, gradients are normalized by the estimated\n"
    "        variance of the gradient; if False, by the uncentered second moment.\n"
    "        Setting this to `True` may help with training, but is slightly more\n"
    "        expensive in terms of computation and memory. Defaults to `False`.\n"
    "      name: Optional name prefix for the operations created when applying\n"
    '        gradients. Defaults to "RMSprop".\n'
    "      **kwargs: keyword arguments. Allowed to be {`clipnorm`, `clipvalue`, `lr`,\n"
    "        `decay`}. `clipnorm` is clip gradients by norm; `clipvalue` is clip\n"
    "        gradients by value, `decay` is included for backward compatibility to\n"
    "        allow time inverse decay of learning rate. `lr` is included for backward\n"
    "        compatibility, recommended to use `learning_rate` instead.\n"
    "\n"
    "    @compatibility(eager)\n"
    "    When eager execution is enabled, `learning_rate`, `decay`, `momentum`, and\n"
    "    `epsilon` can each be a callable that takes no arguments and returns the\n"
    "    actual value to use. This can be useful for changing these values across\n"
    "    different invocations of optimizer functions.\n"
    "    @end_compatibility\n"
    "    "
)

docstring_google_tf_ops_losses__safe_mean_str = (
    "Computes a safe mean of the losses.\n"
    "\n"
    "  Args:\n"
    "    losses: `Tensor` whose elements contain individual loss measurements.\n"
    "    num_present: The number of measurable elements in `losses`.\n"
    "\n"
    "  Returns:\n"
    "    A scalar representing the mean of `losses`. If `num_present` is zero,\n"
    "      then zero is returned.\n"
    "  "
)

docstring_sum_tuple = (
    ":type a: ```int```",
    "",
    ":type b: ```int```",
    "",
    ":rtype: ```int```",
    "",
)

__all__ = [
    "docstring_create_from_attr_google_str",
    "docstring_create_from_attr_str",
    "docstring_extra_colons_str",
    "docstring_google_keras_adadelta",
    "docstring_google_keras_adadelta_str",
    "docstring_google_keras_adam_str",
    "docstring_google_keras_lambda_callback_str",
    "docstring_google_keras_squared_hinge_no_args_doc_str",
    "docstring_google_keras_squared_hinge_no_args_doc_str",
    "docstring_google_keras_squared_hinge_str",
    "docstring_google_keras_tensorboard_return_str",
    "docstring_google_pytorch_lbfgs_str",
    "docstring_google_str",
    "docstring_google_tf_mean_squared_error_args_tuple",
    "docstring_google_tf_mean_squared_error_footer_tuple",
    "docstring_google_tf_mean_squared_error_header_tuple",
    "docstring_google_tf_mean_squared_error_str",
    "docstring_google_tf_ops_losses__safe_mean_str",
    "docstring_header_and_return_no_nl_str",
    "docstring_header_and_return_str",
    "docstring_header_and_return_two_nl_str",
    "docstring_header_no_nl_str",
    "docstring_header_str",
    "docstring_keras_rmsprop_class_str",
    "docstring_keras_rmsprop_method_str",
    "docstring_no_default_doc_str",
    "docstring_no_default_doc_wrapped_str",
    "docstring_no_default_no_nl_str",
    "docstring_no_default_str",
    "docstring_no_nl_no_none_str",
    "docstring_no_nl_str",
    "docstring_no_type_no_default_str",
    "docstring_no_type_no_default_tpl_str",
    "docstring_numpydoc_only_doc_str",
    "docstring_numpydoc_only_params_str",
    "docstring_numpydoc_only_returns_str",
    "docstring_numpydoc_str",
    "docstring_only_return_type_str",
    "docstring_reduction_v2_str",
    "docstring_repr_google_str",
    "docstring_repr_str",
    "docstring_str",
    "docstring_sum_tuple",
]  # type: list[str]
