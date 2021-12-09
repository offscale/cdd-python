"""
IR mocks
"""

from collections import OrderedDict
from copy import deepcopy

from cdd.ast_utils import NoneStr
from cdd.defaults_utils import remove_defaults_from_intermediate_repr
from cdd.pure_utils import deindent, paren_wrap_code
from cdd.tests.mocks.classes import (
    class_torch_nn_l1loss_docstring_str,
    class_torch_nn_one_cycle_lr_docstring_str,
    tensorboard_doc_str_no_args_str,
)
from cdd.tests.mocks.docstrings import (
    docstring_google_pytorch_lbfgs_str,
    docstring_google_tf_adadelta_str,
    docstring_google_tf_adam_str,
    docstring_google_tf_lambda_callback_str,
    docstring_google_tf_squared_hinge_no_args_doc_str,
    docstring_header_no_nl_str,
    docstring_header_str,
    docstring_keras_rmsprop_class_str,
    docstring_keras_rmsprop_method_str,
)
from cdd.tests.utils_for_tests import remove_args_from_docstring

class_google_tf_tensorboard_ir = {
    "doc": tensorboard_doc_str_no_args_str,
    "name": None,
    "params": OrderedDict(
        (
            (
                "log_dir",
                {
                    "default": "logs",
                    "doc": "the path of the directory where to save the log "
                    "files to be parsed by TensorBoard.",
                    "typ": "str",
                },
            ),
            (
                "histogram_freq",
                {
                    "default": 0,
                    "doc": "frequency (in epochs) at which to compute activation "
                    "and weight histograms for the layers of the model. "
                    "If set to 0, histograms won't be computed. "
                    "Validation data (or split) must be specified for "
                    "histogram visualizations.",
                    "typ": "int",
                },
            ),
            (
                "write_graph",
                {
                    "default": True,
                    "doc": "whether to visualize the graph in TensorBoard. The "
                    "log file can become quite large when write_graph is "
                    "set to True.",
                    "typ": "bool",
                },
            ),
            (
                "write_images",
                {
                    "default": False,
                    "doc": "whether to write model weights to visualize as image "
                    "in TensorBoard.",
                    "typ": "bool",
                },
            ),
            (
                "update_freq",
                {
                    "default": "epoch",
                    "doc": "`'batch'` or `'epoch'` or integer. When using "
                    "`'batch'`, writes the losses and metrics to "
                    "TensorBoard after each batch. The same applies for "
                    "`'epoch'`. If using an integer, let's say `1000`, "
                    "the callback will write the metrics and losses to "
                    "TensorBoard every 1000 batches. Note that writing "
                    "too frequently to TensorBoard can slow down your "
                    "training.",
                    "typ": "str",
                },
            ),
            (
                "profile_batch",
                {
                    "default": 2,
                    "doc": "Profile the batch(es) to sample compute "
                    "characteristics. profile_batch must be a "
                    "non-negative integer or a tuple of integers. A pair "
                    "of positive integers signify a range of batches to "
                    "profile. By default, it will profile the second "
                    "batch. Set profile_batch=0 to disable profiling.",
                    "typ": "int",
                },
            ),
            (
                "embeddings_freq",
                {
                    "default": 0,
                    "doc": "frequency (in epochs) at which embedding layers will "
                    "be visualized. If set to 0, embeddings won't be "
                    "visualized.",
                    "typ": "int",
                },
            ),
            (
                "embeddings_metadata",
                {
                    "default": NoneStr,
                    "doc": "a dictionary which maps layer name to a file name in "
                    "which metadata for this embedding layer is saved. "
                    "See the [details]( "
                    "https://www.tensorflow.org/how_tos/embedding_viz/#metadata_optional) "
                    "about metadata files format. In case if the same "
                    "metadata file is used for all embedding layers, "
                    "string can be passed.",
                },
            ),
        )
    ),
    "returns": None,
    "type": "static",
}

class_torch_nn_l1loss_ir = {
    "doc": deindent(remove_args_from_docstring(class_torch_nn_l1loss_docstring_str), 1),
    "name": None,
    "params": OrderedDict(
        (
            (
                "size_average",
                {
                    "default": True,
                    "doc": "Deprecated (see :attr:`reduction`). By default, the "
                    "losses are averaged over each loss element in the "
                    "batch. Note that for some losses, there are multiple "
                    "elements per sample. If the field "
                    ":attr:`size_average` is set to ``False``, the losses "
                    "are instead summed for each minibatch. Ignored when "
                    "reduce is ``False``.",
                    "typ": "Optional[bool]",
                },
            ),
            (
                "reduce",
                {
                    "default": True,
                    "doc": "Deprecated (see :attr:`reduction`). By default, the "
                    "losses are averaged or summed over observations for "
                    "each minibatch depending on :attr:`size_average`. "
                    "When :attr:`reduce` is ``False``, returns a loss per "
                    "batch element instead and ignores "
                    ":attr:`size_average`.",
                    "typ": "Optional[bool]",
                },
            ),
            (
                "reduction",
                {
                    "default": "mean",
                    "doc": "Specifies the reduction to apply to the output: "
                    "``'none'`` | ``'mean'`` | ``'sum'``. ``'none'``: no "
                    "reduction will be applied, ``'mean'``: the sum of "
                    "the output will be divided by the number of elements "
                    "in the output, ``'sum'``: the output will be summed. "
                    "Note: :attr:`size_average` and :attr:`reduce` are in "
                    "the process of being deprecated, and in the "
                    "meantime, specifying either of those two args will "
                    "override :attr:`reduction`.",
                    "typ": "Optional[string]",
                },
            ),
            ("__constants__", {"default": ["reduction"], "typ": "List"}),
        )
    ),
    "returns": OrderedDict((("return_type", {"typ": "None"}),)),
    "type": "static",
}

class_torch_nn_one_cycle_lr_ir = {
    "doc": remove_args_from_docstring(class_torch_nn_one_cycle_lr_docstring_str),
    "name": None,
    "params": OrderedDict(
        (
            ("optimizer", {"doc": "Wrapped optimizer.", "typ": "Optimizer"}),
            (
                "max_lr",
                {
                    "doc": "Upper learning rate boundaries in the cycle "
                    "for each parameter group.",
                    "typ": "Union[float, list]",
                },
            ),
            (
                "total_steps",
                {
                    "default": NoneStr,
                    "doc": "The total number of steps in the cycle. Note "
                    "that if a value is not provided here, then it "
                    "must be inferred by providing a value for "
                    "epochs and steps_per_epoch.",
                    "typ": "int",
                },
            ),
            (
                "epochs",
                {
                    "default": NoneStr,
                    "doc": "The number of epochs to train for. This is "
                    "used along with steps_per_epoch in order to "
                    "infer the total number of steps in the cycle "
                    "if a value for total_steps is not provided.",
                    "typ": "int",
                },
            ),
            (
                "steps_per_epoch",
                {
                    "default": NoneStr,
                    "doc": "The number of steps per epoch to train for. "
                    "This is used along with epochs in order to "
                    "infer the total number of steps in the cycle "
                    "if a value for total_steps is not provided.",
                    "typ": "int",
                },
            ),
            (
                "pct_start",
                {
                    "default": 0.3,
                    "doc": "The percentage of the cycle (in number of "
                    "steps) spent increasing the learning rate.",
                    "typ": "float",
                },
            ),
            (
                "anneal_strategy",
                {
                    "default": "cos",
                    "doc": 'Specifies the annealing strategy: "cos" for '
                    'cosine annealing, "linear" for linear '
                    "annealing.",
                    "typ": "Literal['cos', 'linear']",
                },
            ),
            (
                "cycle_momentum",
                {
                    "default": True,
                    "doc": "If ``True``, momentum is cycled inversely to "
                    "learning rate between 'base_momentum' and "
                    "'max_momentum'.",
                    "typ": "bool",
                },
            ),
            (
                "base_momentum",
                {
                    "default": 0.85,
                    "doc": "Lower momentum boundaries in the cycle for "
                    "each parameter group. Note that momentum is "
                    "cycled inversely to learning rate; at the peak "
                    "of a cycle, momentum is 'base_momentum' and "
                    "learning rate is 'max_lr'.",
                    "typ": "Union[float, list]",
                },
            ),
            (
                "max_momentum",
                {
                    "default": 0.95,
                    "doc": "Upper momentum boundaries in the cycle for "
                    "each parameter group. Functionally, it defines "
                    "the cycle amplitude (max_momentum - "
                    "base_momentum). Note that momentum is cycled "
                    "inversely to learning rate; at the start of a "
                    "cycle, momentum is 'max_momentum' and learning "
                    "rate is 'base_lr'",
                    "typ": "Union[float, list]",
                },
            ),
            (
                "div_factor",
                {
                    "default": 25.0,
                    "doc": "Determines the initial learning rate via "
                    "initial_lr = max_lr/div_factor",
                    "typ": "float",
                },
            ),
            (
                "final_div_factor",
                {
                    "default": 10000.0,
                    "doc": "Determines the minimum learning rate via "
                    "min_lr = initial_lr/final_div_factor",
                    "typ": "float",
                },
            ),
            (
                "last_epoch",
                {
                    "default": -1,
                    "doc": "The index of the last batch. This parameter is "
                    "used when resuming a training job. Since "
                    "`step()` should be invoked after each batch "
                    "instead of after each epoch, this number "
                    "represents the total number of *batches* "
                    "computed, not the total number of epochs "
                    "computed. When last_epoch=-1, the schedule is "
                    "started from the beginning.",
                    "typ": "int",
                },
            ),
            (
                "verbose",
                {
                    "default": False,
                    "doc": "If ``True``, prints a message to stdout for "
                    "each update.",
                    "typ": "bool",
                },
            ),
        )
    ),
    "returns": None,
    "type": "static",
}

docstring_google_tf_adadelta_ir = {
    "doc": remove_args_from_docstring(docstring_google_tf_adadelta_str),
    "name": None,
    "params": OrderedDict(
        (
            (
                "learning_rate",
                {
                    "doc": "A `Tensor`, floating point "
                    "value, or a schedule that is a "
                    "`tf.keras.optimizers.schedules.LearningRateSchedule`. "
                    "The learning rate. To match the "
                    "exact form in the original paper "
                    "use 1.0."
                },
            ),
            (
                "rho",
                {"doc": "A `Tensor` or a floating point " "value. The decay rate."},
            ),
            (
                "epsilon",
                {
                    "doc": "A `Tensor` or a floating point "
                    "value.  A constant epsilon used "
                    "to better conditioning the grad "
                    "update."
                },
            ),
            (
                "name",
                {
                    "default": "Adadelta",
                    "doc": "Optional name prefix for the "
                    "operations created when applying "
                    "gradients.  Defaults to "
                    '`"Adadelta"`.',
                    "typ": "Optional[str]",
                },
            ),
            (
                "kwargs",
                {
                    "default": NoneStr,
                    "doc": "Keyword arguments. Allowed to be "
                    'one of `"clipnorm"` or '
                    '`"clipvalue"`. `"clipnorm"` '
                    "(float) clips gradients by norm; "
                    '`"clipvalue"` (float) clips '
                    "gradients by value.",
                    "typ": "Optional[dict]",
                },
            ),
        )
    ),
    "returns": None,
    "type": "static",
}


docstring_google_tf_lambda_callback_ir = {
    "doc": remove_args_from_docstring(docstring_google_tf_lambda_callback_str),
    "name": None,
    "params": OrderedDict(
        (
            ("on_epoch_begin", {"doc": "called at the beginning of every " "epoch."}),
            ("on_epoch_end", {"doc": "called at the end of every epoch."}),
            ("on_batch_begin", {"doc": "called at the beginning of every " "batch."}),
            ("on_batch_end", {"doc": "called at the end of every batch."}),
            (
                "on_train_begin",
                {"doc": "called at the beginning of model " "training."},
            ),
            ("on_train_end", {"doc": "called at the end of model " "training."}),
        )
    ),
    "returns": None,
    "type": "static",
}

docstring_google_tf_adadelta_function_ir = {
    "doc": remove_args_from_docstring(docstring_google_tf_adadelta_str),
    "name": "Adadelta",
    "params": OrderedDict(
        (
            (
                "learning_rate",
                {
                    "default": 0.001,
                    "doc": "A `Tensor`, floating point "
                    "value, or a schedule that is a "
                    "`tf.keras.optimizers.schedules.LearningRateSchedule`. "
                    "The learning rate. To match the "
                    "exact form in the original paper "
                    "use 1.0.",
                    "typ": "float",
                },
            ),
            (
                "rho",
                {
                    "default": 0.95,
                    "doc": "A `Tensor` or a floating point " "value. The decay rate.",
                    "typ": "float",
                },
            ),
            (
                "epsilon",
                {
                    "default": 1e-07,
                    "doc": "A `Tensor` or a floating point "
                    "value.  A constant epsilon used "
                    "to better conditioning the grad "
                    "update.",
                    "typ": "float",
                },
            ),
            (
                "name",
                {
                    "default": "Adadelta",
                    "doc": "Optional name prefix for the "
                    "operations created when applying "
                    "gradients.",
                    "typ": "Optional[str]",
                },
            ),
            (
                "kwargs",
                {
                    "default": NoneStr,
                    "doc": "Keyword arguments. Allowed to be "
                    'one of `"clipnorm"` or '
                    '`"clipvalue"`. `"clipnorm"` '
                    "(float) clips gradients by norm; "
                    '`"clipvalue"` (float) clips '
                    "gradients by value.",
                    "typ": "Optional[dict]",
                },
            ),
            ("_HAS_AGGREGATE_GRAD", {"default": True, "typ": "bool"}),
        )
    ),
    "returns": None,
}

docstring_google_tf_adam_ir = {
    "doc": remove_args_from_docstring(docstring_google_tf_adam_str),
    "name": None,
    "params": OrderedDict(
        (
            (
                "learning_rate",
                {
                    "default": 0.001,
                    "doc": "A `Tensor`, floating point "
                    "value, or a schedule that is a "
                    "`tf.keras.optimizers.schedules.LearningRateSchedule`, "
                    "or a callable that takes no "
                    "arguments and returns the actual "
                    "value to use, The learning rate. "
                    "Defaults to 0.001.",
                    "typ": "float",
                },
            ),
            (
                "beta_1",
                {
                    "default": 0.9,
                    "doc": "A float value or a constant "
                    "float tensor, or a callable that "
                    "takes no arguments and returns "
                    "the actual value to use. The "
                    "exponential decay rate for the "
                    "1st moment estimates. Defaults "
                    "to 0.9.",
                    "typ": "float",
                },
            ),
            (
                "beta_2",
                {
                    "default": 0.999,
                    "doc": "A float value or a constant "
                    "float tensor, or a callable that "
                    "takes no arguments and returns "
                    "the actual value to use, The "
                    "exponential decay rate for the "
                    "2nd moment estimates. Defaults "
                    "to 0.999.",
                    "typ": "float",
                },
            ),
            (
                "epsilon",
                {
                    "default": 1e-07,
                    "doc": "A small constant for numerical "
                    "stability. This epsilon is "
                    '"epsilon hat" in the Kingma and '
                    "Ba paper (in the formula just "
                    "before Section 2.1), not the "
                    "epsilon in Algorithm 1 of the "
                    "paper. Defaults to 1e-7.",
                    "typ": "float",
                },
            ),
            (
                "amsgrad",
                {
                    "default": False,
                    "doc": "Boolean. Whether to apply "
                    "AMSGrad variant of this "
                    'algorithm from the paper "On the '
                    'Convergence of Adam and beyond". '
                    "Defaults to `False`.",
                    "typ": "bool",
                },
            ),
            (
                "name",
                {
                    "default": "Adam",
                    "doc": "Optional name for the operations "
                    "created when applying gradients. "
                    'Defaults to `"Adam"`.',
                    "typ": "Optional[str]",
                },
            ),
            (
                "kwargs",
                {
                    "default": NoneStr,
                    "doc": "Keyword arguments. Allowed to be "
                    'one of `"clipnorm"` or '
                    '`"clipvalue"`. `"clipnorm"` '
                    "(float) clips gradients by norm; "
                    '`"clipvalue"` (float) clips '
                    "gradients by value.",
                    "typ": "Optional[dict]",
                },
            ),
        )
    ),
    "returns": None,
    "type": "static",
}

docstring_google_tf_squared_hinge_ir = {
    "doc": docstring_google_tf_squared_hinge_no_args_doc_str,
    "name": None,
    "params": OrderedDict(
        (
            (
                "reduction",
                {
                    "default": "AUTO",
                    "doc": "(Optional) Type of "
                    "`tf.keras.losses.Reduction` to "
                    "apply to loss. Default value is "
                    "`AUTO`. `AUTO` indicates that "
                    "the reduction option will be "
                    "determined by the usage context. "
                    "For almost all cases this "
                    "defaults to "
                    "`SUM_OVER_BATCH_SIZE`. When used "
                    "with `tf.distribute.Strategy`, "
                    "outside of built-in training "
                    "loops such as `tf.keras` "
                    "`compile` and `fit`, using "
                    "`AUTO` or `SUM_OVER_BATCH_SIZE` "
                    "will raise an error. Please see "
                    "this custom training [tutorial]( "
                    "https://www.tensorflow.org/tutorials/distribute/custom_training) "
                    "for more details.",
                    "typ": "Optional[str]",
                },
            ),
            (
                "name",
                {
                    "default": "squared_hinge",
                    "doc": "Optional name for the op. " "Defaults to 'squared_hinge'.",
                    "typ": "Optional[str]",
                },
            ),
        )
    ),
    "returns": None,
    "type": "static",
}

docstring_google_pytorch_lbfgs_ir = {
    "doc": remove_args_from_docstring(docstring_google_pytorch_lbfgs_str).strip(),
    "name": None,
    "params": OrderedDict(
        (
            ("lr", {"default": 1.0, "doc": "learning rate", "typ": "float"}),
            (
                "max_iter",
                {
                    "default": 20,
                    "doc": "maximal number of iterations per optimization step",
                    "typ": "int",
                },
            ),
            (
                "max_eval",
                {
                    "default": "```max_iter * 1.25```",
                    "doc": "maximal number of function evaluations per "
                    "optimization step",
                },
            ),
            (
                "tolerance_grad",
                {
                    "default": 1e-05,
                    "doc": "termination tolerance on first order optimality",
                    "typ": "float",
                },
            ),
            (
                "tolerance_change",
                {
                    "default": 1e-09,
                    "doc": "termination tolerance on function value/parameter "
                    "changes",
                    "typ": "float",
                },
            ),
            (
                "history_size",
                {"default": 100, "doc": "update history size", "typ": "int"},
            ),
            (
                "line_search_fn",
                {
                    "default": NoneStr,
                    "doc": "either 'strong_wolfe' or None",
                    "typ": "str",
                },
            ),
        )
    ),
    "returns": None,
    "type": "static",
}

start_args_idx = docstring_keras_rmsprop_class_str.find("  Args:\n")
end_args_idx = docstring_keras_rmsprop_class_str.find("\n\n", start_args_idx) + 2
docstring_keras_rmsprop_class_ir = {
    "doc": docstring_keras_rmsprop_class_str[:start_args_idx]
    + docstring_keras_rmsprop_class_str[end_args_idx:],
    "name": None,
    "params": OrderedDict(
        (
            (
                "learning_rate",
                {
                    "default": 0.001,
                    "doc": "A `Tensor`, floating point value, or a schedule that "
                    "is a "
                    "`tf.keras.optimizers.schedules.LearningRateSchedule`, "
                    "or a callable that takes no arguments and returns "
                    "the actual value to use. The learning rate. Defaults "
                    "to 0.001.",
                    "typ": "float",
                },
            ),
            (
                "rho",
                {
                    "default": 0.9,
                    "doc": "Discounting factor for the history/coming gradient. "
                    "Defaults to 0.9.",
                    "typ": "float",
                },
            ),
            (
                "momentum",
                {
                    "default": 0.0,
                    "doc": "A scalar or a scalar `Tensor`. Defaults to 0.0.",
                    "typ": "float",
                },
            ),
            (
                "epsilon",
                {
                    "default": 1e-07,
                    "doc": "A small constant for numerical stability. This "
                    'epsilon is "epsilon hat" in the Kingma and Ba paper '
                    "(in the formula just before Section 2.1), not the "
                    "epsilon in Algorithm 1 of the paper. Defaults to "
                    "1e-7.",
                    "typ": "float",
                },
            ),
            (
                "centered",
                {
                    "default": False,
                    "doc": "Boolean. If `True`, gradients are normalized by the "
                    "estimated variance of the gradient; if False, by the "
                    "uncentered second moment. Setting this to `True` may "
                    "help with training, but is slightly more expensive "
                    "in terms of computation and memory. Defaults to "
                    "`False`.",
                    "typ": "bool",
                },
            ),
            (
                "name",
                {
                    "default": "RMSprop",
                    "doc": "Optional name prefix for the operations created when "
                    'applying gradients. Defaults to `"RMSprop"`.',
                    "typ": "Optional[str]",
                },
            ),
            (
                "kwargs",
                {
                    "default": NoneStr,
                    "doc": 'Keyword arguments. Allowed to be one of `"clipnorm"` '
                    'or `"clipvalue"`. `"clipnorm"` (float) clips '
                    'gradients by norm; `"clipvalue"` (float) clips '
                    "gradients by value.",
                    "typ": "Optional[dict]",
                },
            ),
        )
    ),
    "returns": None,
    "type": "static",
}

del start_args_idx, end_args_idx

start_args_idx = docstring_keras_rmsprop_method_str.find("  Args:\n")
end_args_idx = docstring_keras_rmsprop_method_str.find("\n\n", start_args_idx + 1) + 4

docstring_keras_rmsprop_method_ir = {
    "doc": docstring_keras_rmsprop_method_str[:start_args_idx]
    + docstring_keras_rmsprop_method_str[end_args_idx:],
    "name": None,
    "params": OrderedDict(
        (
            (
                "learning_rate",
                {
                    "default": 0.001,
                    "doc": "A `Tensor`, floating point value, or a schedule that "
                    "is a "
                    "`tf.keras.optimizers.schedules.LearningRateSchedule`, "
                    "or a callable that takes no arguments and returns "
                    "the actual value to use. The learning rate. Defaults "
                    "to 0.001.",
                    "typ": "float",
                },
            ),
            (
                "rho",
                {
                    "default": 0.9,
                    "doc": "Discounting factor for the history/coming gradient. "
                    "Defaults to 0.9.",
                    "typ": "float",
                },
            ),
            (
                "momentum",
                {
                    "default": 0.0,
                    "doc": "A scalar or a scalar `Tensor`. Defaults to 0.0.",
                    "typ": "float",
                },
            ),
            (
                "epsilon",
                {
                    "default": 1e-07,
                    "doc": "A small constant for numerical stability. This "
                    'epsilon is "epsilon hat" in the Kingma and Ba paper '
                    "(in the formula just before Section 2.1), not the "
                    "epsilon in Algorithm 1 of the paper. Defaults to "
                    "1e-7.",
                    "typ": "float",
                },
            ),
            (
                "centered",
                {
                    "default": False,
                    "doc": "Boolean. If `True`, gradients are normalized by the "
                    "estimated variance of the gradient; if False, by the "
                    "uncentered second moment. Setting this to `True` may "
                    "help with training, but is slightly more expensive "
                    "in terms of computation and memory. Defaults to "
                    "`False`.",
                    "typ": "bool",
                },
            ),
            (
                "name",
                {
                    "default": "RMSprop",
                    "doc": "Optional name prefix for the operations created when "
                    'applying gradients. Defaults to "RMSprop".',
                    "typ": "Optional[str]",
                },
            ),
            (
                "kwargs",
                {
                    "default": NoneStr,
                    "doc": "keyword arguments. Allowed to be {`clipnorm`, "
                    "`clipvalue`, `lr`, `decay`}. `clipnorm` is clip "
                    "gradients by norm; `clipvalue` is clip gradients by "
                    "value, `decay` is included for backward "
                    "compatibility to allow time inverse decay of "
                    "learning rate. `lr` is included for backward "
                    "compatibility, recommended to use `learning_rate` "
                    "instead.",
                    "typ": "Optional[dict]",
                },
            ),
        )
    ),
    "returns": None,
    "type": "static",
}

del start_args_idx, end_args_idx

function_adder_ir = {
    "doc": "",
    "name": "add_6_5",
    "params": OrderedDict(
        (
            ("a", {"default": 6, "doc": "first param", "typ": "int"}),
            ("b", {"default": 5, "doc": "second param", "typ": "int"}),
        )
    ),
    "returns": OrderedDict(
        (
            (
                "return_type",
                {
                    "default": "```operator.add(a, b)```",
                    "doc": "Aggregated summation " "of `a` and `b`.",
                },
            ),
        )
    ),
    "type": "static",
}

function_google_tf_ops_losses__safe_mean_ir = {
    "doc": "Computes a safe mean of the losses.",
    "name": "_safe_mean",
    "params": OrderedDict(
        (
            (
                "losses",
                {
                    "doc": "`Tensor` whose elements contain individual loss "
                    "measurements."
                },
            ),
            (
                "num_present",
                {"doc": "The number of measurable elements in `losses`."},
            ),
        )
    ),
    "returns": OrderedDict(
        (
            (
                "return_type",
                {
                    "default": "```math_ops.div_no_nan(total_loss, num_present, "
                    "name='value')```",
                    "doc": "A scalar representing the mean of `losses`. If `num_present` is "
                    "zero, then zero is returned.",
                },
            ),
        )
    ),
    "type": "static",
}

method_complex_args_variety_ir = {
    "doc": "Call cliff",
    "name": "call_cliff",
    "params": OrderedDict(
        (
            ("dataset_name", {"doc": "name of dataset."}),
            ("as_numpy", {"doc": "Convert to numpy ndarrays."}),
            (
                "K",
                {
                    "doc": "backend engine, e.g., `np` or " "`tf`.",
                    "typ": "Literal['np', 'tf']",
                },
            ),
            (
                "tfds_dir",
                {
                    "default": "~/tensorflow_datasets",
                    "doc": "directory to look for models in.",
                    "typ": "str",
                },
            ),
            (
                "writer",
                {
                    "default": "```{}```".format(paren_wrap_code("stdout")),
                    "doc": "IO object to write out to",
                },
            ),
            (
                "kwargs",
                {
                    "doc": "additional keyword arguments",
                    "typ": "Optional[dict]",
                    "default": NoneStr,
                },
            ),
        )
    ),
    "returns": OrderedDict(
        (
            (
                "return_type",
                {"default": "K", "doc": "backend engine", "typ": "Literal['np', 'tf']"},
            ),
        )
    ),
    "type": "self",
}

intermediate_repr_extra_colons = {
    "name": None,
    "params": OrderedDict(
        (
            (
                "dataset_name",
                {"doc": "Example: foo", "typ": "str"},
            ),
        )
    ),
    "returns": None,
    "doc": "Some comment",
    "type": "static",
}

intermediate_repr_no_default_doc = {
    "name": None,
    "type": "static",
    "doc": docstring_header_str.rstrip("\n"),
    "params": OrderedDict(
        (
            (
                "dataset_name",
                {"default": "mnist", "doc": "name of dataset.", "typ": "str"},
            ),
            (
                "tfds_dir",
                {
                    "default": "~/tensorflow_datasets",
                    "doc": "directory to look for models in.",
                    "typ": "str",
                },
            ),
            (
                "K",
                {
                    "default": "np",
                    "doc": "backend engine, e.g., `np` or " "`tf`.",
                    "typ": "Literal['np', 'tf']",
                },
            ),
            (
                "as_numpy",
                {
                    "default": NoneStr,
                    "doc": "Convert to numpy ndarrays.",
                    "typ": "Optional[bool]",
                },
            ),
            (
                "data_loader_kwargs",
                {
                    "default": NoneStr,
                    "doc": "pass this as arguments to " "data_loader function",
                    "typ": "Optional[dict]",
                },
            ),
        )
    ),
    "returns": OrderedDict(
        (
            (
                "return_type",
                {
                    "default": "(np.empty(0), np.empty(0))",
                    "doc": "Train and tests dataset splits.",
                    "typ": "Union[Tuple[tf.data.Dataset, "
                    "tf.data.Dataset], "
                    "Tuple[np.ndarray, "
                    "np.ndarray]]",
                },
            ),
        )
    ),
}

intermediate_repr_no_default_sql_doc = deepcopy(intermediate_repr_no_default_doc)
intermediate_repr_no_default_sql_doc["params"]["dataset_name"][
    "doc"
] = "[PK] {}".format(
    intermediate_repr_no_default_sql_doc["params"]["dataset_name"]["doc"]
)

intermediate_repr_only_return_type = {
    "name": None,
    "type": "static",
    "doc": "Some comment",
    "params": OrderedDict(
        (("dataset_name", {"doc": "Example: foo"}),),
    ),
    "returns": OrderedDict(
        (
            (
                "return_type",
                {
                    "doc": "Train and tests dataset splits.",
                    "typ": "Union[Tuple[tf.data.Dataset, tf.data.Dataset], "
                    "Tuple[np.ndarray, np.ndarray]]",
                },
            ),
        ),
    ),
}

intermediate_repr = {
    "name": None,
    "type": "static",
    "doc": docstring_header_no_nl_str,
    "params": OrderedDict(
        (
            (
                "dataset_name",
                {
                    "default": "mnist",
                    "doc": "name of dataset. Defaults to " '"mnist"',
                    "typ": "str",
                },
            ),
            (
                "tfds_dir",
                {
                    "default": "~/tensorflow_datasets",
                    "doc": "directory to look for models in. "
                    "Defaults to "
                    '"~/tensorflow_datasets"',
                    "typ": "str",
                },
            ),
            (
                "K",
                {
                    "default": "np",
                    "doc": "backend engine, e.g., `np` or " '`tf`. Defaults to "np"',
                    "typ": "Literal['np', 'tf']",
                },
            ),
            (
                "as_numpy",
                {
                    "default": NoneStr,
                    "doc": "Convert to numpy ndarrays.",
                    "typ": "Optional[bool]",
                },
            ),
            (
                "data_loader_kwargs",
                {
                    "default": NoneStr,
                    "doc": "pass this as arguments to " "data_loader function",
                    "typ": "Optional[dict]",
                },
            ),
        )
    ),
    "returns": OrderedDict(
        (
            (
                "return_type",
                {
                    "default": "(np.empty(0), np.empty(0))",
                    "doc": "Train and tests dataset splits. "
                    "Defaults to (np.empty(0), "
                    "np.empty(0))",
                    "typ": "Union[Tuple[tf.data.Dataset, "
                    "tf.data.Dataset], "
                    "Tuple[np.ndarray, "
                    "np.ndarray]]",
                },
            ),
        )
    ),
}

intermediate_repr_no_default_doc_or_prop = remove_defaults_from_intermediate_repr(
    deepcopy(intermediate_repr), emit_default_prop=False
)

__all__ = [
    "class_google_tf_tensorboard_ir",
    "class_torch_nn_l1loss_ir",
    "class_torch_nn_one_cycle_lr_ir",
    "docstring_google_pytorch_lbfgs_ir",
    "docstring_google_tf_adadelta_function_ir",
    "docstring_google_tf_adadelta_ir",
    "docstring_google_tf_adam_ir",
    "docstring_google_tf_lambda_callback_ir",
    "docstring_google_tf_squared_hinge_ir",
    "function_adder_ir",
    "function_google_tf_ops_losses__safe_mean_ir",
    "intermediate_repr",
    "intermediate_repr_extra_colons",
    "intermediate_repr_no_default_doc",
    "intermediate_repr_no_default_doc_or_prop",
    "intermediate_repr_no_default_sql_doc",
    "intermediate_repr_only_return_type",
    "method_complex_args_variety_ir",
]
