"""
IR mocks
"""

from collections import OrderedDict
from copy import deepcopy

from cdd.shared.ast_utils import NoneStr
from cdd.shared.defaults_utils import remove_defaults_from_intermediate_repr
from cdd.shared.pure_utils import deindent, paren_wrap_code, tab
from cdd.shared.types import IntermediateRepr
from cdd.tests.mocks.classes import (
    class_torch_nn_l1loss_docstring_str,
    class_torch_nn_one_cycle_lr_docstring_str,
    tensorboard_doc_str_no_args_str,
)
from cdd.tests.mocks.docstrings import (
    docstring_google_keras_adadelta_str,
    docstring_google_keras_adam_str,
    docstring_google_keras_lambda_callback_str,
    docstring_google_keras_squared_hinge_no_args_doc_str,
    docstring_google_pytorch_lbfgs_str,
    docstring_header_no_nl_str,
    docstring_keras_rmsprop_class_str,
    docstring_keras_rmsprop_method_str,
)
from cdd.tests.utils_for_tests import remove_args_from_docstring

class_google_keras_tensorboard_ir: IntermediateRepr = {
    "doc": tensorboard_doc_str_no_args_str,
    "name": "TensorBoard",
    "params": OrderedDict(
        (
            (
                "log_dir",
                {
                    "default": "logs",
                    "doc": (
                        "the path of the directory where to save the log "
                        "files to be parsed by TensorBoard. e.g., `log_dir = "
                        "os.path.join(working_dir, 'logs')`. This directory "
                        "should not be reused by any other callbacks."
                    ),
                    "typ": "str",
                },
            ),
            (
                "histogram_freq",
                {
                    "default": 0,
                    "doc": (
                        "frequency (in epochs) at which to compute weight "
                        "histograms for the layers of the model. If set to 0, "
                        "histograms won't be computed. Validation data (or "
                        "split) must be specified for histogram "
                        "visualizations."
                    ),
                    "typ": "int",
                },
            ),
            (
                "write_graph",
                {
                    "default": True,
                    "doc": (
                        "(Not supported at this time) Whether to visualize "
                        "the graph in TensorBoard. Note that the log file can "
                        "become quite large when `write_graph` is set to "
                        "`True`."
                    ),
                    "typ": "bool",
                },
            ),
            (
                "write_images",
                {
                    "default": False,
                    "doc": (
                        "whether to write model weights to visualize as image "
                        "in TensorBoard."
                    ),
                    "typ": "bool",
                },
            ),
            (
                "write_steps_per_second",
                {
                    "default": False,
                    "doc": (
                        "whether to log the training steps per second into "
                        "TensorBoard. This supports both epoch and batch "
                        "frequency logging."
                    ),
                    "typ": "bool",
                },
            ),
            (
                "update_freq",
                {
                    "default": "epoch",
                    "doc": (
                        '`"batch"` or `"epoch"` or integer. When using '
                        '`"epoch"`, writes the losses and metrics to '
                        "TensorBoard after every epoch. If using an integer, "
                        "let's say `1000`, all metrics and losses (including "
                        "custom ones added by `Model.compile`) will be logged "
                        'to TensorBoard every 1000 batches. `"batch"` is a '
                        "synonym for 1, meaning that they will be written "
                        "every batch. Note however that writing too "
                        "frequently to TensorBoard can slow down your "
                        "training, especially when used with distribution "
                        "strategies as it will incur additional "
                        "synchronization overhead. Batch-level summary "
                        "writing is also available via `train_step` override. "
                        "Please see [TensorBoard Scalars tutorial]( "
                        "https://www.tensorflow.org/tensorboard/scalars_and_keras#batch-level_logging)  "
                        "# noqa: E501 for more details."
                    ),
                    "typ": 'Union[Literal["batch", "epoch"], int]',
                },
            ),
            (
                "profile_batch",
                {
                    "default": 0,
                    "doc": (
                        "(Not supported at this time) Profile the batch(es) "
                        "to sample compute characteristics. profile_batch "
                        "must be a non-negative integer or a tuple of "
                        "integers. A pair of positive integers signify a "
                        "range of batches to profile. By default, profiling "
                        "is disabled."
                    ),
                    "typ": "int",
                },
            ),
            (
                "embeddings_freq",
                {
                    "default": 0,
                    "doc": (
                        "frequency (in epochs) at which embedding layers will "
                        "be visualized. If set to 0, embeddings won't be "
                        "visualized."
                    ),
                    "typ": "int",
                },
            ),
            (
                "embeddings_metadata",
                {
                    "default": NoneStr,
                    "doc": (
                        "Dictionary which maps embedding layer names to the "
                        "filename of a file in which to save metadata for the "
                        "embedding layer. In case the same metadata file is "
                        "to be used for all embedding layers, a single "
                        "filename can be passed."
                    ),
                    "typ": "str",
                },
            ),
        )
    ),
    "returns": None,
    "type": "static",
}

# #################
# # PyTorch 2.1.2 #
# #################
# https://github.com/pytorch/pytorch/blob/6c7013a3/torch/nn/modules/loss.py
class_torch_nn_l1loss_ir: IntermediateRepr = {
    "doc": remove_args_from_docstring(class_torch_nn_l1loss_docstring_str),
    "name": "L1Loss",
    "params": OrderedDict(
        (
            (
                "size_average",
                {
                    "default": True,
                    "doc": (
                        "Deprecated (see :attr:`reduction`). By default, the "
                        "losses are averaged over each loss element in the "
                        "batch. Note that for some losses, there are multiple "
                        "elements per sample. If the field "
                        ":attr:`size_average` is set to ``False``, the losses "
                        "are instead summed for each minibatch. Ignored when "
                        ":attr:`reduce` is ``False``."
                    ),
                    "typ": "Optional[bool]",
                },
            ),
            (
                "reduce",
                {
                    "default": True,
                    "doc": (
                        "Deprecated (see :attr:`reduction`). By default, the "
                        "losses are averaged or summed over observations for "
                        "each minibatch depending on :attr:`size_average`. "
                        "When :attr:`reduce` is ``False``, returns a loss per "
                        "batch element instead and ignores "
                        ":attr:`size_average`."
                    ),
                    "typ": "Optional[bool]",
                },
            ),
            (
                "reduction",
                {
                    "default": "mean",
                    "doc": (
                        "Specifies the reduction to apply to the output: "
                        "``'none'`` | ``'mean'`` | ``'sum'``. ``'none'``: no "
                        "reduction will be applied, ``'mean'``: the sum of "
                        "the output will be divided by the number of elements "
                        "in the output, ``'sum'``: the output will be summed. "
                        "Note: :attr:`size_average` and :attr:`reduce` are in "
                        "the process of being deprecated, and in the "
                        "meantime, specifying either of those two args will "
                        "override :attr:`reduction`."
                    ),
                    "typ": "int",
                },
            ),
            ("__constants__", {"default": ["reduction"], "typ": "List"}),
        )
    ),
    "returns": OrderedDict((("return_type", {"typ": "None"}),)),
    "type": "static",
}

class_torch_nn_one_cycle_lr_ir: IntermediateRepr = {
    "doc": remove_args_from_docstring(class_torch_nn_one_cycle_lr_docstring_str),
    "name": "OneCycleLR",
    "params": OrderedDict(
        (
            ("optimizer", {"doc": "Wrapped optimizer.", "typ": "Optimizer"}),
            (
                "max_lr",
                {
                    "doc": (
                        "Upper learning rate boundaries in the cycle for each "
                        "parameter group."
                    ),
                    "typ": "Union[float, list]",
                },
            ),
            (
                "total_steps",
                {
                    "default": NoneStr,
                    "doc": (
                        "The total number of steps in the cycle. Note that if "
                        "a value is not provided here, then it must be "
                        "inferred by providing a value for epochs and "
                        "steps_per_epoch."
                    ),
                    "typ": "Optional[int]",
                },
            ),
            (
                "epochs",
                {
                    "default": NoneStr,
                    "doc": (
                        "The number of epochs to train for. This is used "
                        "along with steps_per_epoch in order to infer the "
                        "total number of steps in the cycle if a value for "
                        "total_steps is not provided."
                    ),
                    "typ": "Optional[int]",
                },
            ),
            (
                "steps_per_epoch",
                {
                    "default": NoneStr,
                    "doc": (
                        "The number of steps per epoch to train for. This is "
                        "used along with epochs in order to infer the total "
                        "number of steps in the cycle if a value for "
                        "total_steps is not provided."
                    ),
                    "typ": "Optional[int]",
                },
            ),
            (
                "pct_start",
                {
                    "default": 0.3,
                    "doc": (
                        "The percentage of the cycle (in number of steps) "
                        "spent increasing the learning rate."
                    ),
                    "typ": "int",
                },
            ),
            (
                "anneal_strategy",
                {
                    "default": "cos",
                    "doc": (
                        'Specifies the annealing strategy: "cos" for cosine '
                        'annealing, "linear" for linear annealing.'
                    ),
                    "typ": "Literal['cos', 'linear']",
                },
            ),
            (
                "cycle_momentum",
                {
                    "default": True,
                    "doc": (
                        "If ``True``, momentum is cycled inversely to "
                        "learning rate between 'base_momentum' and "
                        "'max_momentum'."
                    ),
                    "typ": "bool",
                },
            ),
            (
                "base_momentum",
                {
                    "default": 0.85,
                    "doc": (
                        "Lower momentum boundaries in the cycle for each "
                        "parameter group. Note that momentum is cycled "
                        "inversely to learning rate; at the peak of a cycle, "
                        "momentum is 'base_momentum' and learning rate is "
                        "'max_lr'."
                    ),
                    "typ": "Union[float, list]",
                },
            ),
            (
                "max_momentum",
                {
                    "default": 0.95,
                    "doc": (
                        "Upper momentum boundaries in the cycle for each "
                        "parameter group. Functionally, it defines the cycle "
                        "amplitude (max_momentum - base_momentum). Note that "
                        "momentum is cycled inversely to learning rate; at "
                        "the start of a cycle, momentum is 'max_momentum' and "
                        "learning rate is 'base_lr'"
                    ),
                    "typ": "Union[float, list]",
                },
            ),
            (
                "div_factor",
                {
                    "default": 25.0,
                    "doc": (
                        "Determines the initial learning rate via initial_lr "
                        "= max_lr/div_factor"
                    ),
                    "typ": "float",
                },
            ),
            (
                "final_div_factor",
                {
                    "default": 10000.0,
                    "doc": (
                        "Determines the minimum learning rate via min_lr = "
                        "initial_lr/final_div_factor"
                    ),
                    "typ": "float",
                },
            ),
            (
                "last_epoch",
                {
                    "default": -1,
                    "doc": (
                        "The index of the last batch. This parameter is used "
                        "when resuming a training job. Since `step()` should "
                        "be invoked after each batch instead of after each "
                        "epoch, this number represents the total number of "
                        "*batches* computed, not the total number of epochs "
                        "computed. When last_epoch=-1, the schedule is "
                        "started from the beginning."
                    ),
                    "typ": "int",
                },
            ),
            (
                "verbose",
                {
                    "default": False,
                    "doc": (
                        "If ``True``, prints a message to stdout for each " "update."
                    ),
                    "typ": "bool",
                },
            ),
        )
    ),
    "returns": None,
    "type": "static",
}

# #####################
# # TensorFlow 2.15.0 #
# #####################
# https://github.com/tensorflow/tensorflow/blob/5a56eb1/tensorflow/python/keras/optimizer_v2/adadelta.py#L27-L62
docstring_google_keras_adadelta_ir: IntermediateRepr = {
    "doc": remove_args_from_docstring(docstring_google_keras_adadelta_str) + tab,
    "name": None,
    "params": OrderedDict(
        (
            (
                "learning_rate",
                {
                    "default": 0.001,
                    "doc": (
                        "A float, a "
                        "`keras.optimizers.schedules.LearningRateSchedule` "
                        "instance, or a callable that takes no arguments and "
                        "returns the actual value to use. The learning rate. "
                        "Defaults to `0.001`. Note that `Adadelta` tends to "
                        "benefit from higher initial learning rate values "
                        "compared to other optimizers. To match the exact "
                        "form in the original paper, use 1.0."
                    ),
                    "typ": "float",
                },
            ),
            (
                "rho",
                {
                    "default": 0.95,
                    "doc": (
                        "A floating point value. The decay rate. Defaults to " "`0.95`."
                    ),
                    "typ": "float",
                },
            ),
            (
                "epsilon",
                {
                    "default": NoneStr,
                    "doc": (
                        "Small floating point value for maintaining numerical "
                        "stability."
                    ),
                    "typ": "Optional[float]",
                },
            ),
            (
                "name",
                {
                    "default": NoneStr,
                    "doc": (
                        "String. The name to use for momentum accumulator "
                        "weights created by the optimizer."
                    ),
                    "typ": "Optional[str]",
                },
            ),
            (
                "weight_decay",
                {
                    "default": NoneStr,
                    "doc": "Float. If set, weight decay is applied.",
                    "typ": "Optional[float]",
                },
            ),
            (
                "clipnorm",
                {
                    "default": NoneStr,
                    "doc": (
                        "Float. If set, the gradient of each weight is "
                        "individually clipped so that its norm is no higher "
                        "than this value."
                    ),
                    "typ": "Optional[float]",
                },
            ),
            (
                "clipvalue",
                {
                    "default": NoneStr,
                    "doc": (
                        "Float. If set, the gradient of each weight is "
                        "clipped to be no higher than this value."
                    ),
                    "typ": "Optional[float]",
                },
            ),
            (
                "global_clipnorm",
                {
                    "default": NoneStr,
                    "doc": (
                        "Float. If set, the gradient of all weights is "
                        "clipped so that their global norm is no higher than "
                        "this value."
                    ),
                    "typ": "Optional[float]",
                },
            ),
            (
                "use_ema",
                {
                    "default": False,
                    "doc": (
                        "Boolean, defaults to False. If True, exponential "
                        "moving average (EMA) is applied. EMA consists of "
                        "computing an exponential moving average of the "
                        "weights of the model (as the weight values change "
                        "after each training batch), and periodically "
                        "overwriting the weights with their moving average."
                    ),
                    "typ": "bool",
                },
            ),
            (
                "ema_momentum",
                {
                    "default": 0.99,
                    "doc": (
                        "Float, defaults to 0.99. Only used if "
                        "`use_ema=True`. This is the momentum to use when "
                        "computing the EMA of the model's weights: "
                        "`new_average = ema_momentum * old_average + (1 - "
                        "ema_momentum) * current_variable_value`."
                    ),
                    "typ": "float",
                },
            ),
            (
                "ema_overwrite_frequency",
                {
                    "default": NoneStr,
                    "doc": (
                        "Int or None, defaults to None. Only used if "
                        "`use_ema=True`. Every `ema_overwrite_frequency` "
                        "steps of iterations, we overwrite the model variable "
                        "by its moving average. If None, the optimizer does "
                        "not overwrite model variables in the middle of "
                        "training, and you need to explicitly overwrite the "
                        "variables at the end of training by calling "
                        "`optimizer.finalize_variable_values()` (which "
                        "updates the model variables in-place). When using "
                        "the built-in `fit()` training loop, this happens "
                        "automatically after the last epoch, and you don't "
                        "need to do anything."
                    ),
                    "typ": "Optional[int]",
                },
            ),
            (
                "loss_scale_factor",
                {
                    "default": NoneStr,
                    "doc": (
                        "Float or `None`. If a float, the scale factor will "
                        "be multiplied the loss before computing gradients, "
                        "and the inverse of the scale factor will be "
                        "multiplied by the gradients before updating "
                        "variables. Useful for preventing underflow during "
                        "mixed precision training. Alternately, "
                        "`keras.optimizers.LossScaleOptimizer` will "
                        "automatically set a loss scale factor."
                    ),
                    "typ": "Optional[float]",
                },
            ),
        )
    ),
    "returns": None,
    "type": "static",
}
docstring_google_keras_adadelta_merged_init_ir: IntermediateRepr = {
    "doc": remove_args_from_docstring(docstring_google_keras_adadelta_str),
    "name": None,
    "params": OrderedDict(
        (
            (
                "learning_rate",
                {
                    "default": 0.001,
                    "doc": (
                        "Initial value for the learning rate: either a "
                        "floating point value, or a "
                        "`tf.keras.optimizers.schedules.LearningRateSchedule` "
                        "instance. Defaults to 0.001. Note that `Adadelta` "
                        "tends to benefit from higher initial learning rate "
                        "values compared to other optimizers. To match the "
                        "exact form in the original paper, use 1.0."
                    ),
                    "typ": "float",
                },
            ),
            (
                "rho",
                {
                    "default": NoneStr,
                    "doc": "A `Tensor` or a floating point value. The decay rate.",
                },
            ),
            (
                "epsilon",
                {
                    "default": NoneStr,
                    "doc": (
                        "Small floating point value used to maintain "
                        "numerical stability."
                    ),
                },
            ),
            (
                "name",
                {
                    "default": "Adadelta",
                    "doc": (
                        "Optional name prefix for the operations created when "
                        'applying gradients.  Defaults to `"Adadelta"`.'
                    ),
                    "typ": "Optional[str]",
                },
            ),
            (
                "kwargs",
                {
                    "default": NoneStr,
                    "doc": (
                        'Keyword arguments. Allowed to be one of `"clipnorm"` '
                        'or `"clipvalue"`. `"clipnorm"` (float) clips '
                        "gradients by norm and represents the maximum norm of "
                        'each parameter; `"clipvalue"` (float) clips gradient '
                        "by value and represents the maximum absolute value "
                        "of each parameter."
                    ),
                    "typ": "Optional[dict]",
                },
            ),
        )
    ),
    "returns": None,
    "type": "static",
}

# #####################
# # TensorFlow 2.15.0 #
# #####################
# https://github.com/tensorflow/tensorflow/blob/5a56eb1/tensorflow/python/keras/callbacks.py#L2792-L2840
docstring_google_keras_lambda_callback_ir: IntermediateRepr = {
    "doc": remove_args_from_docstring(docstring_google_keras_lambda_callback_str) + tab,
    "name": None,
    "params": OrderedDict(
        (
            (
                "on_epoch_begin",
                {
                    "doc": "called at the beginning of every epoch.",
                    "typ": "collections.abc.Callable",
                },
            ),
            (
                "on_epoch_end",
                {
                    "doc": "called at the end of every epoch.",
                    "typ": "collections.abc.Callable",
                },
            ),
            (
                "on_train_begin",
                {
                    "doc": "called at the beginning of model training.",
                    "typ": "collections.abc.Callable",
                },
            ),
            (
                "on_train_end",
                {
                    "doc": "called at the end of model training.",
                    "typ": "collections.abc.Callable",
                },
            ),
            (
                "on_train_batch_begin",
                {
                    "doc": "called at the beginning of every train batch.",
                    "typ": "collections.abc.Callable",
                },
            ),
            (
                "on_train_batch_end",
                {
                    "doc": "called at the end of every train batch.",
                    "typ": "collections.abc.Callable",
                },
            ),
            (
                "kwargs",
                {
                    "doc": (
                        "Any function in `Callback` that you want to override "
                        "by passing `function_name=function`. For example, "
                        "`LambdaCallback(.., on_train_end=train_end_fn)`. The "
                        "custom function needs to have same arguments as the "
                        "ones defined in `Callback`."
                    ),
                    "typ": "Optional[dict]",
                },
            ),
        )
    ),
    "returns": None,
    "type": "static",
}

# #####################
# # TensorFlow 2.15.0 #
# #####################
# https://github.com/tensorflow/tensorflow/blob/5a56eb1/tensorflow/python/keras/optimizer_v2/adadelta.py#L27-L62
docstring_google_keras_adadelta_function_ir: IntermediateRepr = {
    "name": "Adadelta",
    "doc": deindent(remove_args_from_docstring(docstring_google_keras_adadelta_str)),
    "params": OrderedDict(
        (
            (
                "learning_rate",
                {
                    "default": 0.001,
                    "doc": (
                        "A float, a "
                        "`keras.optimizers.schedules.LearningRateSchedule` "
                        "instance, or a callable that takes no arguments and "
                        "returns the actual value to use. The learning "
                        "rate.Note that `Adadelta` tends to benefit from "
                        "higher initial learning rate values compared to "
                        "other optimizers. To match the exact form in the "
                        "original paper, use 1.0."
                    ),
                    "typ": "float",
                },
            ),
            (
                "rho",
                {
                    "default": 0.95,
                    "doc": "A floating point value. The decay rate.",
                    "typ": "float",
                },
            ),
            (
                "epsilon",
                {
                    "default": 1e-07,
                    "doc": (
                        "Small floating point value for maintaining numerical "
                        "stability."
                    ),
                    "typ": "Optional[float]",
                },
            ),
            (
                "name",
                {
                    "default": "adadelta",
                    "doc": (
                        "String. The name to use for momentum accumulator "
                        "weights created by the optimizer."
                    ),
                    "typ": "Optional[str]",
                },
            ),
            (
                "weight_decay",
                {
                    "default": NoneStr,
                    "doc": "Float. If set, weight decay is applied.",
                    "typ": "Optional[float]",
                },
            ),
            (
                "clipnorm",
                {
                    "default": NoneStr,
                    "doc": (
                        "Float. If set, the gradient of each weight is "
                        "individually clipped so that its norm is no higher "
                        "than this value."
                    ),
                    "typ": "Optional[float]",
                },
            ),
            (
                "clipvalue",
                {
                    "default": NoneStr,
                    "doc": (
                        "Float. If set, the gradient of each weight is "
                        "clipped to be no higher than this value."
                    ),
                    "typ": "Optional[float]",
                },
            ),
            (
                "global_clipnorm",
                {
                    "default": NoneStr,
                    "doc": (
                        "Float. If set, the gradient of all weights is "
                        "clipped so that their global norm is no higher than "
                        "this value."
                    ),
                    "typ": "Optional[float]",
                },
            ),
            (
                "use_ema",
                {
                    "default": False,
                    "doc": (
                        "Boolean,If True, exponential moving average (EMA) is "
                        "applied. EMA consists of computing an exponential "
                        "moving average of the weights of the model (as the "
                        "weight values change after each training batch), and "
                        "periodically overwriting the weights with their "
                        "moving average."
                    ),
                    "typ": "bool",
                },
            ),
            (
                "ema_momentum",
                {
                    "default": 0.99,
                    "doc": (
                        "Float,Only used if `use_ema=True`. This is the "
                        "momentum to use when computing the EMA of the "
                        "model's weights: `new_average = ema_momentum * "
                        "old_average + (1 - ema_momentum) * "
                        "current_variable_value`."
                    ),
                    "typ": "float",
                },
            ),
            (
                "ema_overwrite_frequency",
                {
                    "default": NoneStr,
                    "doc": (
                        "Int or None,Only used if `use_ema=True`. Every "
                        "`ema_overwrite_frequency` steps of iterations, we "
                        "overwrite the model variable by its moving average. "
                        "If None, the optimizer does not overwrite model "
                        "variables in the middle of training, and you need to "
                        "explicitly overwrite the variables at the end of "
                        "training by calling "
                        "`optimizer.finalize_variable_values()` (which "
                        "updates the model variables in-place). When using "
                        "the built-in `fit()` training loop, this happens "
                        "automatically after the last epoch, and you don't "
                        "need to do anything."
                    ),
                    "typ": "NoneType",
                },
            ),
            (
                "loss_scale_factor",
                {
                    "default": NoneStr,
                    "doc": (
                        "Float or `None`. If a float, the scale factor will "
                        "be multiplied the loss before computing gradients, "
                        "and the inverse of the scale factor will be "
                        "multiplied by the gradients before updating "
                        "variables. Useful for preventing underflow during "
                        "mixed precision training. Alternately, "
                        "`keras.optimizers.LossScaleOptimizer` will "
                        "automatically set a loss scale factor."
                    ),
                    "typ": "Optional[float]",
                },
            ),
        )
    ),
    "returns": None,
}

# ```py
# import ast
# import inspect
#
# import keras.optimizers
# from keras.src.optimizers.base_optimizer import base_optimizer_keyword_args
#
# import cdd.class_.parse
#
# mod: Module = ast.parse(inspect.getsource(keras.optimizers.Adam))
# mod.body[0].body[0].value.value = mod.body[0].body[0].value.value.replace(
#     "{{base_optimizer_keyword_args}}", base_optimizer_keyword_args)
# cdd.class_.parse.class_(node)
# ```
# ###############
# # Keras 3.0.1 #
# ###############
# https://github.com/keras-team/keras/blob/f889c1f/keras/optimizers/adam.py#L7-L40
docstring_google_keras_adam_ir: IntermediateRepr = {
    "name": None,
    "doc": remove_args_from_docstring(docstring_google_keras_adam_str) + tab,
    "params": OrderedDict(
        (
            (
                "learning_rate",
                {
                    "default": 0.001,
                    "doc": (
                        "A float, a "
                        "`keras.optimizers.schedules.LearningRateSchedule` "
                        "instance, or a callable that takes no arguments and "
                        "returns the actual value to use. The learning rate."
                    ),
                    "typ": "float",
                },
            ),
            (
                "beta_1",
                {
                    "default": 0.9,
                    "doc": (
                        "A float value or a constant float tensor, or a "
                        "callable that takes no arguments and returns the "
                        "actual value to use. The exponential decay rate for "
                        "the 1st moment estimates."
                    ),
                    "typ": "float",
                },
            ),
            (
                "beta_2",
                {
                    "default": 0.999,
                    "doc": (
                        "A float value or a constant float tensor, or a "
                        "callable that takes no arguments and returns the "
                        "actual value to use. The exponential decay rate for "
                        "the 2nd moment estimates."
                    ),
                    "typ": "float",
                },
            ),
            (
                "epsilon",
                {
                    "default": 1e-07,
                    "doc": (
                        "A small constant for numerical stability. This "
                        'epsilon is "epsilon hat" in the Kingma and Ba paper '
                        "(in the formula just before Section 2.1), not the "
                        "epsilon in Algorithm 1 of the paper."
                    ),
                    "typ": "float",
                },
            ),
            (
                "amsgrad",
                {
                    "default": False,
                    "doc": (
                        "Boolean. Whether to apply AMSGrad variant of this "
                        'algorithm from the paper "On the Convergence of Adam '
                        'and beyond".'
                    ),
                    "typ": "bool",
                },
            ),
            (
                "name",
                {
                    "default": NoneStr,
                    "doc": (
                        "String. The name to use for momentum accumulator "
                        "weights created by the optimizer."
                    ),
                    "typ": "Optional[str]",
                },
            ),
            (
                "weight_decay",
                {
                    "default": NoneStr,
                    "doc": "Float. If set, weight decay is applied.",
                    "typ": "Optional[float]",
                },
            ),
            (
                "clipnorm",
                {
                    "default": NoneStr,
                    "doc": (
                        "Float. If set, the gradient of each weight is "
                        "individually clipped so that its norm is no higher "
                        "than this value."
                    ),
                    "typ": "Optional[float]",
                },
            ),
            (
                "clipvalue",
                {
                    "default": NoneStr,
                    "doc": (
                        "Float. If set, the gradient of each weight is "
                        "clipped to be no higher than this value."
                    ),
                    "typ": "Optional[float]",
                },
            ),
            (
                "global_clipnorm",
                {
                    "default": NoneStr,
                    "doc": (
                        "Float. If set, the gradient of all weights is "
                        "clipped so that their global norm is no higher than "
                        "this value."
                    ),
                    "typ": "Optional[float]",
                },
            ),
            (
                "use_ema",
                {
                    "default": False,
                    "doc": (
                        "Boolean,If True, exponential moving average (EMA) is "
                        "applied. EMA consists of computing an exponential "
                        "moving average of the weights of the model (as the "
                        "weight values change after each training batch), and "
                        "periodically overwriting the weights with their "
                        "moving average."
                    ),
                    "typ": "bool",
                },
            ),
            (
                "ema_momentum",
                {
                    "default": 0.99,
                    "doc": (
                        "Float,Only used if `use_ema=True`. This is the "
                        "momentum to use when computing the EMA of the "
                        "model's weights: `new_average = ema_momentum * "
                        "old_average + (1 - ema_momentum) * "
                        "current_variable_value`."
                    ),
                    "typ": "float",
                },
            ),
            (
                "ema_overwrite_frequency",
                {
                    "default": NoneStr,
                    "doc": (
                        "Int or None,Only used if `use_ema=True`. Every "
                        "`ema_overwrite_frequency` steps of iterations, we "
                        "overwrite the model variable by its moving average. "
                        "If None, the optimizer does not overwrite model "
                        "variables in the middle of training, and you need to "
                        "explicitly overwrite the variables at the end of "
                        "training by calling "
                        "`optimizer.finalize_variable_values()` (which "
                        "updates the model variables in-place). When using "
                        "the built-in `fit()` training loop, this happens "
                        "automatically after the last epoch, and you don't "
                        "need to do anything."
                    ),
                },
            ),
            (
                "loss_scale_factor",
                {
                    "default": NoneStr,
                    "doc": (
                        "Float or `None`. If a float, the scale factor will "
                        "be multiplied the loss before computing gradients, "
                        "and the inverse of the scale factor will be "
                        "multiplied by the gradients before updating "
                        "variables. Useful for preventing underflow during "
                        "mixed precision training. Alternately, "
                        "`keras.optimizers.LossScaleOptimizer` will "
                        "automatically set a loss scale factor."
                    ),
                    "typ": "Optional[float]",
                },
            ),
        )
    ),
    "returns": None,
    "type": "static",
}

# #####################
# # TensorFlow 2.15.0 #
# #####################
# https://github.com/tensorflow/tensorflow/blob/5a56eb1/tensorflow/python/keras/losses.py#L862-L875
docstring_google_keras_squared_hinge_ir: IntermediateRepr = {
    "doc": docstring_google_keras_squared_hinge_no_args_doc_str + tab,
    "name": None,
    "params": OrderedDict(
        (
            (
                "reduction",
                {
                    "doc": (
                        "Type of reduction to apply to the loss. In almost "
                        'all cases this should be `"sum_over_batch_size"`. '
                        'Supported options are `"sum"`, '
                        '`"sum_over_batch_size"` or `None`.'
                    )
                },
            ),
            ("name", {"doc": "Optional name for the loss instance."}),
        )
    ),
    "returns": None,
    "type": "static",
}

docstring_google_pytorch_lbfgs_ir: IntermediateRepr = {
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
                    "doc": (
                        "maximal number of function evaluations per "
                        "optimization step"
                    ),
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
                    "doc": (
                        "termination tolerance on function value/parameter " "changes"
                    ),
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
                    "typ": "Optional[str]",
                },
            ),
        )
    ),
    "returns": None,
    "type": "static",
}

start_args_idx = docstring_keras_rmsprop_class_str.find("  Args:\n")
end_args_idx = docstring_keras_rmsprop_class_str.find("\n\n", start_args_idx) + 2
docstring_keras_rmsprop_class_ir: IntermediateRepr = {
    "doc": (
        docstring_keras_rmsprop_class_str[:start_args_idx]
        + docstring_keras_rmsprop_class_str[end_args_idx:]
    ),
    "name": None,
    "params": OrderedDict(
        (
            (
                "learning_rate",
                {
                    "default": 0.001,
                    "doc": (
                        "A `Tensor`, floating point value, or a schedule that "
                        "is a "
                        "`tf.keras.optimizers.schedules.LearningRateSchedule`, "
                        "or a callable that takes no arguments and returns "
                        "the actual value to use. The learning rate. Defaults "
                        "to 0.001."
                    ),
                    "typ": "float",
                },
            ),
            (
                "rho",
                {
                    "default": 0.9,
                    "doc": (
                        "Discounting factor for the history/coming gradient. "
                        "Defaults to 0.9."
                    ),
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
                    "doc": (
                        "A small constant for numerical stability. This "
                        'epsilon is "epsilon hat" in the Kingma and Ba paper '
                        "(in the formula just before Section 2.1), not the "
                        "epsilon in Algorithm 1 of the paper. Defaults to "
                        "1e-7."
                    ),
                    "typ": "float",
                },
            ),
            (
                "centered",
                {
                    "default": False,
                    "doc": (
                        "Boolean. If `True`, gradients are normalized by the "
                        "estimated variance of the gradient; if False, by the "
                        "uncentered second moment. Setting this to `True` may "
                        "help with training, but is slightly more expensive "
                        "in terms of computation and memory. Defaults to "
                        "`False`."
                    ),
                    "typ": "bool",
                },
            ),
            (
                "name",
                {
                    "default": "RMSprop",
                    "doc": (
                        "Optional name prefix for the operations created when "
                        'applying gradients. Defaults to `"RMSprop"`.'
                    ),
                    "typ": "Optional[str]",
                },
            ),
            (
                "kwargs",
                {
                    "default": NoneStr,
                    "doc": (
                        'Keyword arguments. Allowed to be one of `"clipnorm"` '
                        'or `"clipvalue"`. `"clipnorm"` (float) clips '
                        'gradients by norm; `"clipvalue"` (float) clips '
                        "gradients by value."
                    ),
                    "typ": 'Optional[Union[Literal["clipnorm", "clipvalue"], float]]',
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

docstring_keras_rmsprop_method_ir: IntermediateRepr = {
    "doc": (
        docstring_keras_rmsprop_method_str[:start_args_idx]
        + docstring_keras_rmsprop_method_str[end_args_idx:]
    ),
    "name": None,
    "params": OrderedDict(
        (
            (
                "learning_rate",
                {
                    "default": 0.001,
                    "doc": (
                        "A `Tensor`, floating point value, or a schedule that "
                        "is a "
                        "`tf.keras.optimizers.schedules.LearningRateSchedule`, "
                        "or a callable that takes no arguments and returns "
                        "the actual value to use. The learning rate. Defaults "
                        "to 0.001."
                    ),
                    "typ": "float",
                },
            ),
            (
                "rho",
                {
                    "default": 0.9,
                    "doc": (
                        "Discounting factor for the history/coming gradient. "
                        "Defaults to 0.9."
                    ),
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
                    "doc": (
                        "A small constant for numerical stability. This "
                        'epsilon is "epsilon hat" in the Kingma and Ba paper '
                        "(in the formula just before Section 2.1), not the "
                        "epsilon in Algorithm 1 of the paper. Defaults to "
                        "1e-7."
                    ),
                    "typ": "float",
                },
            ),
            (
                "centered",
                {
                    "default": False,
                    "doc": (
                        "Boolean. If `True`, gradients are normalized by the "
                        "estimated variance of the gradient; if False, by the "
                        "uncentered second moment. Setting this to `True` may "
                        "help with training, but is slightly more expensive "
                        "in terms of computation and memory. Defaults to "
                        "`False`."
                    ),
                    "typ": "bool",
                },
            ),
            (
                "name",
                {
                    "default": "RMSprop",
                    "doc": (
                        "Optional name prefix for the operations created when "
                        'applying gradients. Defaults to "RMSprop".'
                    ),
                    "typ": "Optional[str]",
                },
            ),
            (
                "kwargs",
                {
                    "default": NoneStr,
                    "doc": (
                        "keyword arguments. Allowed to be {`clipnorm`, "
                        "`clipvalue`, `lr`, `decay`}. `clipnorm` is clip "
                        "gradients by norm; `clipvalue` is clip gradients by "
                        "value, `decay` is included for backward "
                        "compatibility to allow time inverse decay of "
                        "learning rate. `lr` is included for backward "
                        "compatibility, recommended to use `learning_rate` "
                        "instead."
                    ),
                    "typ": "Optional[dict]",
                },
            ),
        )
    ),
    "returns": None,
    "type": "static",
}

del start_args_idx, end_args_idx

function_adder_ir: IntermediateRepr = {
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

function_google_tf_ops_losses__safe_mean_ir: IntermediateRepr = {
    "doc": "Computes a safe mean of the losses.",
    "name": "_safe_mean",
    "params": OrderedDict(
        (
            (
                "losses",
                {
                    "doc": (
                        "`Tensor` whose elements contain individual loss "
                        "measurements."
                    )
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
                    "default": (
                        "```math_ops.div_no_nan(total_loss, num_present, "
                        "name='value')```"
                    ),
                    "doc": (
                        "A scalar representing the mean of `losses`. If `num_present` is "
                        "zero, then zero is returned."
                    ),
                },
            ),
        )
    ),
    "type": "static",
}

method_complex_args_variety_ir: IntermediateRepr = {
    "doc": "Call cliff",
    "name": "call_cliff",
    "params": OrderedDict(
        (
            ("dataset_name", {"doc": "name of dataset."}),
            ("as_numpy", {"doc": "Convert to numpy ndarrays"}),
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

pydantic_ir: IntermediateRepr = {
    "doc": "",
    "name": "Cat",
    "params": OrderedDict(
        (
            ("pet_type", {"typ": "Literal['cat']"}),
            ("cat_name", {"typ": "str"}),
        )
    ),
    "returns": None,
    "type": "static",
}

intermediate_repr_extra_colons: IntermediateRepr = {
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

intermediate_repr_no_default_doc: IntermediateRepr = {
    "name": None,
    "type": "static",
    "doc": docstring_header_no_nl_str,
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
                    "doc": "Convert to numpy ndarrays",
                    "typ": "Optional[bool]",
                },
            ),
            (
                "data_loader_kwargs",
                {
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
                    "typ": (
                        "Union[Tuple[tf.data.Dataset, "
                        "tf.data.Dataset], "
                        "Tuple[np.ndarray, "
                        "np.ndarray]]"
                    ),
                },
            ),
        )
    ),
}

intermediate_repr_no_default_with_nones_doc = deepcopy(intermediate_repr_no_default_doc)
for param in "data_loader_kwargs", "as_numpy":
    intermediate_repr_no_default_with_nones_doc["params"][param]["default"] = NoneStr

intermediate_repr_no_default_sql_doc = deepcopy(intermediate_repr_no_default_doc)
intermediate_repr_no_default_sql_doc["params"]["dataset_name"]["doc"] = (
    "[PK] {}".format(
        intermediate_repr_no_default_sql_doc["params"]["dataset_name"]["doc"]
    )
)

intermediate_repr_no_default_sql_with_nones_doc = deepcopy(
    intermediate_repr_no_default_sql_doc
)
for param in "data_loader_kwargs", "as_numpy":
    intermediate_repr_no_default_sql_with_nones_doc["params"][param][
        "default"
    ] = NoneStr

intermediate_repr_no_default_sql_with_sql_types = deepcopy(
    intermediate_repr_no_default_sql_doc
)
for param, typ in (
    ("dataset_name", "String"),
    ("tfds_dir", "String"),
    ("as_numpy", "Boolean"),
    ("data_loader_kwargs", "JSON"),
):
    intermediate_repr_no_default_sql_with_sql_types["params"][param]["x_typ"] = {
        "sql": {"type": typ}
    }


intermediate_repr_empty = {
    "name": None,
    "type": "static",
    "doc": "",
    "params": OrderedDict(),
    "returns": OrderedDict(),
}

intermediate_repr_only_return_type: IntermediateRepr = {
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
                    "typ": (
                        "Union[Tuple[tf.data.Dataset, tf.data.Dataset], "
                        "Tuple[np.ndarray, np.ndarray]]"
                    ),
                },
            ),
        ),
    ),
}

intermediate_repr: IntermediateRepr = {
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
                    "doc": (
                        "directory to look for models in. "
                        "Defaults to "
                        '"~/tensorflow_datasets"'
                    ),
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
                    "doc": "Convert to numpy ndarrays",
                    "typ": "Optional[bool]",
                },
            ),
            (
                "data_loader_kwargs",
                {
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
                    "doc": (
                        "Train and tests dataset splits. "
                        "Defaults to (np.empty(0), "
                        "np.empty(0))"
                    ),
                    "typ": (
                        "Union[Tuple[tf.data.Dataset, "
                        "tf.data.Dataset], "
                        "Tuple[np.ndarray, "
                        "np.ndarray]]"
                    ),
                },
            ),
        )
    ),
}

intermediate_repr_no_default_doc_or_prop: IntermediateRepr = (
    remove_defaults_from_intermediate_repr(
        deepcopy(intermediate_repr), emit_default_prop=False
    )
)

intermediate_repr_node_pk: IntermediateRepr = {
    "doc": "",
    "name": "node",
    "params": OrderedDict(
        (
            (
                "node_id",
                {"x_typ": {"sql": {"type": "Integer"}}, "doc": "[PK]", "typ": "int"},
            ),
            (
                "primary_element",
                {
                    "x_typ": {"sql": {"type": "Integer"}},
                    "doc": "[FK(element.element_id)]",
                    "typ": "int",
                },
            ),
        )
    ),
    "returns": None,
    "type": None,
}

__all__ = [
    "class_google_keras_tensorboard_ir",
    "class_torch_nn_l1loss_ir",
    "class_torch_nn_one_cycle_lr_ir",
    "docstring_google_keras_adadelta_function_ir",
    "docstring_google_keras_adadelta_ir",
    "docstring_google_keras_adam_ir",
    "docstring_google_keras_lambda_callback_ir",
    "docstring_google_keras_squared_hinge_ir",
    "docstring_google_pytorch_lbfgs_ir",
    "docstring_keras_rmsprop_class_ir",
    "docstring_keras_rmsprop_method_ir",
    "function_adder_ir",
    "function_google_tf_ops_losses__safe_mean_ir",
    "intermediate_repr",
    "intermediate_repr_empty",
    "intermediate_repr_extra_colons",
    "intermediate_repr_no_default_doc",
    "intermediate_repr_no_default_doc_or_prop",
    "intermediate_repr_no_default_sql_doc",
    "intermediate_repr_no_default_sql_with_nones_doc",
    "intermediate_repr_no_default_sql_with_sql_types",
    "intermediate_repr_no_default_with_nones_doc",
    "intermediate_repr_node_pk",
    "intermediate_repr_only_return_type",
    "method_complex_args_variety_ir",
    "pydantic_ir",
]  # type: list[str]
