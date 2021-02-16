"""
IR mocks
"""
from collections import OrderedDict
from copy import deepcopy

from cdd.ast_utils import NoneStr
from cdd.defaults_utils import remove_defaults_from_intermediate_repr
from cdd.pure_utils import paren_wrap_code
from cdd.tests.mocks.docstrings import docstring_header_str

class_google_tf_tensorboard_ir = {
    "doc": "Enable visualizations for TensorBoard.\n"
    "TensorBoard is a visualization tool provided with TensorFlow.\n"
    "This callback logs events for TensorBoard, including:\n"
    "* Metrics summary plots\n"
    "* Training graph visualization\n"
    "* Activation histograms\n"
    "* Sampled profiling\n"
    "If you have installed TensorFlow with pip, you should be able\n"
    "to launch TensorBoard from the command line:\n"
    "```\n"
    "tensorboard --logdir=path_to_your_logs\n"
    "```\n"
    "You can find more information about TensorBoard\n"
    "[here](https://www.tensorflow.org/get_started/summaries_and_tensorboard).\n"
    "\n"
    "\n"
    "Basic usage:\n"
    "```python\n"
    "tensorboard_callback = "
    'tf.keras.callbacks.TensorBoard(log_dir="./logs")\n'
    "model.fit(x_train, y_train, epochs=2, "
    "callbacks=[tensorboard_callback])\n"
    "# Then run the tensorboard command to view the visualizations.\n"
    "```\n"
    "Custom batch-level summaries in a subclassed Model:\n"
    "```python\n"
    "class MyModel(tf.keras.Model):\n"
    "  def build(self, _):\n"
    "    self.dense = tf.keras.layers.Dense(10)\n"
    "  def call(self, x):\n"
    "    outputs = self.dense(x)\n"
    "    tf.summary.histogram('outputs', outputs)\n"
    "    return outputs\n"
    "model = MyModel()\n"
    "model.compile('sgd', 'mse')\n"
    "# Make sure to set `update_freq=N` to log a batch-level summary "
    "every N batches.\n"
    "# In addition to any `tf.summary` contained in `Model.call`, "
    "metrics added in\n"
    "# `Model.compile` will be logged every N batches.\n"
    "tb_callback = tf.keras.callbacks.TensorBoard('./logs', "
    "update_freq=1)\n"
    "model.fit(x_train, y_train, callbacks=[tb_callback])\n"
    "```\n"
    "Custom batch-level summaries in a Functional API Model:\n"
    "```python\n"
    "def my_summary(x):\n"
    "  tf.summary.histogram('x', x)\n"
    "  return x\n"
    "inputs = tf.keras.Input(10)\n"
    "x = tf.keras.layers.Dense(10)(inputs)\n"
    "outputs = tf.keras.layers.Lambda(my_summary)(x)\n"
    "model = tf.keras.Model(inputs, outputs)\n"
    "model.compile('sgd', 'mse')\n"
    "# Make sure to set `update_freq=N` to log a batch-level summary "
    "every N batches.\n"
    "# In addition to any `tf.summary` contained in `Model.call`, "
    "metrics added in\n"
    "# `Model.compile` will be logged every N batches.\n"
    "tb_callback = tf.keras.callbacks.TensorBoard('./logs', "
    "update_freq=1)\n"
    "model.fit(x_train, y_train, callbacks=[tb_callback])\n"
    "```\n"
    "Profiling:\n"
    "```python\n"
    "# Profile a single batch, e.g. the 5th batch.\n"
    "tensorboard_callback = tf.keras.callbacks.TensorBoard(\n"
    "    log_dir='./logs', profile_batch=5)\n"
    "model.fit(x_train, y_train, epochs=2, "
    "callbacks=[tensorboard_callback])\n"
    "# Profile a range of batches, e.g. from 10 to 20.\n"
    "tensorboard_callback = tf.keras.callbacks.TensorBoard(\n"
    "    log_dir='./logs', profile_batch=(10,20))\n"
    "model.fit(x_train, y_train, epochs=2, "
    "callbacks=[tensorboard_callback])\n"
    "```",
    "name": None,
    "params": OrderedDict(
        (
            (
                "log_dir",
                {
                    "default": "logs",
                    "doc": "the path of the directory where "
                    "to save the log files to be "
                    "parsed by TensorBoard.",
                    "typ": "str",
                },
            ),
            (
                "histogram_freq",
                {
                    "default": 0,
                    "doc": "frequency (in epochs) at which "
                    "to compute activation and weight "
                    "histograms for the layers of the "
                    "model. If set to 0, histograms "
                    "won't be computed. Validation "
                    "data (or split) must be "
                    "specified for histogram "
                    "visualizations.",
                    "typ": "int",
                },
            ),
            (
                "write_graph",
                {
                    "default": True,
                    "doc": "whether to visualize the graph "
                    "in TensorBoard. The log file can "
                    "become quite large when "
                    "write_graph is set to True.",
                    "typ": "bool",
                },
            ),
            (
                "write_images",
                {
                    "default": False,
                    "doc": "whether to write model weights "
                    "to visualize as image in "
                    "TensorBoard.",
                    "typ": "bool",
                },
            ),
            (
                "update_freq",
                {
                    "default": "epoch",
                    "doc": "`'batch'` or `'epoch'` or "
                    "integer. When using `'batch'`, "
                    "writes the losses and metrics to "
                    "TensorBoard after each batch. "
                    "The same applies for `'epoch'`. "
                    "If using an integer, let's say "
                    "`1000`, the callback will write "
                    "the metrics and losses to "
                    "TensorBoard every 1000 batches. "
                    "Note that writing too frequently "
                    "to TensorBoard can slow down "
                    "your training.",
                    "typ": "str",
                },
            ),
            (
                "profile_batch",
                {
                    "default": 2,
                    "doc": "Profile the batch(es) to sample "
                    "compute characteristics. "
                    "profile_batch must be a "
                    "non-negative integer or a tuple "
                    "of integers. A pair of positive "
                    "integers signify a range of "
                    "batches to profile. By default, "
                    "it will profile the second "
                    "batch. Set profile_batch=0 to "
                    "disable profiling.",
                    "typ": "int",
                },
            ),
            (
                "embeddings_freq",
                {
                    "default": 0,
                    "doc": "frequency (in epochs) at which "
                    "embedding layers will be "
                    "visualized. If set to 0, "
                    "embeddings won't be visualized.",
                    "typ": "int",
                },
            ),
            (
                "embeddings_metadata",
                {
                    "doc": "a dictionary which maps layer "
                    "name to a file name in which "
                    "metadata for this embedding "
                    "layer is saved. See the "
                    "[details]( "
                    "https://www.tensorflow.org/how_tos/embedding_viz/#metadata_optional) "
                    "about metadata files format. In "
                    "case if the same metadata file "
                    "is used for all embedding "
                    "layers, string can be "
                    "passed.",
                    "default": NoneStr,
                },
            ),
        )
    ),
    "returns": None,
    "type": "static",
}

class_torch_nn_l1loss_ir = {
    "doc": "Creates a criterion that measures the mean absolute error (MAE) "
    "between each element in\n"
    "    the input :math:`x` and target :math:`y`.\n"
    "    The unreduced (i.e. with :attr:`reduction` set to ``'none'``) "
    "loss can be described as:\n"
    "    .. math::\n"
    "        \\ell(x, y) = L = \\{l_1,\\dots,l_N\\}^     op, \\quad\n"
    "        l_n = \\left| x_n - y_n \n"
    "ight|,\n"
    "    where :math:`N` is the batch size. If :attr:`reduction` is not "
    "``'none'``\n"
    "    (default ``'mean'``), then:\n"
    "    .. math::\n"
    "        \\ell(x, y) =\n"
    "        \x08egin{cases}\n"
    "            \\operatorname{mean}(L), &   ext{if reduction} =     "
    "ext{`mean';}\\\n"
    "            \\operatorname{sum}(L),  &   ext{if reduction} =     "
    "ext{`sum'.}\n"
    "        \\end{cases}\n"
    "    :math:`x` and :math:`y` are tensors of arbitrary shapes with a "
    "total\n"
    "    of :math:`n` elements each.\n"
    "    The sum operation still operates over all the elements, and "
    "divides by :math:`n`.\n"
    "    The division by :math:`n` can be avoided if one sets "
    "``reduction = 'sum'``.\n"
    "\n"
    "\n"
    "        reduction (string, optional): Specifies the reduction to "
    "apply to the output:\n"
    "                ``'none'`` | ``'mean'`` | ``'sum'``. ``'none'``: "
    "no reduction will be applied,\n"
    "                ``'mean'``: the sum of the output will be divided "
    "by the number of\n"
    "                elements in the output, ``'sum'``: the output will "
    "be summed. Note: :attr:`size_average`\n"
    "                and :attr:`reduce` are in the process of being "
    "deprecated, and in the meantime,\n"
    "                specifying either of those two args will override "
    ":attr:`reduction`. Default: ``'mean'``\n"
    "\n"
    "\n"
    "        - Input: :math:`(N, *)` where :math:`*` means, any number "
    "of additional\n"
    "          dimensions\n"
    "        - Target: :math:`(N, *)`, same shape as the input\n"
    "        - Output: scalar. If :attr:`reduction` is ``'none'``, "
    "then\n"
    "          :math:`(N, *)`, same shape as the input\n"
    "    Examples::\n"
    "        >>> loss = nn.L1Loss()\n"
    "        >>> input = torch.randn(3, 5, requires_grad=True)\n"
    "        >>> target = torch.randn(3, 5)\n"
    "        >>> output = loss(input, target)\n"
    "        >>> output.backward()\n"
    "    ",
    "name": None,
    "params": OrderedDict(
        (
            (
                "size_average",
                {
                    "default": True,
                    "doc": "Deprecated (see "
                    ":attr:`reduction`). By default, "
                    "the losses are averaged over "
                    "each loss element in the batch. "
                    "Note that for some losses, there "
                    "are multiple elements per "
                    "sample. If the field "
                    ":attr:`size_average` is set to "
                    "``False``, the losses are "
                    "instead summed for each "
                    "minibatch. Ignored when reduce "
                    "is ``False``.",
                    "typ": "Optional[bool]",
                },
            ),
            (
                "reduce",
                {
                    "default": True,
                    "doc": "Deprecated (see "
                    ":attr:`reduction`). By default, "
                    "the losses are averaged or "
                    "summed over observations for "
                    "each minibatch depending on "
                    ":attr:`size_average`. When "
                    ":attr:`reduce` is ``False``, "
                    "returns a loss per batch element "
                    "instead and ignores "
                    ":attr:`size_average`.",
                    "typ": "Optional[bool]",
                },
            ),
            ("__constants__", {"default": ["reduction"], "typ": "List"}),
            ("reduction", {"default": "mean", "typ": "str"}),
        )
    ),
    "returns": OrderedDict((("return_type", {"typ": "None"}),)),
    "type": "static",
}

class_torch_nn_one_cycle_lr_ir = {
    "doc": "Sets the learning rate of each parameter group according to the\n"
    "1cycle learning rate policy.\n"
    "Note also that the total number of steps in the cycle can be "
    "determined in one\n"
    "of two ways (listed in order of precedence):\n"
    "\n"
    "#. A value for total_steps is explicitly provided.\n"
    "#. A number of epochs (epochs) and a number of steps per epoch\n"
    "   (steps_per_epoch) are provided.\n"
    "   In this case, the number of total steps is inferred by\n"
    "   total_steps = epochs * steps_per_epoch\n"
    "\n"
    "You must either provide a value for total_steps or provide a value "
    "for both\n"
    "epochs and steps_per_epoch.\n"
    "\n"
    "\n"
    "Example:\n"
    "    >>> data_loader = torch.utils.data.DataLoader(...)\n"
    "    >>> optimizer = torch.optim.SGD(model.parameters(), lr=0.1, "
    "momentum=0.9)\n"
    "    >>> scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, "
    "max_lr=0.01, steps_per_epoch=len(data_loader), epochs=10)\n"
    "    >>> for epoch in range(10):\n"
    "    >>>     for batch in data_loader:\n"
    "    >>>         train_batch(...)\n"
    "    >>>         scheduler.step()\n"
    "\n"
    "\n"
    ".. _Super-Convergence\\: Very Fast Training of Neural Networks "
    "Using Large Learning Rates:\n"
    "    https://arxiv.org/abs/1708.07120",
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
    "doc": "Optimizer that implements the Adadelta algorithm.\n"
    "\n"
    "Adadelta optimization is a stochastic gradient descent method that is based on\n"
    "adaptive learning rate per dimension to address two drawbacks:\n"
    "\n"
    "- The continual decay of learning rates throughout training\n"
    "- The need for a manually selected global learning rate\n"
    "\n"
    "Adadelta is a more robust extension of Adagrad that adapts learning rates\n"
    "based on a moving window of gradient updates, instead of accumulating all\n"
    "past gradients. This way, Adadelta continues learning even when many updates\n"
    "have been done. Compared to Adagrad, in the original version of Adadelta you\n"
    "don't have to set an initial learning rate. In this version, initial\n"
    "learning rate can be set, as in most other Keras optimizers.\n"
    "\n"
    'According to section 4.3 ("Effective Learning rates"), near the end of\n'
    "training step sizes converge to 1 which is effectively a high learning\n"
    "rate which would cause divergence. This occurs only near the end of the\n"
    "training as gradients and step sizes are small, and the epsilon constant\n"
    "in the numerator and denominator dominate past gradients and parameter\n"
    "updates which converge the learning rate to 1.\n"
    "\n"
    'According to section 4.4("Speech Data"),where a large neural network with\n'
    "4 hidden layers was trained on a corpus of US English data, ADADELTA was\n"
    "used with 100 network replicas.The epsilon used is 1e-6 with rho=0.95\n"
    "which converged faster than ADAGRAD, by the following construction:\n"
    "def __init__(self, lr=1.0, rho=0.95, epsilon=1e-6, decay=0., **kwargs):\n"
    "\n"
    "\n"
    "Reference:\n"
    "    - [Zeiler, 2012](http://arxiv.org/abs/1212.5701)",
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
    "doc": "Optimizer that implements the Adadelta algorithm.\n"
    "\n"
    "Adadelta optimization is a stochastic gradient descent method that is based on\n"
    "adaptive learning rate per dimension to address two drawbacks:\n"
    "\n"
    "- The continual decay of learning rates throughout training\n"
    "- The need for a manually selected global learning rate\n"
    "\n"
    "Adadelta is a more robust extension of Adagrad that adapts learning rates\n"
    "based on a moving window of gradient updates, instead of accumulating all\n"
    "past gradients. This way, Adadelta continues learning even when many updates\n"
    "have been done. Compared to Adagrad, in the original version of Adadelta you\n"
    "don't have to set an initial learning rate. In this version, initial\n"
    "learning rate can be set, as in most other Keras optimizers.\n"
    "\n"
    'According to section 4.3 ("Effective Learning rates"), near the end of\n'
    "training step sizes converge to 1 which is effectively a high learning\n"
    "rate which would cause divergence. This occurs only near the end of the\n"
    "training as gradients and step sizes are small, and the epsilon constant\n"
    "in the numerator and denominator dominate past gradients and parameter\n"
    "updates which converge the learning rate to 1.\n"
    "\n"
    'According to section 4.4("Speech Data"),where a large neural network with\n'
    "4 hidden layers was trained on a corpus of US English data, ADADELTA was\n"
    "used with 100 network replicas.The epsilon used is 1e-6 with rho=0.95\n"
    "which converged faster than ADAGRAD, by the following construction:\n"
    "def __init__(self, lr=1.0, rho=0.95, epsilon=1e-6, decay=0., **kwargs):\n"
    "\n"
    "\n"
    "Reference:\n"
    "    - [Zeiler, 2012](http://arxiv.org/abs/1212.5701)",
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
    "doc": "Optimizer that implements the Adam algorithm.\n"
    "Adam optimization is a stochastic gradient descent method that is based on\n"
    "adaptive estimation of first-order and second-order moments.\n"
    "According to\n"
    "[Kingma et al., 2014](http://arxiv.org/abs/1412.6980),\n"
    'the method is "*computationally\n'
    "efficient, has little memory requirement, invariant to diagonal rescaling of\n"
    "gradients, and is well suited for problems that are large in terms of\n"
    'data/parameters*".\n'
    "\n"
    "\n"
    "Usage:\n"
    "    >>> opt = tf.keras.optimizers.Adam(learning_rate=0.1)\n"
    "    >>> var1 = tf.Variable(10.0)\n"
    "    >>> loss = lambda: (var1 ** 2)/2.0       # d(loss)/d(var1) == var1\n"
    "    >>> step_count = opt.minimize(loss, [var1]).numpy()\n"
    "    >>> # The first step is `-learning_rate*sign(grad)`\n"
    "    >>> var1.numpy()\n"
    "    9.9\n"
    "Reference:\n"
    "    - [Kingma et al., 2014](http://arxiv.org/abs/1412.6980)\n"
    "    - [Reddi et al., 2018](\n"
    "        https://openreview.net/pdf?id=ryQu7f-RZ) for `amsgrad`.\n"
    "Notes:\n"
    "    The default value of 1e-7 for epsilon might not be a good default in\n"
    "    general. For example, when training an Inception network on ImageNet a\n"
    "    current good choice is 1.0 or 0.1. Note that since Adam uses the\n"
    "    formulation just before Section 2.1 of the Kingma and Ba paper rather than\n"
    '    the formulation in Algorithm 1, the "epsilon" referred to here is "epsilon\n'
    '    hat" in the paper.\n'
    "    The sparse implementation of this algorithm (used when the gradient is an\n"
    "    IndexedSlices object, typically because of `tf.gather` or an embedding\n"
    "    lookup in the forward pass) does apply momentum to variable slices even if\n"
    "    they were not used in the forward pass (meaning they have a gradient equal\n"
    "    to zero). Momentum decay (beta1) is also applied to the entire momentum\n"
    "    accumulator. This means that the sparse behavior is equivalent to the dense\n"
    "    behavior (in contrast to some momentum implementations which ignore momentum\n"
    "    unless a variable slice was actually used).",
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
    "doc": "Initializes `SquaredHinge` instance.",
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
    "doc": "Implements L-BFGS algorithm, heavily inspired by `minFunc\n"
    "    <https://www.cs.ubc.ca/~schmidtm/Software/minFunc.html>`.",
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
                    "doc": "maximal number of function evaluations per optimization step",
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
                    "doc": "termination tolerance on function value/parameter changes",
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
    "doc": docstring_header_str.replace("\n", ""),
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
    "doc": docstring_header_str.replace("\n", ""),
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
    "intermediate_repr",
    "intermediate_repr_extra_colons",
    "intermediate_repr_no_default_doc",
    "intermediate_repr_no_default_doc_or_prop",
    "intermediate_repr_no_default_sql_doc",
    "intermediate_repr_only_return_type",
    "method_complex_args_variety_ir",
]
