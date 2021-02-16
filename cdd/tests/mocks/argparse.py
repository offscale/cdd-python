"""
Mocks for the argparse function
"""
from ast import (
    Assign,
    Attribute,
    BinOp,
    Call,
    Expr,
    FunctionDef,
    If,
    Load,
    Mult,
    Name,
    Return,
    Store,
    Tuple,
    arguments,
    fix_missing_locations,
    keyword,
)

from cdd.ast_utils import FALLBACK_TYP, maybe_type_comment, set_arg, set_value
from cdd.pure_utils import tab
from cdd.tests.mocks.docstrings import docstring_header_str

argparse_add_argument_ast = Expr(
    Call(
        args=[set_value("--num")],
        func=Attribute(
            Name("argument_parser", Load()),
            "add_argument",
            Load(),
        ),
        keywords=[
            keyword(arg="type", value=Name("int", Load()), identifier=None),
            keyword(
                arg="required",
                value=set_value(True),
                identifier=None,
            ),
        ],
        expr=None,
        expr_func=None,
    )
)

__cli_doc_head = (
    "Set CLI arguments\n",
    ":param argument_parser: argument parser",
    ":type argument_parser: ```ArgumentParser```\n",
)

_cli_doc_str = "\n{tab}".format(tab=tab).join(
    __cli_doc_head
    + (
        ":returns: argument_parser, Train and tests dataset splits.",
        ":rtype: ```Tuple[ArgumentParser, Union[Tuple[tf.data.Dataset, tf.data.Dataset],"
        " Tuple[np.ndarray,\n{tab}np.ndarray]]]```".format(tab=tab),
    )
)
_cli_doc_expr = Expr(set_value("\n{tab}{}\n{tab}".format(_cli_doc_str, tab=tab)))

_cli_doc_nosplit_str = "\n{tab}".format(tab=tab).join(
    __cli_doc_head
    + (
        ":returns: argument_parser",
        ":rtype: ```ArgumentParser```",
    )
)
_cli_doc_nosplit_expr = Expr(
    set_value("\n{tab}{}\n{tab}".format(_cli_doc_nosplit_str, tab=tab))
)

argparse_func_str = '''
def set_cli_args(argument_parser):
    """
    {_cli_doc_str}
    """
    argument_parser.description = (
        {description!r}
    )
    argument_parser.add_argument(
        "--dataset_name",
        help="name of dataset.",
        required=True,
        default="mnist",
    )
    argument_parser.add_argument(
        "--tfds_dir",
        help="directory to look for models in.",
        required=True,
        default="~/tensorflow_datasets",
    )
    argument_parser.add_argument(
        "--K",
        choices=("np", "tf"),
        help="backend engine, e.g., `np` or `tf`.",
        required=True,
        default="np",
    )
    argument_parser.add_argument(
        "--as_numpy", type=bool, help="Convert to numpy ndarrays.",
    )
    argument_parser.add_argument(
        "--data_loader_kwargs",
        type=loads,
        help="pass this as arguments to data_loader function",
    )
    return argument_parser, (np.empty(0), np.empty(0))
'''.format(
    _cli_doc_str=_cli_doc_str,
    description=docstring_header_str.strip(),
)

argparse_func_with_body_str = '''
def set_cli_args(argument_parser):
    """
    {_cli_doc_str}
    """
    argument_parser.description = (
        {header_doc_str!r}
    )
    argument_parser.add_argument(
        '--dataset_name', type=str, help='name of dataset.', required=True, default='mnist'
    )
    argument_parser.add_argument(
        '--tfds_dir',
        type=str,
        help='directory to look for models in.',
        default='~/tensorflow_datasets',
    )
    argument_parser.add_argument(
        '--K',
        type={FALLBACK_TYP},
        choices=('np', 'tf'),
        help='backend engine, e.g., `np` or `tf`.',
        required=True,
        default='np',
    )
    argument_parser.add_argument('--as_numpy', type=bool, help='Convert to numpy ndarrays.')
    argument_parser.add_argument(
        '--data_loader_kwargs', type=loads, help='pass this as arguments to data_loader function'
    )
    # some comment
    print(5*5)
    if True:
        print(True)
        return 5
    return argument_parser, (np.empty(0), np.empty(0))
'''.format(
    _cli_doc_str=_cli_doc_str,
    FALLBACK_TYP=FALLBACK_TYP,
    header_doc_str=docstring_header_str,
)

argparse_func_action_append_str = '''
def set_cli_action_append(argument_parser):
    """
    {_cli_doc_str}
    """
    argument_parser.description = {header_doc_str!r}
    argument_parser.add_argument(
        "--callbacks",
        type=str,
        choices=(
            "BaseLogger",
            "CSVLogger",
            "Callback",
            "CallbackList",
            "EarlyStopping",
            "History",
            "LambdaCallback",
            "LearningRateScheduler",
            "ModelCheckpoint",
            "ProgbarLogger",
            "ReduceLROnPlateau",
            "RemoteMonitor",
            "TensorBoard",
            "TerminateOnNaN",
        ),
        action="append",
        help="Collection of callables that are run inside the training loop",
    )
    return argument_parser
'''.format(
    _cli_doc_str=_cli_doc_str, header_doc_str=docstring_header_str
)

argparse_func_ast = fix_missing_locations(
    FunctionDef(
        args=arguments(
            args=[set_arg("argument_parser")],
            defaults=[],
            kw_defaults=[],
            kwarg=None,
            kwonlyargs=[],
            posonlyargs=[],
            vararg=None,
            arg=None,
        ),
        body=[
            _cli_doc_expr,
            Assign(
                targets=[
                    Attribute(
                        Name("argument_parser", Load()),
                        "description",
                        Store(),
                    )
                ],
                value=set_value(docstring_header_str.replace("\n", "")),
                expr=None,
                **maybe_type_comment
            ),
            Expr(
                Call(
                    args=[set_value("--dataset_name")],
                    func=Attribute(
                        Name("argument_parser", Load()),
                        "add_argument",
                        Load(),
                    ),
                    keywords=[
                        keyword(
                            arg="help",
                            value=set_value("name of dataset."),
                            identifier=None,
                        ),
                        keyword(
                            arg="required",
                            value=set_value(True),
                            identifier=None,
                        ),
                        keyword(
                            arg="default",
                            value=set_value("mnist"),
                            identifier=None,
                        ),
                    ],
                    expr=None,
                    expr_func=None,
                )
            ),
            Expr(
                Call(
                    args=[set_value("--tfds_dir")],
                    func=Attribute(
                        Name("argument_parser", Load()),
                        "add_argument",
                        Load(),
                    ),
                    keywords=[
                        keyword(
                            arg="help",
                            value=set_value("directory to look for models in."),
                            identifier=None,
                        ),
                        keyword(
                            arg="required",
                            value=set_value(True),
                            identifier=None,
                        ),
                        keyword(
                            arg="default",
                            value=set_value("~/tensorflow_datasets"),
                            identifier=None,
                        ),
                    ],
                    expr=None,
                    expr_func=None,
                )
            ),
            Expr(
                Call(
                    args=[set_value("--K")],
                    func=Attribute(
                        Name("argument_parser", Load()),
                        "add_argument",
                        Load(),
                    ),
                    keywords=[
                        keyword(
                            arg="choices",
                            value=Tuple(
                                ctx=Load(),
                                elts=list(map(set_value, ("np", "tf"))),
                                expr=None,
                            ),
                            identifier=None,
                        ),
                        keyword(
                            arg="help",
                            value=set_value("backend engine, e.g., `np` or `tf`."),
                            identifier=None,
                        ),
                        keyword(
                            arg="required",
                            value=set_value(True),
                            identifier=None,
                        ),
                        keyword(
                            arg="default",
                            value=set_value("np"),
                            identifier=None,
                        ),
                    ],
                    expr=None,
                    expr_func=None,
                )
            ),
            Expr(
                Call(
                    args=[set_value("--as_numpy")],
                    func=Attribute(
                        Name("argument_parser", Load()),
                        "add_argument",
                        Load(),
                    ),
                    keywords=[
                        keyword(
                            arg="type", value=Name("bool", Load()), identifier=None
                        ),
                        keyword(
                            arg="help",
                            value=set_value("Convert to numpy ndarrays."),
                            identifier=None,
                        ),
                    ],
                    expr=None,
                    expr_func=None,
                )
            ),
            Expr(
                Call(
                    args=[set_value("--data_loader_kwargs")],
                    func=Attribute(
                        Name("argument_parser", Load()),
                        "add_argument",
                        Load(),
                    ),
                    keywords=[
                        keyword(
                            arg="type", value=Name("loads", Load()), identifier=None
                        ),
                        keyword(
                            arg="help",
                            value=set_value(
                                "pass this as arguments to data_loader function"
                            ),
                            identifier=None,
                        ),
                    ],
                    expr=None,
                    expr_func=None,
                )
            ),
            Return(
                value=Tuple(
                    ctx=Load(),
                    elts=[
                        Name("argument_parser", Load()),
                        Tuple(
                            ctx=Load(),
                            elts=[
                                Call(
                                    args=[set_value(0)],
                                    func=Attribute(
                                        Name("np", Load()),
                                        "empty",
                                        Load(),
                                    ),
                                    keywords=[],
                                    expr=None,
                                    expr_func=None,
                                )
                            ]
                            * 2,
                            expr=None,
                        ),
                    ],
                    expr=None,
                ),
                expr=None,
            ),
        ],
        decorator_list=[],
        name="set_cli_args",
        returns=None,
        arguments_args=None,
        stmt=None,
        identifier_name=None,
        **maybe_type_comment
    )
)

argparse_func_with_body_ast = fix_missing_locations(
    FunctionDef(
        args=arguments(
            args=[set_arg("argument_parser")],
            defaults=[],
            kw_defaults=[],
            kwarg=None,
            kwonlyargs=[],
            posonlyargs=[],
            vararg=None,
            arg=None,
        ),
        body=[
            _cli_doc_expr,
            Assign(
                targets=[
                    Attribute(
                        Name("argument_parser", Load()),
                        "description",
                        Store(),
                    )
                ],
                value=set_value(docstring_header_str),
                expr=None,
                **maybe_type_comment
            ),
            Expr(
                Call(
                    args=[set_value("--dataset_name")],
                    func=Attribute(
                        Name("argument_parser", Load()),
                        "add_argument",
                        Load(),
                    ),
                    keywords=[
                        keyword(
                            arg="help",
                            value=set_value("name of dataset."),
                            identifier=None,
                        ),
                        keyword(
                            arg="required",
                            value=set_value(True),
                            identifier=None,
                        ),
                        keyword(
                            arg="default",
                            value=set_value("mnist"),
                            identifier=None,
                        ),
                    ],
                    expr=None,
                    expr_func=None,
                )
            ),
            Expr(
                Call(
                    args=[
                        set_value(
                            "--tfds_dir",
                        )
                    ],
                    func=Attribute(
                        Name("argument_parser", Load()),
                        "add_argument",
                        Load(),
                    ),
                    keywords=[
                        keyword(
                            arg="help",
                            value=set_value("directory to look for models in."),
                            identifier=None,
                        ),
                        keyword(
                            arg="required",
                            value=set_value(True),
                            identifier=None,
                        ),
                        keyword(
                            arg="default",
                            value=set_value("~/tensorflow_datasets"),
                            identifier=None,
                        ),
                    ],
                    expr=None,
                    expr_func=None,
                )
            ),
            Expr(
                Call(
                    args=[
                        set_value(
                            "--K",
                        )
                    ],
                    func=Attribute(
                        Name("argument_parser", Load()),
                        "add_argument",
                        Load(),
                    ),
                    keywords=[
                        keyword(
                            arg="choices",
                            value=Tuple(
                                ctx=Load(),
                                elts=list(map(set_value, ("np", "tf"))),
                                expr=None,
                            ),
                            identifier=None,
                        ),
                        keyword(
                            arg="help",
                            value=set_value(
                                "backend engine, e.g., `np` or `tf`.",
                            ),
                            identifier=None,
                        ),
                        keyword(
                            arg="required",
                            value=set_value(True),
                            identifier=None,
                        ),
                        keyword(
                            arg="default",
                            value=set_value("np"),
                            identifier=None,
                        ),
                    ],
                    expr=None,
                    expr_func=None,
                )
            ),
            Expr(
                Call(
                    args=[set_value("--as_numpy")],
                    func=Attribute(
                        Name("argument_parser", Load()),
                        "add_argument",
                        Load(),
                    ),
                    keywords=[
                        keyword(
                            arg="type", value=Name("bool", Load()), identifier=None
                        ),
                        keyword(
                            arg="help",
                            value=set_value(
                                "Convert to numpy ndarrays.",
                            ),
                            identifier=None,
                        ),
                    ],
                    expr=None,
                    expr_func=None,
                )
            ),
            Expr(
                Call(
                    args=[set_value("--data_loader_kwargs")],
                    func=Attribute(
                        Name("argument_parser", Load()),
                        "add_argument",
                        Load(),
                    ),
                    keywords=[
                        keyword(
                            arg="type", value=Name("loads", Load()), identifier=None
                        ),
                        keyword(
                            arg="help",
                            value=set_value(
                                "pass this as arguments to data_loader function",
                            ),
                            identifier=None,
                        ),
                    ],
                    expr=None,
                    expr_func=None,
                )
            ),
            Expr(
                Call(
                    args=[
                        BinOp(
                            set_value(5),
                            Mult(),
                            set_value(5),
                        )
                    ],
                    func=Name("print", Load()),
                    keywords=[],
                    expr=None,
                    expr_func=None,
                )
            ),
            If(
                body=[
                    Expr(
                        Call(
                            args=[set_value(True)],
                            func=Name("print", Load()),
                            keywords=[],
                            expr=None,
                            expr_func=None,
                        )
                    ),
                    Return(
                        value=set_value(5),
                        expr=None,
                    ),
                ],
                orelse=[],
                test=set_value(True),
                expr_test=None,
                stmt=None,
            ),
            Return(
                value=Tuple(
                    ctx=Load(),
                    elts=[
                        Name("argument_parser", Load()),
                        Tuple(
                            ctx=Load(),
                            elts=[
                                Call(
                                    args=[set_value(0)],
                                    func=Attribute(
                                        Name("np", Load()),
                                        "empty",
                                        Load(),
                                    ),
                                    keywords=[],
                                    expr=None,
                                    expr_func=None,
                                )
                            ]
                            * 2,
                            expr=None,
                        ),
                    ],
                    expr=None,
                ),
                expr=None,
            ),
        ],
        decorator_list=[],
        name="set_cli_args",
        returns=None,
        arguments_args=None,
        identifier_name=None,
        stmt=None,
        **maybe_type_comment
    )
)

argparse_func_action_append_ast = fix_missing_locations(
    FunctionDef(
        name="set_cli_action_append",
        args=arguments(
            posonlyargs=[],
            args=[set_arg("argument_parser")],
            kwonlyargs=[],
            kw_defaults=[],
            defaults=[],
            arg=None,
            vararg=None,
            kwarg=None,
        ),
        body=[
            _cli_doc_nosplit_expr,
            Assign(
                targets=[
                    Attribute(Name("argument_parser", Load()), "description", Store())
                ],
                value=set_value(docstring_header_str.rstrip()),
                expr=None,
                **maybe_type_comment
            ),
            Expr(
                Call(
                    func=Attribute(
                        Name("argument_parser", Load()), "add_argument", Load()
                    ),
                    args=[set_value("--callbacks")],
                    keywords=[
                        keyword(
                            arg="type",
                            value=Name("str", Load()),
                            identifier=None,
                        ),
                        keyword(
                            arg="choices",
                            value=Tuple(
                                elts=list(
                                    map(
                                        set_value,
                                        (
                                            "BaseLogger",
                                            "CSVLogger",
                                            "Callback",
                                            "CallbackList",
                                            "EarlyStopping",
                                            "History",
                                            "LambdaCallback",
                                            "LearningRateScheduler",
                                            "ModelCheckpoint",
                                            "ProgbarLogger",
                                            "ReduceLROnPlateau",
                                            "RemoteMonitor",
                                            "TensorBoard",
                                            "TerminateOnNaN",
                                        ),
                                    )
                                ),
                                ctx=Load(),
                                expr=None,
                            ),
                            identifier=None,
                        ),
                        keyword(
                            arg="action",
                            value=set_value("append"),
                            identifier=None,
                        ),
                        keyword(
                            arg="help",
                            value=set_value(
                                "Collection of callables that are run inside the training loop"
                            ),
                            identifier=None,
                        ),
                    ],
                    expr=None,
                    expr_func=None,
                )
            ),
            Return(value=Name("argument_parser", Load()), expr=None),
        ],
        decorator_list=[],
        returns=None,
        arguments_args=None,
        identifier_name=None,
        stmt=None,
        **maybe_type_comment
    )
)

argparse_function_google_tf_tensorboard_ast = FunctionDef(
    args=arguments(
        args=[set_arg("argument_parser")],
        defaults=[],
        kw_defaults=[],
        kwarg=None,
        kwonlyargs=[],
        posonlyargs=[],
        vararg=None,
        arg=None,
    ),
    body=[
        _cli_doc_nosplit_expr,
        Assign(
            targets=[
                Attribute(
                    Name("argument_parser", Load()),
                    "description",
                    Store(),
                )
            ],
            lineno=None,
            type_comment=None,
            value=set_value(
                "Enable visualizations for TensorBoard.\n"
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
                'tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir="./logs")\n'
                "model.fit(x_train, y_train, epochs=2, callbacks=[tensorboard_callback])\n"
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
                "# Make sure to set `update_freq=N` to log a batch-level summary every N batches.\n"
                "# In addition to any `tf.summary` contained in `Model.call`, metrics added in\n"
                "# `Model.compile` will be logged every N batches.\n"
                "tb_callback = tf.keras.callbacks.TensorBoard('./logs', update_freq=1)\n"
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
                "# Make sure to set `update_freq=N` to log a batch-level summary every N batches.\n"
                "# In addition to any `tf.summary` contained in `Model.call`, metrics added in\n"
                "# `Model.compile` will be logged every N batches.\n"
                "tb_callback = tf.keras.callbacks.TensorBoard('./logs', update_freq=1)\n"
                "model.fit(x_train, y_train, callbacks=[tb_callback])\n"
                "```\n"
                "Profiling:\n"
                "```python\n"
                "# Profile a single batch, e.g. the 5th batch.\n"
                "tensorboard_callback = tf.keras.callbacks.TensorBoard(\n"
                "    log_dir='./logs', profile_batch=5)\n"
                "model.fit(x_train, y_train, epochs=2, callbacks=[tensorboard_callback])\n"
                "# Profile a range of batches, e.g. from 10 to 20.\n"
                "tensorboard_callback = tf.keras.callbacks.TensorBoard(\n"
                "    log_dir='./logs', profile_batch=(10,20))\n"
                "model.fit(x_train, y_train, epochs=2, callbacks=[tensorboard_callback])\n"
                "```"
            ),
            expr=None,
        ),
        Expr(
            Call(
                args=[set_value("--log_dir")],
                func=Attribute(
                    Name("argument_parser", Load()),
                    "add_argument",
                    Load(),
                ),
                keywords=[
                    keyword(
                        arg="help",
                        value=set_value(
                            "the path of the directory where to save the log files to be parsed by TensorBoard."
                        ),
                        identifier=None,
                    ),
                    keyword(arg="required", value=set_value(True), identifier=None),
                    keyword(arg="default", value=set_value("logs"), identifier=None),
                ],
                expr=None,
                expr_func=None,
            )
        ),
        Expr(
            Call(
                args=[set_value("--histogram_freq")],
                func=Attribute(
                    Name("argument_parser", Load()),
                    "add_argument",
                    Load(),
                ),
                keywords=[
                    keyword(arg="type", value=Name("int", Load()), identifier=None),
                    keyword(
                        arg="help",
                        value=set_value(
                            "frequency (in epochs) at which to compute activation and weight histograms for the layers"
                            " of the model. If set to 0, histograms won't be computed. Validation data (or split) must"
                            " be specified for histogram visualizations."
                        ),
                        identifier=None,
                    ),
                    keyword(arg="required", value=set_value(True), identifier=None),
                    keyword(arg="default", value=set_value(0), identifier=None),
                ],
                expr=None,
                expr_func=None,
            )
        ),
        Expr(
            Call(
                args=[set_value("--write_graph")],
                func=Attribute(
                    Name("argument_parser", Load()),
                    "add_argument",
                    Load(),
                ),
                keywords=[
                    keyword(arg="type", value=Name("bool", Load()), identifier=None),
                    keyword(
                        arg="help",
                        value=set_value(
                            "whether to visualize the graph in TensorBoard. The log file can become quite large when"
                            " write_graph is set to True."
                        ),
                        identifier=None,
                    ),
                    keyword(arg="required", value=set_value(True), identifier=None),
                    keyword(arg="default", value=set_value(True), identifier=None),
                ],
                expr=None,
                expr_func=None,
            )
        ),
        Expr(
            Call(
                args=[set_value("--write_images")],
                func=Attribute(
                    Name("argument_parser", Load()),
                    "add_argument",
                    Load(),
                ),
                keywords=[
                    keyword(arg="type", value=Name("bool", Load()), identifier=None),
                    keyword(
                        arg="help",
                        value=set_value(
                            "whether to write model weights to visualize as image in TensorBoard."
                        ),
                        identifier=None,
                    ),
                    keyword(arg="required", value=set_value(True), identifier=None),
                    keyword(arg="default", value=set_value(False), identifier=None),
                ],
                expr=None,
                expr_func=None,
            )
        ),
        Expr(
            Call(
                args=[set_value("--update_freq")],
                func=Attribute(
                    Name("argument_parser", Load()),
                    "add_argument",
                    Load(),
                ),
                keywords=[
                    keyword(
                        arg="help",
                        value=set_value(
                            "`'batch'` or `'epoch'` or integer. When using `'batch'`, writes the losses and metrics "
                            "to TensorBoard after each batch. The same applies for `'epoch'`. If using an integer, "
                            "let's say `1000`, the callback will write the metrics and losses to TensorBoard every "
                            "1000 batches. Note that writing too frequently to TensorBoard can slow down your training"
                            "."
                        ),
                        identifier=None,
                    ),
                    keyword(arg="required", value=set_value(True), identifier=None),
                    keyword(arg="default", value=set_value("epoch"), identifier=None),
                ],
                expr=None,
                expr_func=None,
            )
        ),
        Expr(
            Call(
                args=[set_value("--profile_batch")],
                func=Attribute(
                    Name("argument_parser", Load()),
                    "add_argument",
                    Load(),
                ),
                keywords=[
                    keyword(arg="type", value=Name("int", Load()), identifier=None),
                    keyword(
                        arg="help",
                        value=set_value(
                            "Profile the batch(es) to sample compute characteristics. profile_batch must be a"
                            " non-negative integer or a tuple of integers. A pair of positive integers signify a range"
                            " of batches to profile. By default, it will profile the second batch. Set profile_batch=0"
                            " to disable profiling."
                        ),
                        identifier=None,
                    ),
                    keyword(arg="required", value=set_value(True), identifier=None),
                    keyword(arg="default", value=set_value(2), identifier=None),
                ],
                expr=None,
                expr_func=None,
            )
        ),
        Expr(
            Call(
                args=[set_value("--embeddings_freq")],
                func=Attribute(
                    Name("argument_parser", Load()),
                    "add_argument",
                    Load(),
                ),
                keywords=[
                    keyword(arg="type", value=Name("int", Load()), identifier=None),
                    keyword(
                        arg="help",
                        value=set_value(
                            "frequency (in epochs) at which embedding layers will be visualized. "
                            "If set to 0, embeddings won't be visualized."
                        ),
                        identifier=None,
                    ),
                    keyword(arg="required", value=set_value(True), identifier=None),
                    keyword(arg="default", value=set_value(0), identifier=None),
                ],
                expr=None,
                expr_func=None,
            )
        ),
        Expr(
            Call(
                args=[set_value("--embeddings_metadata")],
                func=Attribute(
                    Name("argument_parser", Load()),
                    "add_argument",
                    Load(),
                ),
                keywords=[
                    keyword(
                        arg="help",
                        value=set_value(
                            "a dictionary which maps layer name to a file name in which metadata for this"
                            " embedding layer is saved. See the "
                            "[details]( https://www.tensorflow.org/how_tos/embedding_viz/#metadata_optional) about"
                            " metadata files format. In case if the same metadata file is used for all embedding"
                            " layers, string can be passed."
                        ),
                        identifier=None,
                    ),
                ],
                expr=None,
                expr_func=None,
            )
        ),
        Return(value=Name("argument_parser", Load()), expr=None),
    ],
    decorator_list=[],
    name="set_cli_args",
    returns=None,
    type_comment=None,
    arguments_args=None,
    identifier_name=None,
    stmt=None,
    lineno=None,
)

argparse_func_torch_nn_l1loss_ast = FunctionDef(
    args=arguments(
        args=[set_arg("argument_parser")],
        defaults=[],
        kw_defaults=[],
        kwarg=None,
        kwonlyargs=[],
        posonlyargs=[],
        vararg=None,
        arg=None,
    ),
    body=[
        _cli_doc_nosplit_expr,
        Assign(
            targets=[
                Attribute(
                    Name("argument_parser", Load()),
                    "description",
                    Store(),
                )
            ],
            lineno=None,
            type_comment=None,
            value=set_value(
                "Creates a criterion that measures the mean absolute error (MAE)"
                " between each element in\n"
                "    the input :math:`x` and target :math:`y`.\n"
                "    The unreduced (i.e. with :attr:`reduction` set to ``'none'``) loss"
                " can be described as:\n"
                "    .. math::\n"
                "        \\ell(x, y) = L = \\{l_1,\\dots,l_N\\}^     op, \\quad\n"
                "        l_n = \\left| x_n - y_n \night|,\n"
                "    where :math:`N` is the batch size. "
                "If :attr:`reduction` is not ``'none'``\n"
                "    (default ``'mean'``), then:\n"
                "    .. math::\n"
                "        \\ell(x, y) =\n"
                "        \x08egin{cases}\n"
                "            \\operatorname{mean}(L), &   ext{if reduction} ="
                "     ext{`mean';}\\\n            \\operatorname{sum}(L),  &"
                "   ext{if reduction} =     ext{`sum'.}\n"
                "        \\end{cases}\n"
                "    :math:`x` and :math:`y` are tensors of arbitrary shapes with a total\n"
                "    of :math:`n` elements each.\n"
                "    The sum operation still operates over all the elements, and divides by"
                " :math:`n`.\n    The division by :math:`n` can be avoided if one sets"
                " ``reduction = 'sum'``.\n\n\n"
                "        reduction (string, optional): Specifies the reduction to apply"
                " to the output:\n"
                "                ``'none'`` | ``'mean'`` | ``'sum'``. ``'none'``:"
                " no reduction will be applied,\n"
                "                ``'mean'``: the sum of the output will be divided"
                " by the number of\n                elements in the output, ``'sum'``: the output will be summed."
                " Note: :attr:`size_average`\n"
                "                and :attr:`reduce` are in the process of being deprecated, and in the meantime,"
                "\n                specifying either of those two args will override :attr:`reduction`."
                " Default: ``'mean'``\n\n\n"
                "        - Input: :math:`(N, *)` where :math:`*` means, any number of additional\n"
                "          dimensions\n        - Target: :math:`(N, *)`, same shape as the input\n"
                "        - Output: scalar. If :attr:`reduction` is ``'none'``, then\n"
                "          :math:`(N, *)`, same shape as the input\n"
                "    Examples::\n"
                "        >>> loss = nn.L1Loss()\n"
                "        >>> input = torch.randn(3, 5, requires_grad=True)\n"
                "        >>> target = torch.randn(3, 5)\n"
                "        >>> output = loss(input, target)\n"
                "        >>> output.backward()\n"
                "    "
            ),
            expr=None,
        ),
        Expr(
            Call(
                args=[set_value("--size_average")],
                func=Attribute(
                    Name("argument_parser", Load()),
                    "add_argument",
                    Load(),
                ),
                keywords=[
                    keyword(
                        arg="type",
                        value=Name(
                            "bool",
                            Load(),
                        ),
                        identifier=None,
                    ),
                    keyword(
                        arg="help",
                        value=set_value(
                            "Deprecated (see :attr:`reduction`)."
                            " By default, the losses are averaged over each loss element in the batch. Note that for"
                            " some losses, there are multiple elements per sample. If the field :attr:`size_average`"
                            " is set to ``False``, the losses are instead summed for each minibatch. Ignored when"
                            " reduce is ``False``."
                        ),
                        identifier=None,
                    ),
                    keyword(arg="default", value=set_value(True), identifier=None),
                ],
                expr=None,
                expr_func=None,
            )
        ),
        Expr(
            Call(
                args=[set_value("--reduce")],
                func=Attribute(
                    Name("argument_parser", Load()),
                    "add_argument",
                    Load(),
                ),
                keywords=[
                    keyword(arg="type", value=Name("bool", Load()), identifier=None),
                    keyword(
                        arg="help",
                        value=set_value(
                            "Deprecated (see :attr:`reduction`). "
                            "By default, the losses are averaged or summed over observations for each minibatch "
                            "depending on :attr:`size_average`. When :attr:`reduce` is ``False``, returns a loss "
                            "per batch element instead and ignores :attr:`size_average`."
                        ),
                        identifier=None,
                    ),
                    keyword(arg="default", value=set_value(True), identifier=None),
                ],
                expr=None,
                expr_func=None,
            )
        ),
        Expr(
            Call(
                args=[set_value("--__constants__")],
                func=Attribute(
                    Name(
                        "argument_parser",
                        Load(),
                    ),
                    "add_argument",
                    Load(),
                ),
                keywords=[
                    keyword(
                        arg="type",
                        value=Name(
                            "str",
                            Load(),
                        ),
                        identifier=None,
                    ),
                    keyword(arg="action", value=set_value("append"), identifier=None),
                    keyword(arg="required", value=set_value(True), identifier=None),
                    keyword(
                        arg="default", value=set_value("reduction"), identifier=None
                    ),
                ],
                expr=None,
                expr_func=None,
            )
        ),
        Expr(
            Call(
                args=[set_value("--reduction")],
                func=Attribute(
                    Name("argument_parser", Load()),
                    "add_argument",
                    Load(),
                ),
                keywords=[
                    keyword(arg="required", value=set_value(True), identifier=None),
                    keyword(arg="default", value=set_value("mean"), identifier=None),
                ],
                expr=None,
                expr_func=None,
            )
        ),
        Return(
            value=Name(
                "argument_parser",
                Load(),
            ),
            expr=None,
        ),
    ],
    decorator_list=[],
    name="set_cli_args",
    returns=None,
    type_comment=None,
    arguments_args=None,
    identifier_name=None,
    stmt=None,
    lineno=None,
)

__all__ = [
    "argparse_add_argument_ast",
    "argparse_func_action_append_ast",
    "argparse_func_action_append_str",
    "argparse_func_ast",
    "argparse_func_str",
    "argparse_func_torch_nn_l1loss_ast",
    "argparse_func_with_body_ast",
    "argparse_func_with_body_str",
]
