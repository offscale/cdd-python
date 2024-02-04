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

from cdd.shared.ast_utils import FALLBACK_TYP, maybe_type_comment, set_arg, set_value
from cdd.shared.pure_utils import emit_separating_tabs, tab
from cdd.tests.mocks.classes import tensorboard_doc_str_no_args_str
from cdd.tests.mocks.docstrings import docstring_header_no_nl_str, docstring_header_str
from cdd.tests.mocks.ir import class_torch_nn_l1loss_ir

argparse_add_argument_ast: Expr = Expr(
    Call(
        args=[set_value("--num")],
        func=Attribute(
            Name("argument_parser", Load(), lineno=None, col_offset=None),
            "add_argument",
            Load(),
            lineno=None,
            col_offset=None,
        ),
        keywords=[
            keyword(
                arg="type",
                value=Name("int", Load(), lineno=None, col_offset=None),
                identifier=None,
            ),
            keyword(
                arg="required",
                value=set_value(True),
                identifier=None,
            ),
        ],
        expr=None,
        expr_func=None,
        lineno=None,
        col_offset=None,
    ),
    lineno=None,
    col_offset=None,
)

_argparse_doc_str_tuple = (
    "Set CLI arguments\n",
    ":param argument_parser: argument parser",
    ":type argument_parser: ```ArgumentParser```\n",
)  # type: tuple[str, str, str]

_cli_doc_str: str = "\n{tab}".format(tab=tab).join(
    _argparse_doc_str_tuple
    + (
        ":return: argument_parser, Train and tests dataset splits.",
        ":rtype: ```Tuple[ArgumentParser, Union[Tuple[tf.data.Dataset, tf.data.Dataset],"
        " Tuple[np.ndarray,\n{tab}np.ndarray]]]```".format(tab=tab),
    )
)

_argparse_doc_tuple = (
    _argparse_doc_str_tuple[0].rstrip("\n"),
    "",
    _argparse_doc_str_tuple[1],
    _argparse_doc_str_tuple[2],
    ":return: argument_parser, Train and tests dataset splits.",
    ":rtype: ```Tuple[ArgumentParser, Union[Tuple[tf.data.Dataset, tf.data.Dataset], Tuple[np.ndarray,",
    "np.ndarray]]]```",
)  # type: tuple[str, str, str, str, str, str, str]

_argparse_doc_str: str = "\n".join(_argparse_doc_tuple)

_argparse_description_str: str = emit_separating_tabs(
    "\n{tab}".format(tab=tab).join(
        _argparse_doc_tuple[:-1]
        + (
            "{tab}{last_line_of_doc_str}".format(
                tab=tab, last_line_of_doc_str=_argparse_doc_tuple[-1]
            ),
        )
    )
)
_argparse_doc_stripped_str: str = _argparse_doc_str.replace(
    "{tab}\n{tab}".format(tab=tab), ""
)

_cli_doc_expr: Expr = Expr(
    set_value(_argparse_description_str), lineno=None, col_offset=None
)

_cli_doc_nosplit_str: str = emit_separating_tabs(
    "\n{tab}".format(tab=tab).join(
        _argparse_doc_str_tuple
        + (
            ":return: argument_parser",
            ":rtype: ```ArgumentParser```",
        )
    )
).strip(" \n")

_cli_doc_nosplit_expr: Expr = Expr(
    set_value(
        "\n{tab}{_cli_doc_nosplit_str}\n{tab}".format(
            _cli_doc_nosplit_str=_cli_doc_nosplit_str, tab=tab
        )
    ),
    lineno=None,
    col_offset=None,
)

as_numpy_argparse_call: Expr = Expr(
    Call(
        args=[set_value("--as_numpy")],
        func=Attribute(
            Name("argument_parser", Load(), lineno=None, col_offset=None),
            "add_argument",
            Load(),
            lineno=None,
            col_offset=None,
        ),
        keywords=[
            keyword(
                arg="type",
                value=Name("bool", Load(), lineno=None, col_offset=None),
                identifier=None,
            ),
            keyword(
                arg="help",
                value=set_value(
                    "Convert to numpy ndarrays",
                ),
                identifier=None,
            ),
        ],
        expr=None,
        expr_func=None,
        lineno=None,
        col_offset=None,
    ),
    lineno=None,
    col_offset=None,
)

_argparse_add_arguments = (
    Expr(
        Call(
            args=[set_value("--dataset_name")],
            func=Attribute(
                Name("argument_parser", Load(), lineno=None, col_offset=None),
                "add_argument",
                Load(),
                lineno=None,
                col_offset=None,
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
            lineno=None,
            col_offset=None,
        ),
        lineno=None,
        col_offset=None,
    ),
    Expr(
        Call(
            args=[set_value("--tfds_dir")],
            func=Attribute(
                Name("argument_parser", Load(), lineno=None, col_offset=None),
                "add_argument",
                Load(),
                lineno=None,
                col_offset=None,
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
            lineno=None,
            col_offset=None,
        ),
        lineno=None,
        col_offset=None,
    ),
    Expr(
        Call(
            args=[set_value("--K")],
            func=Attribute(
                Name("argument_parser", Load(), lineno=None, col_offset=None),
                "add_argument",
                Load(),
                lineno=None,
                col_offset=None,
            ),
            keywords=[
                keyword(
                    arg="choices",
                    value=Tuple(
                        ctx=Load(),
                        elts=list(map(set_value, ("np", "tf"))),
                        expr=None,
                        lineno=None,
                        col_offset=None,
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
            lineno=None,
            col_offset=None,
        ),
        lineno=None,
        col_offset=None,
    ),
    as_numpy_argparse_call,
    Expr(
        Call(
            args=[set_value("--data_loader_kwargs")],
            func=Attribute(
                Name("argument_parser", Load(), lineno=None, col_offset=None),
                "add_argument",
                Load(),
                lineno=None,
                col_offset=None,
            ),
            keywords=[
                keyword(
                    arg="type",
                    value=Name("loads", Load(), lineno=None, col_offset=None),
                    identifier=None,
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
            lineno=None,
            col_offset=None,
        ),
        lineno=None,
        col_offset=None,
    ),
)  # type: tuple[Expr, ...]

_argparse_return: Return = Return(
    value=Tuple(
        ctx=Load(),
        elts=[
            Name("argument_parser", Load(), lineno=None, col_offset=None),
            Tuple(
                ctx=Load(),
                elts=[
                    Call(
                        args=[set_value(0)],
                        func=Attribute(
                            Name("np", Load(), lineno=None, col_offset=None),
                            "empty",
                            Load(),
                            lineno=None,
                            col_offset=None,
                        ),
                        keywords=[],
                        expr=None,
                        expr_func=None,
                        lineno=None,
                        col_offset=None,
                    )
                ]
                * 2,
                expr=None,
                lineno=None,
                col_offset=None,
            ),
        ],
        expr=None,
        lineno=None,
        col_offset=None,
    ),
    expr=None,
)

argparse_func_str: str = (
    '''
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
        "--as_numpy", type=bool, help="Convert to numpy ndarrays",
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
)

argparse_func_with_body_str: str = (
    '''
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
    argument_parser.add_argument('--as_numpy', type=bool, help='Convert to numpy ndarrays')
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
)

argparse_func_action_append_str: str = (
    '''
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
)

argparse_func_ast: FunctionDef = fix_missing_locations(
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
                        Name("argument_parser", Load(), lineno=None, col_offset=None),
                        "description",
                        Store(),
                        lineno=None,
                        col_offset=None,
                    )
                ],
                value=set_value(docstring_header_no_nl_str),
                expr=None,
                **maybe_type_comment
            ),
            *_argparse_add_arguments,
            _argparse_return,
        ],
        decorator_list=[],
        type_params=[],
        name="set_cli_args",
        returns=None,
        arguments_args=None,
        stmt=None,
        identifier_name=None,
        **maybe_type_comment
    )
)

argparse_func_with_body_ast: FunctionDef = fix_missing_locations(
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
                        Name("argument_parser", Load(), lineno=None, col_offset=None),
                        "description",
                        Store(),
                        lineno=None,
                        col_offset=None,
                    )
                ],
                value=set_value(_argparse_description_str),
                expr=None,
                **maybe_type_comment
            ),
            *_argparse_add_arguments,
            Expr(
                Call(
                    args=[
                        BinOp(
                            set_value(5),
                            Mult(),
                            set_value(5),
                        )
                    ],
                    func=Name("print", Load(), lineno=None, col_offset=None),
                    keywords=[],
                    expr=None,
                    expr_func=None,
                    lineno=None,
                    col_offset=None,
                ),
                lineno=None,
                col_offset=None,
            ),
            If(
                body=[
                    Expr(
                        Call(
                            args=[set_value(True)],
                            func=Name("print", Load(), lineno=None, col_offset=None),
                            keywords=[],
                            expr=None,
                            expr_func=None,
                            lineno=None,
                            col_offset=None,
                        ),
                        lineno=None,
                        col_offset=None,
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
            _argparse_return,
        ],
        decorator_list=[],
        type_params=[],
        name="set_cli_args",
        returns=None,
        arguments_args=None,
        identifier_name=None,
        stmt=None,
        **maybe_type_comment
    )
)

argparse_func_action_append_ast: FunctionDef = fix_missing_locations(
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
                    Attribute(
                        Name("argument_parser", Load(), lineno=None, col_offset=None),
                        "description",
                        Store(),
                        lineno=None,
                        col_offset=None,
                    )
                ],
                value=set_value(docstring_header_no_nl_str),
                expr=None,
                **maybe_type_comment
            ),
            Expr(
                Call(
                    func=Attribute(
                        Name("argument_parser", Load(), lineno=None, col_offset=None),
                        "add_argument",
                        Load(),
                        lineno=None,
                        col_offset=None,
                    ),
                    args=[set_value("--callbacks")],
                    keywords=[
                        keyword(
                            arg="type",
                            value=Name("str", Load(), lineno=None, col_offset=None),
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
                                lineno=None,
                                col_offset=None,
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
                    lineno=None,
                    col_offset=None,
                ),
                lineno=None,
                col_offset=None,
            ),
            Return(
                value=Name("argument_parser", Load(), lineno=None, col_offset=None),
                expr=None,
            ),
        ],
        decorator_list=[],
        type_params=[],
        returns=None,
        arguments_args=None,
        identifier_name=None,
        stmt=None,
        **maybe_type_comment
    )
)

argparse_function_google_keras_tensorboard_ast: FunctionDef = FunctionDef(
    name="set_cli_args",
    args=arguments(
        posonlyargs=[],
        args=[set_arg("argument_parser")],
        kwonlyargs=[],
        kw_defaults=[],
        defaults=[],
        vararg=None,
        kwarg=None,
        arg=None,
    ),
    body=[
        _cli_doc_nosplit_expr,
        Assign(
            targets=[
                Attribute(
                    value=Name(
                        id="argument_parser", ctx=Load(), lineno=None, col_offset=None
                    ),
                    attr="description",
                    ctx=Store(),
                    lineno=None,
                    col_offset=None,
                )
            ],
            value=set_value(tensorboard_doc_str_no_args_str.rstrip(" ")),
            lineno=None,
            expr=None,
            **maybe_type_comment
        ),
        Expr(
            value=Call(
                func=Attribute(
                    value=Name(
                        id="argument_parser", ctx=Load(), lineno=None, col_offset=None
                    ),
                    attr="add_argument",
                    ctx=Load(),
                    lineno=None,
                    col_offset=None,
                ),
                args=[set_value("--log_dir")],
                keywords=[
                    keyword(
                        arg="help",
                        value=set_value(
                            "the path of the directory where to save the log files to be parsed by TensorBoard. "
                            "e.g., `log_dir = os.path.join(working_dir, 'logs')`. "
                            "This directory should not be reused by any other callbacks."
                        ),
                    ),
                    keyword(arg="required", value=set_value(True)),
                    keyword(arg="default", value=set_value("logs")),
                ],
                lineno=None,
                col_offset=None,
            ),
            lineno=None,
            col_offset=None,
        ),
        Expr(
            value=Call(
                func=Attribute(
                    value=Name(
                        id="argument_parser", ctx=Load(), lineno=None, col_offset=None
                    ),
                    attr="add_argument",
                    ctx=Load(),
                    lineno=None,
                    col_offset=None,
                ),
                args=[set_value("--histogram_freq")],
                keywords=[
                    keyword(arg="type", value=Name(id="int", ctx=Load())),
                    keyword(
                        arg="help",
                        value=set_value(
                            "frequency (in epochs) at which to compute weight histograms for the layers of the model."
                            " If set to 0, histograms won't be computed. Validation data (or split) must be specified"
                            " for histogram visualizations."
                        ),
                    ),
                    keyword(arg="required", value=set_value(True)),
                    keyword(arg="default", value=set_value(0)),
                ],
                lineno=None,
                col_offset=None,
            ),
            lineno=None,
            col_offset=None,
        ),
        Expr(
            value=Call(
                func=Attribute(
                    value=Name(
                        id="argument_parser", ctx=Load(), lineno=None, col_offset=None
                    ),
                    attr="add_argument",
                    ctx=Load(),
                    lineno=None,
                    col_offset=None,
                ),
                args=[set_value("--write_graph")],
                keywords=[
                    keyword(arg="type", value=Name(id="bool", ctx=Load())),
                    keyword(
                        arg="help",
                        value=set_value(
                            "(Not supported at this time) Whether to visualize the graph in TensorBoard. Note that the"
                            " log file can become quite large when `write_graph` is set to `True`."
                        ),
                    ),
                    keyword(arg="required", value=set_value(True)),
                    keyword(arg="default", value=set_value(True)),
                ],
                lineno=None,
                col_offset=None,
            ),
            lineno=None,
            col_offset=None,
        ),
        Expr(
            value=Call(
                func=Attribute(
                    value=Name(
                        id="argument_parser", ctx=Load(), lineno=None, col_offset=None
                    ),
                    attr="add_argument",
                    ctx=Load(),
                    lineno=None,
                    col_offset=None,
                ),
                args=[set_value("--write_images")],
                keywords=[
                    keyword(arg="type", value=Name(id="bool", ctx=Load())),
                    keyword(
                        arg="help",
                        value=set_value(
                            "whether to write model weights to visualize as image in TensorBoard."
                        ),
                    ),
                    keyword(arg="required", value=set_value(True)),
                    keyword(arg="default", value=set_value(False)),
                ],
                lineno=None,
                col_offset=None,
            ),
            lineno=None,
            col_offset=None,
        ),
        Expr(
            value=Call(
                func=Attribute(
                    value=Name(
                        id="argument_parser", ctx=Load(), lineno=None, col_offset=None
                    ),
                    attr="add_argument",
                    ctx=Load(),
                    lineno=None,
                    col_offset=None,
                ),
                args=[set_value("--write_steps_per_second")],
                keywords=[
                    keyword(arg="type", value=Name(id="bool", ctx=Load())),
                    keyword(
                        arg="help",
                        value=set_value(
                            "whether to log the training steps per second into TensorBoard. This supports both epoch"
                            " and batch frequency logging."
                        ),
                    ),
                    keyword(arg="required", value=set_value(True)),
                    keyword(arg="default", value=set_value(False)),
                ],
                lineno=None,
                col_offset=None,
            ),
            lineno=None,
            col_offset=None,
        ),
        Expr(
            value=Call(
                func=Attribute(
                    value=Name(
                        id="argument_parser", ctx=Load(), lineno=None, col_offset=None
                    ),
                    attr="add_argument",
                    ctx=Load(),
                    lineno=None,
                    col_offset=None,
                ),
                args=[set_value("--update_freq")],
                keywords=[
                    keyword(
                        arg="choices",
                        value=Tuple(
                            elts=[set_value("batch"), set_value("epoch")],
                            ctx=Load(),
                            lineno=None,
                            col_offset=None,
                        ),
                    ),
                    keyword(
                        arg="help",
                        value=set_value(
                            '`"batch"` or `"epoch"` or integer. When using `"epoch"`, writes the losses and metrics to TensorBoard after every epoch. If using an integer, let\'s say `1000`, all metrics and losses (including custom ones added by `Model.compile`) will be logged to TensorBoard every 1000 batches. `"batch"` is a synonym for 1, meaning that they will be written every batch. Note however that writing too frequently to TensorBoard can slow down your training, especially when used with distribution strategies as it will incur additional synchronization overhead. Batch-level summary writing is also available via `train_step` override. Please see [TensorBoard Scalars tutorial]( https://www.tensorflow.org/tensorboard/scalars_and_keras#batch-level_logging)  # noqa: E501 for more details.'
                        ),
                    ),
                    keyword(arg="required", value=set_value(True)),
                    keyword(arg="default", value=set_value("epoch")),
                ],
                lineno=None,
                col_offset=None,
            ),
            lineno=None,
            col_offset=None,
        ),
        Expr(
            value=Call(
                func=Attribute(
                    value=Name(
                        id="argument_parser", ctx=Load(), lineno=None, col_offset=None
                    ),
                    attr="add_argument",
                    ctx=Load(),
                    lineno=None,
                    col_offset=None,
                ),
                args=[set_value("--profile_batch")],
                keywords=[
                    keyword(arg="type", value=Name(id="int", ctx=Load())),
                    keyword(
                        arg="help",
                        value=set_value(
                            "(Not supported at this time) Profile the batch(es) to sample compute characteristics."
                            " profile_batch must be a non-negative integer or a tuple of integers. A pair of positive"
                            " integers signify a range of batches to profile. By default, profiling is disabled."
                        ),
                    ),
                    keyword(arg="required", value=set_value(True)),
                    keyword(arg="default", value=set_value(0)),
                ],
                lineno=None,
                col_offset=None,
            ),
            lineno=None,
            col_offset=None,
        ),
        Expr(
            value=Call(
                func=Attribute(
                    value=Name(
                        id="argument_parser", ctx=Load(), lineno=None, col_offset=None
                    ),
                    attr="add_argument",
                    ctx=Load(),
                    lineno=None,
                    col_offset=None,
                ),
                args=[set_value("--embeddings_freq")],
                keywords=[
                    keyword(arg="type", value=Name(id="int", ctx=Load())),
                    keyword(
                        arg="help",
                        value=set_value(
                            "frequency (in epochs) at which embedding layers will be visualized. If set to 0,"
                            " embeddings won't be visualized."
                        ),
                    ),
                    keyword(arg="required", value=set_value(True)),
                    keyword(arg="default", value=set_value(0)),
                ],
                lineno=None,
                col_offset=None,
            ),
            lineno=None,
            col_offset=None,
        ),
        Expr(
            value=Call(
                func=Attribute(
                    value=Name(
                        id="argument_parser", ctx=Load(), lineno=None, col_offset=None
                    ),
                    attr="add_argument",
                    ctx=Load(),
                    lineno=None,
                    col_offset=None,
                ),
                args=[set_value("--embeddings_metadata")],
                keywords=[
                    keyword(
                        arg="help",
                        value=set_value(
                            "Dictionary which maps embedding layer names to the filename of a file in which to save"
                            " metadata for the embedding layer. In case the same metadata file is to be used for all"
                            " embedding layers, a single filename can be passed."
                        ),
                    )
                ],
            ),
            lineno=None,
            col_offset=None,
        ),
        Return(value=Name(id="argument_parser", ctx=Load())),
    ],
    decorator_list=[],
    type_params=[],
    returns=None,
    arguments_args=None,
    identifier_name=None,
    stmt=None,
    lineno=None,
    **maybe_type_comment
)

argparse_func_torch_nn_l1loss_ast: FunctionDef = FunctionDef(
    name="set_cli_args",
    args=arguments(
        posonlyargs=[],
        args=[set_arg("argument_parser")],
        kwonlyargs=[],
        kw_defaults=[],
        defaults=[],
        vararg=None,
        kwarg=None,
        arg=None,
    ),
    body=[
        _cli_doc_nosplit_expr,
        Assign(
            targets=[
                Attribute(
                    value=Name(
                        id="argument_parser", ctx=Load(), lineno=None, col_offset=None
                    ),
                    attr="description",
                    ctx=Store(),
                    lineno=None,
                    col_offset=None,
                )
            ],
            value=set_value(
                "Creates a criterion that measures the mean absolute error (MAE) between each element in\n"
                "the input :math:`x` and target :math:`y`.\n"
                "\n"
                "The unreduced (i.e. with :attr:`reduction` set to ``'none'``) loss can be described as:\n"
                "\n"
                ".. math::\n"
                "\\ell(x, y) = L = \\{l_1,\\dots,l_N\\}^\\top, \\quad\n"
                "l_n = \\left| x_n - y_n \\right|,\n"
                "\n"
                "where :math:`N` is the batch size. If :attr:`reduction` is not ``'none'``\n"
                "(default ``'mean'``), then:\n"
                "\n"
                ".. math::\n"
                "\\ell(x, y) =\n"
                "\\begin{cases}\n"
                "    \\operatorname{mean}(L), & \\text{if reduction} = \\text{`mean';}\\\\\n"
                "    \\operatorname{sum}(L),  & \\text{if reduction} = \\text{`sum'.}\n"
                "\\end{cases}\n"
                "\n"
                ":math:`x` and :math:`y` are tensors of arbitrary shapes with a total\n"
                "of :math:`n` elements each.\n"
                "\n"
                "The sum operation still operates over all the elements, and divides by :math:`n`.\n"
                "\n"
                "The division by :math:`n` can be avoided if one sets ``reduction = 'sum'``.\n"
                "\n"
                "Supports real-valued and complex-valued inputs.\n"
                "\n"
                "Shape:\n"
                "- Input: :math:`(*)`, where :math:`*` means any number of dimensions.\n"
                "- Target: :math:`(*)`, same shape as the input.\n"
                "- Output: scalar. If :attr:`reduction` is ``'none'``, then\n"
                "  :math:`(*)`, same shape as the input.\n"
                "\n"
                "Examples::\n"
                "\n"
                ">>> loss = nn.L1Loss()\n"
                ">>> input = torch.randn(3, 5, requires_grad=True)\n"
                ">>> target = torch.randn(3, 5)\n"
                ">>> output = loss(input, target)\n"
                ">>> output.backward()"
            ),
            lineno=None,
            expr=None,
            **maybe_type_comment
        ),
        Expr(
            value=Call(
                func=Attribute(
                    value=Name(
                        id="argument_parser", ctx=Load(), lineno=None, col_offset=None
                    ),
                    attr="add_argument",
                    ctx=Load(),
                    lineno=None,
                    col_offset=None,
                ),
                args=[set_value("--size_average")],
                keywords=[
                    keyword(arg="type", value=Name(id="bool", ctx=Load())),
                    keyword(
                        arg="help",
                        value=set_value(
                            class_torch_nn_l1loss_ir["params"]["size_average"]["doc"]
                        ),
                    ),
                    keyword(arg="default", value=set_value(True)),
                ],
                expr=None,
                expr_func=None,
            ),
            lineno=None,
            col_offset=None,
        ),
        Expr(
            value=Call(
                func=Attribute(
                    value=Name(
                        id="argument_parser", ctx=Load(), lineno=None, col_offset=None
                    ),
                    attr="add_argument",
                    ctx=Load(),
                    lineno=None,
                    col_offset=None,
                ),
                args=[set_value("--reduce")],
                keywords=[
                    keyword(arg="type", value=Name(id="bool", ctx=Load())),
                    keyword(
                        arg="help",
                        value=set_value(
                            class_torch_nn_l1loss_ir["params"]["reduce"]["doc"]
                        ),
                    ),
                    keyword(arg="default", value=set_value(True)),
                ],
                expr=None,
                expr_func=None,
            ),
            lineno=None,
            col_offset=None,
        ),
        Expr(
            value=Call(
                func=Attribute(
                    value=Name(
                        id="argument_parser", ctx=Load(), lineno=None, col_offset=None
                    ),
                    attr="add_argument",
                    ctx=Load(),
                    lineno=None,
                    col_offset=None,
                ),
                args=[set_value("--reduction")],
                keywords=[
                    keyword(
                        arg="help",
                        value=set_value(
                            class_torch_nn_l1loss_ir["params"]["reduction"]["doc"]
                        ),
                    ),
                    keyword(arg="required", value=set_value(True)),
                    keyword(arg="default", value=set_value("mean")),
                ],
                expr=None,
                expr_func=None,
            ),
            lineno=None,
            col_offset=None,
        ),
        Expr(
            value=Call(
                func=Attribute(
                    value=Name(
                        id="argument_parser", ctx=Load(), lineno=None, col_offset=None
                    ),
                    attr="add_argument",
                    ctx=Load(),
                    lineno=None,
                    col_offset=None,
                ),
                args=[set_value("--__constants__")],
                keywords=[
                    keyword(arg="type", value=Name(id="str", ctx=Load())),
                    keyword(arg="action", value=set_value("append")),
                    keyword(arg="required", value=set_value(True)),
                    keyword(arg="default", value=set_value("reduction")),
                ],
                expr=None,
                expr_func=None,
            ),
            lineno=None,
            col_offset=None,
        ),
        Return(value=Name(id="argument_parser", ctx=Load())),
    ],
    decorator_list=[],
    type_params=[],
    returns=None,
    arguments_args=None,
    identifier_name=None,
    stmt=None,
    lineno=None,
    **maybe_type_comment
)

argparse_add_argument_expr: Expr = Expr(
    Call(
        args=[set_value("--byo")],
        func=Attribute(
            Name("argument_parser", Load(), lineno=None, col_offset=None),
            "add_argument",
            Load(),
            lineno=None,
            col_offset=None,
        ),
        keywords=[
            keyword(
                arg="type",
                value=Name("str", Load(), lineno=None, col_offset=None),
                identifier=None,
            ),
            keyword(arg="action", value=set_value("append"), identifier=None),
            keyword(arg="required", value=set_value(True), identifier=None),
        ],
        expr=None,
        expr_func=None,
    ),
    lineno=None,
    col_offset=None,
)

__all__ = [
    "argparse_add_argument_ast",
    "argparse_add_argument_expr",
    "_argparse_doc_str_tuple",
    "argparse_func_action_append_ast",
    "argparse_func_action_append_str",
    "argparse_func_ast",
    "argparse_func_str",
    "argparse_func_torch_nn_l1loss_ast",
    "argparse_func_with_body_ast",
    "argparse_func_with_body_str",
    "argparse_function_google_keras_tensorboard_ast",
]  # type: list[str]
