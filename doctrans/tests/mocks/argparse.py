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

from doctrans.ast_utils import (
    FALLBACK_ARGPARSE_TYP,
    FALLBACK_TYP,
    maybe_type_comment,
    set_arg,
    set_value,
)

argparse_func_str = '''
def set_cli_args(argument_parser):
    """
    Set CLI arguments

    :param argument_parser: argument parser
    :type argument_parser: ```ArgumentParser```

    :return: argument_parser, Train and tests dataset splits.
    :rtype: ```Tuple[ArgumentParser, Union[Tuple[tf.data.Dataset, tf.data.Dataset], Tuple[np.ndarray, np.ndarray]]]```
    """
    argument_parser.description = "{description}"
    argument_parser.add_argument(
        "--dataset_name",
        type=str,
        help="name of dataset.",
        required=True,
        default="mnist",
    )
    argument_parser.add_argument(
        "--tfds_dir",
        type=str,
        help="directory to look for models in.",
        default="~/tensorflow_datasets",
    )
    argument_parser.add_argument(
        "--K",
        type={FALLBACK_TYP},
        choices=("np", "tf"),
        help="backend engine, e.g., `np` or `tf`.",
        required=True,
        default="np",
    )
    argument_parser.add_argument(
        "--as_numpy", type=bool, help="Convert to numpy ndarrays"
    )
    argument_parser.add_argument(
        "--data_loader_kwargs",
        type=loads,
        help="pass this as arguments to data_loader function",
    )
    return argument_parser, (np.empty(0), np.empty(0))
'''.format(
    description="Acquire from the official tensorflow_datasets model zoo,"
    " or the ophthalmology focussed ml-prepare library",
    FALLBACK_TYP=FALLBACK_TYP,
)

argparse_func_with_body_str = '''
def set_cli_args(argument_parser):
    """
    Set CLI arguments

    :param argument_parser: argument parser
    :type argument_parser: ```ArgumentParser```

    :return: argument_parser, Train and tests dataset splits.
    :rtype: ```Tuple[ArgumentParser, Union[Tuple[tf.data.Dataset, tf.data.Dataset], Tuple[np.ndarray, np.ndarray]]]```
    """
    argument_parser.description = (
        'Acquire from the official tensorflow_datasets model zoo, or the ophthalmology focussed ml-prepare library'
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
    FALLBACK_TYP=FALLBACK_TYP
)

argparse_func_action_append_str = '''
def set_cli_action_append(argument_parser):
    """
    Set CLI arguments

    :param argument_parser: argument parser
    :type argument_parser: ```ArgumentParser```

    :return: argument_parser
    :rtype: ```ArgumentParser```
    """
    argument_parser.description = "{description}"
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
    description="Acquire from the official tensorflow_datasets model zoo,"
    " or the ophthalmology focussed ml-prepare library",
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
            Expr(
                set_value(
                    "\n    Set CLI arguments\n\n    "
                    ":param argument_parser: argument parser\n    "
                    ":type argument_parser: ```ArgumentParser```\n\n    "
                    ":return: argument_parser, Train and tests dataset splits.\n    "
                    ":rtype: ```Tuple[ArgumentParser,"
                    " Union[Tuple[tf.data.Dataset, tf.data.Dataset],"
                    " Tuple[np.ndarray, np.ndarray]]]```\n    "
                )
            ),
            Assign(
                targets=[
                    Attribute(
                        Name("argument_parser", Load()),
                        "description",
                        Store(),
                    )
                ],
                value=set_value(
                    "Acquire from the official tensorflow_datasets model zoo,"
                    " or the ophthalmology focussed ml-prepare library"
                ),
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
                        keyword(arg="type", value=Name("str", Load()), identifier=None),
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
                        keyword(arg="type", value=Name("str", Load()), identifier=None),
                        keyword(
                            arg="help",
                            value=set_value("directory to look for models in."),
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
                            arg="type",
                            value=FALLBACK_ARGPARSE_TYP,
                            identifier=None,
                        ),
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
                            value=set_value("Convert to numpy ndarrays"),
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
            Expr(
                set_value(
                    "\n    Set CLI arguments\n\n    "
                    ":param argument_parser: argument parser\n    "
                    ":type argument_parser: ```ArgumentParser```\n\n    "
                    ":return: argument_parser, Train and tests dataset splits.\n    "
                    ":rtype: ```Tuple[ArgumentParser, Union[Tuple[tf.data.Dataset, tf.data.Dataset],"
                    " Tuple[np.ndarray, np.ndarray]]]```\n    "
                )
            ),
            Assign(
                targets=[
                    Attribute(
                        Name("argument_parser", Load()),
                        "description",
                        Store(),
                    )
                ],
                value=set_value(
                    "Acquire from the official tensorflow_datasets model zoo,"
                    " or the ophthalmology focussed ml-prepare library"
                ),
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
                            arg="type",
                            value=Name(
                                "str",
                                Load(),
                            ),
                            identifier=None,
                        ),
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
                            arg="type",
                            value=Name(
                                "str",
                                Load(),
                            ),
                            identifier=None,
                        ),
                        keyword(
                            arg="help",
                            value=set_value("directory to look for models in."),
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
                            arg="type",
                            value=FALLBACK_ARGPARSE_TYP,
                            identifier=None,
                        ),
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
                                "Convert to numpy ndarrays",
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
            Expr(
                set_value(
                    "\n    Set CLI arguments\n\n    "
                    ":param argument_parser: argument parser\n    "
                    ":type argument_parser: ```ArgumentParser```\n\n    "
                    ":return: argument_parser\n    "
                    ":rtype: ```ArgumentParser```\n    "
                )
            ),
            Assign(
                targets=[
                    Attribute(Name("argument_parser", Load()), "description", Store())
                ],
                value=set_value(
                    "Acquire from the official tensorflow_datasets model zoo,"
                    " or the ophthalmology focussed ml-prepare library"
                ),
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
                        keyword(arg="type", value=Name("str", Load()), identifier=None),
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
        Expr(
            set_value(
                "\n    Set CLI arguments\n\n"
                "    :param argument_parser: argument parser\n"
                "    :type argument_parser: ```ArgumentParser```\n\n"
                "    :return: argument_parser\n"
                "    :rtype: ```ArgumentParser```\n    "
            )
        ),
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
                            "Deprecated (see :attr:`reduction`). By default,\n            the losses are averaged over each loss element in the batch. Note that for\n            some losses, there are multiple elements per sample. If the field :attr:`size_average`\n            is set to ``False``, the losses are instead summed for each minibatch. Ignored\n            when reduce is ``False``."
                        ),
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
                            "Deprecated (see :attr:`reduction`)."
                            " By default, the\n"
                            "            losses are averaged or summed over observations for each minibatch depending\n            on :attr:`size_average`. When :attr:`reduce` is ``False``, returns a loss per\n            batch element instead and ignores :attr:`size_average`."
                        ),
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
                    keyword(arg="help", value=set_value(""), identifier=None),
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
                    keyword(
                        arg="type",
                        value=Name(
                            "str",
                            Load(),
                        ),
                        identifier=None,
                    ),
                    keyword(arg="help", value=set_value(None), identifier=None),
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
    "argparse_func_with_body_ast",
    "argparse_func_with_body_str",
    "argparse_func_torch_nn_l1loss_ast",
]
