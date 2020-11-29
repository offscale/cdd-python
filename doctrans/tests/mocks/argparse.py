"""
Mocks for the argparse function
"""
import ast
from ast import (
    FunctionDef,
    arguments,
    arg,
    Expr,
    Constant,
    Assign,
    Store,
    Attribute,
    Name,
    Load,
    Call,
    keyword,
    Return,
    Tuple,
    BinOp,
    Mult,
    If,
)

from doctrans.ast_utils import FALLBACK_TYP, FALLBACK_ARGPARSE_TYP, set_value
from doctrans.pure_utils import PY3_8

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

argparse_func_ast = (
    FunctionDef(
        args=arguments(
            args=[
                arg(
                    annotation=None,
                    arg="argument_parser",
                    type_comment=None,
                    expr=None,
                    identifier_arg=None,
                )
            ],
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
                Constant(
                    kind=None,
                    value="\n    Set CLI arguments\n\n    "
                    ":param argument_parser: argument parser\n    "
                    ":type argument_parser: ```ArgumentParser```\n\n    "
                    ":return: argument_parser, Train and tests dataset splits.\n    "
                    ":rtype: ```Tuple[ArgumentParser,"
                    " Union[Tuple[tf.data.Dataset, tf.data.Dataset],"
                    " Tuple[np.ndarray, np.ndarray]]]```\n    ",
                    constant_value=None,
                    string=None,
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
                type_comment=None,
                value=Constant(
                    kind=None,
                    value="Acquire from the official tensorflow_datasets model zoo,"
                    " or the ophthalmology focussed ml-prepare library",
                    constant_value=None,
                    string=None,
                ),
                expr=None,
            ),
            Expr(
                Call(
                    args=[
                        Constant(
                            kind=None,
                            value="--dataset_name",
                            constant_value=None,
                            string=None,
                        )
                    ],
                    func=Attribute(
                        Name("argument_parser", Load()),
                        "add_argument",
                        Load(),
                    ),
                    keywords=[
                        keyword(arg="type", value=Name("str", Load()), identifier=None),
                        keyword(
                            arg="help",
                            value=Constant(
                                kind=None,
                                value="name of dataset.",
                                constant_value=None,
                                string=None,
                            ),
                            identifier=None,
                        ),
                        keyword(
                            arg="required",
                            value=Constant(
                                kind=None, value=True, constant_value=None, string=None
                            ),
                            identifier=None,
                        ),
                        keyword(
                            arg="default",
                            value=Constant(
                                kind=None,
                                value="mnist",
                                constant_value=None,
                                string=None,
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
                        Constant(
                            kind=None,
                            value="--tfds_dir",
                            constant_value=None,
                            string=None,
                        )
                    ],
                    func=Attribute(
                        Name("argument_parser", Load()),
                        "add_argument",
                        Load(),
                    ),
                    keywords=[
                        keyword(arg="type", value=Name("str", Load()), identifier=None),
                        keyword(
                            arg="help",
                            value=Constant(
                                kind=None,
                                value="directory to look for models in.",
                                constant_value=None,
                                string=None,
                            ),
                            identifier=None,
                        ),
                        keyword(
                            arg="default",
                            value=Constant(
                                kind=None,
                                value="~/tensorflow_datasets",
                                constant_value=None,
                                string=None,
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
                        Constant(
                            kind=None, value="--K", constant_value=None, string=None
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
                                elts=[
                                    Constant(
                                        kind=None,
                                        value="np",
                                        constant_value=None,
                                        string=None,
                                    ),
                                    Constant(
                                        kind=None,
                                        value="tf",
                                        constant_value=None,
                                        string=None,
                                    ),
                                ],
                                expr=None,
                            ),
                            identifier=None,
                        ),
                        keyword(
                            arg="help",
                            value=Constant(
                                kind=None,
                                value="backend engine, e.g., `np` or `tf`.",
                                constant_value=None,
                                string=None,
                            ),
                            identifier=None,
                        ),
                        keyword(
                            arg="required",
                            value=Constant(
                                kind=None, value=True, constant_value=None, string=None
                            ),
                            identifier=None,
                        ),
                        keyword(
                            arg="default",
                            value=Constant(
                                kind=None, value="np", constant_value=None, string=None
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
                        Constant(
                            kind=None,
                            value="--as_numpy",
                            constant_value=None,
                            string=None,
                        )
                    ],
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
                            value=Constant(
                                kind=None,
                                value="Convert to numpy ndarrays",
                                constant_value=None,
                                string=None,
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
                        Constant(
                            kind=None,
                            value="--data_loader_kwargs",
                            constant_value=None,
                            string=None,
                        )
                    ],
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
                            value=Constant(
                                kind=None,
                                value="pass this as arguments to data_loader function",
                                constant_value=None,
                                string=None,
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
                                    args=[
                                        Constant(
                                            kind=None,
                                            value=0,
                                            constant_value=None,
                                            string=None,
                                        )
                                    ],
                                    func=Attribute(
                                        Name("np", Load()),
                                        "empty",
                                        Load(),
                                    ),
                                    keywords=[],
                                    expr=None,
                                    expr_func=None,
                                ),
                                Call(
                                    args=[
                                        Constant(
                                            kind=None,
                                            value=0,
                                            constant_value=None,
                                            string=None,
                                        )
                                    ],
                                    func=Attribute(
                                        Name("np", Load()),
                                        "empty",
                                        Load(),
                                    ),
                                    keywords=[],
                                    expr=None,
                                    expr_func=None,
                                ),
                            ],
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
        type_comment=None,
        arguments_args=None,
        stmt=None,
        identifier_name=None,
    )
    if PY3_8
    else ast.parse(argparse_func_str).body[0]
)

argparse_func_with_body_ast = (
    FunctionDef(
        args=arguments(
            args=[
                arg(
                    annotation=None,
                    arg="argument_parser",
                    type_comment=None,
                    expr=None,
                    identifier_arg=None,
                )
            ],
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
                Constant(
                    kind=None,
                    value="\n    Set CLI arguments\n\n    "
                    ":param argument_parser: argument parser\n    "
                    ":type argument_parser: ```ArgumentParser```\n\n    "
                    ":return: argument_parser, Train and tests dataset splits.\n    "
                    ":rtype: ```Tuple[ArgumentParser, Union[Tuple[tf.data.Dataset, tf.data.Dataset],"
                    " Tuple[np.ndarray, np.ndarray]]]```\n    ",
                    constant_value=None,
                    string=None,
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
                type_comment=None,
                value=Constant(
                    kind=None,
                    value="Acquire from the official tensorflow_datasets model zoo,"
                    " or the ophthalmology focussed ml-prepare library",
                    constant_value=None,
                    string=None,
                ),
                expr=None,
            ),
            Expr(
                Call(
                    args=[
                        Constant(
                            kind=None,
                            value="--dataset_name",
                            constant_value=None,
                            string=None,
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
                            value=Constant(
                                kind=None,
                                value="name of dataset.",
                                constant_value=None,
                                string=None,
                            ),
                            identifier=None,
                        ),
                        keyword(
                            arg="required",
                            value=Constant(
                                kind=None, value=True, constant_value=None, string=None
                            ),
                            identifier=None,
                        ),
                        keyword(
                            arg="default",
                            value=Constant(
                                kind=None,
                                value="mnist",
                                constant_value=None,
                                string=None,
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
                        Constant(
                            kind=None,
                            value="--tfds_dir",
                            constant_value=None,
                            string=None,
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
                            value=Constant(
                                kind=None,
                                value="directory to look for models in.",
                                constant_value=None,
                                string=None,
                            ),
                            identifier=None,
                        ),
                        keyword(
                            arg="default",
                            value=Constant(
                                kind=None,
                                value="~/tensorflow_datasets",
                                constant_value=None,
                                string=None,
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
                        Constant(
                            kind=None, value="--K", constant_value=None, string=None
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
                                elts=[
                                    Constant(
                                        kind=None,
                                        value="np",
                                        constant_value=None,
                                        string=None,
                                    ),
                                    Constant(
                                        kind=None,
                                        value="tf",
                                        constant_value=None,
                                        string=None,
                                    ),
                                ],
                                expr=None,
                            ),
                            identifier=None,
                        ),
                        keyword(
                            arg="help",
                            value=Constant(
                                kind=None,
                                value="backend engine, e.g., `np` or `tf`.",
                                constant_value=None,
                                string=None,
                            ),
                            identifier=None,
                        ),
                        keyword(
                            arg="required",
                            value=Constant(
                                kind=None, value=True, constant_value=None, string=None
                            ),
                            identifier=None,
                        ),
                        keyword(
                            arg="default",
                            value=Constant(
                                kind=None, value="np", constant_value=None, string=None
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
                        Constant(
                            kind=None,
                            value="--as_numpy",
                            constant_value=None,
                            string=None,
                        )
                    ],
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
                            value=Constant(
                                kind=None,
                                value="Convert to numpy ndarrays",
                                constant_value=None,
                                string=None,
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
                        Constant(
                            kind=None,
                            value="--data_loader_kwargs",
                            constant_value=None,
                            string=None,
                        )
                    ],
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
                            value=Constant(
                                kind=None,
                                value="pass this as arguments to data_loader function",
                                constant_value=None,
                                string=None,
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
                            Constant(
                                kind=None, value=5, constant_value=None, string=None
                            ),
                            Mult(),
                            Constant(
                                kind=None, value=5, constant_value=None, string=None
                            ),
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
                            args=[
                                Constant(
                                    kind=None,
                                    value=True,
                                    constant_value=None,
                                    string=None,
                                )
                            ],
                            func=Name("print", Load()),
                            keywords=[],
                            expr=None,
                            expr_func=None,
                        )
                    ),
                    Return(
                        value=Constant(
                            kind=None, value=5, constant_value=None, string=None
                        ),
                        expr=None,
                    ),
                ],
                orelse=[],
                test=Constant(kind=None, value=True, constant_value=None, string=None),
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
                                    args=[
                                        Constant(
                                            kind=None,
                                            value=0,
                                            constant_value=None,
                                            string=None,
                                        )
                                    ],
                                    func=Attribute(
                                        Name("np", Load()),
                                        "empty",
                                        Load(),
                                    ),
                                    keywords=[],
                                    expr=None,
                                    expr_func=None,
                                ),
                                Call(
                                    args=[
                                        Constant(
                                            kind=None,
                                            value=0,
                                            constant_value=None,
                                            string=None,
                                        )
                                    ],
                                    func=Attribute(
                                        Name("np", Load()),
                                        "empty",
                                        Load(),
                                    ),
                                    keywords=[],
                                    expr=None,
                                    expr_func=None,
                                ),
                            ],
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
        type_comment=None,
        arguments_args=None,
        identifier_name=None,
        stmt=None,
    )
    if PY3_8
    else ast.parse(argparse_func_with_body_str).body[0]
)

argparse_add_argument_ast = Expr(
    Call(
        args=[Constant(kind=None, value="--num", constant_value=None, string=None)],
        func=Attribute(
            Name("argument_parser", Load()),
            "add_argument",
            Load(),
        ),
        keywords=[
            keyword(arg="type", value=Name("int", Load()), identifier=None),
            keyword(
                arg="required",
                value=Constant(kind=None, value=True, constant_value=None, string=None),
                identifier=None,
            ),
        ],
        expr=None,
        expr_func=None,
    )
)

argparse_func_action_append_ast = (
    FunctionDef(
        name="set_cli_action_append",
        args=arguments(
            posonlyargs=[],
            args=[
                arg("argument_parser", expr=None, annotation=None, type_comment=None)
            ],
            kwonlyargs=[],
            kw_defaults=[],
            defaults=[],
            arg=None,
            vararg=None,
            kwarg=None,
        ),
        body=[
            Expr(
                Constant(
                    "\n    Set CLI arguments\n\n    "
                    ":param argument_parser: argument parser\n    "
                    ":type argument_parser: ```ArgumentParser```\n\n    "
                    ":return: argument_parser\n    "
                    ":rtype: ```ArgumentParser```\n    ",
                    string=None,
                )
            ),
            Assign(
                targets=[
                    Attribute(Name("argument_parser", Load()), "description", Store())
                ],
                value=Constant(
                    "Acquire from the official tensorflow_datasets model zoo,"
                    " or the ophthalmology focussed ml-prepare library",
                    string=None,
                    kind=None,
                ),
                expr=None,
                type_comment=None,
            ),
            Expr(
                Call(
                    func=Attribute(
                        Name("argument_parser", Load()), "add_argument", Load()
                    ),
                    args=[
                        Constant(
                            value="--callbacks",
                            constant_value=None,
                            string=None,
                            kind=None,
                        )
                    ],
                    keywords=[
                        keyword(arg="type", value=Name("str", Load()), identifier=None),
                        keyword(
                            arg="choices",
                            value=Tuple(
                                elts=[
                                    Constant(
                                        kind=None,
                                        value="BaseLogger",
                                        constant_value=None,
                                        string=None,
                                    ),
                                    Constant(
                                        kind=None,
                                        value="CSVLogger",
                                        constant_value=None,
                                        string=None,
                                    ),
                                    Constant(
                                        kind=None,
                                        value="Callback",
                                        constant_value=None,
                                        string=None,
                                    ),
                                    Constant(
                                        kind=None,
                                        value="CallbackList",
                                        constant_value=None,
                                        string=None,
                                    ),
                                    Constant(
                                        kind=None,
                                        value="EarlyStopping",
                                        constant_value=None,
                                        string=None,
                                    ),
                                    Constant(
                                        kind=None,
                                        value="History",
                                        constant_value=None,
                                        string=None,
                                    ),
                                    Constant(
                                        kind=None,
                                        value="LambdaCallback",
                                        constant_value=None,
                                        string=None,
                                    ),
                                    Constant(
                                        kind=None,
                                        value="LearningRateScheduler",
                                        constant_value=None,
                                        string=None,
                                    ),
                                    Constant(
                                        kind=None,
                                        value="ModelCheckpoint",
                                        constant_value=None,
                                        string=None,
                                    ),
                                    Constant(
                                        kind=None,
                                        value="ProgbarLogger",
                                        constant_value=None,
                                        string=None,
                                    ),
                                    Constant(
                                        kind=None,
                                        value="ReduceLROnPlateau",
                                        constant_value=None,
                                        string=None,
                                    ),
                                    Constant(
                                        kind=None,
                                        value="RemoteMonitor",
                                        constant_value=None,
                                        string=None,
                                    ),
                                    Constant(
                                        kind=None,
                                        value="TensorBoard",
                                        constant_value=None,
                                        string=None,
                                    ),
                                    Constant(
                                        kind=None,
                                        value="TerminateOnNaN",
                                        constant_value=None,
                                        string=None,
                                    ),
                                ],
                                ctx=Load(),
                                expr=None,
                            ),
                            identifier=None,
                        ),
                        keyword(
                            arg="action",
                            value=set_value(kind=None, value="append"),
                            identifier=None,
                        ),
                        keyword(
                            arg="help",
                            value=Constant(
                                kind=None,
                                value="Collection of callables that are run inside the training loop",
                                constant_value=None,
                                string=None,
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
        type_comment=None,
        arguments_args=None,
        identifier_name=None,
        stmt=None,
    )
    if PY3_8
    else ast.parse(argparse_func_action_append_str).body[0]
)

__all__ = [
    "argparse_add_argument_ast",
    "argparse_func_action_append_ast",
    "argparse_func_action_append_str",
    "argparse_func_ast",
    "argparse_func_str",
    "argparse_func_with_body_ast",
    "argparse_func_with_body_str",
]
