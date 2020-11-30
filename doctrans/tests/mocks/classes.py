"""
Mocks for the `class`
"""
import ast
from ast import (
    ClassDef,
    Name,
    Load,
    Expr,
    Constant,
    AnnAssign,
    Store,
    Subscript,
    Tuple,
    Dict,
    Attribute,
    Index,
    Call,
    Assign,
    FunctionDef,
    arguments,
    arg,
    BinOp,
    Sub,
    Mult,
    keyword,
    USub,
    UnaryOp,
    Return,
)

from doctrans.ast_utils import set_value
from doctrans.pure_utils import PY3_8

class_str = '''
class ConfigClass(object):
    """
    Acquire from the official tensorflow_datasets model zoo, or the ophthalmology focussed ml-prepare library

    :cvar dataset_name: name of dataset. Defaults to "mnist"
    :cvar tfds_dir: directory to look for models in. Defaults to "~/tensorflow_datasets"
    :cvar K: backend engine, e.g., `np` or `tf`. Defaults to "np"
    :cvar as_numpy: Convert to numpy ndarrays
    :cvar data_loader_kwargs: pass this as arguments to data_loader function
    :cvar return_type: Train and tests dataset splits. Defaults to (np.empty(0), np.empty(0))"""

    dataset_name: str = "mnist"
    tfds_dir: Optional[str] = "~/tensorflow_datasets"
    K: Literal["np", "tf"] = "np"
    as_numpy: Optional[bool] = None
    data_loader_kwargs: dict = {}
    return_type: Union[
        Tuple[tf.data.Dataset, tf.data.Dataset], Tuple[np.ndarray, np.ndarray]
    ] = (
        np.empty(0),
        np.empty(0),
    )
'''

class_nargs_str = '''
class ConfigClass(object):
    """
    Acquire from the official tensorflow_datasets model zoo, or the ophthalmology focussed ml-prepare library

    :cvar callbacks: Collection of callables that are run inside the training loop"""

    callbacks: Optional[
        List[
            Literal[
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
            ]
        ]
    ] = None
'''

class_ast = (
    ClassDef(
        bases=[Name("object", Load())],
        body=[
            Expr(
                Constant(
                    kind=None,
                    value="\n    Acquire from the official tensorflow_datasets model zoo,"
                    " or the ophthalmology focussed ml-prepare library\n\n    "
                    ':cvar dataset_name: name of dataset. Defaults to "mnist"\n    '
                    ':cvar tfds_dir: directory to look for models in. Defaults to "~/tensorflow_datasets"\n    '
                    ':cvar K: backend engine, e.g., `np` or `tf`. Defaults to "np"\n    '
                    ":cvar as_numpy: Convert to numpy ndarrays\n    "
                    ":cvar data_loader_kwargs: pass this as arguments to data_loader function\n    "
                    ":cvar return_type: Train and tests dataset splits. Defaults to (np.empty(0), np.empty(0))",
                    constant_value=None,
                    string=None,
                )
            ),
            AnnAssign(
                annotation=Name(
                    "str",
                    Load(),
                ),
                simple=1,
                target=Name("dataset_name", Store()),
                value=Constant(
                    kind=None, value="mnist", constant_value=None, string=None
                ),
                expr=None,
                expr_annotation=None,
                expr_target=None,
            ),
            AnnAssign(
                annotation=Subscript(
                    Name("Optional", Load()),
                    Index(
                        value=Name(
                            "str",
                            Load(),
                        )
                    ),
                    Load(),
                ),
                simple=1,
                target=Name("tfds_dir", Store()),
                value=Constant(
                    kind=None,
                    value="~/tensorflow_datasets",
                    constant_value=None,
                    string=None,
                ),
                expr=None,
                expr_annotation=None,
                expr_target=None,
            ),
            AnnAssign(
                annotation=Subscript(
                    Name(
                        "Literal",
                        Load(),
                    ),
                    Index(
                        value=Tuple(
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
                            ctx=Load(),
                            expr=None,
                        )
                    ),
                    Load(),
                ),
                simple=1,
                target=Name("K", Store()),
                value=Constant(kind=None, value="np", constant_value=None, string=None),
                expr=None,
                expr_target=None,
                expr_annotation=None,
            ),
            AnnAssign(
                annotation=Subscript(
                    Name(
                        "Optional",
                        Load(),
                    ),
                    Index(value=Name("bool", Load())),
                    Load(),
                ),
                simple=1,
                target=Name("as_numpy", Store()),
                value=Constant(kind=None, value=None, constant_value=None, string=None),
                expr=None,
                expr_target=None,
                expr_annotation=None,
            ),
            AnnAssign(
                annotation=Name("dict", Load()),
                simple=1,
                target=Name(
                    "data_loader_kwargs",
                    Store(),
                ),
                value=Dict(keys=[], values=[], expr=None),
                expr=None,
                expr_target=None,
                expr_annotation=None,
            ),
            AnnAssign(
                annotation=Subscript(
                    Name("Union", Load()),
                    Index(
                        value=Tuple(
                            ctx=Load(),
                            elts=[
                                Subscript(
                                    Name("Tuple", Load()),
                                    Index(
                                        value=Tuple(
                                            ctx=Load(),
                                            elts=[
                                                Attribute(
                                                    Attribute(
                                                        Name("tf", Load()),
                                                        "data",
                                                        Load(),
                                                    ),
                                                    "Dataset",
                                                    Load(),
                                                ),
                                                Attribute(
                                                    Attribute(
                                                        Name("tf", Load()),
                                                        "data",
                                                        Load(),
                                                    ),
                                                    "Dataset",
                                                    Load(),
                                                ),
                                            ],
                                            expr=None,
                                        )
                                    ),
                                    Load(),
                                ),
                                Subscript(
                                    Name("Tuple", Load()),
                                    Index(
                                        Tuple(
                                            ctx=Load(),
                                            elts=[
                                                Attribute(
                                                    Name("np", Load()),
                                                    "ndarray",
                                                    Load(),
                                                ),
                                                Attribute(
                                                    Name("np", Load()),
                                                    "ndarray",
                                                    Load(),
                                                ),
                                            ],
                                            expr=None,
                                        )
                                    ),
                                    Load(),
                                ),
                            ],
                            expr=None,
                        )
                    ),
                    Load(),
                ),
                simple=1,
                target=Name("return_type", Store()),
                value=Tuple(
                    ctx=Load(),
                    elts=[
                        Call(
                            args=[
                                Constant(
                                    kind=None, value=0, constant_value=None, string=None
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
                                    kind=None, value=0, constant_value=None, string=None
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
                expr=None,
                expr_target=None,
                expr_annotation=None,
            ),
        ],
        decorator_list=[],
        keywords=[],
        name="ConfigClass",
        expr=None,
        identifier_name=None,
    )
    if PY3_8
    else ast.parse(class_str).body[0]
)

class_nargs_ast = (
    ClassDef(
        bases=[Name("object", Load())],
        body=[
            Expr(
                Constant(
                    kind=None,
                    value="\n    Acquire from the official tensorflow_datasets model zoo,"
                    " or the ophthalmology focussed ml-prepare library\n\n    "
                    ":cvar callbacks: Collection of callables that are run inside the training loop",
                    constant_value=None,
                    string=None,
                )
            ),
            AnnAssign(
                annotation=Subscript(
                    Name("Optional", Load()),
                    Index(
                        value=Subscript(
                            Name("List", Load()),
                            Index(
                                value=Subscript(
                                    Name("Literal", Load()),
                                    Index(
                                        value=Tuple(
                                            ctx=Load(),
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
                                            expr=None,
                                        )
                                    ),
                                    Load(),
                                )
                            ),
                            Load(),
                        )
                    ),
                    Load(),
                ),
                simple=1,
                target=Name("callbacks", Store()),
                value=Constant(kind=None, value=None, constant_value=None, string=None),
                expr=None,
                expr_annotation=None,
                expr_target=None,
            ),
        ],
        decorator_list=[],
        keywords=[],
        name="ConfigClass",
        expr=None,
        identifier_name=None,
    )
    if PY3_8
    else ast.parse(class_nargs_str).body[0]
)

class_squared_hinge_config_ast = ClassDef(
    bases=[Name("object", Load())],
    body=[
        Expr(
            set_value(
                "\n    Computes the squared hinge loss between `y_true` and `y_pred`.\n    \n"
                "    `loss = mean(square(maximum(1 - y_true * y_pred, 0)), axis=-1)`\n    \n"
                "    Standalone usage:\n    \n"
                "    >>> y_true = np.random.choice([-1, 1], size=(2, 3))\n"
                "    >>> y_pred = np.random.random(size=(2, 3))\n"
                "    >>> loss = tf.keras.losses.squared_hinge(y_true, y_pred)\n"
                "    >>> assert loss.shape == (2,)\n"
                "    >>> assert np.array_equal(\n"
                "    ...     loss.numpy(),\n"
                "    ...     np.mean(np.square(np.maximum(1. - y_true * y_pred, 0.)), axis=-1))\n\n"
                "    :cvar y_true: The ground truth values. `y_true` values are expected to be -1 or 1.\n"
                "    If binary (0 or 1) labels are provided we will convert them to -1 or 1.\n"
                "    shape = `[batch_size, d0, .. dN]`.\n"
                "    :cvar y_pred: The predicted values. shape = `[batch_size, d0, .. dN]`.\n"
                "    :cvar return_type: None"
            )
        ),
        Assign(
            targets=[Name("y_true", Store())],
            value=set_value(value=None),
            expr=None,
            lineno=None,
        ),
        Assign(
            targets=[Name("y_pred", Store())],
            value=set_value(value=None),
            expr=None,
            lineno=None,
        ),
        Assign(
            targets=[Name("return_type", Store())],
            value=set_value(
                "```K.mean(math_ops.square(math_ops.maximum(1.0 - y_true * y_pred, 0.0)), axis=-1)```"
            ),
            expr=None,
            lineno=None,
        ),
        FunctionDef(
            args=arguments(
                args=[arg(annotation=None, arg="self", expr=None, identifier_arg=None)],
                defaults=[],
                kw_defaults=[],
                kwarg=None,
                kwonlyargs=[],
                vararg=None,
                arg=None,
                posonlyargs=[],
            ),
            body=[
                Assign(
                    targets=[
                        Attribute(
                            Name("self", Load()),
                            "y_pred",
                            Load(),
                        )
                    ],
                    value=Call(
                        args=[Attribute(Name("self", Load()), "y_pred", Load())],
                        func=Attribute(
                            Name("ops", Load()),
                            "convert_to_tensor_v2",
                            Load(),
                        ),
                        keywords=[],
                        expr=None,
                        expr_func=None,
                    ),
                    expr=None,
                    lineno=None,
                ),
                Assign(
                    targets=[Attribute(Name("self", Load()), "y_true", Load())],
                    value=Call(
                        args=[
                            Attribute(
                                Name("self", Load()),
                                "y_true",
                                Load(),
                            ),
                            Attribute(
                                Attribute(
                                    Name("self", Load()),
                                    "y_pred",
                                    Load(),
                                ),
                                "dtype",
                                Load(),
                            ),
                        ],
                        func=Attribute(
                            Name("math_ops", Load()),
                            "cast",
                            Load(),
                        ),
                        keywords=[],
                        expr=None,
                        expr_func=None,
                    ),
                    expr=None,
                    lineno=None,
                ),
                Assign(
                    targets=[
                        Attribute(
                            Name("self", Load()),
                            "y_true",
                            Load(),
                        )
                    ],
                    value=Call(
                        args=[
                            Attribute(
                                Name("self", Load()),
                                "y_true",
                                Load(),
                            )
                        ],
                        func=Name("_maybe_convert_labels", Load()),
                        keywords=[],
                        expr=None,
                        expr_func=None,
                    ),
                    expr=None,
                    lineno=None,
                ),
                Return(
                    value=Call(
                        args=[
                            Call(
                                args=[
                                    Call(
                                        args=[
                                            BinOp(
                                                set_value(1.0),
                                                Sub(),
                                                BinOp(
                                                    Attribute(
                                                        Name("self", Load()),
                                                        "y_true",
                                                        Load(),
                                                    ),
                                                    Mult(),
                                                    Attribute(
                                                        Name("self", Load()),
                                                        "y_pred",
                                                        Load(),
                                                    ),
                                                ),
                                            ),
                                            set_value(0.0),
                                        ],
                                        func=Attribute(
                                            Name("math_ops", Load()),
                                            "maximum",
                                            Load(),
                                        ),
                                        keywords=[],
                                        expr=None,
                                        expr_func=None,
                                    )
                                ],
                                func=Attribute(
                                    Name("math_ops", Load()),
                                    "square",
                                    Load(),
                                ),
                                keywords=[],
                                expr=None,
                                expr_func=None,
                            )
                        ],
                        func=Attribute(Name("K", Load()), "mean", Load()),
                        keywords=[
                            keyword(
                                arg="axis",
                                value=UnaryOp(USub(), set_value(1)),
                                identifier=None,
                            )
                        ],
                        expr=None,
                        expr_func=None,
                    ),
                    expr=None,
                ),
            ],
            decorator_list=[],
            name="__call__",
            returns=None,
            arguments_args=None,
            identifier_name=None,
            stmt=None,
            lineno=None,
        ),
    ],
    decorator_list=[],
    keywords=[],
    expr=None,
    identifier_name=None,
    name="SquaredHingeConfig",
)

__all__ = ["class_ast", "class_str", "class_nargs_ast", "class_nargs_str"]
