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
)

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
                    string=None
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

__all__ = ["class_str", "class_ast", "class_nargs_str", "class_nargs_ast"]
