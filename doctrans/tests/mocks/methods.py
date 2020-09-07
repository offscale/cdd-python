"""
Mocks for methods
"""
import ast
from ast import (
    Return,
    Tuple,
    Load,
    Call,
    Constant,
    Expr,
    Index,
    arguments,
    arg,
    FunctionDef,
    ClassDef,
    Attribute,
    Name,
    Subscript,
    BinOp,
    Mult,
    If,
)

from doctrans.pure_utils import PY3_8

return_ast = Return(
    value=Tuple(
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
    expr=None,
)

class_with_method_str = '''
class C(object):
    """ C class (mocked!) """

    def function_name(
        self,
        dataset_name="mnist",
        tfds_dir="~/tensorflow_datasets",
        K="np",
        as_numpy=None,
        **data_loader_kwargs
    ):
        """
        Acquire from the official tensorflow_datasets model zoo, or the ophthalmology focussed ml-prepare library

        :param dataset_name: name of dataset.
        :type dataset_name: ```str```

        :param tfds_dir: directory to look for models in.
        :type tfds_dir: ```Optional[str]```

        :param K: backend engine, e.g., `np` or `tf`.
        :type K: ```Literal['np', 'tf']```

        :param as_numpy: Convert to numpy ndarrays
        :type as_numpy: ```Optional[bool]```

        :param data_loader_kwargs: pass this as arguments to data_loader function
        :type data_loader_kwargs: ```**data_loader_kwargs```

        :return: Train and tests dataset splits.
        :rtype: ```Union[Tuple[tf.data.Dataset, tf.data.Dataset], Tuple[np.ndarray, np.ndarray]]```
        """
        return np.empty(0), np.empty(0)
'''

class_with_method_types_str = '''
class C(object):
    """ C class (mocked!) """

    def function_name(
        self,
        dataset_name: str = "mnist",
        tfds_dir: Optional[str] = "~/tensorflow_datasets",
        K: Literal["np", "tf"] = "np",
        as_numpy: Optional[bool] = None,
        **data_loader_kwargs
    ) -> Union[Tuple[tf.data.Dataset, tf.data.Dataset], Tuple[np.ndarray, np.ndarray]]:
        """
        Acquire from the official tensorflow_datasets model zoo, or the ophthalmology focussed ml-prepare library

        :param dataset_name: name of dataset.

        :param tfds_dir: directory to look for models in.

        :param K: backend engine, e.g., `np` or `tf`.

        :param as_numpy: Convert to numpy ndarrays

        :param data_loader_kwargs: pass this as arguments to data_loader function

        :return: Train and tests dataset splits.
        """
        return np.empty(0), np.empty(0)
'''

class_with_method_and_body_types_str = '''
class C(object):
    """ C class (mocked!) """

    def function_name(
        self,
        dataset_name: str = "mnist",
        tfds_dir: Optional[str] = "~/tensorflow_datasets",
        K: Literal["np", "tf"] = "np",
        as_numpy: Optional[bool] = None,
        **data_loader_kwargs
    ) -> Union[Tuple[tf.data.Dataset, tf.data.Dataset], Tuple[np.ndarray, np.ndarray]]:
        """
        Acquire from the official tensorflow_datasets model zoo, or the ophthalmology focussed ml-prepare library

        :param dataset_name: name of dataset.

        :param tfds_dir: directory to look for models in.

        :param K: backend engine, e.g., `np` or `tf`.

        :param as_numpy: Convert to numpy ndarrays

        :param data_loader_kwargs: pass this as arguments to data_loader function

        :return: Train and tests dataset splits.
        """
        # some comment
        print(5 * 5)
        if True:
            print(True)
            return 5
        return np.empty(0), np.empty(0)
'''

class_with_optional_arg_method_str = '''
class C(object):
    """ C class (mocked!) """

    def function_name(
        self,
        dataset_name: str,
        K: Optional[Literal["np", "tf"]] = None
    ) -> Union[Tuple[tf.data.Dataset, tf.data.Dataset], Tuple[np.ndarray, np.ndarray]]:
        """
        Acquire from the official tensorflow_datasets model zoo, or the ophthalmology focussed ml-prepare library

        :param dataset_name: name of dataset.

        :param K: backend engine, e.g., `np` or `tf`.

        :return: Train and tests dataset splits.
        """
        return np.empty(0), np.empty(0)
'''

class_with_method_and_body_types_ast = (
    ClassDef(
        name="C",
        bases=[Name("object", Load())],
        keywords=[],
        body=[
            Expr(
                Constant(value=" C class (mocked!) ", constant_value=None, string=None)
            ),
            FunctionDef(
                name="function_name",
                args=arguments(
                    posonlyargs=[],
                    vararg=None,
                    args=[
                        arg(
                            arg="self", annotation=None, expr=None, identifier_arg=None
                        ),
                        arg(
                            arg="dataset_name",
                            annotation=Name("str", Load()),
                            expr=None,
                            identifier_arg=None,
                        ),
                        arg(
                            arg="tfds_dir",
                            annotation=Subscript(
                                Name("Optional", Load()), Name("str", Load()), Load()
                            ),
                            expr=None,
                            identifier_arg=None,
                        ),
                        arg(
                            arg="K",
                            annotation=Subscript(
                                Name("Literal", Load()),
                                Tuple(
                                    elts=[
                                        Constant(
                                            value="np", constant_value=None, string=None
                                        ),
                                        Constant(
                                            value="tf", constant_value=None, string=None
                                        ),
                                    ],
                                    ctx=Load(),
                                    expr=None,
                                ),
                                Load(),
                            ),
                            expr=None,
                            identifier_arg=None,
                        ),
                        arg(
                            arg="as_numpy",
                            annotation=Subscript(
                                Name("Optional", Load()), Name("bool", Load()), Load()
                            ),
                            expr=None,
                            identifier_arg=None,
                        ),
                    ],
                    kwonlyargs=[],
                    kw_defaults=[],
                    kwarg=arg(
                        arg="data_loader_kwargs",
                        annotation=None,
                        expr=None,
                        identifier_arg=None,
                    ),
                    defaults=[
                        Constant(value="mnist", constant_value=None, string=None),
                        Constant(
                            value="~/tensorflow_datasets",
                            constant_value=None,
                            string=None,
                        ),
                        Constant(value="np", constant_value=None, string=None),
                        Constant(value=None, constant_value=None, string=None),
                    ],
                    arg=None,
                ),
                body=[
                    Expr(
                        Constant(
                            value="\n        Acquire from the official tensorflow_datasets model zoo,"
                            " or the ophthalmology focussed ml-prepare library\n\n        "
                            ":param dataset_name: name of dataset.\n\n        "
                            ":param tfds_dir: directory to look for models in.\n\n        "
                            ":param K: backend engine, e.g., `np` or `tf`.\n\n        "
                            ":param as_numpy: Convert to numpy ndarrays\n\n        "
                            ":param data_loader_kwargs: pass this as arguments to data_loader function\n\n        "
                            ":return: Train and tests dataset splits.\n        ",
                            constant_value=None,
                            string=None,
                        )
                    ),
                    Expr(
                        Call(
                            func=Name("print", Load()),
                            args=[
                                BinOp(
                                    Constant(value=5, constant_value=None, string=None),
                                    Mult(),
                                    Constant(value=5, constant_value=None, string=None),
                                )
                            ],
                            keywords=[],
                            expr=None,
                            expr_func=None,
                        )
                    ),
                    If(
                        test=Constant(value=True, constant_value=None, string=None),
                        body=[
                            Expr(
                                Call(
                                    func=Name("print", Load()),
                                    args=[
                                        Constant(
                                            value=True, constant_value=None, string=None
                                        )
                                    ],
                                    keywords=[],
                                    expr=None,
                                    expr_func=None,
                                )
                            ),
                            Return(
                                value=Constant(
                                    value=5, constant_value=None, string=None
                                )
                            ),
                        ],
                        orelse=[],
                        expr_test=None,
                        stmt=None,
                    ),
                    Return(
                        value=Tuple(
                            elts=[
                                Call(
                                    func=Attribute(Name("np", Load()), "empty", Load()),
                                    args=[
                                        Constant(
                                            value=0, constant_value=None, string=None
                                        )
                                    ],
                                    keywords=[],
                                    expr=None,
                                    expr_func=None,
                                ),
                                Call(
                                    func=Attribute(Name("np", Load()), "empty", Load()),
                                    args=[
                                        Constant(
                                            value=0, constant_value=None, string=None
                                        )
                                    ],
                                    keywords=[],
                                    expr=None,
                                    expr_func=None,
                                ),
                            ],
                            ctx=Load(),
                            expr=None,
                        ),
                        expr=None,
                    ),
                ],
                decorator_list=[],
                returns=Subscript(
                    Name("Union", Load()),
                    Tuple(
                        [
                            Subscript(
                                Name("Tuple", Load()),
                                Tuple(
                                    [
                                        Attribute(
                                            Attribute(
                                                Name("tf", Load()), "data", Load()
                                            ),
                                            "Dataset",
                                            Load(),
                                        ),
                                        Attribute(
                                            Attribute(
                                                Name("tf", Load()), "data", Load()
                                            ),
                                            "Dataset",
                                            Load(),
                                        ),
                                    ],
                                    Load(),
                                    expr=None,
                                ),
                                Load(),
                            ),
                            Subscript(
                                Name("Tuple", Load()),
                                Tuple(
                                    [
                                        Attribute(
                                            Name("np", Load()), "ndarray", Load()
                                        ),
                                        Attribute(
                                            Name("np", Load()), "ndarray", Load()
                                        ),
                                    ],
                                    Load(),
                                    expr=None,
                                ),
                                Load(),
                            ),
                        ],
                        Load(),
                    ),
                    Load(),
                ),
                arguments_args=None,
                identifier_name=None,
                stmt=None,
            ),
        ],
        decorator_list=[],
        expr=None,
        identifier_name=None,
    )
    if PY3_8
    else ast.parse(class_with_method_and_body_types_str).body[0]
)

class_with_method_ast = (
    ClassDef(
        bases=[Name("object", Load())],
        body=[
            Expr(
                Constant(
                    kind=None,
                    value=" C class (mocked!) ",
                    constant_value=None,
                    string=None,
                )
            ),
            FunctionDef(
                args=arguments(
                    args=[
                        arg(
                            annotation=None,
                            arg="self",
                            type_comment=None,
                            expr=None,
                            identifier_arg=None,
                        ),
                        arg(
                            annotation=None,
                            arg="dataset_name",
                            type_comment=None,
                            expr=None,
                            identifier_arg=None,
                        ),
                        arg(
                            annotation=None,
                            arg="tfds_dir",
                            type_comment=None,
                            expr=None,
                            identifier_arg=None,
                        ),
                        arg(
                            annotation=None,
                            arg="K",
                            type_comment=None,
                            expr=None,
                            identifier_arg=None,
                        ),
                        arg(
                            annotation=None,
                            arg="as_numpy",
                            type_comment=None,
                            expr=None,
                            identifier_arg=None,
                        ),
                    ],
                    defaults=[
                        Constant(
                            kind=None, value="mnist", constant_value=None, string=None
                        ),
                        Constant(
                            kind=None,
                            value="~/tensorflow_datasets",
                            constant_value=None,
                            string=None,
                        ),
                        Constant(
                            kind=None, value="np", constant_value=None, string=None
                        ),
                        Constant(
                            kind=None, value=None, constant_value=None, string=None
                        ),
                    ],
                    kw_defaults=[],
                    kwarg=arg(
                        annotation=None,
                        arg="data_loader_kwargs",
                        type_comment=None,
                        expr=None,
                        identifier_arg=None,
                    ),
                    kwonlyargs=[],
                    posonlyargs=[],
                    vararg=None,
                    arg=None,
                ),
                body=[
                    Expr(
                        Constant(
                            kind=None,
                            value="\n        Acquire from the official tensorflow_datasets model zoo,"
                            " or the ophthalmology focussed ml-prepare library\n\n        "
                            ":param dataset_name: name of dataset.\n        "
                            ":type dataset_name: ```str```\n\n        "
                            ":param tfds_dir: directory to look for models in.\n        "
                            ":type tfds_dir: ```Optional[str]```\n\n        "
                            ":param K: backend engine, e.g., `np` or `tf`.\n        "
                            ":type K: ```Literal['np', 'tf']```\n\n        "
                            ":param as_numpy: Convert to numpy ndarrays\n        "
                            ":type as_numpy: ```Optional[bool]```\n\n        "
                            ":param data_loader_kwargs: pass this as arguments to data_loader function\n        "
                            ":type data_loader_kwargs: ```**data_loader_kwargs```\n\n        "
                            ":return: Train and tests dataset splits.\n        "
                            ":rtype: ```Union[Tuple[tf.data.Dataset, tf.data.Dataset], Tuple[np.ndarray, np.ndarray]]"
                            "```\n        ",
                            constant_value=None,
                            string=None,
                        )
                    ),
                    return_ast,
                ],
                decorator_list=[],
                name="function_name",
                returns=None,
                type_comment=None,
                arguments_args=None,
                identifier_name=None,
                stmt=None,
            ),
        ],
        decorator_list=[],
        keywords=[],
        name="C",
        expr=None,
        identifier_name=None,
    )
    if PY3_8
    else ast.parse(class_with_method_str).body[0]
)

class_with_method_types_ast = (
    ClassDef(
        bases=[Name("object", Load())],
        body=[
            Expr(
                Constant(
                    kind=None,
                    value=" C class (mocked!) ",
                    constant_value=None,
                    string=None,
                )
            ),
            FunctionDef(
                args=arguments(
                    args=[
                        arg(
                            annotation=None,
                            arg="self",
                            type_comment=None,
                            expr=None,
                            identifier_arg=None,
                        ),
                        arg(
                            annotation=Name(
                                "str",
                                Load(),
                            ),
                            arg="dataset_name",
                            type_comment=None,
                            expr=None,
                            identifier_arg=None,
                        ),
                        arg(
                            annotation=Subscript(
                                Name(
                                    "Optional",
                                    Load(),
                                ),
                                Index(
                                    value=Name(
                                        "str",
                                        Load(),
                                    )
                                ),
                                Load(),
                            ),
                            arg="tfds_dir",
                            type_comment=None,
                            expr=None,
                            identifier_arg=None,
                        ),
                        arg(
                            annotation=Subscript(
                                Name(
                                    "Literal",
                                    Load(),
                                ),
                                Index(
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
                                    )
                                ),
                                Load(),
                            ),
                            arg="K",
                            type_comment=None,
                            expr=None,
                            identifier_arg=None,
                        ),
                        arg(
                            annotation=Subscript(
                                Name(
                                    "Optional",
                                    Load(),
                                ),
                                Index(value=Name("bool", Load())),
                                Load(),
                            ),
                            arg="as_numpy",
                            type_comment=None,
                            expr=None,
                            identifier_arg=None,
                        ),
                    ],
                    defaults=[
                        Constant(
                            kind=None, value="mnist", constant_value=None, string=None
                        ),
                        Constant(
                            kind=None,
                            value="~/tensorflow_datasets",
                            constant_value=None,
                            string=None,
                        ),
                        Constant(
                            kind=None, value="np", constant_value=None, string=None
                        ),
                        Constant(
                            kind=None, value=None, constant_value=None, string=None
                        ),
                    ],
                    kw_defaults=[],
                    kwarg=arg(
                        annotation=None,
                        arg="data_loader_kwargs",
                        type_comment=None,
                        expr=None,
                        identifier_arg=None,
                    ),
                    kwonlyargs=[],
                    posonlyargs=[],
                    vararg=None,
                    arg=None,
                ),
                body=[
                    Expr(
                        Constant(
                            kind=None,
                            value="\n        Acquire from the official tensorflow_datasets"
                            " model zoo, or the ophthalmology focussed ml-prepare library\n"
                            "    \n        "
                            ":param dataset_name: name of dataset.\n    \n        "
                            ":param tfds_dir: directory to look for models in.\n    \n        "
                            ":param K: backend engine, e.g., `np` or `tf`.\n    \n        "
                            ":param as_numpy: Convert to numpy ndarrays\n    \n        "
                            ":param data_loader_kwargs: pass this as arguments to data_loader function\n    \n        "
                            ":return: Train and tests dataset splits.\n        ",
                            constant_value=None,
                            string=None,
                        )
                    ),
                    return_ast,
                ],
                decorator_list=[],
                name="function_name",
                returns=Subscript(
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
                                        value=Tuple(
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
                type_comment=None,
                arguments_args=None,
                identifier_name=None,
                stmt=None,
            ),
        ],
        decorator_list=[],
        keywords=[],
        name="C",
        expr=None,
        identifier_name=None,
    )
    if PY3_8
    else ast.parse(class_with_method_types_str).body[0]
)

class_with_optional_arg_method_ast = (
    ClassDef(
        bases=[Name("object", Load())],
        body=[
            Expr(
                Constant(
                    kind=None,
                    value=" C class (mocked!) ",
                    constant_value=None,
                    string=None,
                )
            ),
            FunctionDef(
                args=arguments(
                    args=[
                        arg(
                            annotation=None,
                            arg="self",
                            type_comment=None,
                            expr=None,
                            identifier_arg=None,
                        ),
                        arg(
                            annotation=Name(
                                "str",
                                Load(),
                            ),
                            arg="dataset_name",
                            type_comment=None,
                            expr=None,
                            identifier_arg=None,
                        ),
                        arg(
                            annotation=Subscript(
                                Name(
                                    "Optional",
                                    Load(),
                                ),
                                Index(
                                    value=Subscript(
                                        Name(
                                            "Literal",
                                            Load(),
                                        ),
                                        Index(
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
                                            )
                                        ),
                                        Load(),
                                    )
                                ),
                                Load(),
                            ),
                            arg="K",
                            type_comment=None,
                            expr=None,
                            identifier_arg=None,
                        ),
                    ],
                    defaults=[
                        Constant(
                            kind=None, value=None, constant_value=None, string=None
                        )
                    ],
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
                            value="\n        Acquire from the official tensorflow_datasets model zoo,"
                            " or the ophthalmology focussed ml-prepare library\n\n        "
                            ":param dataset_name: name of dataset.\n\n        "
                            ":param K: backend engine, e.g., `np` or `tf`.\n\n        "
                            ":return: Train and tests dataset splits.\n        ",
                            constant_value=None,
                            string=None,
                        )
                    ),
                    return_ast,
                ],
                decorator_list=[],
                name="function_name",
                returns=Subscript(
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
                                        value=Tuple(
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
                type_comment=None,
                arguments_args=None,
                identifier_name=None,
                stmt=None,
            ),
        ],
        decorator_list=[],
        keywords=[],
        name="C",
        expr=None,
        identifier_name=None,
    )
    if PY3_8
    else ast.parse(class_with_optional_arg_method_str).body[0]
)

__all__ = [
    "class_with_method_str",
    "class_with_method_types_str",
    "class_with_optional_arg_method_str",
    "class_with_method_and_body_types_str",
    "class_with_method_and_body_types_ast",
    "class_with_method_ast",
    "class_with_method_types_ast",
    "class_with_optional_arg_method_ast",
]
