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

class_with_method_str = '''
class C(object):
    """ C class (mocked!) """

    def method_name(
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

    def method_name(
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

    def method_name(
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

    def method_name(
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

class_with_method_and_body_types_ast = ClassDef(
    bases=[Name(ctx=Load(), id="object")],
    body=[
        Expr(value=Constant(kind=None, value=" C class (mocked!) ")),
        FunctionDef(
            args=arguments(
                args=[
                    arg(annotation=None, arg="self", type_comment=None),
                    arg(
                        annotation=Name(ctx=Load(), id="str"),
                        arg="dataset_name",
                        type_comment=None,
                    ),
                    arg(
                        annotation=Subscript(
                            ctx=Load(),
                            slice=Index(value=Name(ctx=Load(), id="str")),
                            value=Name(ctx=Load(), id="Optional"),
                        ),
                        arg="tfds_dir",
                        type_comment=None,
                    ),
                    arg(
                        annotation=Subscript(
                            ctx=Load(),
                            slice=Index(
                                value=Tuple(
                                    ctx=Load(),
                                    elts=[
                                        Constant(kind=None, value="np"),
                                        Constant(kind=None, value="tf"),
                                    ],
                                )
                            ),
                            value=Name(ctx=Load(), id="Literal"),
                        ),
                        arg="K",
                        type_comment=None,
                    ),
                    arg(
                        annotation=Subscript(
                            ctx=Load(),
                            slice=Index(value=Name(ctx=Load(), id="bool")),
                            value=Name(ctx=Load(), id="Optional"),
                        ),
                        arg="as_numpy",
                        type_comment=None,
                    ),
                ],
                defaults=[
                    Constant(kind=None, value="mnist"),
                    Constant(kind=None, value="~/tensorflow_datasets"),
                    Constant(kind=None, value="np"),
                    Constant(kind=None, value=None),
                ],
                kw_defaults=[],
                kwarg=arg(annotation=None, arg="data_loader_kwargs", type_comment=None),
                kwonlyargs=[],
                posonlyargs=[],
                vararg=None,
            ),
            body=[
                Expr(
                    value=Constant(
                        kind=None,
                        value="\n        Acquire from the official tensorflow_datasets model zoo,"
                        " or the ophthalmology focussed ml-prepare library\n\n        "
                        ":param dataset_name: name of dataset.\n\n        "
                        ":param tfds_dir: directory to look for models in.\n\n        "
                        ":param K: backend engine, e.g., `np` or `tf`.\n\n        "
                        ":param as_numpy: Convert to numpy ndarrays\n\n        "
                        ":param data_loader_kwargs: pass this as arguments to data_loader function\n\n        "
                        ":return: Train and tests dataset splits.\n        ",
                    )
                ),
                Expr(
                    value=Call(
                        args=[
                            BinOp(
                                left=Constant(kind=None, value=5),
                                op=Mult(),
                                right=Constant(kind=None, value=5),
                            )
                        ],
                        func=Name(ctx=Load(), id="print"),
                        keywords=[],
                    )
                ),
                If(
                    body=[
                        Expr(
                            value=Call(
                                args=[Constant(kind=None, value=True)],
                                func=Name(ctx=Load(), id="print"),
                                keywords=[],
                            )
                        ),
                        Return(value=Constant(kind=None, value=5)),
                    ],
                    orelse=[],
                    test=Constant(kind=None, value=True),
                ),
                Return(
                    value=Tuple(
                        ctx=Load(),
                        elts=[
                            Call(
                                args=[Constant(kind=None, value=0)],
                                func=Attribute(
                                    attr="empty",
                                    ctx=Load(),
                                    value=Name(ctx=Load(), id="np"),
                                ),
                                keywords=[],
                            ),
                            Call(
                                args=[Constant(kind=None, value=0)],
                                func=Attribute(
                                    attr="empty",
                                    ctx=Load(),
                                    value=Name(ctx=Load(), id="np"),
                                ),
                                keywords=[],
                            ),
                        ],
                    )
                ),
            ],
            decorator_list=[],
            name="method_name",
            returns=Subscript(
                ctx=Load(),
                slice=Index(
                    value=Tuple(
                        ctx=Load(),
                        elts=[
                            Subscript(
                                ctx=Load(),
                                slice=Index(
                                    value=Tuple(
                                        ctx=Load(),
                                        elts=[
                                            Attribute(
                                                attr="Dataset",
                                                ctx=Load(),
                                                value=Attribute(
                                                    attr="data",
                                                    ctx=Load(),
                                                    value=Name(ctx=Load(), id="tf"),
                                                ),
                                            ),
                                            Attribute(
                                                attr="Dataset",
                                                ctx=Load(),
                                                value=Attribute(
                                                    attr="data",
                                                    ctx=Load(),
                                                    value=Name(ctx=Load(), id="tf"),
                                                ),
                                            ),
                                        ],
                                    )
                                ),
                                value=Name(ctx=Load(), id="Tuple"),
                            ),
                            Subscript(
                                ctx=Load(),
                                slice=Index(
                                    value=Tuple(
                                        ctx=Load(),
                                        elts=[
                                            Attribute(
                                                attr="ndarray",
                                                ctx=Load(),
                                                value=Name(ctx=Load(), id="np"),
                                            ),
                                            Attribute(
                                                attr="ndarray",
                                                ctx=Load(),
                                                value=Name(ctx=Load(), id="np"),
                                            ),
                                        ],
                                    )
                                ),
                                value=Name(ctx=Load(), id="Tuple"),
                            ),
                        ],
                    )
                ),
                value=Name(ctx=Load(), id="Union"),
            ),
            type_comment=None,
            lineno=None,
        ),
    ],
    decorator_list=[],
    keywords=[],
    name="C",
)

class_with_method_ast = (
    ClassDef(
        bases=[Name(ctx=Load(), id="object")],
        body=[
            Expr(value=Constant(kind=None, value=" C class (mocked!) ")),
            FunctionDef(
                args=arguments(
                    args=[
                        arg(annotation=None, arg="self", type_comment=None),
                        arg(annotation=None, arg="dataset_name", type_comment=None),
                        arg(annotation=None, arg="tfds_dir", type_comment=None),
                        arg(annotation=None, arg="K", type_comment=None),
                        arg(annotation=None, arg="as_numpy", type_comment=None),
                    ],
                    defaults=[
                        Constant(kind=None, value="mnist"),
                        Constant(kind=None, value="~/tensorflow_datasets"),
                        Constant(kind=None, value="np"),
                        Constant(kind=None, value=None),
                    ],
                    kw_defaults=[],
                    kwarg=arg(
                        annotation=None, arg="data_loader_kwargs", type_comment=None
                    ),
                    kwonlyargs=[],
                    posonlyargs=[],
                    vararg=None,
                ),
                body=[
                    Expr(
                        value=Constant(
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
                        )
                    ),
                    Return(
                        value=Tuple(
                            ctx=Load(),
                            elts=[
                                Call(
                                    args=[Constant(kind=None, value=0)],
                                    func=Attribute(
                                        attr="empty",
                                        ctx=Load(),
                                        value=Name(ctx=Load(), id="np"),
                                    ),
                                    keywords=[],
                                ),
                                Call(
                                    args=[Constant(kind=None, value=0)],
                                    func=Attribute(
                                        attr="empty",
                                        ctx=Load(),
                                        value=Name(ctx=Load(), id="np"),
                                    ),
                                    keywords=[],
                                ),
                            ],
                        )
                    ),
                ],
                decorator_list=[],
                name="method_name",
                returns=None,
                type_comment=None,
            ),
        ],
        decorator_list=[],
        keywords=[],
        name="C",
    )
    if PY3_8
    else ast.parse(class_with_method_str).body[0]
)

class_with_method_types_ast = (
    ClassDef(
        bases=[Name(ctx=Load(), id="object")],
        body=[
            Expr(value=Constant(kind=None, value=" C class (mocked!) ")),
            FunctionDef(
                args=arguments(
                    args=[
                        arg(annotation=None, arg="self", type_comment=None),
                        arg(
                            annotation=Name(ctx=Load(), id="str"),
                            arg="dataset_name",
                            type_comment=None,
                        ),
                        arg(
                            annotation=Subscript(
                                ctx=Load(),
                                slice=Index(value=Name(ctx=Load(), id="str")),
                                value=Name(ctx=Load(), id="Optional"),
                            ),
                            arg="tfds_dir",
                            type_comment=None,
                        ),
                        arg(
                            annotation=Subscript(
                                ctx=Load(),
                                slice=Index(
                                    value=Tuple(
                                        ctx=Load(),
                                        elts=[
                                            Constant(kind=None, value="np"),
                                            Constant(kind=None, value="tf"),
                                        ],
                                    )
                                ),
                                value=Name(ctx=Load(), id="Literal"),
                            ),
                            arg="K",
                            type_comment=None,
                        ),
                        arg(
                            annotation=Subscript(
                                ctx=Load(),
                                slice=Index(value=Name(ctx=Load(), id="bool")),
                                value=Name(ctx=Load(), id="Optional"),
                            ),
                            arg="as_numpy",
                            type_comment=None,
                        ),
                    ],
                    defaults=[
                        Constant(kind=None, value="mnist"),
                        Constant(kind=None, value="~/tensorflow_datasets"),
                        Constant(kind=None, value="np"),
                        Constant(kind=None, value=None),
                    ],
                    kw_defaults=[],
                    kwarg=arg(
                        annotation=None, arg="data_loader_kwargs", type_comment=None
                    ),
                    kwonlyargs=[],
                    posonlyargs=[],
                    vararg=None,
                ),
                body=[
                    Expr(
                        value=Constant(
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
                        )
                    ),
                    Return(
                        value=Tuple(
                            ctx=Load(),
                            elts=[
                                Call(
                                    args=[Constant(kind=None, value=0)],
                                    func=Attribute(
                                        attr="empty",
                                        ctx=Load(),
                                        value=Name(ctx=Load(), id="np"),
                                    ),
                                    keywords=[],
                                ),
                                Call(
                                    args=[Constant(kind=None, value=0)],
                                    func=Attribute(
                                        attr="empty",
                                        ctx=Load(),
                                        value=Name(ctx=Load(), id="np"),
                                    ),
                                    keywords=[],
                                ),
                            ],
                        )
                    ),
                ],
                decorator_list=[],
                name="method_name",
                returns=Subscript(
                    ctx=Load(),
                    slice=Index(
                        value=Tuple(
                            ctx=Load(),
                            elts=[
                                Subscript(
                                    ctx=Load(),
                                    slice=Index(
                                        value=Tuple(
                                            ctx=Load(),
                                            elts=[
                                                Attribute(
                                                    attr="Dataset",
                                                    ctx=Load(),
                                                    value=Attribute(
                                                        attr="data",
                                                        ctx=Load(),
                                                        value=Name(ctx=Load(), id="tf"),
                                                    ),
                                                ),
                                                Attribute(
                                                    attr="Dataset",
                                                    ctx=Load(),
                                                    value=Attribute(
                                                        attr="data",
                                                        ctx=Load(),
                                                        value=Name(ctx=Load(), id="tf"),
                                                    ),
                                                ),
                                            ],
                                        )
                                    ),
                                    value=Name(ctx=Load(), id="Tuple"),
                                ),
                                Subscript(
                                    ctx=Load(),
                                    slice=Index(
                                        value=Tuple(
                                            ctx=Load(),
                                            elts=[
                                                Attribute(
                                                    attr="ndarray",
                                                    ctx=Load(),
                                                    value=Name(ctx=Load(), id="np"),
                                                ),
                                                Attribute(
                                                    attr="ndarray",
                                                    ctx=Load(),
                                                    value=Name(ctx=Load(), id="np"),
                                                ),
                                            ],
                                        )
                                    ),
                                    value=Name(ctx=Load(), id="Tuple"),
                                ),
                            ],
                        )
                    ),
                    value=Name(ctx=Load(), id="Union"),
                ),
                type_comment=None,
            ),
        ],
        decorator_list=[],
        keywords=[],
        name="C",
    )
    if PY3_8
    else ast.parse(class_with_method_types_str).body[0]
)

class_with_optional_arg_method_ast = (
    ClassDef(
        bases=[Name(ctx=Load(), id="object")],
        body=[
            Expr(value=Constant(kind=None, value=" C class (mocked!) ")),
            FunctionDef(
                args=arguments(
                    args=[
                        arg(annotation=None, arg="self", type_comment=None),
                        arg(
                            annotation=Name(ctx=Load(), id="str"),
                            arg="dataset_name",
                            type_comment=None,
                        ),
                        arg(
                            annotation=Subscript(
                                ctx=Load(),
                                slice=Index(
                                    value=Subscript(
                                        ctx=Load(),
                                        slice=Index(
                                            value=Tuple(
                                                ctx=Load(),
                                                elts=[
                                                    Constant(kind=None, value="np"),
                                                    Constant(kind=None, value="tf"),
                                                ],
                                            )
                                        ),
                                        value=Name(ctx=Load(), id="Literal"),
                                    )
                                ),
                                value=Name(ctx=Load(), id="Optional"),
                            ),
                            arg="K",
                            type_comment=None,
                        ),
                    ],
                    defaults=[Constant(kind=None, value=None)],
                    kw_defaults=[],
                    kwarg=None,
                    kwonlyargs=[],
                    posonlyargs=[],
                    vararg=None,
                ),
                body=[
                    Expr(
                        value=Constant(
                            kind=None,
                            value="\n        Acquire from the official tensorflow_datasets model zoo,"
                            " or the ophthalmology focussed ml-prepare library\n\n        "
                            ":param dataset_name: name of dataset.\n\n        "
                            ":param K: backend engine, e.g., `np` or `tf`.\n\n        "
                            ":return: Train and tests dataset splits.\n        ",
                        )
                    ),
                    Return(
                        value=Tuple(
                            ctx=Load(),
                            elts=[
                                Call(
                                    args=[Constant(kind=None, value=0)],
                                    func=Attribute(
                                        attr="empty",
                                        ctx=Load(),
                                        value=Name(ctx=Load(), id="np"),
                                    ),
                                    keywords=[],
                                ),
                                Call(
                                    args=[Constant(kind=None, value=0)],
                                    func=Attribute(
                                        attr="empty",
                                        ctx=Load(),
                                        value=Name(ctx=Load(), id="np"),
                                    ),
                                    keywords=[],
                                ),
                            ],
                        )
                    ),
                ],
                decorator_list=[],
                name="method_name",
                returns=Subscript(
                    ctx=Load(),
                    slice=Index(
                        value=Tuple(
                            ctx=Load(),
                            elts=[
                                Subscript(
                                    ctx=Load(),
                                    slice=Index(
                                        value=Tuple(
                                            ctx=Load(),
                                            elts=[
                                                Attribute(
                                                    attr="Dataset",
                                                    ctx=Load(),
                                                    value=Attribute(
                                                        attr="data",
                                                        ctx=Load(),
                                                        value=Name(ctx=Load(), id="tf"),
                                                    ),
                                                ),
                                                Attribute(
                                                    attr="Dataset",
                                                    ctx=Load(),
                                                    value=Attribute(
                                                        attr="data",
                                                        ctx=Load(),
                                                        value=Name(ctx=Load(), id="tf"),
                                                    ),
                                                ),
                                            ],
                                        )
                                    ),
                                    value=Name(ctx=Load(), id="Tuple"),
                                ),
                                Subscript(
                                    ctx=Load(),
                                    slice=Index(
                                        value=Tuple(
                                            ctx=Load(),
                                            elts=[
                                                Attribute(
                                                    attr="ndarray",
                                                    ctx=Load(),
                                                    value=Name(ctx=Load(), id="np"),
                                                ),
                                                Attribute(
                                                    attr="ndarray",
                                                    ctx=Load(),
                                                    value=Name(ctx=Load(), id="np"),
                                                ),
                                            ],
                                        )
                                    ),
                                    value=Name(ctx=Load(), id="Tuple"),
                                ),
                            ],
                        )
                    ),
                    value=Name(ctx=Load(), id="Union"),
                ),
                type_comment=None,
            ),
        ],
        decorator_list=[],
        keywords=[],
        name="C",
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
