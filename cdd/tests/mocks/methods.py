"""
Mocks for methods

Note: TensorFlow code is taken from `5a56eb1`; the same that tf 2.15.0 was released with on 14/11/2023.
"""

from ast import (
    Assign,
    Attribute,
    BinOp,
    Call,
    ClassDef,
    Expr,
    FunctionDef,
    If,
    Index,
    Load,
    Mult,
    Name,
    Pass,
    Return,
    Store,
    Subscript,
    Tuple,
    arguments,
    fix_missing_locations,
    keyword,
)
from functools import partial
from operator import add
from textwrap import indent

from cdd.shared.ast_utils import maybe_type_comment, set_arg, set_slice, set_value
from cdd.shared.pure_utils import emit_separating_tabs, tab
from cdd.tests.mocks.docstrings import (
    docstring_google_keras_adadelta,
    docstring_google_tf_mean_squared_error_str,
    docstring_google_tf_ops_losses__safe_mean_str,
    docstring_header_str,
    docstring_no_default_doc_wrapped_str,
    docstring_no_type_no_default_str,
    docstring_str,
)

return_ast: Return = Return(
    value=Tuple(
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
    expr=None,
)

class_with_method_str: str = (
    '''
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
{header_doc_str}{sep}
        :param dataset_name: name of dataset.
        :type dataset_name: ```str```
{sep}
        :param tfds_dir: directory to look for models in.
        :type tfds_dir: ```str```
{sep}
        :param K: backend engine, e.g., `np` or `tf`.
        :type K: ```Literal['np', 'tf']```
{sep}
        :param as_numpy: Convert to numpy ndarrays
        :type as_numpy: ```Optional[bool]```
{sep}
        :param data_loader_kwargs: pass this as arguments to data_loader function
        :type data_loader_kwargs: ```Optional[dict]```
{sep}
        :return: Train and tests dataset splits.
        :rtype: ```Union[Tuple[tf.data.Dataset, tf.data.Dataset], Tuple[np.ndarray,
{indent}np.ndarray]]```
        """
        return np.empty(0), np.empty(0)
'''.format(
        header_doc_str=indent(docstring_header_str, tab * 2),
        sep=tab * 2,
        indent=" " * 12,
    )
)

class_with_method_types_str: str = (
    '''
class C(object):
    """ C class (mocked!) """

    def function_name(
        self,
        dataset_name: str = "mnist",
        tfds_dir: str = "~/tensorflow_datasets",
        K: Literal["np", "tf"] = "np",
        as_numpy: Optional[bool] = None,
        **data_loader_kwargs
    ) -> Union[Tuple[tf.data.Dataset, tf.data.Dataset], Tuple[np.ndarray, np.ndarray]]:
        """
{header_doc_str}
        :param dataset_name: name of dataset.

        :param tfds_dir: directory to look for models in.

        :param K: backend engine, e.g., `np` or `tf`.

        :param as_numpy: Convert to numpy ndarrays

        :param data_loader_kwargs: pass this as arguments to data_loader function

        :return: Train and tests dataset splits.
        """
        return np.empty(0), np.empty(0)
'''.format(
        header_doc_str=indent(docstring_header_str, tab * 2)
    )
)

class_with_method_and_body_types_str: str = (
    '''
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
        {header_doc_str}

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
'''.format(
        header_doc_str=indent(docstring_header_str, tab * 2)
    )
)

class_with_optional_arg_method_str: str = (
    '''
class C(object):
    """ C class (mocked!) """

    def function_name(
        self,
        dataset_name: str,
        K: Optional[Literal["np", "tf"]] = None
    ) -> Union[Tuple[tf.data.Dataset, tf.data.Dataset], Tuple[np.ndarray, np.ndarray]]:
        """
        {header_doc_str}

        :param dataset_name: name of dataset.

        :param K: backend engine, e.g., `np` or `tf`.

        :return: Train and tests dataset splits.
        """
        return np.empty(0), np.empty(0)
'''.format(
        header_doc_str=indent(docstring_header_str, tab * 2)
    )
)


returns_subscript: Subscript = Subscript(
    Name("Union", Load(), lineno=None, col_offset=None),
    set_slice(
        Tuple(
            [
                Subscript(
                    Name("Tuple", Load(), lineno=None, col_offset=None),
                    set_slice(
                        Tuple(
                            [
                                Attribute(
                                    Attribute(
                                        Name(
                                            "tf", Load(), lineno=None, col_offset=None
                                        ),
                                        "data",
                                        Load(),
                                        lineno=None,
                                        col_offset=None,
                                    ),
                                    "Dataset",
                                    Load(),
                                    lineno=None,
                                    col_offset=None,
                                )
                            ]
                            * 2,
                            Load(),
                            expr=None,
                            lineno=None,
                            col_offset=None,
                        )
                    ),
                    Load(),
                    lineno=None,
                    col_offset=None,
                ),
                Subscript(
                    Name("Tuple", Load(), lineno=None, col_offset=None),
                    set_slice(
                        Tuple(
                            [
                                Attribute(
                                    Name("np", Load(), lineno=None, col_offset=None),
                                    "ndarray",
                                    Load(),
                                    lineno=None,
                                    col_offset=None,
                                )
                            ]
                            * 2,
                            Load(),
                            expr=None,
                            lineno=None,
                            col_offset=None,
                        )
                    ),
                    Load(),
                    lineno=None,
                    col_offset=None,
                ),
            ],
            Load(),
            lineno=None,
            col_offset=None,
        )
    ),
    Load(),
    lineno=None,
    col_offset=None,
)


class_with_method_and_body_types_ast: ClassDef = fix_missing_locations(
    ClassDef(
        name="C",
        bases=[Name("object", Load(), lineno=None, col_offset=None)],
        keywords=[],
        body=[
            Expr(set_value(" C class (mocked!) "), lineno=None, col_offset=None),
            FunctionDef(
                name="function_name",
                args=arguments(
                    posonlyargs=[],
                    vararg=None,
                    args=[
                        set_arg("self"),
                        set_arg(
                            arg="dataset_name",
                            annotation=Name(
                                "str", Load(), lineno=None, col_offset=None
                            ),
                        ),
                        set_arg(
                            arg="tfds_dir",
                            annotation=Subscript(
                                Name("Optional", Load(), lineno=None, col_offset=None),
                                set_slice(
                                    Name("str", Load(), lineno=None, col_offset=None)
                                ),
                                Load(),
                                lineno=None,
                                col_offset=None,
                            ),
                        ),
                        set_arg(
                            arg="K",
                            annotation=Subscript(
                                Name("Literal", Load(), lineno=None, col_offset=None),
                                set_slice(
                                    Tuple(
                                        elts=list(map(set_value, ("np", "tf"))),
                                        ctx=Load(),
                                        expr=None,
                                        lineno=None,
                                        col_offset=None,
                                    )
                                ),
                                Load(),
                                lineno=None,
                                col_offset=None,
                            ),
                        ),
                        set_arg(
                            arg="as_numpy",
                            annotation=Subscript(
                                Name("Optional", Load(), lineno=None, col_offset=None),
                                set_slice(
                                    Name("bool", Load(), lineno=None, col_offset=None)
                                ),
                                Load(),
                                lineno=None,
                                col_offset=None,
                            ),
                        ),
                    ],
                    kwonlyargs=[],
                    kw_defaults=[],
                    kwarg=set_arg("data_loader_kwargs"),
                    defaults=list(
                        map(set_value, ("mnist", "~/tensorflow_datasets", "np", None))
                    ),
                    arg=None,
                ),
                body=[
                    Expr(
                        set_value(
                            docstring_no_type_no_default_str,
                        ),
                        lineno=None,
                        col_offset=None,
                    ),
                    Expr(
                        Call(
                            func=Name("print", Load(), lineno=None, col_offset=None),
                            args=[
                                BinOp(
                                    set_value(5),
                                    Mult(),
                                    set_value(5),
                                )
                            ],
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
                        test=set_value(True),
                        body=[
                            Expr(
                                Call(
                                    func=Name(
                                        "print", Load(), lineno=None, col_offset=None
                                    ),
                                    args=[set_value(True)],
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
                        expr_test=None,
                        stmt=None,
                    ),
                    Return(
                        value=Tuple(
                            elts=[
                                Call(
                                    func=Attribute(
                                        Name(
                                            "np", Load(), lineno=None, col_offset=None
                                        ),
                                        "empty",
                                        Load(),
                                        lineno=None,
                                        col_offset=None,
                                    ),
                                    args=[set_value(0)],
                                    keywords=[],
                                    expr=None,
                                    expr_func=None,
                                    lineno=None,
                                    col_offset=None,
                                )
                            ]
                            * 2,
                            ctx=Load(),
                            expr=None,
                            lineno=None,
                            col_offset=None,
                        ),
                        expr=None,
                    ),
                ],
                decorator_list=[],
                type_params=[],
                returns=returns_subscript,
                arguments_args=None,
                identifier_name=None,
                stmt=None,
                lineno=None,
                **maybe_type_comment
            ),
        ],
        decorator_list=[],
        type_params=[],
        expr=None,
        identifier_name=None,
        lineno=None,
        col_offset=None,
    )
)

class_with_method_ast: ClassDef = fix_missing_locations(
    ClassDef(
        bases=[Name("object", Load(), lineno=None, col_offset=None)],
        body=[
            Expr(
                set_value(
                    " C class (mocked!) ",
                ),
                lineno=None,
                col_offset=None,
            ),
            FunctionDef(
                args=arguments(
                    args=list(
                        map(
                            set_arg,
                            ("self", "dataset_name", "tfds_dir", "K", "as_numpy"),
                        )
                    ),
                    defaults=list(
                        map(set_value, ("mnist", "~/tensorflow_datasets", "np", None))
                    ),
                    kw_defaults=[],
                    kwarg=set_arg("data_loader_kwargs"),
                    kwonlyargs=[],
                    posonlyargs=[],
                    vararg=None,
                    arg=None,
                ),
                body=[
                    Expr(
                        set_value(
                            emit_separating_tabs(
                                indent(docstring_no_default_doc_wrapped_str, tab * 2), 2
                            )
                        ),
                        lineno=None,
                        col_offset=None,
                    ),
                    return_ast,
                ],
                decorator_list=[],
                type_params=[],
                name="function_name",
                returns=None,
                arguments_args=None,
                identifier_name=None,
                stmt=None,
                **maybe_type_comment
            ),
        ],
        decorator_list=[],
        type_params=[],
        keywords=[],
        name="C",
        expr=None,
        identifier_name=None,
        lineno=None,
        col_offset=None,
    )
)

class_with_method_types_ast: ClassDef = fix_missing_locations(
    ClassDef(
        bases=[Name("object", Load(), lineno=None, col_offset=None)],
        body=[
            Expr(
                set_value(
                    " C class (mocked!) ",
                ),
                lineno=None,
                col_offset=None,
            ),
            FunctionDef(
                args=arguments(
                    args=[
                        set_arg("self"),
                        set_arg(
                            annotation=Name(
                                "str",
                                Load(),
                            ),
                            arg="dataset_name",
                        ),
                        set_arg(
                            annotation=Name(
                                "str",
                                Load(),
                            ),
                            arg="tfds_dir",
                        ),
                        set_arg(
                            annotation=Subscript(
                                Name(
                                    "Literal",
                                    Load(),
                                ),
                                Index(
                                    value=Tuple(
                                        ctx=Load(),
                                        elts=[
                                            set_value(
                                                "np",
                                            ),
                                            set_value(
                                                "tf",
                                            ),
                                        ],
                                        expr=None,
                                        lineno=None,
                                        col_offset=None,
                                    )
                                ),
                                Load(),
                                lineno=None,
                                col_offset=None,
                            ),
                            arg="K",
                        ),
                        set_arg(
                            annotation=Subscript(
                                Name(
                                    "Optional",
                                    Load(),
                                ),
                                Index(value=Name("bool", Load())),
                                Load(),
                                lineno=None,
                                col_offset=None,
                            ),
                            arg="as_numpy",
                        ),
                    ],
                    defaults=list(
                        map(set_value, ("mnist", "~/tensorflow_datasets", "np", None))
                    ),
                    kw_defaults=[],
                    kwarg=set_arg("data_loader_kwargs"),
                    kwonlyargs=[],
                    posonlyargs=[],
                    vararg=None,
                    arg=None,
                ),
                body=[
                    Expr(
                        set_value(indent(docstring_no_type_no_default_str, tab * 2)),
                        lineno=None,
                        col_offset=None,
                    ),
                    return_ast,
                ],
                decorator_list=[],
                type_params=[],
                name="function_name",
                returns=returns_subscript,
                arguments_args=None,
                identifier_name=None,
                stmt=None,
                **maybe_type_comment
            ),
        ],
        decorator_list=[],
        type_params=[],
        keywords=[],
        name="C",
        expr=None,
        identifier_name=None,
        lineno=None,
        col_offset=None,
    )
)

class_with_optional_arg_method_ast: ClassDef = ClassDef(
    bases=[Name("object", Load(), lineno=None, col_offset=None)],
    body=[
        Expr(
            set_value(
                " C class (mocked!) ",
            ),
            lineno=None,
            col_offset=None,
        ),
        FunctionDef(
            args=arguments(
                args=[
                    set_arg("self"),
                    set_arg(
                        annotation=Name(
                            "str",
                            Load(),
                        ),
                        arg="dataset_name",
                    ),
                    set_arg(
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
                                            elts=list(
                                                map(
                                                    set_value,
                                                    (
                                                        "np",
                                                        "tf",
                                                    ),
                                                )
                                            ),
                                            expr=None,
                                            lineno=None,
                                            col_offset=None,
                                        )
                                    ),
                                    Load(),
                                    lineno=None,
                                    col_offset=None,
                                )
                            ),
                            Load(),
                            lineno=None,
                            col_offset=None,
                        ),
                        arg="K",
                    ),
                ],
                defaults=[set_value(None)],
                kw_defaults=[],
                kwarg=None,
                kwonlyargs=[],
                posonlyargs=[],
                vararg=None,
                arg=None,
            ),
            body=[
                Expr(
                    set_value(indent(docstring_str, tab * 2)),
                    lineno=None,
                    col_offset=None,
                ),
                return_ast,
            ],
            decorator_list=[],
            type_params=[],
            name="function_name",
            returns=returns_subscript,
            arguments_args=None,
            identifier_name=None,
            stmt=None,
            **maybe_type_comment
        ),
    ],
    decorator_list=[],
    type_params=[],
    keywords=[],
    name="C",
    expr=None,
    identifier_name=None,
    lineno=None,
    col_offset=None,
)

function_adder_str: str = '''
def add_6_5(*, a=6, b=5):
    """
    :param a: first param
    :type a: ```int```

    :param b: second param
    :type b: ```int```

    :return: Aggregated summation of `a` and `b`.
    :rtype: ```int```
    """
    return operator.add(a, b)
'''

function_adder_ast: FunctionDef = FunctionDef(
    name="add_6_5",
    args=arguments(
        vararg=None,
        kwarg=None,
        posonlyargs=[],
        args=[],
        kwonlyargs=list(map(set_arg, ("a", "b"))),
        kw_defaults=list(map(set_value, (6, 5))),
        defaults=[],
        arg=None,
    ),
    body=[
        Expr(
            set_value(
                "\n    :param a: first param\n    "
                ":type a: ```int```\n\n    "
                ":param b: second param\n    "
                ":type b: ```int```\n\n    "
                ":return: Aggregated summation of `a` and `b`.\n    "
                ":rtype: ```int```\n    ",
            ),
            lineno=None,
            col_offset=None,
        ),
        Return(
            value=Call(
                func=Attribute(
                    Name("operator", Load(), lineno=None, col_offset=None),
                    "add",
                    Load(),
                    lineno=None,
                    col_offset=None,
                ),
                args=[
                    Name("a", Load(), lineno=None, col_offset=None),
                    Name("b", Load(), lineno=None, col_offset=None),
                ],
                keywords=[],
                expr=None,
                expr_func=None,
                lineno=None,
                col_offset=None,
            ),
            expr=None,
        ),
    ],
    decorator_list=[],
    type_params=[],
    arguments_args=None,
    identifier_name=None,
    stmt=None,
    lineno=None,
    returns=None,
)

function_default_complex_default_arg_str: str = (
    "def call_peril(dataset_name: str='mnist', writer=stdout):\n\tpass"
)

function_default_complex_default_arg_ast: FunctionDef = FunctionDef(
    name="call_peril",
    args=arguments(
        args=[
            set_arg(
                annotation=Name(
                    "str",
                    Load(),
                ),
                arg="dataset_name",
            ),
            set_arg("writer"),
        ],
        defaults=[
            set_value("mnist"),
            Name("stdout", Load(), lineno=None, col_offset=None),
        ],
        kw_defaults=[],
        kwarg=None,
        kwonlyargs=[],
        posonlyargs=[],
        vararg=None,
        arg=None,
    ),
    body=[Pass()],
    decorator_list=[],
    type_params=[],
    lineno=None,
    arguments_args=None,
    identifier_name=None,
    stmt=None,
)

method_complex_args_variety_str: str = '''
def call_cliff(
    self,
    dataset_name,
    *,
    as_numpy,
    K: Literal["np", "tf"],
    tfds_dir="~/tensorflow_datasets",
    writer=stdout,
    **kwargs
) -> Literal["np", "tf"]:
    """
    Call cliff

    :param dataset_name: name of dataset.

    :param as_numpy: Convert to numpy ndarrays

    :param K: backend engine, e.g., `np` or `tf`.

    :param tfds_dir: directory to look for models in.

    :param writer: IO object to write out to

    :param **kwargs: additional keyword arguments

    :return: backend engine
    """
    return K
'''

method_complex_args_variety_ast: FunctionDef = FunctionDef(
    name="call_cliff",
    args=arguments(
        posonlyargs=[],
        args=list(map(set_arg, ("self", "dataset_name"))),
        kwonlyargs=[
            set_arg("as_numpy"),
            set_arg(
                arg="K",
                annotation=Subscript(
                    Name("Literal", Load(), lineno=None, col_offset=None),
                    Tuple(
                        elts=list(
                            map(
                                set_value,
                                (
                                    "np",
                                    "tf",
                                ),
                            )
                        ),
                        ctx=Load(),
                        expr=None,
                        lineno=None,
                        col_offset=None,
                    ),
                    Load(),
                    lineno=None,
                    col_offset=None,
                ),
            ),
            set_arg("tfds_dir"),
            set_arg("writer"),
        ],
        kw_defaults=[
            None,
            None,
            set_value("~/tensorflow_datasets"),
            Name("stdout", Load(), lineno=None, col_offset=None),
        ],
        kwarg=set_arg("kwargs"),
        defaults=[],
        arg=None,
        vararg=None,
    ),
    body=[
        Expr(
            set_value(
                "\n    Call cliff\n\n    "
                ":param dataset_name: name of dataset.\n\n    "
                ":param as_numpy: Convert to numpy ndarrays\n\n    "
                ":param K: backend engine, e.g., `np` or `tf`.\n\n    "
                ":param tfds_dir: directory to look for models in.\n\n    "
                ":param writer: IO object to write out to\n\n    "
                ":param **kwargs: additional keyword arguments\n\n    "
                ":return: backend engine\n    ",
            ),
            lineno=None,
            col_offset=None,
        ),
        Return(value=Name("K", Load(), lineno=None, col_offset=None), expr=None),
    ],
    decorator_list=[],
    type_params=[],
    returns=Subscript(
        Name("Literal", Load(), lineno=None, col_offset=None),
        Tuple(
            elts=list(
                map(
                    set_value,
                    (
                        "np",
                        "tf",
                    ),
                )
            ),
            ctx=Load(),
            expr=None,
            lineno=None,
            col_offset=None,
        ),
        Load(),
        lineno=None,
        col_offset=None,
    ),
    arguments_args=None,
    identifier_name=None,
    stmt=None,
)

# #####################
# # TensorFlow 2.15.0 #
# #####################
# https://github.com/tensorflow/tensorflow/blob/5a56eb1/tensorflow/python/keras/losses.py#L1433-L1454
function_google_tf_squared_hinge_docstring = (
    "Computes the squared hinge loss between `y_true` and `y_pred`.",
    "",
    "  `loss = mean(square(maximum(1 - y_true * y_pred, 0)), axis=-1)`",
    "",
    "  Standalone usage:",
    "",
    "  >>> y_true = np.random.choice([-1, 1], size=(2, 3))",
    "  >>> y_pred = np.random.random(size=(2, 3))",
    "  >>> loss = tf.keras.losses.squared_hinge(y_true, y_pred)",
    "  >>> assert loss.shape == (2,)",
    "  >>> assert np.array_equal(",
    "  ...     loss.numpy(),",
    "  ...     np.mean(np.square(np.maximum(1. - y_true * y_pred, 0.)), axis=-1))",
    "",
    "  Args:",
    "    y_true: The ground truth values. `y_true` values are expected to be -1 or 1.",
    "      If binary (0 or 1) labels are provided we will convert them to -1 or 1.",
    "      shape = `[batch_size, d0, .. dN]`.",
    "    y_pred: The predicted values. shape = `[batch_size, d0, .. dN]`.",
    "",
    "  Returns:",
    "     Squared hinge loss values. shape = `[batch_size, d0, .. dN-1]`.",
    "  ",
)
function_google_tf_squared_hinge_docstring_str: str = "\n".join(
    function_google_tf_squared_hinge_docstring
)
function_google_tf_squared_hinge = (
    "def squared_hinge(y_true, y_pred):",
    '  """{}"""'.format(function_google_tf_squared_hinge_docstring_str),
    "  "
    + "\n  ".join(
        (
            "y_pred = ops.convert_to_tensor_v2(y_pred)",
            "y_true = math_ops.cast(y_true, y_pred.dtype)",
            "y_true = _maybe_convert_labels(y_true)",
            "return K.mean(",
            "    math_ops.square(math_ops.maximum(1. - y_true * y_pred, 0.)), axis=-1)",
        )
    ),
)

function_google_tf_squared_hinge_str: str = "\n".join(function_google_tf_squared_hinge)

# `from tensorflow.python.ops.losses.losses_impl import _safe_mean` @ tf-nightly:2.7.0.dev20210908
function_google_tf_ops_losses__safe_mean_ast: FunctionDef = FunctionDef(
    name="_safe_mean",
    args=arguments(
        args=list(map(set_arg, ("losses", "num_present"))),
        arg=None,
        defaults=[],
        kw_defaults=[],
        kwarg=None,
        kwonlyargs=[],
        posonlyargs=[],
        vararg=None,
    ),
    body=[
        Expr(
            value=set_value(docstring_google_tf_ops_losses__safe_mean_str),
            lineno=None,
            col_offset=None,
        ),
        Assign(
            targets=[Name(id="total_loss", ctx=Store(), lineno=None, col_offset=None)],
            value=Call(
                func=Attribute(
                    value=Name(id="math_ops", ctx=Load(), lineno=None, col_offset=None),
                    attr="reduce_sum",
                    ctx=Load(),
                    lineno=None,
                    col_offset=None,
                ),
                args=[Name(id="losses", ctx=Load(), lineno=None, col_offset=None)],
                keywords=[],
                lineno=None,
                col_offset=None,
            ),
            **maybe_type_comment
        ),
        Return(
            value=Call(
                func=Attribute(
                    value=Name(id="math_ops", ctx=Load(), lineno=None, col_offset=None),
                    attr="div_no_nan",
                    ctx=Load(),
                    lineno=None,
                    col_offset=None,
                ),
                args=[
                    Name(id="total_loss", ctx=Load(), lineno=None, col_offset=None),
                    Name(id="num_present", ctx=Load(), lineno=None, col_offset=None),
                ],
                keywords=[
                    keyword(arg="name", value=set_value("value"), identifier=None)
                ],
                lineno=None,
                col_offset=None,
            )
        ),
    ],
    decorator_list=[],
)

# ```py
# import ast
# import inspect
# from tensorflow.python.ops.losses.losses_impl import mean_squared_error
#
# print(ast.dump(ast.parse(inspect.getsource(mean_squared_error)).body[0], indent=4)
# ```
# #####################
# # TensorFlow 2.15.0 #
# #####################
# https://github.com/tensorflow/tensorflow/blob/5a56eb1/tensorflow/python/ops/losses/losses_impl.py#L627-L755
# - (minus non-docstring body and `decorator_list`)
function_google_tf_mean_squared_error_ast: FunctionDef = FunctionDef(
    name="mean_squared_error",
    args=arguments(
        posonlyargs=[],
        args=list(
            map(
                set_arg,
                (
                    "labels",
                    "predictions",
                    "weights",
                    "scope",
                    "loss_collection",
                    "reduction",
                ),
            )
        ),
        kwonlyargs=[],
        kw_defaults=[],
        defaults=[
            set_value(1.0),
            set_value(None),
            Attribute(
                value=Attribute(
                    value=Name(id="ops", ctx=Load(), lineno=None, col_offset=None),
                    attr="GraphKeys",
                    ctx=Load(),
                    lineno=None,
                    col_offset=None,
                ),
                attr="LOSSES",
                ctx=Load(),
                lineno=None,
                col_offset=None,
            ),
            Attribute(
                value=Name(id="Reduction", ctx=Load(), lineno=None, col_offset=None),
                attr="SUM_BY_NONZERO_WEIGHTS",
                ctx=Load(),
                lineno=None,
                col_offset=None,
            ),
        ],
        vararg=None,
        kwarg=None,
        arg=None,
    ),
    body=[
        Expr(
            value=set_value(docstring_google_tf_mean_squared_error_str),
            lineno=None,
            col_offset=None,
        )
    ],
    decorator_list=[],
)

docstring_google_tf_adadelta_function = (
    "",
    "class Adadelta(object):",
    '{tab}"""{docstr}"""'.format(
        tab=tab,
        docstr="\n".join(map(partial(add, " " * 2), docstring_google_keras_adadelta)),
    ),
    "",
    "    def __init__(",
    "        self,",
    "        learning_rate=0.001,",
    "        rho=0.95,",
    "        epsilon=1e-7,",
    "        weight_decay=None,",
    "        clipnorm=None,",
    "        clipvalue=None,",
    "        global_clipnorm=None,",
    "        use_ema=False,",
    "        ema_momentum=0.99,",
    "        ema_overwrite_frequency=None,",
    '        name="adadelta",',
    "        **kwargs,",
    "    ):",
    "        super().__init__(",
    "            learning_rate=learning_rate,",
    "            weight_decay=weight_decay,",
    "            clipnorm=clipnorm,",
    "            clipvalue=clipvalue,",
    "            global_clipnorm=global_clipnorm,",
    "            use_ema=use_ema,",
    "            ema_momentum=ema_momentum,",
    "            ema_overwrite_frequency=ema_overwrite_frequency,",
    "            name=name,",
    "            **kwargs,",
    "        )",
    "        self.rho = rho",
    "        self.epsilon = epsilon",
    "",
    "    def build(self, var_list):",
    "        if self.built:",
    "            return",
    "        super().build(var_list)",
    "        self._accumulated_grads = []",
    "        self._accumulated_delta_vars = []",
    "        for var in var_list:",
    "            self._accumulated_grads.append(",
    '                self.add_variable_from_reference(var, "accumulated_grad")',
    "            )",
    "            self._accumulated_delta_vars.append(",
    '                self.add_variable_from_reference(var, "accumulated_delta_var")',
    "            )",
    "",
    "    def update_step(self, grad, variable, learning_rate):",
    '        """Update step given gradient and the associated model variable."""',
    "        lr = ops.cast(learning_rate, variable.dtype)",
    "        grad = ops.cast(grad, variable.dtype)",
    "",
    "        rho = self.rho",
    "        accumulated_grad = self._accumulated_grads[",
    "            self._get_variable_index(variable)",
    "        ]",
    "        accumulated_delta_var = self._accumulated_delta_vars[",
    "            self._get_variable_index(variable)",
    "        ]",
    "",
    "        def rms(x):",
    "            return ops.sqrt(ops.add(x, self.epsilon))",
    "",
    "        self.assign(",
    "            accumulated_grad,",
    "            ops.add(",
    "                rho * accumulated_grad, ops.multiply(1 - rho, ops.square(grad))",
    "            ),",
    "        )",
    "        delta_var = ops.negative(",
    "            ops.divide(",
    "                ops.multiply(rms(accumulated_delta_var), grad),",
    "                rms(accumulated_grad),",
    "            )",
    "        )",
    "        self.assign(",
    "            accumulated_delta_var,",
    "            ops.add(",
    "                ops.multiply(rho, accumulated_delta_var),",
    "                ops.multiply(1 - rho, ops.square(delta_var)),",
    "            ),",
    "        )",
    "        self.assign_add(variable, ops.multiply(lr, delta_var))",
    "",
    "    def get_config(self):",
    "        config = super().get_config()",
    "",
    "        config.update(",
    "            {",
    '                "rho": self.rho,',
    '                "epsilon": self.epsilon,',
    "            }",
    "        )",
    "        return config",
)
docstring_google_keras_adadelta_function_str: str = "\n".join(
    docstring_google_tf_adadelta_function
)

__all__ = [
    "class_with_method_and_body_types_ast",
    "class_with_method_and_body_types_str",
    "class_with_method_ast",
    "class_with_method_str",
    "class_with_method_types_ast",
    "class_with_method_types_str",
    "class_with_optional_arg_method_ast",
    "class_with_optional_arg_method_str",
    "docstring_google_keras_adadelta_function_str",
    "function_adder_ast",
    "function_adder_str",
    "function_default_complex_default_arg_ast",
    "function_default_complex_default_arg_str",
    "function_google_tf_mean_squared_error_ast",
    "function_google_tf_ops_losses__safe_mean_ast",
    "function_google_tf_squared_hinge_docstring_str",
    "function_google_tf_squared_hinge_str",
    "method_complex_args_variety_ast",
    "method_complex_args_variety_str",
    "returns_subscript",
]  # type: list[str]
