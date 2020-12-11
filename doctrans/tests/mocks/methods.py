"""
Mocks for methods
"""
from ast import (
    Return,
    Tuple,
    Load,
    Call,
    Expr,
    Index,
    arguments,
    FunctionDef,
    ClassDef,
    Attribute,
    Name,
    Subscript,
    BinOp,
    Mult,
    If,
    Pass,
    fix_missing_locations,
)
from functools import partial
from operator import add

from doctrans.ast_utils import set_value, set_slice, set_arg, maybe_type_comment
from doctrans.tests.mocks.docstrings import docstring_google_tf_adadelta_str

return_ast = Return(
    value=Tuple(
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

class_with_method_and_body_types_ast = fix_missing_locations(
    ClassDef(
        name="C",
        bases=[Name("object", Load())],
        keywords=[],
        body=[
            Expr(set_value(" C class (mocked!) ")),
            FunctionDef(
                name="function_name",
                args=arguments(
                    posonlyargs=[],
                    vararg=None,
                    args=[
                        set_arg("self"),
                        set_arg(arg="dataset_name", annotation=Name("str", Load())),
                        set_arg(
                            arg="tfds_dir",
                            annotation=Subscript(
                                Name("Optional", Load()),
                                set_slice(Name("str", Load())),
                                Load(),
                            ),
                        ),
                        set_arg(
                            arg="K",
                            annotation=Subscript(
                                Name("Literal", Load()),
                                set_slice(
                                    Tuple(
                                        elts=list(map(set_value, ("np", "tf"))),
                                        ctx=Load(),
                                        expr=None,
                                    )
                                ),
                                Load(),
                            ),
                        ),
                        set_arg(
                            arg="as_numpy",
                            annotation=Subscript(
                                Name("Optional", Load()),
                                set_slice(Name("bool", Load())),
                                Load(),
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
                            "\n        Acquire from the official tensorflow_datasets model zoo,"
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
                        Call(
                            func=Name("print", Load()),
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
                        )
                    ),
                    If(
                        test=set_value(True),
                        body=[
                            Expr(
                                Call(
                                    func=Name("print", Load()),
                                    args=[set_value(True)],
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
                        expr_test=None,
                        stmt=None,
                    ),
                    Return(
                        value=Tuple(
                            elts=[
                                Call(
                                    func=Attribute(Name("np", Load()), "empty", Load()),
                                    args=[set_value(0)],
                                    keywords=[],
                                    expr=None,
                                    expr_func=None,
                                )
                            ]
                            * 2,
                            ctx=Load(),
                            expr=None,
                        ),
                        expr=None,
                    ),
                ],
                decorator_list=[],
                returns=Subscript(
                    Name("Union", Load()),
                    set_slice(
                        Tuple(
                            [
                                Subscript(
                                    Name("Tuple", Load()),
                                    set_slice(
                                        Tuple(
                                            [
                                                Attribute(
                                                    Attribute(
                                                        Name("tf", Load()),
                                                        "data",
                                                        Load(),
                                                    ),
                                                    "Dataset",
                                                    Load(),
                                                )
                                            ]
                                            * 2,
                                            Load(),
                                            expr=None,
                                        )
                                    ),
                                    Load(),
                                ),
                                Subscript(
                                    Name("Tuple", Load()),
                                    set_slice(
                                        Tuple(
                                            [
                                                Attribute(
                                                    Name("np", Load()),
                                                    "ndarray",
                                                    Load(),
                                                )
                                            ]
                                            * 2,
                                            Load(),
                                            expr=None,
                                        )
                                    ),
                                    Load(),
                                ),
                            ],
                            Load(),
                        )
                    ),
                    Load(),
                ),
                arguments_args=None,
                identifier_name=None,
                stmt=None,
                lineno=None,
                **maybe_type_comment
            ),
        ],
        decorator_list=[],
        expr=None,
        identifier_name=None,
    )
)


class_with_method_ast = fix_missing_locations(
    ClassDef(
        bases=[Name("object", Load())],
        body=[
            Expr(
                set_value(
                    " C class (mocked!) ",
                )
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
                            "\n        Acquire from the official tensorflow_datasets model zoo,"
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
                    return_ast,
                ],
                decorator_list=[],
                name="function_name",
                returns=None,
                arguments_args=None,
                identifier_name=None,
                stmt=None,
                **maybe_type_comment
            ),
        ],
        decorator_list=[],
        keywords=[],
        name="C",
        expr=None,
        identifier_name=None,
    )
)

class_with_method_types_ast = fix_missing_locations(
    ClassDef(
        bases=[Name("object", Load())],
        body=[
            Expr(
                set_value(
                    " C class (mocked!) ",
                )
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
                                    value=Name(
                                        "str",
                                        Load(),
                                    )
                                ),
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
                                    )
                                ),
                                Load(),
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
                        set_value(
                            "\n        Acquire from the official tensorflow_datasets"
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
                                                )
                                            ]
                                            * 2,
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
                                                )
                                            ]
                                            * 2,
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
                arguments_args=None,
                identifier_name=None,
                stmt=None,
                **maybe_type_comment
            ),
        ],
        decorator_list=[],
        keywords=[],
        name="C",
        expr=None,
        identifier_name=None,
    )
)

class_with_optional_arg_method_ast = ClassDef(
    bases=[Name("object", Load())],
    body=[
        Expr(
            set_value(
                " C class (mocked!) ",
            )
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
                                        )
                                    ),
                                    Load(),
                                )
                            ),
                            Load(),
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
                    set_value(
                        "\n        Acquire from the official tensorflow_datasets model zoo,"
                        " or the ophthalmology focussed ml-prepare library\n\n        "
                        ":param dataset_name: name of dataset.\n\n        "
                        ":param K: backend engine, e.g., `np` or `tf`.\n\n        "
                        ":return: Train and tests dataset splits.\n        ",
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
                                            )
                                        ]
                                        * 2,
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
                                            )
                                        ]
                                        * 2,
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
            arguments_args=None,
            identifier_name=None,
            stmt=None,
            **maybe_type_comment
        ),
    ],
    decorator_list=[],
    keywords=[],
    name="C",
    expr=None,
    identifier_name=None,
)

function_adder_str = '''
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

function_adder_ast = FunctionDef(
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
            )
        ),
        Return(
            value=Call(
                func=Attribute(Name("operator", Load()), "add", Load()),
                args=[Name("a", Load()), Name("b", Load())],
                keywords=[],
                expr=None,
                expr_func=None,
            ),
            expr=None,
        ),
    ],
    decorator_list=[],
    arguments_args=None,
    identifier_name=None,
    stmt=None,
)

function_adder_ir = {
    "name": "add_6_5",
    "params": [
        {"default": 6, "doc": "first param", "name": "a", "typ": "int"},
        {"default": 5, "doc": "second param", "name": "b", "typ": "int"},
    ],
    "returns": {
        "default": "```operator.add(a, b)```",
        "doc": "Aggregated summation of `a` and `b`.",
        "name": "return_type",
        "typ": "int",
    },
    "doc": "",
    "type": "static",
}

function_default_complex_default_arg_str = (
    "def call_peril(dataset_name: str='mnist', writer=stdout):\n\tpass"
)

function_default_complex_default_arg_ast = FunctionDef(
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
            set_value(
                "mnist",
            ),
            Name("stdout", Load()),
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
    lineno=None,
    arguments_args=None,
    identifier_name=None,
    stmt=None,
)

method_complex_args_variety_str = '''
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

method_complex_args_variety_ast = FunctionDef(
    name="call_cliff",
    args=arguments(
        posonlyargs=[],
        args=list(map(set_arg, ("self", "dataset_name"))),
        kwonlyargs=[
            set_arg("as_numpy"),
            set_arg(
                arg="K",
                annotation=Subscript(
                    Name("Literal", Load()),
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
                    ),
                    Load(),
                ),
            ),
            set_arg("tfds_dir"),
            set_arg("writer"),
        ],
        kw_defaults=[
            None,
            None,
            set_value(
                "~/tensorflow_datasets",
            ),
            Name("stdout", Load()),
        ],
        kwarg=set_arg("kwargs"),
        defaults=[],
        arg=None,
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
            )
        ),
        Return(value=Name("K", Load()), expr=None),
    ],
    decorator_list=[],
    returns=Subscript(
        Name("Literal", Load()),
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
        ),
        Load(),
    ),
    arguments_args=None,
    identifier_name=None,
    stmt=None,
)

# https://github.com/tensorflow/tensorflow/blob/7ad2723/tensorflow/python/keras/losses.py#L1327-L1355
function_google_tf_squared_hinge_str = '''def squared_hinge(y_true, y_pred):
  """Computes the squared hinge loss between `y_true` and `y_pred`.

  `loss = mean(square(maximum(1 - y_true * y_pred, 0)), axis=-1)`

  Standalone usage:

  >>> y_true = np.random.choice([-1, 1], size=(2, 3))
  >>> y_pred = np.random.random(size=(2, 3))
  >>> loss = tf.keras.losses.squared_hinge(y_true, y_pred)
  >>> assert loss.shape == (2,)
  >>> assert np.array_equal(
  ...     loss.numpy(),
  ...     np.mean(np.square(np.maximum(1. - y_true * y_pred, 0.)), axis=-1))

  Args:
    y_true: The ground truth values. `y_true` values are expected to be -1 or 1.
      If binary (0 or 1) labels are provided we will convert them to -1 or 1.
      shape = `[batch_size, d0, .. dN]`.
    y_pred: The predicted values. shape = `[batch_size, d0, .. dN]`.

  Returns:
     Squared hinge loss values. shape = `[batch_size, d0, .. dN-1]`.
  """
  y_pred = ops.convert_to_tensor_v2(y_pred)
  y_true = math_ops.cast(y_true, y_pred.dtype)
  y_true = _maybe_convert_labels(y_true)
  return K.mean(
      math_ops.square(math_ops.maximum(1. - y_true * y_pred, 0.)), axis=-1)
'''

docstring_google_tf_adadelta_function_str = """
class Adadelta(object):
  \"\"\"\n{docstring_google_tf_adadelta_str}\n  \"\"\"\n{body}""".format(
    docstring_google_tf_adadelta_str="\n".join(
        map(partial(add, " " * 2), docstring_google_tf_adadelta_str.splitlines())
    ),
    body="""
  _HAS_AGGREGATE_GRAD = True

  def __init__(self,
               learning_rate=0.001,
               rho=0.95,
               epsilon=1e-7,
               name='Adadelta',
               **kwargs):
    # super(Adadelta, self).__init__(name, **kwargs)
    self._set_hyper('learning_rate', kwargs.get('lr', learning_rate))
    self._set_hyper('decay', self._initial_decay)
    self._set_hyper('rho', rho)
    self.epsilon = epsilon or backend_config.epsilon()

  def _create_slots(self, var_list):
    # Separate for-loops to respect the ordering of slot variables from v1.
    for v in var_list:
      self.add_slot(v, 'accum_grad')
    for v in var_list:
      self.add_slot(v, 'accum_var')

  def _prepare_local(self, var_device, var_dtype, apply_state):
    # super(Adadelta, self)._prepare_local(var_device, var_dtype, apply_state)
    apply_state[(var_device, var_dtype)].update(
        dict(
            epsilon=ops.convert_to_tensor_v2_with_dispatch(
                self.epsilon, var_dtype),
            rho=array_ops.identity(self._get_hyper('rho', var_dtype))))

  def set_weights(self, weights):
    params = self.weights
    # Override set_weights for backward compatibility of Keras V1 optimizer
    # since it does not include iteration at head of the weight list. Set
    # iteration to 0.
    if len(params) == len(weights) + 1:
      weights = [np.array(0)] + weights
    # super(Adadelta, self).set_weights(weights)

  def _resource_apply_dense(self, grad, var, apply_state=None):
    var_device, var_dtype = var.device, var.dtype.base_dtype
    coefficients = ((apply_state or {}).get((var_device, var_dtype))
                    or self._fallback_apply_state(var_device, var_dtype))

    accum_grad = self.get_slot(var, 'accum_grad')
    accum_var = self.get_slot(var, 'accum_var')
    return gen_training_ops.ResourceApplyAdadelta(
        var=var.handle,
        accum=accum_grad.handle,
        accum_update=accum_var.handle,
        lr=coefficients['lr_t'],
        rho=coefficients['rho'],
        epsilon=coefficients['epsilon'],
        grad=grad,
        use_locking=self._use_locking)

  def _resource_apply_sparse(self, grad, var, indices, apply_state=None):
    var_device, var_dtype = var.device, var.dtype.base_dtype
    coefficients = ((apply_state or {}).get((var_device, var_dtype))
                    or self._fallback_apply_state(var_device, var_dtype))

    accum_grad = self.get_slot(var, 'accum_grad')
    accum_var = self.get_slot(var, 'accum_var')
    return gen_training_ops.ResourceSparseApplyAdadelta(
        var=var.handle,
        accum=accum_grad.handle,
        accum_update=accum_var.handle,
        lr=coefficients['lr_t'],
        rho=coefficients['rho'],
        epsilon=coefficients['epsilon'],
        grad=grad,
        indices=indices,
        use_locking=self._use_locking)

  def get_config(self):
    config = super(Adadelta, self).get_config()
    config.update({
        'learning_rate': self._serialize_hyperparameter('learning_rate'),
        'decay': self._serialize_hyperparameter('decay'),
        'rho': self._serialize_hyperparameter('rho'),
        'epsilon': self.epsilon,
    })
    return config
""",
)

docstring_google_tf_adadelta_function_ir = {
    "doc": "Optimizer that implements the Adadelta algorithm.\n"
    "\n"
    "Adadelta optimization is a stochastic gradient descent method that "
    "is based on\n"
    "adaptive learning rate per dimension to address two drawbacks:\n"
    "\n"
    "- The continual decay of learning rates throughout training\n"
    "- The need for a manually selected global learning rate\n"
    "\n"
    "Adadelta is a more robust extension of Adagrad that adapts "
    "learning rates\n"
    "based on a moving window of gradient updates, instead of "
    "accumulating all\n"
    "past gradients. This way, Adadelta continues learning even when "
    "many updates\n"
    "have been done. Compared to Adagrad, in the original version of "
    "Adadelta you\n"
    "don't have to set an initial learning rate. In this version, "
    "initial\n"
    "learning rate can be set, as in most other Keras optimizers.\n"
    "\n"
    'According to section 4.3 ("Effective Learning rates"), near the '
    "end of\n"
    "training step sizes converge to 1 which is effectively a high "
    "learning\n"
    "rate which would cause divergence. This occurs only near the end "
    "of the\n"
    "training as gradients and step sizes are small, and the epsilon "
    "constant\n"
    "in the numerator and denominator dominate past gradients and "
    "parameter\n"
    "updates which converge the learning rate to 1.\n"
    "\n"
    'According to section 4.4("Speech Data"),where a large neural '
    "network with\n"
    "4 hidden layers was trained on a corpus of US English data, "
    "ADADELTA was\n"
    "used with 100 network replicas.The epsilon used is 1e-6 with "
    "rho=0.95\n"
    "which converged faster than ADAGRAD, by the following "
    "construction:\n"
    "def __init__(self, lr=1.0, rho=0.95, epsilon=1e-6, decay=0., "
    "**kwargs):\n"
    "\n"
    "\n"
    "Reference:\n"
    "    - [Zeiler, 2012](http://arxiv.org/abs/1212.5701)",
    "name": "Adadelta",
    "params": [
        {
            "default": 0.001,
            "doc": "A `Tensor`, floating point value, or a schedule "
            "that is a\n"
            "  "
            "`tf.keras.optimizers.schedules.LearningRateSchedule`. "
            "The learning rate.\n"
            "  To match the exact form in the original paper "
            "use 1.0.",
            "name": "learning_rate",
            "typ": "float",
        },
        {
            "default": 0.95,
            "doc": "A `Tensor` or a floating point value. The decay " "rate.",
            "name": "rho",
            "typ": "float",
        },
        {
            "default": 1e-07,
            "doc": "A `Tensor` or a floating point value.  A "
            "constant epsilon used\n"
            "         to better conditioning the grad update.",
            "name": "epsilon",
            "typ": "float",
        },
        {
            "default": "Adadelta",
            "doc": "Optional name prefix for the operations created "
            "when applying\n"
            "  gradients. ",
            "name": "name",
            "typ": "str",
        },
        {
            "doc": "Keyword arguments. Allowed to be one of\n"
            '  `"clipnorm"` or `"clipvalue"`.\n'
            '  `"clipnorm"` (float) clips gradients by norm; '
            '`"clipvalue"` (float) clips\n'
            "  gradients by value.",
            "name": "kwargs",
            "typ": "dict",
        },
        {"default": True, "name": "_HAS_AGGREGATE_GRAD"},
    ],
    "returns": None,
}

__all__ = [
    "class_with_method_and_body_types_ast",
    "class_with_method_and_body_types_str",
    "class_with_method_ast",
    "class_with_method_str",
    "class_with_method_types_ast",
    "class_with_method_types_str",
    "class_with_optional_arg_method_ast",
    "class_with_optional_arg_method_str",
    "docstring_google_tf_adadelta_function_str",
    "function_adder_ast",
    "function_adder_ir",
    "function_adder_str",
    "function_default_complex_default_arg_ast",
    "function_default_complex_default_arg_str",
    "function_google_tf_squared_hinge_str",
    "method_complex_args_variety_ast",
    "method_complex_args_variety_str",
]
