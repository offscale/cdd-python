"""
Mocks for methods
"""

from ast import (
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
    Subscript,
    Tuple,
    arguments,
    fix_missing_locations,
)
from functools import partial
from operator import add
from textwrap import indent

from cdd.ast_utils import maybe_type_comment, set_arg, set_slice, set_value
from cdd.pure_utils import emit_separating_tabs, tab
from cdd.tests.mocks.docstrings import (
    docstring_google_tf_adadelta,
    docstring_header_str,
    docstring_no_default_doc_wrapped_str,
    docstring_no_type_no_default_str,
    docstring_str,
)

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
        :param as_numpy: Convert to numpy ndarrays.
        :type as_numpy: ```Optional[bool]```
{sep}
        :param data_loader_kwargs: pass this as arguments to data_loader function
        :type data_loader_kwargs: ```Optional[dict]```
{sep}
        :returns: Train and tests dataset splits.
        :rtype: ```Union[Tuple[tf.data.Dataset, tf.data.Dataset], Tuple[np.ndarray,
{indent}np.ndarray]]```
        """
        return np.empty(0), np.empty(0)
'''.format(
    header_doc_str=indent(docstring_header_str, tab * 2),
    sep=tab * 2,
    indent=" " * 12,
)

class_with_method_types_str = '''
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

        :param as_numpy: Convert to numpy ndarrays.

        :param data_loader_kwargs: pass this as arguments to data_loader function

        :returns: Train and tests dataset splits.
        """
        return np.empty(0), np.empty(0)
'''.format(
    header_doc_str=indent(docstring_header_str, tab * 2)
)

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
        {header_doc_str}

        :param dataset_name: name of dataset.

        :param tfds_dir: directory to look for models in.

        :param K: backend engine, e.g., `np` or `tf`.

        :param as_numpy: Convert to numpy ndarrays.

        :param data_loader_kwargs: pass this as arguments to data_loader function

        :returns: Train and tests dataset splits.
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

class_with_optional_arg_method_str = '''
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

        :returns: Train and tests dataset splits.
        """
        return np.empty(0), np.empty(0)
'''.format(
    header_doc_str=indent(docstring_header_str, tab * 2)
)


returns_subscript = Subscript(
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
)


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
                            docstring_no_type_no_default_str,
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
                returns=returns_subscript,
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
                            emit_separating_tabs(
                                indent(docstring_no_default_doc_wrapped_str, tab * 2), 2
                            )
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
                    Expr(set_value(indent(docstring_no_type_no_default_str, tab * 2))),
                    return_ast,
                ],
                decorator_list=[],
                name="function_name",
                returns=returns_subscript,
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
                Expr(set_value(indent(docstring_str, tab * 2))),
                return_ast,
            ],
            decorator_list=[],
            name="function_name",
            returns=returns_subscript,
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

    :returns: Aggregated summation of `a` and `b`.
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
                ":returns: Aggregated summation of `a` and `b`.\n    "
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
            set_value("mnist"),
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

    :param as_numpy: Convert to numpy ndarrays.

    :param K: backend engine, e.g., `np` or `tf`.

    :param tfds_dir: directory to look for models in.

    :param writer: IO object to write out to

    :param **kwargs: additional keyword arguments

    :returns: backend engine
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
            set_value("~/tensorflow_datasets"),
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
                ":param as_numpy: Convert to numpy ndarrays.\n\n    "
                ":param K: backend engine, e.g., `np` or `tf`.\n\n    "
                ":param tfds_dir: directory to look for models in.\n\n    "
                ":param writer: IO object to write out to\n\n    "
                ":param **kwargs: additional keyword arguments\n\n    "
                ":returns: backend engine\n    ",
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
function_google_tf_squared_hinge_docstring_str = "\n".join(
    function_google_tf_squared_hinge_docstring
)
function_google_tf_squared_hinge = (
    "def squared_hinge(y_true, y_pred):",
    '  """{}"""'.format(function_google_tf_squared_hinge_docstring_str),
    "  y_pred = ops.convert_to_tensor_v2(y_pred)",
    "  y_true = math_ops.cast(y_true, y_pred.dtype)",
    "  y_true = _maybe_convert_labels(y_true)",
    "  return K.mean(",
    "      math_ops.square(math_ops.maximum(1. - y_true * y_pred, 0.)), axis=-1)",
)
function_google_tf_squared_hinge_str = "\n".join(function_google_tf_squared_hinge)

docstring_google_tf_adadelta_function = (
    "",
    "class Adadelta(object):",
    '  """{}"""'.format(
        "\n".join(map(partial(add, " " * 2), docstring_google_tf_adadelta))
    ),
    "",
    "  _HAS_AGGREGATE_GRAD = True",
    "",
    "  def __init__(self,",
    "               learning_rate=0.001,",
    "               rho=0.95,",
    "               epsilon=1e-7,",
    "               name='Adadelta',",
    "               **kwargs):",
    "    # super(Adadelta, self).__init__(name, **kwargs)",
    "    self._set_hyper('learning_rate', kwargs.get('lr', learning_rate))",
    "    self._set_hyper('decay', self._initial_decay)",
    "    self._set_hyper('rho', rho)",
    "    self.epsilon = epsilon or backend_config.epsilon()",
    "",
    "  def _create_slots(self, var_list):",
    "    # Separate for-loops to respect the ordering of slot variables from v1.",
    "    for v in var_list:",
    "      self.add_slot(v, 'accum_grad')",
    "    for v in var_list:",
    "      self.add_slot(v, 'accum_var')",
    "",
    "  def _prepare_local(self, var_device, var_dtype, apply_state):",
    "    # super(Adadelta, self)._prepare_local(var_device, var_dtype, apply_state)",
    "    apply_state[(var_device, var_dtype)].update(",
    "        dict(",
    "            epsilon=ops.convert_to_tensor_v2_with_dispatch(",
    "                self.epsilon, var_dtype),",
    "            rho=array_ops.identity(self._get_hyper('rho', var_dtype))))",
    "",
    "  def set_weights(self, weights):",
    "    params = self.weights",
    "    # Override set_weights for backward compatibility of Keras V1 optimizer",
    "    # since it does not include iteration at head of the weight list. Set",
    "    # iteration to 0.",
    "    if len(params) == len(weights) + 1:",
    "      weights = [np.array(0)] + weights",
    "    # super(Adadelta, self).set_weights(weights)",
    "",
    "  def _resource_apply_dense(self, grad, var, apply_state=None):",
    "    var_device, var_dtype = var.device, var.dtype.base_dtype",
    "    coefficients = ((apply_state or {}).get((var_device, var_dtype))",
    "                    or self._fallback_apply_state(var_device, var_dtype))",
    "",
    "    accum_grad = self.get_slot(var, 'accum_grad')",
    "    accum_var = self.get_slot(var, 'accum_var')",
    "    return gen_training_ops.ResourceApplyAdadelta(",
    "        var=var.handle,",
    "        accum=accum_grad.handle,",
    "        accum_update=accum_var.handle,",
    "        lr=coefficients['lr_t'],",
    "        rho=coefficients['rho'],",
    "        epsilon=coefficients['epsilon'],",
    "        grad=grad,",
    "        use_locking=self._use_locking)",
    "",
    "  def _resource_apply_sparse(self, grad, var, indices, apply_state=None):",
    "    var_device, var_dtype = var.device, var.dtype.base_dtype",
    "    coefficients = ((apply_state or {}).get((var_device, var_dtype))",
    "                    or self._fallback_apply_state(var_device, var_dtype))",
    "",
    "    accum_grad = self.get_slot(var, 'accum_grad')",
    "    accum_var = self.get_slot(var, 'accum_var')",
    "    return gen_training_ops.ResourceSparseApplyAdadelta(",
    "        var=var.handle,",
    "        accum=accum_grad.handle,",
    "        accum_update=accum_var.handle,",
    "        lr=coefficients['lr_t'],",
    "        rho=coefficients['rho'],",
    "        epsilon=coefficients['epsilon'],",
    "        grad=grad,",
    "        indices=indices,",
    "        use_locking=self._use_locking)",
    "",
    "  def get_config(self):",
    "    config = super(Adadelta, self).get_config()",
    "    config.update({",
    "        'learning_rate': self._serialize_hyperparameter('learning_rate'),",
    "        'decay': self._serialize_hyperparameter('decay'),",
    "        'rho': self._serialize_hyperparameter('rho'),",
    "        'epsilon': self.epsilon,",
    "    })",
    "    return config",
)
docstring_google_tf_adadelta_function_str = "\n".join(
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
    "docstring_google_tf_adadelta_function_str",
    "function_adder_ast",
    "function_adder_str",
    "function_default_complex_default_arg_ast",
    "function_default_complex_default_arg_str",
    "function_google_tf_squared_hinge_str",
    "method_complex_args_variety_ast",
    "method_complex_args_variety_str",
]
