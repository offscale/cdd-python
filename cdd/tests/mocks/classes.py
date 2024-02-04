"""
Mocks for the `class`

Note: TensorFlow code is taken from `5a56eb1`; the same that tf 2.15.0 was released with on 14/11/2023.
"""

from ast import (
    AnnAssign,
    Assign,
    Attribute,
    BinOp,
    Call,
    ClassDef,
    Expr,
    FunctionDef,
    Index,
    List,
    Load,
    Mult,
    Name,
    Pass,
    Return,
    Store,
    Sub,
    Subscript,
    Tuple,
    UnaryOp,
    USub,
    arguments,
    keyword,
)
from copy import deepcopy
from functools import partial
from itertools import chain
from operator import add, itemgetter
from textwrap import indent

from cdd.shared.ast_utils import maybe_type_comment, set_arg, set_slice, set_value
from cdd.shared.defaults_utils import extract_default
from cdd.shared.pure_utils import strip_starting, tab
from cdd.tests.mocks.docstrings import docstring_header_str, docstring_reduction_v2_str
from cdd.tests.mocks.methods import (
    function_google_tf_squared_hinge_docstring_str,
    returns_subscript,
)
from cdd.tests.utils_for_tests import remove_args_from_docstring

class_doc_str: str = tab.join(
    (
        "\n",
        "{header_doc_str}{tab}\n".format(header_doc_str=docstring_header_str, tab=tab),
        ':cvar dataset_name: name of dataset. Defaults to "mnist"\n',
        ':cvar tfds_dir: directory to look for models in. Defaults to "~/tensorflow_datasets"\n',
        ':cvar K: backend engine, e.g., `np` or `tf`. Defaults to "np"\n',
        ":cvar as_numpy: Convert to numpy ndarrays\n",
        ":cvar data_loader_kwargs: pass this as arguments to data_loader function\n",
        ":cvar return_type: Train and tests dataset splits. Defaults to (np.empty(0), np.empty(0))",
    )
)
class_doc_str_expr: Expr = Expr(set_value(class_doc_str), lineno=None, col_offset=None)

class_str: str = (
    '''
class ConfigClass(object):
    """
{header_doc_str}{tab}
    :cvar dataset_name: name of dataset. Defaults to "mnist"
    :cvar tfds_dir: directory to look for models in. Defaults to "~/tensorflow_datasets"
    :cvar K: backend engine, e.g., `np` or `tf`. Defaults to "np"
    :cvar as_numpy: Convert to numpy ndarrays
    :cvar data_loader_kwargs: pass this as arguments to data_loader function
    :cvar return_type: Train and tests dataset splits. Defaults to (np.empty(0), np.empty(0))"""

    dataset_name: str = "mnist"
    tfds_dir: str = "~/tensorflow_datasets"
    K: Literal["np", "tf"] = "np"
    as_numpy: Optional[bool]
    data_loader_kwargs: Optional[dict]
    return_type: Union[
        Tuple[tf.data.Dataset, tf.data.Dataset], Tuple[np.ndarray, np.ndarray]
    ] = (
        np.empty(0),
        np.empty(0),
    )
'''.format(
        header_doc_str=indent(docstring_header_str, tab), tab=tab
    )
)

class_nargs_str: str = (
    '''
class ConfigClass(object):
    """
    {header_doc_str}

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
'''.format(
        header_doc_str=indent(docstring_header_str, tab)
    )
)

class_ast: ClassDef = ClassDef(
    bases=[Name("object", Load(), lineno=None, col_offset=None)],
    body=[
        class_doc_str_expr,
        AnnAssign(
            annotation=Name("str", Load(), lineno=None, col_offset=None),
            simple=1,
            target=Name("dataset_name", Store(), lineno=None, col_offset=None),
            value=set_value("mnist"),
            expr=None,
            expr_annotation=None,
            expr_target=None,
            col_offset=None,
            lineno=None,
        ),
        AnnAssign(
            annotation=Name("str", Load(), lineno=None, col_offset=None),
            simple=1,
            target=Name("tfds_dir", Store(), lineno=None, col_offset=None),
            value=set_value(
                "~/tensorflow_datasets",
            ),
            expr=None,
            expr_annotation=None,
            expr_target=None,
            col_offset=None,
            lineno=None,
        ),
        AnnAssign(
            annotation=Subscript(
                Name("Literal", Load(), lineno=None, col_offset=None),
                Index(
                    value=Tuple(
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
                    )
                ),
                Load(),
                lineno=None,
                col_offset=None,
            ),
            simple=1,
            target=Name("K", Store(), lineno=None, col_offset=None),
            value=set_value("np"),
            expr=None,
            expr_target=None,
            expr_annotation=None,
            col_offset=None,
            lineno=None,
        ),
        AnnAssign(
            annotation=Subscript(
                Name("Optional", Load(), lineno=None, col_offset=None),
                Index(value=Name("bool", Load(), lineno=None, col_offset=None)),
                Load(),
                lineno=None,
                col_offset=None,
            ),
            simple=1,
            target=Name("as_numpy", Store(), lineno=None, col_offset=None),
            value=None,
            expr=None,
            expr_target=None,
            expr_annotation=None,
            col_offset=None,
            lineno=None,
        ),
        AnnAssign(
            annotation=Subscript(
                Name("Optional", Load(), lineno=None, col_offset=None),
                set_slice(Name("dict", Load(), lineno=None, col_offset=None)),
                Load(),
                lineno=None,
                col_offset=None,
            ),
            simple=1,
            target=Name("data_loader_kwargs", Store(), lineno=None, col_offset=None),
            value=None,
            expr=None,
            expr_target=None,
            expr_annotation=None,
            col_offset=None,
            lineno=None,
        ),
        AnnAssign(
            annotation=returns_subscript,
            simple=1,
            target=Name("return_type", Store(), lineno=None, col_offset=None),
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
            expr_target=None,
            expr_annotation=None,
            col_offset=None,
            lineno=None,
        ),
    ],
    decorator_list=[],
    type_params=[],
    keywords=[],
    name="ConfigClass",
    expr=None,
    identifier_name=None,
    lineno=None,
    col_offset=None,
)

class_ast_no_default_doc: ClassDef = deepcopy(class_ast)
class_ast_no_default_doc.body[0] = Expr(
    set_value(
        "\n".join(
            map(
                itemgetter(0),
                map(
                    partial(extract_default, emit_default_doc=False),
                    class_doc_str.splitlines(),
                ),
            )
        )
    ),
    lineno=None,
    col_offset=None,
)

class_ast_with_none: ClassDef = deepcopy(class_ast)
assert (
    isinstance(class_ast_with_none.body[4], AnnAssign)
    and class_ast_with_none.body[4].target.id == "as_numpy"
)
class_ast_with_none.body[4].value = set_value(
    None
)  # E.g., because argparse has a default set to `None`

class_nargs_ast: ClassDef = ClassDef(
    bases=[Name("object", Load(), lineno=None, col_offset=None)],
    body=[
        Expr(
            set_value(
                "\n{tab}{header_doc_str}{tab}\n{tab}"
                ":cvar callbacks: Collection of callables that are run inside the training loop".format(
                    tab=tab, header_doc_str=docstring_header_str
                ),
            ),
            lineno=None,
            col_offset=None,
        ),
        AnnAssign(
            annotation=Subscript(
                Name("Optional", Load(), lineno=None, col_offset=None),
                Index(
                    value=Subscript(
                        Name("List", Load(), lineno=None, col_offset=None),
                        Index(
                            value=Subscript(
                                Name("Literal", Load(), lineno=None, col_offset=None),
                                Index(
                                    value=Tuple(
                                        ctx=Load(),
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
                    )
                ),
                Load(),
                lineno=None,
                col_offset=None,
            ),
            simple=1,
            target=Name("callbacks", Store()),
            value=None,
            expr=None,
            expr_annotation=None,
            expr_target=None,
            col_offset=None,
            lineno=None,
        ),
    ],
    decorator_list=[],
    type_params=[],
    keywords=[],
    name="ConfigClass",
    expr=None,
    identifier_name=None,
    lineno=None,
    col_offset=None,
)

class_squared_hinge_config_ast: ClassDef = ClassDef(
    bases=[Name("object", Load(), lineno=None, col_offset=None)],
    body=[
        Expr(
            set_value(
                "\n{tab}{doc}{tab}\n{args}".format(
                    tab=tab,
                    doc=remove_args_from_docstring(
                        function_google_tf_squared_hinge_docstring_str
                    ),
                    args="\n".join(
                        map(
                            partial(add, tab),
                            (
                                ":cvar y_true: The ground truth values. `y_true` values are expected to be -1 or 1."
                                " If binary (0 or 1) labels are provided we will convert them to -1 or 1."
                                " shape = `[batch_size, d0, .. dN]`.",
                                ":cvar y_pred: The predicted values. shape = `[batch_size, d0, .. dN]`.",
                                ":cvar return_type: Squared hinge loss values. shape = `[batch_size, d0, .. dN-1]`.",
                            ),
                        )
                    ),
                ),
            ),
            lineno=None,
            col_offset=None,
        ),
        AnnAssign(
            annotation=Name("object", Load(), lineno=None, col_offset=None),
            simple=1,
            target=Name("y_true", Store()),
            value=set_value(None),
            expr=None,
            expr_target=None,
            expr_annotation=None,
            col_offset=None,
            lineno=None,
        ),
        AnnAssign(
            annotation=Name("object", Load(), lineno=None, col_offset=None),
            simple=1,
            target=Name("y_pred", Store()),
            value=set_value(None),
            expr=None,
            expr_target=None,
            expr_annotation=None,
            col_offset=None,
            lineno=None,
        ),
        AnnAssign(
            annotation=Name("str", Load(), lineno=None, col_offset=None),
            simple=1,
            target=Name("return_type", Store()),
            value=set_value(
                "```K.mean(math_ops.square(math_ops.maximum(1.0 - y_true * y_pred, 0.0)), axis=-1)```"
            ),
            expr=None,
            expr_target=None,
            expr_annotation=None,
            col_offset=None,
            lineno=None,
        ),
        FunctionDef(
            args=arguments(
                args=[set_arg("self")],
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
                            Name("self", Load(), lineno=None, col_offset=None),
                            "y_pred",
                            Load(),
                            lineno=None,
                            col_offset=None,
                        )
                    ],
                    value=Call(
                        args=[
                            Attribute(
                                Name("self", Load(), lineno=None, col_offset=None),
                                "y_pred",
                                Load(),
                                lineno=None,
                                col_offset=None,
                            )
                        ],
                        func=Attribute(
                            Name("ops", Load(), lineno=None, col_offset=None),
                            "convert_to_tensor_v2",
                            Load(),
                            lineno=None,
                            col_offset=None,
                        ),
                        keywords=[],
                        expr=None,
                        expr_func=None,
                        lineno=None,
                        col_offset=None,
                    ),
                    expr=None,
                    lineno=None,
                    **maybe_type_comment,
                ),
                Assign(
                    targets=[
                        Attribute(
                            Name("self", Load(), lineno=None, col_offset=None),
                            "y_true",
                            Load(),
                            lineno=None,
                            col_offset=None,
                        )
                    ],
                    value=Call(
                        args=[
                            Attribute(
                                Name("self", Load(), lineno=None, col_offset=None),
                                "y_true",
                                Load(),
                                lineno=None,
                                col_offset=None,
                            ),
                            Attribute(
                                Attribute(
                                    Name("self", Load(), lineno=None, col_offset=None),
                                    "y_pred",
                                    Load(),
                                    lineno=None,
                                    col_offset=None,
                                ),
                                "dtype",
                                Load(),
                            ),
                        ],
                        func=Attribute(
                            Name("math_ops", Load(), lineno=None, col_offset=None),
                            "cast",
                            Load(),
                            lineno=None,
                            col_offset=None,
                        ),
                        keywords=[],
                        expr=None,
                        expr_func=None,
                        lineno=None,
                        col_offset=None,
                    ),
                    expr=None,
                    lineno=None,
                    **maybe_type_comment,
                ),
                Assign(
                    targets=[
                        Attribute(
                            Name("self", Load(), lineno=None, col_offset=None),
                            "y_true",
                            Load(),
                            lineno=None,
                            col_offset=None,
                        )
                    ],
                    value=Call(
                        args=[
                            Attribute(
                                Name("self", Load(), lineno=None, col_offset=None),
                                "y_true",
                                Load(),
                                lineno=None,
                                col_offset=None,
                            )
                        ],
                        func=Name(
                            "_maybe_convert_labels",
                            Load(),
                            lineno=None,
                            col_offset=None,
                        ),
                        keywords=[],
                        expr=None,
                        expr_func=None,
                        lineno=None,
                        col_offset=None,
                    ),
                    expr=None,
                    lineno=None,
                    **maybe_type_comment,
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
                                                        Name(
                                                            "self",
                                                            Load(),
                                                            lineno=None,
                                                            col_offset=None,
                                                        ),
                                                        "y_true",
                                                        Load(),
                                                        lineno=None,
                                                        col_offset=None,
                                                    ),
                                                    Mult(),
                                                    Attribute(
                                                        Name(
                                                            "self",
                                                            Load(),
                                                            lineno=None,
                                                            col_offset=None,
                                                        ),
                                                        "y_pred",
                                                        Load(),
                                                        lineno=None,
                                                        col_offset=None,
                                                    ),
                                                ),
                                            ),
                                            set_value(0.0),
                                        ],
                                        func=Attribute(
                                            Name(
                                                "math_ops",
                                                Load(),
                                                lineno=None,
                                                col_offset=None,
                                            ),
                                            "maximum",
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
                                ],
                                func=Attribute(
                                    Name(
                                        "math_ops", Load(), lineno=None, col_offset=None
                                    ),
                                    "square",
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
                        ],
                        func=Attribute(
                            Name("K", Load(), lineno=None, col_offset=None),
                            "mean",
                            Load(),
                            lineno=None,
                            col_offset=None,
                        ),
                        keywords=[
                            keyword(
                                arg="axis",
                                value=UnaryOp(USub(), set_value(1)),
                                identifier=None,
                            )
                        ],
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
            name="__call__",
            returns=None,
            arguments_args=None,
            identifier_name=None,
            stmt=None,
            lineno=None,
            **maybe_type_comment,
        ),
    ],
    decorator_list=[],
    type_params=[],
    keywords=[],
    expr=None,
    identifier_name=None,
    name="SquaredHingeConfig",
    lineno=None,
    col_offset=None,
)

# ```py
# import ast
# import inspect
#
# import keras.callbacks
#
# ast.parse(inspect.getsource(keras.callbacks.TensorBoard)).body[0].body[0].value.value.splitlines()
# ```
# ###############
# # Keras 3.0.1 #
# ###############
# https://github.com/keras-team/keras/blob/f889c1f/keras/callbacks/tensorboard.py#L20-L157 [- args]
tensorboard_doc_str_no_args = (
    "Enable visualizations for TensorBoard.",
    "",
    "    TensorBoard is a visualization tool provided with TensorFlow. A TensorFlow",
    "    installation is required to use this callback.",
    "",
    "    This callback logs events for TensorBoard, including:",
    "",
    "    * Metrics summary plots",
    "    * Training graph visualization",
    "    * Weight histograms",
    "    * Sampled profiling",
    "",
    "    When used in `model.evaluate()` or regular validation",
    "    in addition to epoch summaries, there will be a summary that records",
    "    evaluation metrics vs `model.optimizer.iterations` written. The metric names",
    "    will be prepended with `evaluation`, with `model.optimizer.iterations` being",
    "    the step in the visualized TensorBoard.",
    "",
    "    If you have installed TensorFlow with pip, you should be able",
    "    to launch TensorBoard from the command line:",
    "",
    "    ```",
    "    tensorboard --logdir=path_to_your_logs",
    "    ```",
    "",
    "    You can find more information about TensorBoard",
    "    [here](https://www.tensorflow.org/get_started/summaries_and_tensorboard).",
    "",
    "",
    "    Examples:",
    "",
    "    Basic usage:",
    "",
    "    ```python",
    '    tensorboard_callback = keras.callbacks.TensorBoard(log_dir="./logs")',
    "    model.fit(x_train, y_train, epochs=2, callbacks=[tensorboard_callback])",
    "    # Then run the tensorboard command to view the visualizations.",
    "    ```",
    "",
    "    Custom batch-level summaries in a subclassed Model:",
    "",
    "    ```python",
    "    class MyModel(keras.Model):",
    "",
    "        def build(self, _):",
    "            self.dense = keras.layers.Dense(10)",
    "",
    "        def call(self, x):",
    "            outputs = self.dense(x)",
    "            tf.summary.histogram('outputs', outputs)",
    "            return outputs",
    "",
    "    model = MyModel()",
    "    model.compile('sgd', 'mse')",
    "",
    "    # Make sure to set `update_freq=N` to log a batch-level summary every N",
    "    # batches.  In addition to any `tf.summary` contained in `model.call()`,",
    "    # metrics added in `Model.compile` will be logged every N batches.",
    "    tb_callback = keras.callbacks.TensorBoard('./logs', update_freq=1)",
    "    model.fit(x_train, y_train, callbacks=[tb_callback])",
    "    ```",
    "",
    "    Custom batch-level summaries in a Functional API Model:",
    "",
    "    ```python",
    "    def my_summary(x):",
    "        tf.summary.histogram('x', x)",
    "        return x",
    "",
    "    inputs = keras.Input(10)",
    "    x = keras.layers.Dense(10)(inputs)",
    "    outputs = keras.layers.Lambda(my_summary)(x)",
    "    model = keras.Model(inputs, outputs)",
    "    model.compile('sgd', 'mse')",
    "",
    "    # Make sure to set `update_freq=N` to log a batch-level summary every N",
    "    # batches. In addition to any `tf.summary` contained in `Model.call`,",
    "    # metrics added in `Model.compile` will be logged every N batches.",
    "    tb_callback = keras.callbacks.TensorBoard('./logs', update_freq=1)",
    "    model.fit(x_train, y_train, callbacks=[tb_callback])",
    "    ```",
    "",
    "    Profiling:",
    "",
    "    ```python",
    "    # Profile a single batch, e.g. the 5th batch.",
    "    tensorboard_callback = keras.callbacks.TensorBoard(",
    "        log_dir='./logs', profile_batch=5)",
    "    model.fit(x_train, y_train, epochs=2, callbacks=[tensorboard_callback])",
    "",
    "    # Profile a range of batches, e.g. from 10 to 20.",
    "    tensorboard_callback = keras.callbacks.TensorBoard(",
    "        log_dir='./logs', profile_batch=(10,20))",
    "    model.fit(x_train, y_train, epochs=2, callbacks=[tensorboard_callback])",
    "    ```",
    "    ",
)  # type: tuple[str, ...]

tensorboard_doc_str_no_args_str: str = "\n".join(
    map(
        partial(strip_starting, str_to_strip="  "),
        tensorboard_doc_str_no_args,
    )
)

tensorboard_doc_str_no_args_examples_idx = tensorboard_doc_str_no_args_str.index(
    "  Examples:"
)

tensorboard_doc_str_args = (
    "    Args:",
    "        log_dir: the path of the directory where to save the log files to be",
    "            parsed by TensorBoard. e.g.,",
    "            `log_dir = os.path.join(working_dir, 'logs')`.",
    "            This directory should not be reused by any other callbacks.",
    "        histogram_freq: frequency (in epochs) at which to compute",
    "            weight histograms for the layers of the model. If set to 0,",
    "            histograms won't be computed. Validation data (or split) must be",
    "            specified for histogram visualizations.",
    "        write_graph:  (Not supported at this time)",
    "            Whether to visualize the graph in TensorBoard.",
    "            Note that the log file can become quite large",
    "            when `write_graph` is set to `True`.",
    "        write_images: whether to write model weights to visualize as image in",
    "            TensorBoard.",
    "        write_steps_per_second: whether to log the training steps per second",
    "            into TensorBoard. This supports both epoch and batch frequency",
    "            logging.",
    '        update_freq: `"batch"` or `"epoch"` or integer. When using `"epoch"`,',
    "            writes the losses and metrics to TensorBoard after every epoch.",
    "            If using an integer, let's say `1000`, all metrics and losses",
    "            (including custom ones added by `Model.compile`) will be logged to",
    '            TensorBoard every 1000 batches. `"batch"` is a synonym for 1,',
    "            meaning that they will be written every batch.",
    "            Note however that writing too frequently to TensorBoard can slow",
    "            down your training, especially when used with distribution",
    "            strategies as it will incur additional synchronization overhead.",
    "            Batch-level summary writing is also available via `train_step`",
    "            override. Please see",
    "            [TensorBoard Scalars tutorial](",
    "                "
    "https://www.tensorflow.org/tensorboard/scalars_and_keras#batch-level_logging)  # noqa: E501",
    "            for more details.",
    "        profile_batch: (Not supported at this time)",
    "            Profile the batch(es) to sample compute characteristics.",
    "            profile_batch must be a non-negative integer or a tuple of integers.",
    "            A pair of positive integers signify a range of batches to profile.",
    "            By default, profiling is disabled.",
    "        embeddings_freq: frequency (in epochs) at which embedding layers will be",
    "            visualized. If set to 0, embeddings won't be visualized.",
    "        embeddings_metadata: Dictionary which maps embedding layer names to the",
    "            filename of a file in which to save metadata for the embedding layer.",
    "            In case the same metadata file is to be",
    "            used for all embedding layers, a single filename can be passed.",
)

tensorboard_doc_str: str = "\n".join(
    map(
        partial(strip_starting, str_to_strip="  "),
        chain.from_iterable(
            (
                tensorboard_doc_str_no_args[:tensorboard_doc_str_no_args_examples_idx],
                "",
                tensorboard_doc_str_args,
                ("\n",),
                tensorboard_doc_str_no_args[tensorboard_doc_str_no_args_examples_idx:],
            )
        ),
    )
)
del tensorboard_doc_str_no_args_examples_idx

# Minus a lot of functions, just includes args and first line of `__init__` and `set_model`
class_google_keras_tensorboard_str: str = (
    '''
class TensorBoard(Callback):
    """{tensorboard_doc_str}"""

    def __init__(
        self,
        log_dir="logs",
        histogram_freq=0,
        write_graph=True,
        write_images=False,
        write_steps_per_second=False,
        update_freq="epoch",
        profile_batch=0,
        embeddings_freq=0,
        embeddings_metadata=None,
    ):
        super().__init__()

    def set_model(self, model):
        """Sets Keras model and writes graph if specified."""
'''.format(
        tensorboard_doc_str=tensorboard_doc_str
    )
)

class_google_keras_tensorboard_ast: ClassDef = ClassDef(
    name="TensorBoard",
    bases=[Name(id="Callback", ctx=Load(), lineno=None, col_offset=None)],
    keywords=[],
    body=[
        Expr(set_value(tensorboard_doc_str), lineno=None, col_offset=None),
        FunctionDef(
            name="__init__",
            args=arguments(
                posonlyargs=[],
                args=list(
                    map(
                        set_arg,
                        (
                            "self",
                            "log_dir",
                            "histogram_freq",
                            "write_graph",
                            "write_images",
                            "write_steps_per_second",
                            "update_freq",
                            "profile_batch",
                            "embeddings_freq",
                            "embeddings_metadata",
                        ),
                    )
                ),
                kwonlyargs=[],
                kw_defaults=[],
                defaults=list(
                    map(set_value, ("logs", 0, True, False, False, "epoch", 0, 0, None))
                ),
                kwarg=None,
                vararg=None,
                arg=None,
            ),
            body=[
                Expr(
                    value=Call(
                        func=Attribute(
                            value=Call(
                                func=Name(
                                    id="super", ctx=Load(), lineno=None, col_offset=None
                                ),
                                args=[],
                                keywords=[],
                                lineno=None,
                                col_offset=None,
                            ),
                            attr="__init__",
                            ctx=Load(),
                            lineno=None,
                            col_offset=None,
                        ),
                        args=[],
                        keywords=[],
                        lineno=None,
                        col_offset=None,
                    ),
                    lineno=None,
                    col_offset=None,
                ),
            ],
            decorator_list=[],
            type_params=[],
            arguments_args=None,
            identifier_name=None,
            stmt=None,
            lineno=None,
            returns=None,
            **maybe_type_comment,
        ),
        FunctionDef(
            name="set_model",
            args=arguments(
                posonlyargs=[],
                args=list(map(set_arg, ("self", "model"))),
                kwonlyargs=[],
                kw_defaults=[],
                defaults=[],
                kwarg=None,
                vararg=None,
                arg=None,
            ),
            body=[
                Expr(
                    set_value("Sets Keras model and writes graph if specified."),
                    lineno=None,
                    col_offset=None,
                )
            ],
            decorator_list=[],
            type_params=[],
            arguments_args=None,
            identifier_name=None,
            stmt=None,
            lineno=None,
            returns=None,
            **maybe_type_comment,
        ),
    ],
    decorator_list=[],
    type_params=[],
    expr=None,
    identifier_name=None,
    lineno=None,
    col_offset=None,
)

# #################
# # PyTorch 2.1.2 #
# #################
# import inspect
# import torch.nn.modules.loss
#
# inspect.getdoc(torch.nn.modules.loss.L1Loss).splitlines()
class_torch_nn_l1loss_docstring = (
    "Creates a criterion that measures the mean absolute error (MAE) between each element in",
    "the input :math:`x` and target :math:`y`.",
    "",
    "The unreduced (i.e. with :attr:`reduction` set to ``'none'``) loss can be described as:",
    "",
    ".. math::",
    "    \\ell(x, y) = L = \\{l_1,\\dots,l_N\\}^\\top, \\quad",
    "    l_n = \\left| x_n - y_n \\right|,",
    "",
    "where :math:`N` is the batch size. If :attr:`reduction` is not ``'none'``",
    "(default ``'mean'``), then:",
    "",
    ".. math::",
    "    \\ell(x, y) =",
    "    \\begin{cases}",
    "        \\operatorname{mean}(L), & \\text{if reduction} = \\text{`mean';}\\\\",
    "        \\operatorname{sum}(L),  & \\text{if reduction} = \\text{`sum'.}",
    "    \\end{cases}",
    "",
    ":math:`x` and :math:`y` are tensors of arbitrary shapes with a total",
    "of :math:`n` elements each.",
    "",
    "The sum operation still operates over all the elements, and divides by :math:`n`.",
    "",
    "The division by :math:`n` can be avoided if one sets ``reduction = 'sum'``.",
    "",
    "Supports real-valued and complex-valued inputs.",
    "",
    "Args:",
    "    size_average (bool, optional): Deprecated (see :attr:`reduction`). By default,",
    "        the losses are averaged over each loss element in the batch. Note that for",
    "        some losses, there are multiple elements per sample. If the field :attr:`size_average`",
    "        is set to ``False``, the losses are instead summed for each minibatch. Ignored",
    "        when :attr:`reduce` is ``False``. Default: ``True``",
    "    reduce (bool, optional): Deprecated (see :attr:`reduction`). By default, the",
    "        losses are averaged or summed over observations for each minibatch depending",
    "        on :attr:`size_average`. When :attr:`reduce` is ``False``, returns a loss per",
    "        batch element instead and ignores :attr:`size_average`. Default: ``True``",
    "    reduction (str, optional): Specifies the reduction to apply to the output:",
    "        ``'none'`` | ``'mean'`` | ``'sum'``. ``'none'``: no reduction will be applied,",
    "        ``'mean'``: the sum of the output will be divided by the number of",
    "        elements in the output, ``'sum'``: the output will be summed. Note: :attr:`size_average`",
    "        and :attr:`reduce` are in the process of being deprecated, and in the meantime,",
    "        specifying either of those two args will override :attr:`reduction`. Default: ``'mean'``",
    "",
    "Shape:",
    "    - Input: :math:`(*)`, where :math:`*` means any number of dimensions.",
    "    - Target: :math:`(*)`, same shape as the input.",
    "    - Output: scalar. If :attr:`reduction` is ``'none'``, then",
    "      :math:`(*)`, same shape as the input.",
    "",
    "Examples::",
    "",
    "    >>> loss = nn.L1Loss()",
    "    >>> input = torch.randn(3, 5, requires_grad=True)",
    "    >>> target = torch.randn(3, 5)",
    "    >>> output = loss(input, target)",
    "    >>> output.backward()",
)
class_torch_nn_l1loss_docstring_str: str = "\n".join(class_torch_nn_l1loss_docstring)

class_torch_nn_l1loss_str: str = (
    '''
class L1Loss(_Loss):
    r"""{class_torch_nn_l1loss_docstring_str!s}"""
    __constants__ = ['reduction']

    def __init__(self, size_average=None, reduce=None, reduction: str = 'mean') -> None:
        super().__init__(size_average, reduce, reduction)

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        return F.l1_loss(input, target, reduction=self.reduction)
'''.format(
        class_torch_nn_l1loss_docstring_str=class_torch_nn_l1loss_docstring_str
    )
)

class_torch_nn_l1loss_ast: ClassDef = ClassDef(
    bases=[
        Name(
            "_Loss",
            Load(),
        )
    ],
    body=[
        Expr(
            set_value(class_torch_nn_l1loss_docstring_str), lineno=None, col_offset=None
        ),
        Assign(
            targets=[Name("__constants__", Store())],
            value=List([set_value("reduction")], Load(), lineno=None, col_offset=None),
            expr=None,
            lineno=None,
            **maybe_type_comment,
        ),
        FunctionDef(
            args=arguments(
                args=[
                    set_arg("self"),
                    set_arg("size_average"),
                    set_arg("reduce"),
                    set_arg(
                        annotation=Name("str", Load(), lineno=None, col_offset=None),
                        arg="reduction",
                    ),
                ],
                defaults=list(map(set_value, (None, None, "mean"))),
                kw_defaults=[],
                kwarg=None,
                kwonlyargs=[],
                posonlyargs=[],
                vararg=None,
                arg=None,
            ),
            body=[
                Expr(
                    Call(
                        args=[
                            Name(
                                "size_average",
                                Load(),
                            ),
                            Name("reduce", Load(), lineno=None, col_offset=None),
                            Name(
                                "reduction",
                                Load(),
                            ),
                        ],
                        func=Attribute(
                            Call(
                                args=[],
                                func=Name(
                                    "super", Load(), lineno=None, col_offset=None
                                ),
                                keywords=[],
                                expr=None,
                                expr_func=None,
                            ),
                            "__init__",
                            Load(),
                            lineno=None,
                            col_offset=None,
                        ),
                        keywords=[],
                        expr=None,
                        expr_func=None,
                        lineno=None,
                        col_offset=None,
                    ),
                    lineno=None,
                    col_offset=None,
                )
            ],
            decorator_list=[],
            type_params=[],
            name="__init__",
            returns=set_value(None),
            arguments_args=None,
            identifier_name=None,
            stmt=None,
            lineno=None,
            **maybe_type_comment,
        ),
        FunctionDef(
            args=arguments(
                args=[
                    set_arg("self"),
                    set_arg(
                        annotation=Name("Tensor", Load(), lineno=None, col_offset=None),
                        arg="input",
                    ),
                    set_arg(
                        annotation=Name("Tensor", Load(), lineno=None, col_offset=None),
                        arg="target",
                    ),
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
                Return(
                    value=Call(
                        args=[
                            Name("input", Load(), lineno=None, col_offset=None),
                            Name("target", Load(), lineno=None, col_offset=None),
                        ],
                        func=Attribute(
                            Name("F", Load(), lineno=None, col_offset=None),
                            "l1_loss",
                            Load(),
                            lineno=None,
                            col_offset=None,
                        ),
                        keywords=[
                            keyword(
                                arg="reduction",
                                value=Attribute(
                                    Name("self", Load(), lineno=None, col_offset=None),
                                    "reduction",
                                    Load(),
                                    lineno=None,
                                    col_offset=None,
                                ),
                                identifier=None,
                            )
                        ],
                        expr=None,
                        expr_func=None,
                    ),
                    expr=None,
                )
            ],
            decorator_list=[],
            type_params=[],
            name="forward",
            returns=Name(
                "Tensor",
                Load(),
            ),
            arguments_args=None,
            identifier_name=None,
            stmt=None,
            lineno=None,
            **maybe_type_comment,
        ),
    ],
    decorator_list=[],
    type_params=[],
    keywords=[],
    name="L1Loss",
    expr=None,
    identifier_name=None,
    lineno=None,
    col_offset=None,
)

class_torch_nn_one_cycle_lr_docstring = (
    "Sets the learning rate of each parameter group according to the",
    "    1cycle learning rate policy. The 1cycle policy anneals the learning",
    "    rate from an initial learning rate to some maximum learning rate and then",
    "    from that maximum learning rate to some minimum learning rate much lower",
    "    than the initial learning rate.",
    "    This policy was initially described in the paper `Super-Convergence:",
    "    Very Fast Training of Neural Networks Using Large Learning Rates`_.",
    "",
    "    The 1cycle learning rate policy changes the learning rate after every batch.",
    "    `step` should be called after a batch has been used for training.",
    "",
    "    This scheduler is not chainable.",
    "",
    "    Note also that the total number of steps in the cycle can be determined in one",
    "    of two ways (listed in order of precedence):",
    "",
    "    #. A value for total_steps is explicitly provided.",
    "    #. A number of epochs (epochs) and a number of steps per epoch",
    "       (steps_per_epoch) are provided.",
    "       In this case, the number of total steps is inferred by",
    "       total_steps = epochs * steps_per_epoch",
    "",
    "    You must either provide a value for total_steps or provide a value for both",
    "    epochs and steps_per_epoch.",
    "",
    "    Args:",
    "        optimizer (Optimizer): Wrapped optimizer.",
    "        max_lr (float or list): Upper learning rate boundaries in the cycle",
    "            for each parameter group.",
    "        total_steps (int): The total number of steps in the cycle. Note that",
    "            if a value is not provided here, then it must be inferred by providing",
    "            a value for epochs and steps_per_epoch.",
    "            Default: None",
    "        epochs (int): The number of epochs to train for. This is used along",
    "            with steps_per_epoch in order to infer the total number of steps in the cycle",
    "            if a value for total_steps is not provided.",
    "            Default: None",
    "        steps_per_epoch (int): The number of steps per epoch to train for. This is",
    "            used along with epochs in order to infer the total number of steps in the",
    "            cycle if a value for total_steps is not provided.",
    "            Default: None",
    "        pct_start (float): The percentage of the cycle (in number of steps) spent",
    "            increasing the learning rate.",
    "            Default: 0.3",
    "        anneal_strategy (str): {'cos', 'linear'}",
    '            Specifies the annealing strategy: "cos" for cosine annealing, "linear" for',
    "            linear annealing.",
    "            Default: 'cos'",
    "        cycle_momentum (bool): If ``True``, momentum is cycled inversely",
    "            to learning rate between 'base_momentum' and 'max_momentum'.",
    "            Default: True",
    "        base_momentum (float or list): Lower momentum boundaries in the cycle",
    "            for each parameter group. Note that momentum is cycled inversely",
    "            to learning rate; at the peak of a cycle, momentum is",
    "            'base_momentum' and learning rate is 'max_lr'.",
    "            Default: 0.85",
    "        max_momentum (float or list): Upper momentum boundaries in the cycle",
    "            for each parameter group. Functionally,",
    "            it defines the cycle amplitude (max_momentum - base_momentum).",
    "            Note that momentum is cycled inversely",
    "            to learning rate; at the start of a cycle, momentum is 'max_momentum'",
    "            and learning rate is 'base_lr'",
    "            Default: 0.95",
    "        div_factor (float): Determines the initial learning rate via",
    "            initial_lr = max_lr/div_factor",
    "            Default: 25",
    "        final_div_factor (float): Determines the minimum learning rate via",
    "            min_lr = initial_lr/final_div_factor",
    "            Default: 1e4",
    "        last_epoch (int): The index of the last batch. This parameter is used when",
    "            resuming a training job. Since `step()` should be invoked after each",
    "            batch instead of after each epoch, this number represents the total",
    "            number of *batches* computed, not the total number of epochs computed.",
    "            When last_epoch=-1, the schedule is started from the beginning.",
    "            Default: -1",
    "        verbose (bool): If ``True``, prints a message to stdout for",
    "            each update. Default: ``False``.",
    "",
    "    Example:",
    "        >>> data_loader = torch.utils.data.DataLoader(...)",
    "        >>> optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)",
    "        >>> scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=0.01,"
    " steps_per_epoch=len(data_loader), epochs=10)",
    "        >>> for epoch in range(10):",
    "        >>>     for batch in data_loader:",
    "        >>>         train_batch(...)",
    "        >>>         scheduler.step()",
    "",
    "",
    "    .. _Super-Convergence\\: Very Fast Training of Neural Networks Using Large Learning Rates:",
    "        https://arxiv.org/abs/1708.07120",
    "    ",
)
class_torch_nn_one_cycle_lr_docstring_str = "\n".join(
    class_torch_nn_one_cycle_lr_docstring
)[: -len(tab)]

class_torch_nn_one_cycle_lr = (
    "class OneCycleLR(_LRScheduler):",
    '    r"""{}"""'.format(class_torch_nn_one_cycle_lr_docstring_str),
    "    def __init__(self,",
    "                 optimizer,",
    "                 max_lr,",
    "                 total_steps=None,",
    "                 epochs=None,",
    "                 steps_per_epoch=None,",
    "                 pct_start=0.3,",
    "                 anneal_strategy='cos',",
    "                 cycle_momentum=True,",
    "                 base_momentum=0.85,",
    "                 max_momentum=0.95,",
    "                 div_factor=25.,",
    "                 final_div_factor=1e4,",
    "                 last_epoch=-1,",
    "                 verbose=False):",
    "",
    "        pass",
)
class_torch_nn_one_cycle_lr_str = "\n".join(class_torch_nn_one_cycle_lr)


class_torch_nn_one_cycle_lr_ast: ClassDef = ClassDef(
    bases=[Name("_LRScheduler", Load(), lineno=None, col_offset=None)],
    body=[
        Expr(
            set_value(class_torch_nn_one_cycle_lr_docstring_str),
            lineno=None,
            col_offset=None,
        ),
        FunctionDef(
            args=arguments(
                args=list(
                    map(
                        set_arg,
                        (
                            "self",
                            "optimizer",
                            "max_lr",
                            "total_steps",
                            "epochs",
                            "steps_per_epoch",
                            "pct_start",
                            "anneal_strategy",
                            "cycle_momentum",
                            "base_momentum",
                            "max_momentum",
                            "div_factor",
                            "final_div_factor",
                            "last_epoch",
                            "verbose",
                        ),
                    )
                ),
                defaults=list(
                    chain.from_iterable(
                        (
                            map(
                                set_value,
                                (
                                    None,
                                    None,
                                    None,
                                    0.3,
                                    "cos",
                                    True,
                                    0.85,
                                    0.95,
                                    25.0,
                                    10000.0,
                                ),
                            ),
                            iter((UnaryOp(USub(), set_value(1)), set_value(False))),
                        )
                    )
                ),
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
            name="__init__",
            returns=None,
            arguments_args=None,
            identifier_name=None,
            stmt=None,
            lineno=None,
            **maybe_type_comment,
        ),
    ],
    decorator_list=[],
    type_params=[],
    keywords=[],
    expr=None,
    identifier_name=None,
    name="OneCycleLR",
    lineno=None,
    col_offset=None,
)

# From `tf.keras.losses.Reduction` @ tf-nightly:2.7.0.dev20210908, minus methods and decorator
class_reduction_v2: ClassDef = ClassDef(
    name="ReductionV2",
    bases=[],
    keywords=[],
    body=[
        Expr(value=set_value(docstring_reduction_v2_str), lineno=None, col_offset=None),
        Assign(targets=[Name(id="AUTO", ctx=Store())], value=set_value("auto")),
        Assign(targets=[Name(id="NONE", ctx=Store())], value=set_value("none")),
        Assign(targets=[Name(id="SUM", ctx=Store())], value=set_value("sum")),
        Assign(
            targets=[Name(id="SUM_OVER_BATCH_SIZE", ctx=Store())],
            value=set_value("sum_over_batch_size"),
        ),
    ],
    decorator_list=[],
    lineno=None,
    col_offset=None,
)

__all__ = [
    "class_ast",
    "class_ast_no_default_doc",
    "class_ast_with_none",
    "class_doc_str",
    "class_google_keras_tensorboard_ast",
    "class_google_keras_tensorboard_str",
    "class_nargs_ast",
    "class_nargs_str",
    "class_reduction_v2",
    "class_squared_hinge_config_ast",
    "class_str",
    "class_torch_nn_l1loss_ast",
    "class_torch_nn_l1loss_docstring_str",
    "class_torch_nn_l1loss_str",
    "class_torch_nn_one_cycle_lr_ast",
    "class_torch_nn_one_cycle_lr_docstring_str",
    "class_torch_nn_one_cycle_lr_str",
    "tensorboard_doc_str",
    "tensorboard_doc_str_no_args_str",
]  # type: list[str]
