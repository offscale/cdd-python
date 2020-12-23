"""
Mocks for the `class`
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
    Return,
    Store,
    Sub,
    Subscript,
    Tuple,
    UnaryOp,
    USub,
    arguments,
    get_docstring,
    keyword,
)
from copy import deepcopy

from doctrans.ast_utils import (
    NoneStr,
    maybe_type_comment,
    set_arg,
    set_slice,
    set_value,
)
from doctrans.docstring_parsers import parse_docstring
from doctrans.emit import docstring
from doctrans.pure_utils import PY_GTE_3_8, PY_GTE_3_9

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
    data_loader_kwargs: Optional[dict] = None
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

class_ast = ClassDef(
    bases=[Name("object", Load())],
    body=[
        Expr(
            set_value(
                "\n    Acquire from the official tensorflow_datasets model zoo,"
                " or the ophthalmology focussed ml-prepare library\n\n    "
                ':cvar dataset_name: name of dataset. Defaults to "mnist"\n    '
                ':cvar tfds_dir: directory to look for models in. Defaults to "~/tensorflow_datasets"\n    '
                ':cvar K: backend engine, e.g., `np` or `tf`. Defaults to "np"\n    '
                ":cvar as_numpy: Convert to numpy ndarrays\n    "
                ":cvar data_loader_kwargs: pass this as arguments to data_loader function\n    "
                ":cvar return_type: Train and tests dataset splits. Defaults to (np.empty(0), np.empty(0))",
            )
        ),
        AnnAssign(
            annotation=Name(
                "str",
                Load(),
            ),
            simple=1,
            target=Name("dataset_name", Store()),
            value=set_value("mnist"),
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
            value=set_value(
                "~/tensorflow_datasets",
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
                    )
                ),
                Load(),
            ),
            simple=1,
            target=Name("K", Store()),
            value=set_value("np"),
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
            value=set_value(None),
            expr=None,
            expr_target=None,
            expr_annotation=None,
        ),
        AnnAssign(
            annotation=Subscript(
                Name("Optional", Load()), set_slice(Name("dict", Load())), Load()
            ),
            simple=1,
            target=Name(
                "data_loader_kwargs",
                Store(),
            ),
            value=set_value(None),
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
                                    Tuple(
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
            simple=1,
            target=Name("return_type", Store()),
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

class_ast_no_default_doc = deepcopy(class_ast)
_doc_val = list(
    filter(
        None,
        docstring(
            parse_docstring(get_docstring(class_ast_no_default_doc))
        ).splitlines(),
    )
)
_doc_val[0] = "\n{}\n".format(_doc_val[0])
class_ast_no_default_doc.body[0].value = set_value(
    "\n".join(_doc_val[:-2] + _doc_val[-1:]).replace(":param", ":cvar")
)
del _doc_val

class_nargs_ast = ClassDef(
    bases=[Name("object", Load())],
    body=[
        Expr(
            set_value(
                "\n    Acquire from the official tensorflow_datasets model zoo,"
                " or the ophthalmology focussed ml-prepare library\n\n    "
                ":cvar callbacks: Collection of callables that are run inside the training loop",
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
            value=set_value(None),
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
                "    :cvar return_type: Squared hinge loss values. shape = `[batch_size, d0, .. dN-1]`."
                " Defaults to ```K.mean(math_ops.square(math_ops.maximum(1.0 - y_true * y_pred, 0.0)), axis=-1)```"
            )
        ),
        AnnAssign(
            annotation=Name("object", Load()),
            simple=1,
            target=Name("y_true", Store()),
            value=set_value(None),
            expr=None,
            expr_target=None,
            expr_annotation=None,
        ),
        AnnAssign(
            annotation=Name("object", Load()),
            simple=1,
            target=Name("y_pred", Store()),
            value=set_value(None),
            expr=None,
            expr_target=None,
            expr_annotation=None,
        ),
        AnnAssign(
            annotation=Name("str", Load()),
            simple=1,
            target=Name("return_type", Store()),
            value=set_value(
                "```K.mean(math_ops.square(math_ops.maximum(1.0 - y_true * y_pred, 0.0)), axis=-1)```"
            ),
            expr=None,
            expr_target=None,
            expr_annotation=None,
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
                    **maybe_type_comment,
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
                    **maybe_type_comment,
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
            **maybe_type_comment,
        ),
    ],
    decorator_list=[],
    keywords=[],
    expr=None,
    identifier_name=None,
    name="SquaredHingeConfig",
)

# https://github.com/tensorflow/tensorflow/blob/7ad2723/tensorflow/python/keras/callbacks.py#L1924-L2430 [- many funcs]
class_google_tf_tensorboard_str = '''
class TensorBoard(Callback, version_utils.TensorBoardVersionSelector):
  # pylint: disable=line-too-long
  """Enable visualizations for TensorBoard.
  TensorBoard is a visualization tool provided with TensorFlow.
  This callback logs events for TensorBoard, including:
  * Metrics summary plots
  * Training graph visualization
  * Activation histograms
  * Sampled profiling
  If you have installed TensorFlow with pip, you should be able
  to launch TensorBoard from the command line:
  ```
  tensorboard --logdir=path_to_your_logs
  ```
  You can find more information about TensorBoard
  [here](https://www.tensorflow.org/get_started/summaries_and_tensorboard).
  Arguments:
      log_dir: the path of the directory where to save the log files to be
        parsed by TensorBoard.
      histogram_freq: frequency (in epochs) at which to compute activation and
        weight histograms for the layers of the model. If set to 0, histograms
        won't be computed. Validation data (or split) must be specified for
        histogram visualizations.
      write_graph: whether to visualize the graph in TensorBoard. The log file
        can become quite large when write_graph is set to True.
      write_images: whether to write model weights to visualize as image in
        TensorBoard.
      update_freq: `'batch'` or `'epoch'` or integer. When using `'batch'`,
        writes the losses and metrics to TensorBoard after each batch. The same
        applies for `'epoch'`. If using an integer, let's say `1000`, the
        callback will write the metrics and losses to TensorBoard every 1000
        batches. Note that writing too frequently to TensorBoard can slow down
        your training.
      profile_batch: Profile the batch(es) to sample compute characteristics.
        profile_batch must be a non-negative integer or a tuple of integers.
        A pair of positive integers signify a range of batches to profile.
        By default, it will profile the second batch. Set profile_batch=0
        to disable profiling.
      embeddings_freq: frequency (in epochs) at which embedding layers will be
        visualized. If set to 0, embeddings won't be visualized.
      embeddings_metadata: a dictionary which maps layer name to a file name in
        which metadata for this embedding layer is saved. See the
        [details](
          https://www.tensorflow.org/how_tos/embedding_viz/#metadata_optional)
        about metadata files format. In case if the same metadata file is
        used for all embedding layers, string can be passed.
  Examples:
  Basic usage:
  ```python
  tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir="./logs")
  model.fit(x_train, y_train, epochs=2, callbacks=[tensorboard_callback])
  # Then run the tensorboard command to view the visualizations.
  ```
  Custom batch-level summaries in a subclassed Model:
  ```python
  class MyModel(tf.keras.Model):
    def build(self, _):
      self.dense = tf.keras.layers.Dense(10)
    def call(self, x):
      outputs = self.dense(x)
      tf.summary.histogram('outputs', outputs)
      return outputs
  model = MyModel()
  model.compile('sgd', 'mse')
  # Make sure to set `update_freq=N` to log a batch-level summary every N batches.
  # In addition to any `tf.summary` contained in `Model.call`, metrics added in
  # `Model.compile` will be logged every N batches.
  tb_callback = tf.keras.callbacks.TensorBoard('./logs', update_freq=1)
  model.fit(x_train, y_train, callbacks=[tb_callback])
  ```
  Custom batch-level summaries in a Functional API Model:
  ```python
  def my_summary(x):
    tf.summary.histogram('x', x)
    return x
  inputs = tf.keras.Input(10)
  x = tf.keras.layers.Dense(10)(inputs)
  outputs = tf.keras.layers.Lambda(my_summary)(x)
  model = tf.keras.Model(inputs, outputs)
  model.compile('sgd', 'mse')
  # Make sure to set `update_freq=N` to log a batch-level summary every N batches.
  # In addition to any `tf.summary` contained in `Model.call`, metrics added in
  # `Model.compile` will be logged every N batches.
  tb_callback = tf.keras.callbacks.TensorBoard('./logs', update_freq=1)
  model.fit(x_train, y_train, callbacks=[tb_callback])
  ```
  Profiling:
  ```python
  # Profile a single batch, e.g. the 5th batch.
  tensorboard_callback = tf.keras.callbacks.TensorBoard(
      log_dir='./logs', profile_batch=5)
  model.fit(x_train, y_train, epochs=2, callbacks=[tensorboard_callback])
  # Profile a range of batches, e.g. from 10 to 20.
  tensorboard_callback = tf.keras.callbacks.TensorBoard(
      log_dir='./logs', profile_batch=(10,20))
  model.fit(x_train, y_train, epochs=2, callbacks=[tensorboard_callback])
  ```
  """

  # pylint: enable=line-too-long

  def __init__(self,
               log_dir='logs',
               histogram_freq=0,
               write_graph=True,
               write_images=False,
               update_freq='epoch',
               profile_batch=2,
               embeddings_freq=0,
               embeddings_metadata=None,
               **kwargs):
    super(TensorBoard, self).__init__()

  def set_model(self, model):
    """Sets Keras model and writes graph if specified."""
'''.replace(
    "Arguments:", "Args:"
)

class_google_tf_tensorboard_ast = ClassDef(
    bases=[
        Name("Callback", Load()),
        Attribute(Name("version_utils", Load()), "TensorBoardVersionSelector", Load()),
    ],
    body=[
        Expr(
            set_value(
                "Enable visualizations for TensorBoard.\n"
                "  TensorBoard is a visualization tool provided with TensorFlow.\n"
                "  This callback logs events for TensorBoard, including:\n"
                "  * Metrics summary plots\n"
                "  * Training graph visualization\n"
                "  * Activation histograms\n"
                "  * Sampled profiling\n"
                "  If you have installed TensorFlow with pip, you should be able\n"
                "  to launch TensorBoard from the command line:\n"
                "  ```\n"
                "  tensorboard --logdir=path_to_your_logs\n"
                "  ```\n"
                "  You can find more information about TensorBoard\n"
                "  [here](https://www.tensorflow.org/get_started/summaries_and_tensorboard).\n"
                "  Arguments:\n"
                "      log_dir: the path of the directory where to save the log files to be\n"
                "        parsed by TensorBoard.\n"
                "      histogram_freq: frequency (in epochs) at which to compute activation and\n"
                "        weight histograms for the layers of the model. If set to 0, histograms\n"
                "        won't be computed. Validation data (or split) must be specified for\n"
                "        histogram visualizations.\n"
                "      write_graph: whether to visualize the graph in TensorBoard. The log file\n"
                "        can become quite large when write_graph is set to True.\n"
                "      write_images: whether to write model weights to visualize as image in\n"
                "        TensorBoard.\n"
                "      update_freq: `'batch'` or `'epoch'` or integer. When using `'batch'`,\n"
                "        writes the losses and metrics to TensorBoard after each batch. The same\n"
                "        applies for `'epoch'`. If using an integer, let's say `1000`, the\n"
                "        callback will write the metrics and losses to TensorBoard every 1000\n"
                "        batches. Note that writing too frequently to TensorBoard can slow down\n"
                "        your training.\n"
                "      profile_batch: Profile the batch(es) to sample compute characteristics.\n"
                "        profile_batch must be a non-negative integer or a tuple of integers.\n"
                "        A pair of positive integers signify a range of batches to profile.\n"
                "        By default, it will profile the second batch. Set profile_batch=0\n"
                "        to disable profiling.\n"
                "      embeddings_freq: frequency (in epochs) at which embedding layers will be\n"
                "        visualized. If set to 0, embeddings won't be visualized.\n"
                "      embeddings_metadata: a dictionary which maps layer name to a file name in\n"
                "        which metadata for this embedding layer is saved. See the\n"
                "        [details](\n"
                "          https://www.tensorflow.org/how_tos/embedding_viz/#metadata_optional)\n"
                "        about metadata files format. In case if the same metadata file is\n"
                "        used for all embedding layers, string can be passed.\n"
                "  Examples:\n"
                "  Basic usage:\n"
                "  ```python\n"
                '  tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir="./logs")\n'
                "  model.fit(x_train, y_train, epochs=2, callbacks=[tensorboard_callback])\n"
                "  # Then run the tensorboard command to view the visualizations.\n"
                "  ```\n"
                "  Custom batch-level summaries in a subclassed Model:\n"
                "  ```python\n"
                "  class MyModel(tf.keras.Model):\n"
                "    def build(self, _):\n"
                "      self.dense = tf.keras.layers.Dense(10)\n"
                "    def call(self, x):\n"
                "      outputs = self.dense(x)\n"
                "      tf.summary.histogram('outputs', outputs)\n"
                "      return outputs\n"
                "  model = MyModel()\n"
                "  model.compile('sgd', 'mse')\n"
                "  # Make sure to set `update_freq=N` to log a batch-level summary every N batches.\n"
                "  # In addition to any `tf.summary` contained in `Model.call`, metrics added in\n"
                "  # `Model.compile` will be logged every N batches.\n"
                "  tb_callback = tf.keras.callbacks.TensorBoard('./logs', update_freq=1)\n"
                "  model.fit(x_train, y_train, callbacks=[tb_callback])\n"
                "  ```\n"
                "  Custom batch-level summaries in a Functional API Model:\n"
                "  ```python\n"
                "  def my_summary(x):\n"
                "    tf.summary.histogram('x', x)\n"
                "    return x\n"
                "  inputs = tf.keras.Input(10)\n"
                "  x = tf.keras.layers.Dense(10)(inputs)\n"
                "  outputs = tf.keras.layers.Lambda(my_summary)(x)\n"
                "  model = tf.keras.Model(inputs, outputs)\n"
                "  model.compile('sgd', 'mse')\n"
                "  # Make sure to set `update_freq=N` to log a batch-level summary every N batches.\n"
                "  # In addition to any `tf.summary` contained in `Model.call`, metrics added in\n"
                "  # `Model.compile` will be logged every N batches.\n"
                "  tb_callback = tf.keras.callbacks.TensorBoard('./logs', update_freq=1)\n"
                "  model.fit(x_train, y_train, callbacks=[tb_callback])\n"
                "  ```\n"
                "  Profiling:\n"
                "  ```python\n  # Profile a single batch, e.g. the 5th batch.\n"
                "  tensorboard_callback = tf.keras.callbacks.TensorBoard(\n"
                "      log_dir='./logs', profile_batch=5)\n"
                "  model.fit(x_train, y_train, epochs=2, callbacks=[tensorboard_callback])\n"
                "  # Profile a range of batches, e.g. from 10 to 20.\n"
                "  tensorboard_callback = tf.keras.callbacks.TensorBoard(\n"
                "      log_dir='./logs', profile_batch=(10,20))\n"
                "  model.fit(x_train, y_train, epochs=2, callbacks=[tensorboard_callback])\n"
                "  ```\n  ".replace("Arguments:", "Args:")
            )
        ),
        FunctionDef(
            args=arguments(
                args=list(
                    map(
                        set_arg,
                        (
                            "self",
                            "log_dir",
                            "histogram_freq",
                            "write_graph",
                            "write_images",
                            "update_freq",
                            "profile_batch",
                            "embeddings_freq",
                            "embeddings_metadata",
                        ),
                    )
                ),
                defaults=list(
                    map(
                        set_value,
                        (
                            "logs",
                            0,
                            True,
                            False,
                            "epoch",
                            2,
                            0,
                            None,
                        ),
                    )
                ),
                kw_defaults=[],
                kwarg=set_arg("kwargs"),
                kwonlyargs=[],
                posonlyargs=[],
                vararg=None,
                arg=None,
            ),
            body=[
                Expr(
                    Call(
                        args=[],
                        func=Attribute(
                            Call(
                                args=[
                                    Name("TensorBoard", Load()),
                                    Name("self", Load()),
                                ],
                                func=Name("super", Load()),
                                keywords=[],
                                expr=None,
                                expr_func=None,
                            ),
                            "__init__",
                            Load(),
                        ),
                        keywords=[],
                        expr=None,
                        expr_func=None,
                    )
                )
            ],
            decorator_list=[],
            name="__init__",
            returns=None,
            arguments_args=None,
            identifier_name=None,
            lineno=None,
            stmt=None,
            **maybe_type_comment,
        ),
        FunctionDef(
            args=arguments(
                args=list(map(set_arg, ("self", "model"))),
                defaults=[],
                kw_defaults=[],
                kwarg=None,
                kwonlyargs=[],
                posonlyargs=[],
                vararg=None,
                arg=None,
            ),
            body=[Expr(set_value("Sets Keras model and writes graph if specified."))],
            decorator_list=[],
            name="set_model",
            returns=None,
            arguments_args=None,
            identifier_name=None,
            stmt=None,
            lineno=None,
            **maybe_type_comment,
        ),
    ],
    decorator_list=[],
    keywords=[],
    expr=None,
    identifier_name=None,
    name="TensorBoard",
)


class_google_tf_tensorboard_ir = {
    "doc": "Enable visualizations for TensorBoard.\n"
    "TensorBoard is a visualization tool provided with TensorFlow.\n"
    "This callback logs events for TensorBoard, including:\n"
    "* Metrics summary plots\n"
    "* Training graph visualization\n"
    "* Activation histograms\n"
    "* Sampled profiling\n"
    "If you have installed TensorFlow with pip, you should be able\n"
    "to launch TensorBoard from the command line:\n"
    "```\n"
    "tensorboard --logdir=path_to_your_logs\n"
    "```\n"
    "You can find more information about TensorBoard\n"
    "[here](https://www.tensorflow.org/get_started/summaries_and_tensorboard).\n"
    "\n"
    "\n"
    "Basic usage:\n"
    "```python\n"
    "tensorboard_callback = "
    'tf.keras.callbacks.TensorBoard(log_dir="./logs")\n'
    "model.fit(x_train, y_train, epochs=2, "
    "callbacks=[tensorboard_callback])\n"
    "# Then run the tensorboard command to view the visualizations.\n"
    "```\n"
    "Custom batch-level summaries in a subclassed Model:\n"
    "```python\n"
    "class MyModel(tf.keras.Model):\n"
    "  def build(self, _):\n"
    "    self.dense = tf.keras.layers.Dense(10)\n"
    "  def call(self, x):\n"
    "    outputs = self.dense(x)\n"
    "    tf.summary.histogram('outputs', outputs)\n"
    "    return outputs\n"
    "model = MyModel()\n"
    "model.compile('sgd', 'mse')\n"
    "# Make sure to set `update_freq=N` to log a batch-level summary "
    "every N batches.\n"
    "# In addition to any `tf.summary` contained in `Model.call`, "
    "metrics added in\n"
    "# `Model.compile` will be logged every N batches.\n"
    "tb_callback = tf.keras.callbacks.TensorBoard('./logs', "
    "update_freq=1)\n"
    "model.fit(x_train, y_train, callbacks=[tb_callback])\n"
    "```\n"
    "Custom batch-level summaries in a Functional API Model:\n"
    "```python\n"
    "def my_summary(x):\n"
    "  tf.summary.histogram('x', x)\n"
    "  return x\n"
    "inputs = tf.keras.Input(10)\n"
    "x = tf.keras.layers.Dense(10)(inputs)\n"
    "outputs = tf.keras.layers.Lambda(my_summary)(x)\n"
    "model = tf.keras.Model(inputs, outputs)\n"
    "model.compile('sgd', 'mse')\n"
    "# Make sure to set `update_freq=N` to log a batch-level summary "
    "every N batches.\n"
    "# In addition to any `tf.summary` contained in `Model.call`, "
    "metrics added in\n"
    "# `Model.compile` will be logged every N batches.\n"
    "tb_callback = tf.keras.callbacks.TensorBoard('./logs', "
    "update_freq=1)\n"
    "model.fit(x_train, y_train, callbacks=[tb_callback])\n"
    "```\n"
    "Profiling:\n"
    "```python\n"
    "# Profile a single batch, e.g. the 5th batch.\n"
    "tensorboard_callback = tf.keras.callbacks.TensorBoard(\n"
    "    log_dir='./logs', profile_batch=5)\n"
    "model.fit(x_train, y_train, epochs=2, "
    "callbacks=[tensorboard_callback])\n"
    "# Profile a range of batches, e.g. from 10 to 20.\n"
    "tensorboard_callback = tf.keras.callbacks.TensorBoard(\n"
    "    log_dir='./logs', profile_batch=(10,20))\n"
    "model.fit(x_train, y_train, epochs=2, "
    "callbacks=[tensorboard_callback])\n"
    "```",
    "name": None,
    "params": [
        {
            "default": "logs",
            "doc": "the path of the directory where to save the log "
            "files to be\n"
            "      parsed by TensorBoard.",
            "name": "log_dir",
            "typ": "str",
        },
        {
            "default": 0,
            "doc": "frequency (in epochs) at which to compute "
            "activation and\n"
            "      weight histograms for the layers of the "
            "model. If set to 0, histograms\n"
            "      won't be computed. Validation data (or "
            "split) must be specified for\n"
            "      histogram visualizations.",
            "name": "histogram_freq",
            "typ": "int",
        },
        {
            "default": True,
            "doc": "whether to visualize the graph in TensorBoard. "
            "The log file\n"
            "      can become quite large when write_graph is "
            "set to True.",
            "name": "write_graph",
            "typ": "bool",
        },
        {
            "default": False,
            "doc": "whether to write model weights to visualize as "
            "image in\n"
            "      TensorBoard.",
            "name": "write_images",
            "typ": "bool",
        },
        {
            "default": "epoch",
            "doc": "`'batch'` or `'epoch'` or integer. When using "
            "`'batch'`,\n"
            "      writes the losses and metrics to "
            "TensorBoard after each batch. The same\n"
            "      applies for `'epoch'`. If using an "
            "integer, let's say `1000`, the\n"
            "      callback will write the metrics and losses "
            "to TensorBoard every 1000\n"
            "      batches. Note that writing too frequently "
            "to TensorBoard can slow down\n"
            "      your training.",
            "name": "update_freq",
            "typ": "str",
        },
        {
            "default": 2,
            "doc": "Profile the batch(es) to sample compute "
            "characteristics.\n"
            "      profile_batch must be a non-negative "
            "integer or a tuple of integers.\n"
            "      A pair of positive integers signify a "
            "range of batches to profile.\n"
            "      By default, it will profile the second "
            "batch. Set profile_batch=0\n"
            "      to disable profiling.",
            "name": "profile_batch",
            "typ": "int",
        },
        {
            "default": 0,
            "doc": "frequency (in epochs) at which embedding layers "
            "will be\n"
            "      visualized. If set to 0, embeddings won't "
            "be visualized.",
            "name": "embeddings_freq",
            "typ": "int",
        },
        {
            "default": NoneStr,
            "doc": "a dictionary which maps layer name to a file "
            "name in\n"
            "      which metadata for this embedding layer is "
            "saved. See the\n"
            "      [details](\n"
            "        "
            "https://www.tensorflow.org/how_tos/embedding_viz/#metadata_optional)\n"
            "      about metadata files format. In case if "
            "the same metadata file is\n"
            "      used for all embedding layers, string can "
            "be passed.",
            "name": "embeddings_metadata",
            "typ": "Constant" if PY_GTE_3_8 else "NameConstant",
        },
        {"name": "kwargs", "typ": "Optional[dict]"},
    ],
    "returns": None,
    "type": "static",
}

class_torch_nn_l1loss_str = '''
class L1Loss(_Loss):
    r"""Creates a criterion that measures the mean absolute error (MAE) between each element in
    the input :math:`x` and target :math:`y`.
    The unreduced (i.e. with :attr:`reduction` set to ``'none'``) loss can be described as:
    .. math::
        \ell(x, y) = L = \{l_1,\dots,l_N\}^\top, \quad
        l_n = \left| x_n - y_n \right|,
    where :math:`N` is the batch size. If :attr:`reduction` is not ``'none'``
    (default ``'mean'``), then:
    .. math::
        \ell(x, y) =
        \begin{cases}
            \operatorname{mean}(L), & \text{if reduction} = \text{`mean';}\\
            \operatorname{sum}(L),  & \text{if reduction} = \text{`sum'.}
        \end{cases}
    :math:`x` and :math:`y` are tensors of arbitrary shapes with a total
    of :math:`n` elements each.
    The sum operation still operates over all the elements, and divides by :math:`n`.
    The division by :math:`n` can be avoided if one sets ``reduction = 'sum'``.
    Args:
        size_average (bool, optional): Deprecated (see :attr:`reduction`). By default,
            the losses are averaged over each loss element in the batch. Note that for
            some losses, there are multiple elements per sample. If the field :attr:`size_average`
            is set to ``False``, the losses are instead summed for each minibatch. Ignored
            when reduce is ``False``. Default: ``True``
        reduce (bool, optional): Deprecated (see :attr:`reduction`). By default, the
            losses are averaged or summed over observations for each minibatch depending
            on :attr:`size_average`. When :attr:`reduce` is ``False``, returns a loss per
            batch element instead and ignores :attr:`size_average`. Default: ``True``
        reduction (string, optional): Specifies the reduction to apply to the output:
            ``'none'`` | ``'mean'`` | ``'sum'``. ``'none'``: no reduction will be applied,
            ``'mean'``: the sum of the output will be divided by the number of
            elements in the output, ``'sum'``: the output will be summed. Note: :attr:`size_average`
            and :attr:`reduce` are in the process of being deprecated, and in the meantime,
            specifying either of those two args will override :attr:`reduction`. Default: ``'mean'``
    Shape:
        - Input: :math:`(N, *)` where :math:`*` means, any number of additional
          dimensions
        - Target: :math:`(N, *)`, same shape as the input
        - Output: scalar. If :attr:`reduction` is ``'none'``, then
          :math:`(N, *)`, same shape as the input
    Examples::
        >>> loss = nn.L1Loss()
        >>> input = torch.randn(3, 5, requires_grad=True)
        >>> target = torch.randn(3, 5)
        >>> output = loss(input, target)
        >>> output.backward()
    """
    __constants__ = ['reduction']

    def __init__(self, size_average=None, reduce=None, reduction: str = 'mean') -> None:
        super(L1Loss, self).__init__(size_average, reduce, reduction)

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        return F.l1_loss(input, target, reduction=self.reduction)
'''  # noqa: W605

class_torch_nn_l1loss_ast = ClassDef(
    bases=[
        Name(
            "_Loss",
            Load(),
        )
    ],
    body=[
        Expr(
            set_value(
                "Creates a criterion that measures the mean absolute error (MAE) between each element in\n"
                "    the input :math:`x` and target :math:`y`.\n"
                "    The unreduced (i.e. with :attr:`reduction` set to ``'none'``) loss can be described as:\n"
                "    .. math::\n"
                "        \\ell(x, y) = L = \\{l_1,\\dots,l_N\\}^\top, \\quad\n"
                "        l_n = \\left| x_n - y_n \night|,\n"
                "    where :math:`N` is the batch size. If :attr:`reduction` is not ``'none'``\n"
                "    (default ``'mean'``), then:\n"
                "    .. math::\n"
                "        \\ell(x, y) =\n"
                "        \x08egin{cases}\n"
                "            \\operatorname{mean}(L), & \text{if reduction} = \text{`mean';}\\\n"
                "            \\operatorname{sum}(L),  & \text{if reduction} = \text{`sum'.}\n"
                "        \\end{cases}\n    :math:`x` and :math:`y` are tensors of arbitrary shapes with a total\n"
                "    of :math:`n` elements each.\n"
                "    The sum operation still operates over all the elements, and divides by :math:`n`.\n"
                "    The division by :math:`n` can be avoided if one sets ``reduction = 'sum'``.\n"
                "    Args:\n"
                "        size_average (bool, optional): Deprecated (see :attr:`reduction`). By default,\n"
                "            the losses are averaged over each loss element in the batch. Note that for\n"
                "            some losses, there are multiple elements per sample. If the field :attr:`size_average`\n"
                "            is set to ``False``, the losses are instead summed for each minibatch. Ignored\n"
                "            when reduce is ``False``. Default: ``True``\n"
                "        reduce (bool, optional): Deprecated (see :attr:`reduction`). By default, the\n"
                "            losses are averaged or summed over observations for each minibatch depending\n"
                "            on :attr:`size_average`. When :attr:`reduce` is ``False``, returns a loss per\n"
                "            batch element instead and ignores :attr:`size_average`. Default: ``True``\n"
                "        reduction (string, optional): Specifies the reduction to apply to the output:\n"
                "            ``'none'`` | ``'mean'`` | ``'sum'``. ``'none'``: no reduction will be applied,\n"
                "            ``'mean'``: the sum of the output will be divided by the number of\n"
                "            elements in the output, ``'sum'``: the output will be summed. Note: :attr:`size_average`"
                "\n"
                "            and :attr:`reduce` are in the process of being deprecated, and in the meantime,\n"
                "            specifying either of those two args will override :attr:`reduction`"
                ". Default: ``'mean'``\n"
                "    Shape:\n"
                "        - Input: :math:`(N, *)` where :math:`*` means, any number of additional\n"
                "          dimensions\n"
                "        - Target: :math:`(N, *)`, same shape as the input\n"
                "        - Output: scalar. If :attr:`reduction` is ``'none'``, then\n"
                "          :math:`(N, *)`, same shape as the input\n"
                "    Examples::\n"
                "        >>> loss = nn.L1Loss()\n"
                "        >>> input = torch.randn(3, 5, requires_grad=True)\n"
                "        >>> target = torch.randn(3, 5)\n"
                "        >>> output = loss(input, target)\n"
                "        >>> output.backward()\n"
                "    "
            )
        ),
        Assign(
            targets=[Name("__constants__", Store())],
            type_comment=None,
            value=List([set_value("reduction")], Load()),
            expr=None,
            lineno=None,
        ),
        FunctionDef(
            args=arguments(
                args=[
                    set_arg(annotation=None, arg="self"),
                    set_arg(annotation=None, arg="size_average"),
                    set_arg(annotation=None, arg="reduce"),
                    set_arg(annotation=Name("str", Load()), arg="reduction"),
                ],
                defaults=[set_value(None), set_value(None), set_value("mean")],
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
                            Name("reduce", Load()),
                            Name(
                                "reduction",
                                Load(),
                            ),
                        ],
                        func=Attribute(
                            Call(
                                args=[
                                    Name("L1Loss", Load()),
                                    Name(
                                        "self",
                                        Load(),
                                    ),
                                ],
                                func=Name("super", Load()),
                                keywords=[],
                                expr=None,
                                expr_func=None,
                            ),
                            "__init__",
                            Load(),
                        ),
                        keywords=[],
                        expr=None,
                        expr_func=None,
                    )
                )
            ],
            decorator_list=[],
            name="__init__",
            returns=set_value(None),
            type_comment=None,
            arguments_args=None,
            identifier_name=None,
            stmt=None,
            lineno=None,
        ),
        FunctionDef(
            args=arguments(
                args=[
                    set_arg(annotation=None, arg="self"),
                    set_arg(annotation=Name("Tensor", Load()), arg="input"),
                    set_arg(annotation=Name("Tensor", Load()), arg="target"),
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
                        args=[Name("input", Load()), Name("target", Load())],
                        func=Attribute(
                            Name("F", Load()),
                            "l1_loss",
                            Load(),
                        ),
                        keywords=[
                            keyword(
                                arg="reduction",
                                value=Attribute(
                                    Name("self", Load()),
                                    "reduction",
                                    Load(),
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
            name="forward",
            returns=Name(
                "Tensor",
                Load(),
            ),
            type_comment=None,
            arguments_args=None,
            identifier_name=None,
            stmt=None,
            lineno=None,
        ),
    ],
    decorator_list=[],
    keywords=[],
    name="L1Loss",
    expr=None,
    identifier_name=None,
)

class_torch_nn_l1loss_ir = {
    "doc": "Creates a criterion that measures the mean absolute error (MAE) "
    "between each element in\n"
    "    the input :math:`x` and target :math:`y`.\n"
    "    The unreduced (i.e. with :attr:`reduction` set to ``'none'``) "
    "loss can be described as:\n"
    "    .. math::\n"
    "        \\ell(x, y) = L = \\{l_1,\\dots,l_N\\}^     op, \\quad\n"
    "        l_n = \\left| x_n - y_n \n"
    "ight|,\n"
    "    where :math:`N` is the batch size. If :attr:`reduction` is not "
    "``'none'``\n"
    "    (default ``'mean'``), then:\n"
    "    .. math::\n"
    "        \\ell(x, y) =\n"
    "        \x08egin{cases}\n"
    "            \\operatorname{mean}(L), &   ext{if reduction} =     "
    "ext{`mean';}\\\n"
    "            \\operatorname{sum}(L),  &   ext{if reduction} =     "
    "ext{`sum'.}\n"
    "        \\end{cases}\n"
    "    :math:`x` and :math:`y` are tensors of arbitrary shapes with a "
    "total\n"
    "    of :math:`n` elements each.\n"
    "    The sum operation still operates over all the elements, and "
    "divides by :math:`n`.\n"
    "    The division by :math:`n` can be avoided if one sets "
    "``reduction = 'sum'``.\n"
    "\n"
    "\n"
    "        reduction (string, optional): Specifies the reduction to "
    "apply to the output:\n"
    "                ``'none'`` | ``'mean'`` | ``'sum'``. ``'none'``: "
    "no reduction will be applied,\n"
    "                ``'mean'``: the sum of the output will be divided "
    "by the number of\n"
    "                elements in the output, ``'sum'``: the output will "
    "be summed. Note: :attr:`size_average`\n"
    "                and :attr:`reduce` are in the process of being "
    "deprecated, and in the meantime,\n"
    "                specifying either of those two args will override "
    ":attr:`reduction`. Default: ``'mean'``\n"
    "\n"
    "\n"
    "        - Input: :math:`(N, *)` where :math:`*` means, any number "
    "of additional\n"
    "          dimensions\n"
    "        - Target: :math:`(N, *)`, same shape as the input\n"
    "        - Output: scalar. If :attr:`reduction` is ``'none'``, "
    "then\n"
    "          :math:`(N, *)`, same shape as the input\n"
    "    Examples::\n"
    "        >>> loss = nn.L1Loss()\n"
    "        >>> input = torch.randn(3, 5, requires_grad=True)\n"
    "        >>> target = torch.randn(3, 5)\n"
    "        >>> output = loss(input, target)\n"
    "        >>> output.backward()\n"
    "    ",
    "name": None,
    "params": [
        {
            "default": True,
            "doc": "Deprecated (see :attr:`reduction`). By default,\n"
            "            the losses are averaged over each "
            "loss element in the batch. Note that for\n"
            "            some losses, there are multiple "
            "elements per sample. If the field "
            ":attr:`size_average`\n"
            "            is set to ``False``, the losses are "
            "instead summed for each minibatch. Ignored\n"
            "            when reduce is ``False``.",
            "name": "size_average",
            "typ": "Optional[bool]",
        },
        {
            "default": True,
            "doc": "Deprecated (see :attr:`reduction`). By default, "
            "the\n"
            "            losses are averaged or summed over "
            "observations for each minibatch depending\n"
            "            on :attr:`size_average`. When "
            ":attr:`reduce` is ``False``, returns a loss per\n"
            "            batch element instead and ignores "
            ":attr:`size_average`.",
            "name": "reduce",
            "typ": "Optional[bool]",
        },
        {
            "default": "```{parens[0]}['reduction']{parens[1]}```".format(
                parens=("(", ")") if PY_GTE_3_9 else ("", "")
            ),
            "name": "__constants__",
            "typ": "List",
        },
        {"default": "mean", "doc": None, "name": "reduction", "typ": "str"},
    ],
    "returns": {"name": "return_type", "typ": "None"},
    "type": "static",
}

__all__ = [
    "class_ast",
    "class_google_tf_tensorboard_ast",
    "class_google_tf_tensorboard_ir",
    "class_google_tf_tensorboard_str",
    "class_nargs_ast",
    "class_nargs_str",
    "class_squared_hinge_config_ast",
    "class_str",
    "class_torch_nn_l1loss_ast",
    "class_torch_nn_l1loss_str",
    "class_torch_nn_l1loss_ir",
]
