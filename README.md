cdd-python
==========
![Python version range](https://img.shields.io/badge/python-3.6%20|%203.7%20|%203.8%20|%203.9%20|%203.10b1-blue.svg)
![Python implementation](https://img.shields.io/badge/implementation-cpython-blue.svg)
[![License](https://img.shields.io/badge/license-Apache--2.0%20OR%20MIT-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Linting, testing, coverage, and release](https://github.com/offscale/cdd-python/workflows/Linting,%20testing,%20coverage,%20and%20release/badge.svg)](https://github.com/offscale/cdd-python/actions)
![Tested OSs, others may work](https://img.shields.io/badge/Tested%20on-Linux%20|%20macOS%20|%20Windows-green)
![Documentation coverage](https://raw.githubusercontent.com/offscale/cdd-python/master/.github/doccoverage.svg)
[![codecov](https://codecov.io/gh/offscale/cdd-python/branch/master/graph/badge.svg)](https://codecov.io/gh/offscale/cdd-python)
[![black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Imports: isort](https://img.shields.io/badge/%20imports-isort-%231674b1?style=flat&labelColor=ef8336)](https://pycqa.github.io/isort)
[![PyPi: release](https://img.shields.io/pypi/v/python-cdd.svg?maxAge=3600)](https://pypi.org/project/python-cdd)

Open API to/fro routes, models, and tests. Convert between docstrings, classes, methods, argparse, and SQLalchemy.

Public SDK works with filenames, source code, and even in memory constructs (e.g., as imported into your REPL).

## Install package

### PyPi

    pip install python-cdd

### Master

    pip install -r https://raw.githubusercontent.com/offscale/cdd-python/master/requirements.txt
    pip install https://api.github.com/repos/offscale/cdd-python/zipball#egg=cdd

## Goal

Easily create and maintain Database / ORM models and REST APIs out of existing Python SDKs.

For example, this can be used to expose TensorFlow in a REST API and store its parameters in an SQL database.

## Relation to other projects

This was created to aid in the `ml_params` project. It exposes an `@abstractclass` which is implemented [officially] by more than 8 projects.

Due to the nature of ML frameworks, `ml_params`' `def train(self, <these>)` has a potentially large number of arguments.
Accumulate the complexity of maintaining interfaces as the underlying release changes (e.g, new version of PyTorch), 
add in the extra interfaces folks find useful (CLIs, REST APIs, SQL models, &etc.); and you end up needing a team to maintain it.

That's unacceptable. The only existing solutions maintainable by one engineer involve dynamic generation, 
with no static, editable interfaces available. This means developer tooling becomes useless for debugging, introspection, and documentation.

To break it down, with current tooling there is no way to know:

  - What arguments can be provided to `train`
  - What CLI arguments are available
  - What 'shape' the `Config` takes

Some of these problems can be solved dynamically, however in doing so one loses developer-tool insights.
There is no code-completion, and likely the CLI parser won't provide you with the enumeration of possibilities.

## SDK example (REPL)

To create a `class` from [`tf.keras.optimizers.Adam`](https://www.tensorflow.org/api_docs/python/tf/keras/optimizers/Adam):

```python
>>> from cdd.source_transformer import to_code

>>> from cdd import emit, parse

>>> import tensorflow as tf

>>> from typing import Optional

>>> print(to_code(emit.class_(parse.class_(tf.keras.optimizers.Adam,
                                           merge_inner_function="__init__"),
                              class_name="AdamConfig")))


class AdamConfig(object):
    """
    Optimizer that implements the Adam algorithm.

    Adam optimization is a stochastic gradient descent method that is based on
    adaptive estimation of first-order and second-order moments.

    According to
    [Kingma et al., 2014](http://arxiv.org/abs/1412.6980),
    the method is "*computationally
    efficient, has little memory requirement, invariant to diagonal rescaling of
    gradients, and is well suited for problems that are large in terms of
    data/parameters*".


    Usage:

    >>> opt = tf.keras.optimizers.Adam(learning_rate=0.1)
    >>> var1 = tf.Variable(10.0)
    >>> loss = lambda: (var1 ** 2)/2.0       # d(loss)/d(var1) == var1
    >>> step_count = opt.minimize(loss, [var1]).numpy()
    >>> # The first step is `-learning_rate*sign(grad)`
    >>> var1.numpy()
    9.9

    Reference:
      - [Kingma et al., 2014](http://arxiv.org/abs/1412.6980)
      - [Reddi et al., 2018](
          https://openreview.net/pdf?id=ryQu7f-RZ) for `amsgrad`.

    Notes:

    The default value of 1e-7 for epsilon might not be a good default in
    general. For example, when training an Inception network on ImageNet a
    current good choice is 1.0 or 0.1. Note that since Adam uses the
    formulation just before Section 2.1 of the Kingma and Ba paper rather than
    the formulation in Algorithm 1, the "epsilon" referred to here is "epsilon
    hat" in the paper.

    The sparse implementation of this algorithm (used when the gradient is an
    IndexedSlices object, typically because of `tf.gather` or an embedding
    lookup in the forward pass) does apply momentum to variable slices even if
    they were not used in the forward pass (meaning they have a gradient equal
    to zero). Momentum decay (beta1) is also applied to the entire momentum
    accumulator. This means that the sparse behavior is equivalent to the dense
    behavior (in contrast to some momentum implementations which ignore momentum
    unless a variable slice was actually used).

    :cvar learning_rate: A `Tensor`, floating point value, or a schedule that is a
        `tf.keras.optimizers.schedules.LearningRateSchedule`, or a callable that takes no arguments and
        returns the actual value to use, The learning rate.
    :cvar beta_1: A float value or a constant float tensor, or a callable that takes no arguments and
        returns the actual value to use. The exponential decay rate for the 1st moment estimates.
    :cvar beta_2: A float value or a constant float tensor, or a callable that takes no arguments and
        returns the actual value to use, The exponential decay rate for the 2nd moment estimates.
    :cvar epsilon: A small constant for numerical stability. This epsilon is "epsilon hat" in the
        Kingma and Ba paper (in the formula just before Section 2.1), not the epsilon in Algorithm 1 of the
        paper.
    :cvar amsgrad: Boolean. Whether to apply AMSGrad variant of this algorithm from the paper "On the
        Convergence of Adam and beyond".
    :cvar name: Optional name for the operations created when applying gradients.
    :cvar kwargs: Keyword arguments. Allowed to be one of `"clipnorm"` or `"clipvalue"`. `"clipnorm"`
        (float) clips gradients by norm; `"clipvalue"` (float) clips gradients by value."""
    learning_rate: float = 0.001
    beta_1: float = 0.9
    beta_2: float = 0.999
    epsilon: float = 1e-07
    amsgrad: bool = False
    name: Optional[str] = 'Adam'
    kwargs: Optional[dict] = None
    _HAS_AGGREGATE_GRAD: bool = True
```

### Approach

Traverse the AST, and emit the modifications, such that each "format" can convert to each other.
Type asymmetries are added to the docstrings, e.g., "primary_key" has no equivalent in a regular python func argument, 
so is added as `":param my_id: [PK] The unique identifier"`.

The following are the different formats supported, all of which can convert betwixt eachother:

#### Docstring

```reStructuredText
Acquire from the official tensorflow_datasets model zoo, or the ophthalmology focussed ml-prepare library

:param dataset_name: name of dataset. Defaults to mnist
:type dataset_name: ```str```

:param tfds_dir: directory to look for models in. Defaults to ~/tensorflow_datasets
:type tfds_dir: ```Optional[str]```

:param K: backend engine, e.g., `np` or `tf`. Defaults to np
:type K: ```Union[np, tf]```

:param as_numpy: Convert to numpy ndarrays
:type as_numpy: ```Optional[bool]```

:param data_loader_kwargs: pass this as arguments to data_loader function
:type data_loader_kwargs: ```**data_loader_kwargs```

:return: Train and tests dataset splits. Defaults to (np.empty(0), np.empty(0))
:rtype: ```Union[Tuple[tf.data.Dataset, tf.data.Dataset], Tuple[np.ndarray, np.ndarray]]```
```

##### `class`

```python
from typing import Optional, Union, Tuple, Literal

import numpy as np
import tensorflow as tf


class TargetClass(object):
    """
    Acquire from the official tensorflow_datasets model zoo, or the ophthalmology focussed ml-prepare library

    :cvar dataset_name: name of dataset. Defaults to mnist
    :cvar tfds_dir: directory to look for models in. Defaults to ~/tensorflow_datasets
    :cvar K: backend engine, e.g., `np` or `tf`. Defaults to np
    :cvar as_numpy: Convert to numpy ndarrays
    :cvar data_loader_kwargs: pass this as arguments to data_loader function
    :cvar return_type: Train and tests dataset splits. Defaults to (np.empty(0), np.empty(0))"""

    dataset_name: str = 'mnist'
    tfds_dir: Optional[str] = '~/tensorflow_datasets'
    K: Literal['np', 'tf'] = 'np'
    as_numpy: Optional[bool] = None
    data_loader_kwargs: dict = {}
    return_type: Union[Tuple[tf.data.Dataset, tf.data.Dataset], Tuple[np.ndarray, np.ndarray]] = (
        np.empty(0),
        np.empty(0),
    )
```

##### `class` method

```python
from typing import Optional, Union, Tuple, Literal

import numpy as np
import tensorflow as tf

class C(object):
    """ C class (mocked!) """

    def method_name(
        self,
        dataset_name: str = 'mnist',
        tfds_dir: Optional[str] = '~/tensorflow_datasets',
        K: Literal['np', 'tf'] = 'np',
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
```

##### Argparse augmenting function

```python
from typing import Union, Tuple
from json import loads

import numpy as np
import tensorflow as tf


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
        type=globals().__getitem__,
        choices=('np', 'tf'),
        help='backend engine, expr.g., `np` or `tf`.',
        required=True,
        default='np',
    )
    argument_parser.add_argument('--as_numpy', type=bool, help='Convert to numpy ndarrays')
    argument_parser.add_argument(
        '--data_loader_kwargs', type=loads, help='pass this as arguments to data_loader function'
    )
    return argument_parser, (np.empty(0), np.empty(0))
```

##### SQLalchemy
There are two variants in the latest SQLalchemy, both are supported:

```py
from sqlalchemy import JSON, Boolean, Column, Enum, MetaData, String, Table, create_engine

engine = create_engine("sqlite://", echo=True, future=True)
metadata = MetaData()

config_tbl = Table(
    "config_tbl",
    metadata,
    Column(
        "dataset_name",
        String,
        doc="name of dataset",
        default="mnist",
        primary_key=True,
    ),
    Column(
        "tfds_dir",
        String,
        doc="directory to look for models in",
        default="~/tensorflow_datasets",
        nullable=False,
    ),
    Column(
        "K",
        Enum("np", "tf", name="K"),
        doc="backend engine, e.g., `np` or `tf`",
        default="np",
        nullable=False,
    ),
    Column(
        "as_numpy",
        Boolean,
        doc="Convert to numpy ndarrays",
        default=None,
        nullable=True,
    ),
    Column(
        "data_loader_kwargs",
        JSON,
        doc="pass this as arguments to data_loader function",
        default=None,
        nullable=True,
    ),
    comment='Acquire from the official tensorflow_datasets model zoo, or the ophthalmology focussed ml-prepare\n'
            '\n'
            ':returns: Train and tests dataset splits. Defaults to (np.empty(0), np.empty(0))\n'
            ':rtype: ```Union[Tuple[tf.data.Dataset, tf.data.Dataset], Tuple[np.ndarray, np.ndarray]]```',
)

metadata.create_all(engine)
```

```py
from sqlalchemy.orm import declarative_base
from sqlalchemy import JSON, Boolean, Column, Enum, String

Base = declarative_base()

class Config(Base):
    """
    Acquire from the official tensorflow_datasets model zoo, or the ophthalmology focussed ml-prepare
    
    :returns: Train and tests dataset splits. Defaults to (np.empty(0), np.empty(0))
    :rtype: ```Union[Tuple[tf.data.Dataset, tf.data.Dataset], Tuple[np.ndarray, np.ndarray]]```
    """
    __tablename__ = "config_tbl"

    dataset_name = Column(
        String,
        doc="name of dataset",
        default="mnist",
        primary_key=True,
    )

    tfds_dir = Column(
        String,
        doc="directory to look for models in",
        default="~/tensorflow_datasets",
        nullable=False,
    )

    K = Column(
        Enum("np", "tf", name="K"),
        doc="backend engine, e.g., `np` or `tf`",
        default="np",
        nullable=False,
    )

    as_numpy = Column(
        Boolean,
        doc="Convert to numpy ndarrays",
        default=None,
        nullable=True,
    )

    data_loader_kwargs = Column(
        JSON,
        doc="pass this as arguments to data_loader function",
        default=None,
        nullable=True,
    )

    def __repr__(self):
        """
        Emit a string representation of the current instance
        
        :returns: String representation of instance
        :rtype: ```str```
        """
    
        return ("Config(dataset_name={dataset_name!r}, tfds_dir={tfds_dir!r}, "
                "K={K!r}, as_numpy={as_numpy!r}, data_loader_kwargs={data_loader_kwargs!r})").format(
            dataset_name=self.dataset_name, tfds_dir=self.tfds_dir, K=self.K,
            as_numpy=self.as_numpy, data_loader_kwargs=self.data_loader_kwargs
        )
```

## Advantages

  - CLI gives proper `--help` messages
  - IDE and console gives proper insights to function, and arguments, including on type
  - `class`–based interface opens this up to clean object passing
  - Rather than passing around odd ORM class entities, you can use POPO (Plain Old Python Objects) and serialise easily
  - `@abstractmethod` can add—remove, and change—as many arguments as it wants; including required arguments; without worry
  - Verbosity of output removes the magic. It's always clear what's going on.
  - Outputting regular code means things can be composed and extended as normally.

## Disadvantages

  - You have to run a tool to synchronise your various formats.
  - Duplication (but the tool handles this)

## Alternatives

  - Slow, manual duplication; or
  - Dynamic code generation, e.g., with a singular interface for everything; so everything is in one place without duplication

## Minor other use-cases this facilitates

  - Switch between having types in the docstring and having the types inline ([PEP484](https://python.org/dev/peps/pep-0484)–style))
  - Switch between docstring formats (to/from {numpy, ReST, google})
  - Desktop GUI with wxWidgets, from the argparse layer through [Gooey](https://github.com/chriskiehl/Gooey) [one liner]

## CLI for this project

    $ python -m cdd --help
    
    usage: python -m cdd [-h] [--version]
                         {sync_properties,sync,gen,gen_routes,openapi,doctrans,exmod}
                         ...
    
    Open API to/fro routes, models, and tests. Convert between docstrings,
    classes, methods, argparse, and SQLalchemy.
    
    positional arguments:
      {sync_properties,sync,gen,gen_routes,openapi,doctrans,exmod}
        sync_properties     Synchronise one or more properties between input and
                            input_str Python files
        sync                Force argparse, classes, and/or methods to be
                            equivalent
        gen                 Generate classes, functions, argparse function,
                            sqlalchemy tables and/or sqlalchemy classes from the
                            input mapping
        gen_routes          Generate per model route(s)
        openapi             Generate OpenAPI schema from specified project(s)
        doctrans            Convert docstring format of all classes and functions
                            within target file
        exmod               Expose module hierarchy->{functions,classes,vars} for
                            parameterisation via {REST API + database,CLI,SDK}
    
    options:
      -h, --help            show this help message and exit
      --version             show program's version number and exit

### `sync`

    $ python -m cdd sync --help

    usage: python -m cdd sync [-h] [--argparse-function ARGPARSE_FUNCTIONS]
                              [--argparse-function-name ARGPARSE_FUNCTION_NAMES]
                              [--class CLASSES] [--class-name CLASS_NAMES]
                              [--function FUNCTIONS]
                              [--function-name FUNCTION_NAMES] --truth
                              {argparse_function,class,function}
    
    options:
      -h, --help            show this help message and exit
      --argparse-function ARGPARSE_FUNCTIONS
                            File where argparse function is `def`ined.
      --argparse-function-name ARGPARSE_FUNCTION_NAMES
                            Name of argparse function.
      --class CLASSES       File where class `class` is declared.
      --class-name CLASS_NAMES
                            Name of `class`
      --function FUNCTIONS  File where function is `def`ined.
      --function-name FUNCTION_NAMES
                            Name of Function. If method, use Python resolution
                            syntax, i.e., ClassName.function_name
      --truth {argparse_function,class,function}
                            Single source of truth. Others will be generated from
                            this. Will run with first found choice.

### `sync_properties`

    $ python -m cdd sync_properties --help

    usage: python -m cdd sync_properties [-h] --input-filename INPUT_FILENAME
                                         --input-param INPUT_PARAMS [--input-eval]
                                         --output-filename OUTPUT_FILENAME
                                         --output-param OUTPUT_PARAMS
                                         [--output-param-wrap OUTPUT_PARAM_WRAP]
    
    options:
      -h, --help            show this help message and exit
      --input-filename INPUT_FILENAME
                            File to find `--input-param` from
      --input-param INPUT_PARAMS
                            Location within file of property. Can be top level
                            like `a` for `a=5` or with the `.` syntax as in
                            `--output-param`.
      --input-eval          Whether to evaluate the input-param, or just leave it
      --output-filename OUTPUT_FILENAME
                            Edited in place, the property within this file (to
                            update) is selected by --output-param
      --output-param OUTPUT_PARAMS
                            Parameter to update. E.g., `A.F` for `class A: F`,
                            `f.g` for `def f(g): pass`
      --output-param-wrap OUTPUT_PARAM_WRAP
                            Wrap all input_str params with this. E.g.,
                            `Optional[Union[{output_param}, str]]`

### `gen`

    $ python -m cdd gen --help

    usage: python -m cdd gen [-h] --name-tpl NAME_TPL --input-mapping
                             INPUT_MAPPING [--prepend PREPEND]
                             [--imports-from-file IMPORTS_FROM_FILE]
                             [--parse {argparse,class,function,sqlalchemy,sqlalchemy_table}]
                             --emit
                             {argparse,class,function,sqlalchemy,sqlalchemy_table}
                             --output-filename OUTPUT_FILENAME [--emit-call]
                             [--decorator DECORATOR_LIST]
    
    optional arguments:
      -h, --help            show this help message and exit
      --name-tpl NAME_TPL   Template for the name, e.g., `{name}Config`.
      --input-mapping INPUT_MAPPING
                            Import location of dictionary/mapping/2-tuple
                            collection.
      --prepend PREPEND     Prepend file with this. Use '\n' for newlines.
      --imports-from-file IMPORTS_FROM_FILE
                            Extract imports from file and append to `output_file`.
                            If module or other symbol path given, resolve file
                            then use it.
      --parse {argparse,class,function,sqlalchemy,sqlalchemy_table}
                            What type the input is.
      --emit {argparse,class,function,sqlalchemy,sqlalchemy_table}
                            What type to generate.
      --output-filename OUTPUT_FILENAME, -o OUTPUT_FILENAME
                            Output file to write to.
      --emit-call           Whether to place all the previous body into a new
                            `__call__` internal function
      --decorator DECORATOR_LIST
                            List of decorators.

### `gen_routes`

    $ python -m cdd gen_routes --help

    usage: python -m cdd gen_routes [-h] --crud {CRUD,CR,C,R,U,D,CR,CU,CD,CRD}
                                    [--app-name APP_NAME] --model-path MODEL_PATH
                                    --model-name MODEL_NAME --routes-path
                                    ROUTES_PATH [--route ROUTE]
    
    options:
      -h, --help            show this help message and exit
      --crud {CRUD,CR,C,R,U,D,CR,CU,CD,CRD}
                            What of (C)reate, (R)ead, (U)pdate, (D)elete to
                            generate
      --app-name APP_NAME   Name of app (e.g., `app_name = Bottle();
                            @app_name.get('/api') def slash(): pass`)
      --model-path MODEL_PATH
                            Python module resolution (foo.models) or filepath
                            (foo/models)
      --model-name MODEL_NAME
                            Name of model to generate from
      --routes-path ROUTES_PATH
                            Python module resolution 'foo.routes' or filepath
                            'foo/routes'
      --route ROUTE         Name of the route, defaults to
                            `/api/{model_name.lower()}`

### `openapi`

    $ python -m cdd openapi --help

    usage: python -m cdd openapi [-h] [--app-name APP_NAME] --model-paths
                                 MODEL_PATHS --routes-paths
                                 [ROUTES_PATHS [ROUTES_PATHS ...]]
    
    optional arguments:
      -h, --help            show this help message and exit
      --app-name APP_NAME   Name of app (e.g., `app_name = Bottle();
                            @app_name.get('/api') def slash(): pass`)
      --model-paths MODEL_PATHS
                            Python module resolution (foo.models) or filepath
                            (foo/models)
      --routes-paths [ROUTES_PATHS [ROUTES_PATHS ...]]
                            Python module resolution 'foo.routes' or filepath
                            'foo/routes'

### `doctrans`

    $ python -m cdd doctrans --help
    
    usage: python -m cdd doctrans [-h] --filename FILENAME --format
                                  {rest,google,numpydoc}
                                  (--type-annotations | --no-type-annotations)
    
    options:
      -h, --help            show this help message and exit
      --filename FILENAME   Python file to convert docstrings within. Edited in
                            place.
      --format {rest,google,numpydoc}
                            The docstring format to replace existing format with.
      --type-annotations    Inline the type, i.e., annotate PEP484 (outside
                            docstring. Requires 3.6+)
      --no-type-annotations
                            Ensure all types are in docstring (rather than a
                            PEP484 type annotation)

### `exmod`

    $ python -m cdd exmod --help
    
    usage: python -m cdd exmod [-h] --module MODULE --emit
                               {argparse,class,function,sqlalchemy,sqlalchemy_table}
                               [--blacklist BLACKLIST] [--whitelist WHITELIST]
                               --output-directory OUTPUT_DIRECTORY [--dry-run]
    
    options:
      -h, --help            show this help message and exit
      --module MODULE, -m MODULE
                            The module or fully-qualified name (FQN) to expose.
      --emit {argparse,class,function,sqlalchemy,sqlalchemy_table}
                            What type to generate.
      --blacklist BLACKLIST
                            Modules/FQN to omit. If unspecified will emit all
                            (unless whitelist).
      --whitelist WHITELIST
                            Modules/FQN to emit. If unspecified will emit all
                            (minus blacklist).
      --output-directory OUTPUT_DIRECTORY, -o OUTPUT_DIRECTORY
                            Where to place the generated exposed interfaces to the
                            given `--module`.
      --dry-run             Show what would be created; don't actually write to
                            the filesystem.

---

## License

Licensed under either of

- Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE) or <https://www.apache.org/licenses/LICENSE-2.0>)
- MIT license ([LICENSE-MIT](LICENSE-MIT) or <https://opensource.org/licenses/MIT>)

at your option.

### Contribution

Unless you explicitly state otherwise, any contribution intentionally submitted
for inclusion in the work by you, as defined in the Apache-2.0 license, shall be
dual licensed as above, without any additional terms or conditions.
