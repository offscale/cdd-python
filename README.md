doctrans
========
![Python version range](https://img.shields.io/badge/python-3.6%20|%203.7%20|%203.8%20|%203.9%20|%203.10a2-blue.svg)
![Python implementation](https://img.shields.io/badge/implementation-cpython-blue.svg)
[![License](https://img.shields.io/badge/license-Apache--2.0%20OR%20MIT-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Linting, testing, and coverage](https://github.com/SamuelMarks/doctrans/workflows/Linting,%20testing,%20and%20coverage/badge.svg)](https://github.com/SamuelMarks/doctrans/actions)
![Tested OSs, others may work](https://img.shields.io/badge/Tested%20on-Linux%20|%20macOS%20|%20Windows-green)
![Documentation coverage](.github/doccoverage.svg)
[![codecov](https://codecov.io/gh/SamuelMarks/doctrans/branch/master/graph/badge.svg)](https://codecov.io/gh/SamuelMarks/doctrans)
[![black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

Translate between docstrings, classes, methods, and argparse.

Public SDK works with filenames, source code, and even in memory constructs (e.g., as imported into your REPL).

## Install dependencies

    pip install -r requirements.txt

## Install package

    pip install .

## Purpose

This was created to aid in the `ml_params` project. It exposes an `@abstractclass` which is implemented [officially] by more than 8 projects.

Its `def train(self, <these>)` has a potentially large number of arguments.
Additionally, there is a `def train_c(self, config)`, which accepts an instance of a `Config` class, or a dictionary.
Finally: `ml_params` defines a CLI interface.

With current tooling there is no way to know:

  - What arguments can be provided to `train`
  - What CLI arguments are available
  - What 'shape' the `Config` takes

Some of these problems can be solved dynamically, however in doing so one loses developer-tool insights. There is no code-completion, and likely the CLI parser won't provide you with the enumeration of possibilities.

### Approach

Traverse the AST, and emit the modifications, such that each of these 4 can convert to each other.

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

## Advantages

  - CLI gives proper `--help` messages
  - IDE and console gives proper insights to function, and arguments, including on type
  - `class`–based interface opens this up to clean object passing
  - `@abstractmethod` can add—remove, and change—as many arguments as it wants; including required arguments; without worry
  - Verbosity of output removes the magic. It's always clear what's going on.
  - Outputting regular code means things can be composed and extended as normally.

## Disadvantages

  - You have to run a tool to synchronise your: docstring(s), config `class`(es), and argparse augmenting function.
  - Duplication (but the tool handles this)

## Alternatives

  - Dynamic code generation, e.g., with a singular interface for everything; so everything is in one place without duplication

## Minor other use-cases this facilitates

  - Switch between having types in the docstring and having the types inline ([PEP484](https://python.org/dev/peps/pep-0484)–style))
  - Switch between docstring formats (WiP, SDK can go from {numpy, ReST, google} to ReST; but not to others)
  - Desktop GUI with wxWidgets, from the argparse layer through [Gooey](https://github.com/chriskiehl/Gooey) [one liner]

## CLI for this project

    $ python -m doctrans --help

    usage: python -m doctrans [-h] [--version] {sync_properties,sync,gen} ...
    
    Translate between docstrings, classes, methods, and argparse.
    
    positional arguments:
      {sync_properties,sync,gen}
        sync_properties     Synchronise one or more properties between input and
                            input_str Python files
        sync                Force argparse, classes, and/or methods to be
                            equivalent
        gen                 Generate classes, functions, and/or argparse functions
                            from the input mapping
    
    optional arguments:
      -h, --help            show this help message and exit
      --version             show program's version number and exit

### `sync`

    $ python -m doctrans sync --help

    usage: python -m doctrans sync [-h] [--argparse-function ARGPARSE_FUNCTIONS]
                                   [--argparse-function-name ARGPARSE_FUNCTION_NAMES]
                                   [--class CLASSES] [--class-name CLASS_NAMES]
                                   [--function FUNCTIONS]
                                   [--function-name FUNCTION_NAMES] --truth
                                   {argparse_function,class,function}
    
    optional arguments:
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

    $ python -m doctrans sync_properties --help

    usage: python -m doctrans sync_properties [-h] --input-filename INPUT_FILENAME
                                              --input-param INPUT_PARAMS
                                              [--input-eval] --output-filename
                                              OUTPUT_FILENAME --output-param
                                              OUTPUT_PARAMS
                                              [--output-param-wrap OUTPUT_PARAM_WRAP]
    
    optional arguments:
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

    $ python -m doctrans gen --help

    usage: python -m doctrans gen [-h] --name-tpl NAME_TPL --input-mapping
                                  INPUT_MAPPING [--prepend PREPEND]
                                  [--imports-from-file IMPORTS_FROM_FILE] --type
                                  {argparse,class,function} --output-filename
                                  OUTPUT_FILENAME
    
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
      --type {argparse,class,function}
                            What type to generate.
      --output-filename OUTPUT_FILENAME, -o OUTPUT_FILENAME
                            Output file to write to.

## Future work

  0. Add 4th 'type' of JSON-schema so it becomes useful in JSON-RPC, REST-API, and GUI environments
  1. Add 5th type of SQLalchemy model
  2. Add 6th type of routing layer
  3. Add 7th type of Open API (at this point, rename to `cdd-python`)
  4. Move to https://github.com/offscale then upload to PyPi

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
