doctrans
========
![Python version range](https://img.shields.io/badge/python-3.8-blue.svg)
[![License](https://img.shields.io/badge/license-Apache--2.0%20OR%20MIT-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Linting, testing, and coverage](https://github.com/SamuelMarks/doctrans/workflows/Linting,%20testing,%20and%20coverage/badge.svg)](https://github.com/SamuelMarks/doctrans/actions)
![Documentation coverage](.github/doccoverage.svg)
[![codecov](https://codecov.io/gh/SamuelMarks/doctrans/branch/master/graph/badge.svg)](https://codecov.io/gh/SamuelMarks/doctrans)

Translate between docstrings, classes, methods, and argparse.

## Install dependencies

    pip install -r requirements.txt

## Install package

    pip install .

## Purpose

This was created to aid in the `ml_params` project. It exposes an `@abstractclass` which is implemented [officially] by more than 8 projects.

Its `def train(self, <these>)` has a potentially large number of arguments.
Additionally there is a `def train_c(self, config)`, which accepts an instance of a `Config` class, or a dictionary.
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

## Future work

  - Proper CLI to manage what function, class, and argparse is generated, and from which source-of-truth
  - Choosing between having the types in the docstring and having the types inline ([PEP484](https://python.org/dev/peps/pep-0484)–style)
  - Generating JSON-schema from `class` so it becomes useful in JSON-RPC, REST-API, and GUI environments
  - Add 5th 'type' of SQLalchemy model
  - Add 6th type of routing layer
  - Add 7th type of Open API (at this point, rename to `cdd-python`)
  - Backporting below Python 3.8 (argparse might stop this… the only other thing I use is `Literal`; but this has been [backported](https://pypi.org/project/typing-extensions))
  - Move to https://github.com/offscale then upload to PyPi

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
