doctrans
========
![Python version range](https://img.shields.io/badge/python-3.8-blue.svg)
[![Linting & testing](https://github.com/SamuelMarks/doctrans/workflows/Linting%20&%20testing/badge.svg)](https://github.com/SamuelMarks/doctrans/actions)

Translate between docstrings, classes, and argparse.

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

### Example
All 3 of these can convert to each other.

#### Docstring
```reStructuredText
Acquire from the official tensorflow_datasets model zoo, or the ophthalmology focussed ml-prepare library

:param dataset_name: name of dataset
:type dataset_name: ```str```

:param tfds_dir: directory to look for models in. Default is ~/tensorflow_datasets.
:type tfds_dir: ```Optional[str]```

:param K: backend engine, e.g., `np` or `tf`
:type K: ```Optional[Literal[np, tf]]```

:param as_numpy: Convert to numpy ndarrays
:type as_numpy: ```bool```

:param data_loader_kwargs: pass this as arguments to data_loader function
:type data_loader_kwargs: ```**data_loader_kwargs```

:return: Train and tests dataset splits
:rtype: ```Union[Tuple[tf.data.Dataset, tf.data.Dataset], Tuple[np.ndarray, np.ndarray]]```
```

##### `class`
```python
class TargetClass(object):
    """
    Acquire from the official tensorflow_datasets model zoo, or the ophthalmology focussed ml-prepare library

    :cvar dataset_name: name of dataset
    :cvar tfds_dir: directory to look for models in. Default is ~/tensorflow_datasets.
    :cvar K: backend engine, e.g., `np` or `tf`
    :cvar as_numpy: Convert to numpy ndarrays
    :cvar data_loader_kwargs: pass this as arguments to data_loader function
    :cvar return_type: Train and tests dataset splits"""

    dataset_name: str = ''
    tfds_dir: Optional[str] = None
    K: Optional[Literal[np, tf]] = None
    as_numpy: bool = True
    data_loader_kwargs: dict = {}
    return_type: Union[Tuple[tf.data.Dataset, tf.data.Dataset], Tuple[np.ndarray, np.ndarray]] = None
```

##### Argparse augmenting function
```python
def set_cli_args(argument_parser):
    """
    Set CLI arguments

    :param argument_parser: argument parser
    :type argument_parser: ```ArgumentParser```

    :return: argument parser and return type
    :rtype: ```Tuple[ArgumentParser, Union[Tuple[tf.data.Dataset, tf.data.Dataset], Tuple[np.ndarray, np.ndarray]]]```
    """

    argument_parser.description = 'Acquire from the official tensorflow_datasets model zoo, ' \
                                  'or the ophthalmology focussed ml-prepare library'

    argument_parser.add_argument('--dataset_name', type=str, help='name of dataset', required=True)
    argument_parser.add_argument('--tfds_dir', type=str, default='~/tensorflow_datasets',
                                 help='directory to look for models in.')
    argument_parser.add_argument('--K', type=str,
                                 choices=('np', 'tf'),
                                 help='backend engine, e.g., `np` or `tf`',
                                 required=True)
    argument_parser.add_argument('--as_numpy', type=bool, default=True, help='Convert to numpy ndarrays')
    argument_parser.add_argument('--data_loader_kwargs', type=loads,
                                 help='pass this as arguments to data_loader function')

    return argument_parser, (Union[Tuple[tf.data.Dataset, tf.data.Dataset], Tuple[np.ndarray, np.ndarray]])
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

## Future work

  - Handling of default arguments
  - Get tests to succeed on GitHub Actions (they succeed locally)
  - More docstring support ([docstring-parser](https://github.com/rr-/docstring_parser) doesn't support this [sphinx ReST format with types](https://www.sphinx-doc.org/en/master/usage/restructuredtext/domains.html#info-field-lists))
  - Proper CLI to manage what function, class, and argparse is generated, and from which source-of-truth
  - Choosing between having the types in the docstring and having the types inline ([PEP484](https://python.org/dev/peps/pep-0484)–style)
  - Generating JSON-schema from `class` so it becomes useful in JSON-RPC, REST-API, and GUI environments
  - Backporting below Python 3.8
  - Move to https://github.com/offscale then upload to PyPi
