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

## Advantages

  - CLI gives proper `--help` messages
  - IDE and console gives proper insights to function, and arguments, including on type
  - `class`–based interface opens this up to clean object passing
  - `@abstractmethod` can add—remove, and change—as many arguments as it wants; including required arguments; without worry

## Disadvantages

  - You have to run a tool to synchronise your: docstring(s), config `class`(es), and argparse augmenting function.
  - Duplication (but the tool handles this)

## Future work

  - More docstring support ([docstring-parser](https://github.com/rr-/docstring_parser) doesn't support this [sphinx ReST format with types](https://www.sphinx-doc.org/en/master/usage/restructuredtext/domains.html#info-field-lists))
  - Choosing between having the types in the docstring and having the types inline ([PEP484](https://python.org/dev/peps/pep-0484)–style)
  - Generating JSON-schema from `class` so it becomes useful in JSON-RPC, REST-API, and GUI environments
  - Backporting below Python 3.8
