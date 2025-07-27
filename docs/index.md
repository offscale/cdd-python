cdd-python
==========

![Python version range](https://img.shields.io/badge/python-3.6%20|%203.7%20|%203.8%20|%203.9%20|%203.10%20|%203.11%20|%203.12%20|%203.13-blue.svg)
![Python implementation](https://img.shields.io/badge/implementation-cpython-blue.svg)
[![License](https://img.shields.io/badge/license-Apache--2.0%20OR%20MIT-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Linting, testing, coverage, and release](https://github.com/offscale/cdd-python/workflows/Linting,%20testing,%20coverage,%20and%20release/badge.svg)](https://github.com/offscale/cdd-python/actions)
![Tested OSs, others may work](https://img.shields.io/badge/Tested%20on-Linux%20|%20macOS%20|%20Windows-green)
![Documentation coverage](https://raw.githubusercontent.com/offscale/cdd-python/master/.github/doccoverage.svg)
[![codecov](https://codecov.io/gh/offscale/cdd-python/graph/badge.svg?token=TndsFusENZ)](https://codecov.io/gh/offscale/cdd-python)
[![black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Imports: isort](https://img.shields.io/badge/%20imports-isort-%231674b1?style=flat&labelColor=ef8336)](https://pycqa.github.io/isort)
[![PyPi: release](https://img.shields.io/pypi/v/python-cdd.svg?maxAge=3600)](https://pypi.org/project/python-cdd)

[OpenAPI](https://openapis.org) to/fro routes, models, and tests. Convert between docstrings, `class`es, methods, [argparse](https://docs.python.org/3/library/argparse.html), pydantic, and [SQLalchemy](https://sqlalchemy.org).

Public SDK works with filenames, source code, and even in memory constructs (e.g., as imported into your REPL).

## Features

| Type                                                                                                                                                    | Parse | Emit | Convert to all other Types |
|---------------------------------------------------------------------------------------------------------------------------------------------------------|-------|------|----------------------------|
| docstrings (between Google, NumPy, ReST formats; and betwixt type annotations and docstring)                                                            | ✅     | ✅    | ✅                          |
| `class`es                                                                                                                                               | ✅     | ✅    | ✅                          |
| functions                                                                                                                                               | ✅     | ✅    | ✅                          |
| [`argparse` CLI generating](https://docs.python.org/3/library/argparse.html#argumentparser-objects) functions                                           | ✅     | ✅    | ✅                          |
| JSON-schema                                                                                                                                             | ✅     | ✅    | ✅                          |
| [SQLalchemy `class`es](https://docs.sqlalchemy.org/en/14/orm/mapping_styles.html#orm-declarative-mapping)                                               | ✅     | ✅    | ✅                          |
| [SQLalchemy `Table`s](https://docs.sqlalchemy.org/en/14/core/metadata.html#sqlalchemy.schema.Table)                                                     | ✅     | ✅    | ✅                          |
| [SQLalchemy hybrid `class`es](https://docs.sqlalchemy.org/en/14/orm/declarative_tables.html#declarative-with-imperative-table-a-k-a-hybrid-declarative) | ✅     | ✅    | ✅                          |
| [pydantic `class`es](https://pydantic-docs.helpmanual.io/usage/schema/)                                                                                 | ✅     | ✅    | ✅                          |

### [OpenAPI](https://openapis.org) composite

The [OpenAPI](https://swagger.io/specification/) parser and emitter
utilises:

| Type                                                                                                      | Parse | Emit |
|-----------------------------------------------------------------------------------------------------------|-------|------|
| [Bottle route functions](https://bottlepy.org/docs/dev/api.html#routing)                                  | WiP   | WiP  |
| [FastAPI route functions](https://fastapi.tiangolo.com/tutorial/body/#request-body-path-query-parameters) | ✅     | ❌    |
| JSON-schema (e.g., from [SQLalchemy](https://docs.sqlalchemy.org))                                        | ✅     | ✅    |


Navigate to the [API Reference](api.md) to see the API documentation.
