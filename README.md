# cdd-python

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
[![hosted documentation](https://img.shields.io/badge/hosted-docs-white)](https://offscale.io/cdd-python/)

OpenAPI ↔ Python. This is one compiler in a suite, all focussed on the same task: Compiler Driven Development (CDD).

Each compiler is written in its target language, is whitespace and comment sensitive, and has both an SDK and CLI.

The CLI—at a minimum—has:

- `cdd-python --help`
- `cdd-python --version`
- `cdd-python from_openapi to_sdk_cli -i spec.json`
- `cdd-python from_openapi to_sdk -i spec.json`
- `cdd-python from_openapi to_server -i spec.json`
- `cdd-python to_openapi -f path/to/code`
- `cdd-python to_docs_json --no-imports --no-wrapping -i spec.json`
- `cdd-python serve_json_rpc --port 8080 --listen 0.0.0.0`

The goal of this project is to enable rapid application development without tradeoffs. Tradeoffs of Protocol Buffers / Thrift etc. are an untouchable "generated" directory and package, compile-time and/or runtime overhead. Tradeoffs of Java or JavaScript for everything are: overhead in hardware access, offline mode, ML inefficiency, and more. And neither of these alternative approaches are truly integrated into your target system, test frameworks, and bigger abstractions you build in your app. Tradeoffs in CDD are code duplication (but CDD handles the synchronisation for you).

## 🚀 Capabilities

The `cdd-python` compiler leverages a unified architecture to support various facets of API and code lifecycle management.

- **Compilation**:
    - **OpenAPI → `Python`**: Generate idiomatic native models, network routes, client SDKs, and boilerplate directly from OpenAPI (`.json` / `.yaml`) specifications.
    - **`Python` → OpenAPI**: Statically parse existing `Python` source code and emit compliant OpenAPI specifications.
- **AST-Driven & Safe**: Employs static analysis instead of unsafe dynamic execution or reflection, allowing it to safely parse and emit code even for incomplete or un-compilable project states.
- **Seamless Sync**: Keep your docs, tests, database, clients, and routing in perfect harmony. Update your code, and generate the docs; or update the docs, and generate the code.

## 📦 Installation & Build

### Native Tooling

```bash
python -m pip install -e .
python -m pytest
```

### Makefile / make.bat

You can also use the included cross-platform Makefiles to fetch dependencies, build, and test:

```bash
# Install dependencies
make deps

# Build the project
make build

# Run tests
make test
```

## 🛠 Usage

### Command Line Interface

```bash
# Generate Python models from an OpenAPI spec
cdd-python from_openapi to_sdk -i spec.json -o src/models

# Generate an OpenAPI spec from your Python code
cdd-python to_openapi -f src/models -o openapi.json
```

### Programmatic SDK / Library

```py
from cdd import generate_sdk, Config

if __name__ == '__main__':
    config = Config(input_path='spec.json', output_dir='src/models')
    generate_sdk(config)
    print("SDK generation complete.")
```

## 🏗 Supported Conversions for Python

*(The boxes below reflect the features supported by this specific `cdd-python` implementation)*

| Features | Parse (From) | Emit (To) |
| --- | --- | --- |
| OpenAPI 3.2.0 | ✅ | ✅ |
| API Client SDK | ✅ | ✅ |
| API Client CLI | ✅ | ✅ |
| Server Routes / Endpoints | ✅ | [ ] |
| ORM / DB Schema | ✅ | ✅ |
| Mocks + Tests | ✅ | ✅ |
| Model Context Protocol (MCP) | [ ] | [ ] |

### Uncommon Features

`cdd-python` supports additional integrations natively not found in standard API generators:
- **Google Discovery JSON:** Native support for parsing Google Discovery documents in addition to OpenAPI specs.
- **Legacy Swagger:** First-class compatibility and fallback for Swagger 2.0 schemas.
- **Deep Python Ecosystem Integration:** Native parsing and emitting betwixt Python functions, `class`es, docstrings (Google, NumPy, ReST), `argparse`, `pydantic`, and `SQLAlchemy`.

---

## CLI Help

$ python -m cdd --help
    usage: python -m cdd [-h] [--version]
                     {sync_properties,sync,gen,gen_routes,openapi,doctrans,exmod}
                     ...

    Open API to/fro routes, models, and tests. Convert between docstrings,
    classes, methods, argparse, pydantic, and SQLalchemy.
    
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
                              [--function-name FUNCTION_NAMES] [--no-word-wrap]
                              --truth
                              {argparse_function,class,function,sqlalchemy,sqlalchemy_hybrid,sqlalchemy_table}
    
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
      --no-word-wrap        Whether word-wrap is disabled (on emission). None
                            enables word-wrap. Defaults to None.
      --truth {argparse_function,class,function,sqlalchemy,sqlalchemy_hybrid,sqlalchemy_table}
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
                             [--parse {argparse,class,function,json_schema,pydantic,sqlalchemy,sqlalchemy_hybrid,sqlalchemy_table,infer}]
                             --emit
                             {argparse,class,function,json_schema,pydantic,sqlalchemy,sqlalchemy_hybrid,sqlalchemy_table}
                             -o OUTPUT_FILENAME [--emit-call]
                             [--emit-and-infer-imports] [--no-word-wrap]
                             [--decorator DECORATOR_LIST] [--phase PHASE]
    
    options:
      -h, --help            show this help message and exit
      --name-tpl NAME_TPL   Template for the name, e.g., `{name}Config`.
      --input-mapping INPUT_MAPPING
                            Fully-qualified module, filepath, or directory.
      --prepend PREPEND     Prepend file with this. Use '\n' for newlines.
      --imports-from-file IMPORTS_FROM_FILE
                            Extract imports from file and append to `output_file`.
                            If module or other symbol path given, resolve file
                            then use it.
      --parse {argparse,class,function,json_schema,pydantic,sqlalchemy,sqlalchemy_hybrid,sqlalchemy_table,infer}
                            What type the input is.
      --emit {argparse,class,function,json_schema,pydantic,sqlalchemy,sqlalchemy_hybrid,sqlalchemy_table}
                            Which type to generate.
      -o OUTPUT_FILENAME, --output-filename OUTPUT_FILENAME
                            Output file to write to.
      --emit-call           Whether to place all the previous body into a new
                            `__call__` internal function
      --emit-and-infer-imports
                            Whether to emit and infer imports at the top of the
                            generated code
      --no-word-wrap        Whether word-wrap is disabled (on emission). None
                            enables word-wrap. Defaults to None.
      --decorator DECORATOR_LIST
                            List of decorators.
      --phase PHASE         Which phase to run through. E.g., SQLalchemy may
                            require multiple phases to resolve foreign keys.

PS: If you're outputting JSON-schema and want a file per schema then:

    python -c 'import sys,json,os; f=open(sys.argv[1], "rt"); d=json.load(f); f.close(); [(lambda f: json.dump(sc,f) or f.close())(open(os.path.join(os.path.dirname(sys.argv[1]), sc["$id"].rpartition("/")[2]), "wt")) for sc in d["schemas"]]' <path_to_json_file>

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
                                 MODEL_PATHS --routes-paths [ROUTES_PATHS ...]
    
    options:
      -h, --help            show this help message and exit
      --app-name APP_NAME   Name of app (e.g., `app_name = Bottle();
                            @app_name.get('/api') def slash(): pass`)
      --model-paths MODEL_PATHS
                            Python module resolution (foo.models) or filepath
                            (foo/models)
      --routes-paths [ROUTES_PATHS ...]
                            Python module resolution 'foo.routes' or filepath
                            'foo/routes'

### `doctrans`

    $ python -m cdd doctrans --help
    usage: python -m cdd doctrans [-h] --filename FILENAME --format
                                  {rest,google,numpydoc}
                                  (--type-annotations | --no-type-annotations | --no-word-wrap)
    
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
      --no-word-wrap        Whether word-wrap is disabled (on emission). None
                            enables word-wrap. Defaults to None.

### `exmod`

    $ python -m cdd exmod --help
    usage: python -m cdd exmod [-h] -m MODULE --emit
                               {argparse,class,function,json_schema,pydantic,sqlalchemy,sqlalchemy_hybrid,sqlalchemy_table}
                               [--emit-sqlalchemy-submodule]
                               [--extra-module [EXTRA_MODULES]] [--no-word-wrap]
                               [--blacklist BLACKLIST] [--whitelist WHITELIST] -o
                               OUTPUT_DIRECTORY
                               [--target-module-name TARGET_MODULE_NAME] [-r]
                               [--dry-run]
    
    options:
      -h, --help            show this help message and exit
      -m MODULE, --module MODULE
                            The module or fully-qualified name (FQN) to expose.
      --emit {argparse,class,function,json_schema,pydantic,sqlalchemy,sqlalchemy_hybrid,sqlalchemy_table}
                            Which type to generate.
      --emit-sqlalchemy-submodule
                            Whether to; for sqlalchemy*; emit submodule "sqlalchem
                            y_mod/{__init__,connection,create_tables}.py"
      --extra-module [EXTRA_MODULES]
                            Additional module(s) to expose; specifiable multiple
                            times. Added to symbol auto-import resolver.
      --no-word-wrap        Whether word-wrap is disabled (on emission). None
                            enables word-wrap. Defaults to None.
      --blacklist BLACKLIST
                            Modules/FQN to omit. If unspecified will emit all
                            (unless whitelist).
      --whitelist WHITELIST
                            Modules/FQN to emit. If unspecified will emit all
                            (minus blacklist).
      -o OUTPUT_DIRECTORY, --output-directory OUTPUT_DIRECTORY
                            Where to place the generated exposed interfaces to the
                            given `--module`.
      --target-module-name TARGET_MODULE_NAME
                            Target module name. Defaults to `${module}___gold`.
      -r, --recursive       Recursively traverse module hierarchy and recreate
                            hierarchy with exposed interfaces
      --dry-run             Show what would be created; don't actually write to
                            the filesystem.

PS: Below is a temporary hack to run on the SQLalchemy output to make it work; until the `tuple`|`Tuple`|`List`|`list`|
`name` as column-type bug is resolved:

    fastmod --accept-all -iF 'tuple, comment=' 'LargeBinary, comment=' ; fastmod --accept-all -iF 'tuple,
            comment=' 'LargeBinary, comment=' ; fastmod --accept-all -iF 'list, comment=' 'LargeBinary, comment=' ; fastmod --accept-all -iF 'list,
            comment=' 'LargeBinary, comment=' ; fastmod --accept-all -iF 'name, comment=' 'String, comment='

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
