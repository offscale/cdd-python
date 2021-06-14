#!/usr/bin/env python

"""
`__main__` implementation, can be run directly or with `python -m cdd`
"""

from argparse import ArgumentParser, Namespace
from codecs import decode
from collections import deque
from itertools import filterfalse
from operator import eq
from os import path

from cdd import __description__, __version__
from cdd.conformance import ground_truth
from cdd.docstring_parsers import Style
from cdd.doctrans import doctrans
from cdd.exmod import exmod
from cdd.gen import gen
from cdd.openapi.gen_openapi import openapi_bulk
from cdd.openapi.gen_routes import gen_routes, upsert_routes
from cdd.pure_utils import pluralise, rpartial
from cdd.sync_properties import sync_properties


def _build_parser():
    """
    Parser builder

    :returns: instanceof ArgumentParser
    :rtype: ```ArgumentParser```
    """
    parser = ArgumentParser(
        prog="python -m cdd",
        description=__description__,
    )
    parser.add_argument(
        "--version",
        action="version",
        version="%(prog)s {__version__}".format(__version__=__version__),
    )

    subparsers = parser.add_subparsers()
    subparsers.required = True
    subparsers.dest = "command"

    parse_emit_types = "argparse", "class", "function", "sqlalchemy", "sqlalchemy_table"

    ############
    # Property #
    ############
    property_parser = subparsers.add_parser(
        "sync_properties",
        help=(
            "Synchronise one or more properties between input and input_str Python"
            " files"
        ),
    )

    property_parser.add_argument(
        "--input-filename",
        help="File to find `--input-param` from",
        required=True,
        type=str,
    )
    property_parser.add_argument(
        "--input-param",
        help=(
            "Location within file of property. Can be top level like `a` for `a=5` or"
            " with the `.` syntax as in `--output-param`."
        ),
        required=True,
        action="append",
        type=str,
        dest="input_params",
    )
    property_parser.add_argument(
        "--input-eval",
        help="Whether to evaluate the input-param, or just leave it",
        action="store_true",
    )
    property_parser.add_argument(
        "--output-filename",
        help=(
            "Edited in place, the property within this file (to update) is selected by"
            " --output-param"
        ),
        type=str,
        required=True,
    )
    property_parser.add_argument(
        "--output-param",
        help=(
            "Parameter to update. E.g., `A.F` for `class A: F`, `f.g` for `def f(g):"
            " pass`"
        ),
        required=True,
        action="append",
        type=str,
        dest="output_params",
    )
    property_parser.add_argument(
        "--output-param-wrap",
        type=str,
        help=(
            "Wrap all input_str params with this. E.g., `Optional[Union[{output_param},"
            " str]]`"
        ),
    )

    ########
    # Sync #
    ########
    sync_parser = subparsers.add_parser(
        "sync", help="Force argparse, classes, and/or methods to be equivalent"
    )

    sync_parser.add_argument(
        "--argparse-function",
        help="File where argparse function is `def`ined.",
        action="append",
        type=str,
        dest="argparse_functions",
    )
    sync_parser.add_argument(
        "--argparse-function-name",
        help="Name of argparse function.",
        action="append",
        type=str,
        dest="argparse_function_names",
    )
    sync_parser.add_argument(
        "--class",
        help="File where class `class` is declared.",
        action="append",
        type=str,
        dest="classes",
    )
    sync_parser.add_argument(
        "--class-name",
        help="Name of `class`",
        action="append",
        type=str,
        dest="class_names",
    )
    sync_parser.add_argument(
        "--function",
        help="File where function is `def`ined.",
        action="append",
        type=str,
        dest="functions",
    )
    sync_parser.add_argument(
        "--function-name",
        help=(
            "Name of Function. If method, use Python resolution syntax,"
            " i.e., ClassName.function_name"
        ),
        action="append",
        type=str,
        dest="function_names",
    )
    sync_parser.add_argument(
        "--truth",
        help=(
            "Single source of truth. Others will be generated from this. Will run with"
            " first found choice."
        ),
        choices=(
            "argparse_function",
            "class",
            "function",
            "sqlalchemy",
            "sqlalchemy_table",
        ),
        type=str,
        required=True,
    )

    #######
    # Gen #
    #######
    gen_parser = subparsers.add_parser(
        "gen",
        help=(
            "Generate classes, functions, argparse function, sqlalchemy tables and/or sqlalchemy classes"
            " from the input mapping"
        ),
    )

    gen_parser.add_argument(
        "--name-tpl", help="Template for the name, e.g., `{name}Config`.", required=True
    )
    gen_parser.add_argument(
        "--input-mapping",
        help="Import location of dictionary/mapping/2-tuple collection.",
        required=True,
    )
    gen_parser.add_argument(
        "--prepend",
        help="Prepend file with this. Use '\\n' for newlines.",
        type=lambda arg: decode(str(arg), "unicode_escape"),
    )
    gen_parser.add_argument(
        "--imports-from-file",
        help=(
            "Extract imports from file and append to `output_file`. "
            "If module or other symbol path given, resolve file then use it."
        ),
    )
    gen_parser.add_argument(
        "--parse",
        help="What type the input is.",
        choices=parse_emit_types,
        default="infer",
        dest="parse_name",
    )
    gen_parser.add_argument(
        "--emit",
        help="What type to generate.",
        choices=parse_emit_types,
        required=True,
        dest="emit_name",
    )
    gen_parser.add_argument(
        "--output-filename", "-o", help="Output file to write to.", required=True
    )
    gen_parser.add_argument(
        "--emit-call",
        action="store_true",
        help=(
            "Whether to place all the previous body into a new `__call__` internal"
            " function"
        ),
    )
    gen_parser.add_argument(
        "--decorator",
        help="List of decorators.",
        action="append",
        type=str,
        dest="decorator_list",
    )

    ##############
    # gen_routes #
    ##############
    routes_parser = subparsers.add_parser(
        "gen_routes", help="Generate per model route(s)"
    )

    routes_parser.add_argument(
        "--crud",
        help="What of (C)reate, (R)ead, (U)pdate, (D)elete to generate",
        choices=("CRUD", "CR", "C", "R", "U", "D", "CR", "CU", "CD", "CRD"),
        required=True,
    )
    routes_parser.add_argument(
        "--app-name",
        help="Name of app (e.g., `app_name = Bottle();\n@app_name.get('/api')\ndef slash(): pass`)",
        default="rest_api",
    )
    routes_parser.add_argument(
        "--model-path",
        help="Python module resolution (foo.models) or filepath (foo/models)",
        required=True,
    )
    routes_parser.add_argument(
        "--model-name", help="Name of model to generate from", required=True
    )
    routes_parser.add_argument(
        "--routes-path",
        help="Python module resolution 'foo.routes' or filepath 'foo/routes'",
        required=True,
    )
    routes_parser.add_argument(
        "--route",
        help="Name of the route, defaults to `/api/{model_name.lower()}`",
    )

    ###########
    # openapi #
    ###########
    openapi_parser = subparsers.add_parser(
        "openapi", help="Generate OpenAPI schema from specified project(s)"
    )

    openapi_parser.add_argument(
        "--app-name",
        help="Name of app (e.g., `app_name = Bottle();\n@app_name.get('/api')\ndef slash(): pass`)",
        default="rest_api",
    )
    openapi_parser.add_argument(
        "--model-paths",
        help="Python module resolution (foo.models) or filepath (foo/models)",
        required=True,
    )
    openapi_parser.add_argument(
        "--routes-paths",
        help="Python module resolution 'foo.routes' or filepath 'foo/routes'",
        nargs="*",
        required=True,
    )

    ############
    # doctrans #
    ############
    doctrans_parser = subparsers.add_parser(
        "doctrans",
        help=(
            "Convert docstring format of all classes and functions within target file"
        ),
    )

    doctrans_parser.add_argument(
        "--filename",
        help="Python file to convert docstrings within. Edited in place.",
        type=str,
        required=True,
    )
    doctrans_parser.add_argument(
        "--format",
        help="The docstring format to replace existing format with.",
        type=str,
        choices=tuple(filterfalse(rpartial(eq, "auto"), Style.__members__.keys())),
        required=True,
    )

    doctrans_parser_group = doctrans_parser.add_mutually_exclusive_group(required=True)
    doctrans_parser_group.add_argument(
        "--type-annotations",
        help="Inline the type, i.e., annotate PEP484 (outside docstring. Requires 3.6+)",
        dest="type_annotations",
        action="store_true",
    )
    doctrans_parser_group.add_argument(
        "--no-type-annotations",
        help="Ensure all types are in docstring (rather than a PEP484 type annotation)",
        dest="type_annotations",
        action="store_false",
    )

    #########
    # exmod #
    #########
    exmod_parser = subparsers.add_parser(
        "exmod",
        help=(
            "Expose module hierarchy->{functions,classes,vars} for parameterisation "
            "via {REST API + database,CLI,SDK}"
        ),
    )

    exmod_parser.add_argument(
        "--module",
        "-m",
        help="The module or fully-qualified name (FQN) to expose.",
        required=True,
    )
    exmod_parser.add_argument(
        "--emit",
        help="What type to generate.",
        choices=parse_emit_types,
        required=True,
        action="append",
    )
    exmod_parser.add_argument(
        "--blacklist",
        help="Modules/FQN to omit. If unspecified will emit all (unless whitelist).",
        action="append",
    )
    exmod_parser.add_argument(
        "--whitelist",
        help="Modules/FQN to emit. If unspecified will emit all (minus blacklist).",
        action="append",
    )
    exmod_parser.add_argument(
        "--output-directory",
        "-o",
        help="Where to place the generated exposed interfaces to the given `--module`.",
        required=True,
    )
    exmod_parser.add_argument(
        "--dry-run",
        help="Show what would be created; don't actually write to the filesystem.",
        action="store_true",
    )

    return parser


def main(cli_argv=None, return_args=False):
    """
    Run the CLI parser

    :param cli_argv: CLI arguments. If None uses `sys.argv`.
    :type cli_argv: ```Optional[List[str]]```

    :param return_args: Primarily use is for tests. Returns the args rather than executing anything.
    :type return_args: ```bool```

    :returns: the args if `return_args`, else None
    :rtype: ```Optional[Namespace]```
    """
    _parser = _build_parser()
    args = _parser.parse_args(args=cli_argv)
    command = args.command
    args_dict = {k: v for k, v in vars(args).items() if k != "command"}
    if command == "sync":
        args = Namespace(
            **{
                k: v if k == "truth" or isinstance(v, list) or v is None else [v]
                for k, v in args_dict.items()
            }
        )

        truth_file = getattr(args, pluralise(args.truth))
        require_file_existent(
            _parser, truth_file[0] if truth_file else truth_file, name="truth"
        )
        truth_file = path.realpath(path.expanduser(truth_file[0]))

        number_of_files = sum(
            len(val)
            for key, val in vars(args).items()
            if isinstance(val, list) and not key.endswith("_names")
        )

        if number_of_files < 2:
            _parser.error(
                "Two or more of `--argparse-function`, `--class`, and `--function` must"
                " be specified"
            )
        require_file_existent(_parser, truth_file, name="truth")

        return args if return_args else ground_truth(args, truth_file)
    elif command == "sync_properties":
        deque(
            (
                setattr(
                    args, fname, path.realpath(path.expanduser(getattr(args, fname)))
                )
                for fname in ("input_filename", "output_filename")
                if path.isfile(getattr(args, fname))
            ),
            maxlen=0,
        )

        for filename, arg_name in (args.input_filename, "input-file"), (
            args.output_filename,
            "output-file",
        ):
            require_file_existent(_parser, filename, name=arg_name)
        sync_properties(**args_dict)
    elif command == "gen":
        if path.isfile(args.output_filename):
            raise IOError(
                "File exists and this is a destructive operation. Delete/move {output_filename!r} then"
                " rerun.".format(output_filename=args.output_filename)
            )
        gen(**args_dict)
    elif command == "gen_routes":
        if args.route is None:
            args.route = "/api/{model_name}".format(model_name=args.model_name.lower())

        (
            lambda routes__primary_key: upsert_routes(
                app=args.app_name,
                route=args.route,
                routes=routes__primary_key[0],
                routes_path=getattr(args, "routes_path", None),
                primary_key=routes__primary_key[1],
            )
        )(
            gen_routes(
                app=args.app_name,
                crud=args.crud,
                model_name=args.model_name,
                model_path=args.model_path,
                route=args.route,
            )
        )
    elif command == "openapi":
        openapi_bulk(
            app_name=args.app_name,
            model_paths=args.model_paths,
            routes_paths=args.routes_paths,
        )
    elif command == "doctrans":
        require_file_existent(_parser, args.filename, name="filename")
        doctrans(
            filename=args.filename,
            docstring_format=args.format,
            type_annotations=args.type_annotations,
        )
    elif command == "exmod":
        exmod(
            module=args.module,
            emit_name=args.emit,
            blacklist=args.blacklist,
            whitelist=args.whitelist,
            output_directory=args.output_directory,
            dry_run=args.dry_run,
        )


def require_file_existent(_parser, filename, name):
    """
    Raise SystemExit(2) if filename is None or not found

    :param _parser: The argparse parser
    :type _parser: ```ArgumentParser```

    :param filename: The filename
    :type filename: ```Optional[str]```

    :param name: Argument name
    :type name: ```str```
    """
    if filename is None or not path.isfile(filename):
        _parser.error(
            "--{name} must be an existent file. Got: {filename!r}".format(
                name=name, filename=filename
            )
        )


if __name__ == "__main__":
    main()

__all__ = ["main"]
