# !/usr/bin/env python

"""
`__main__` implementation, can be run directly or with `python -m doctrans`
"""

from argparse import ArgumentParser, Namespace
from os import path

from doctrans import __version__
from doctrans.conformance import ground_truth
from doctrans.pure_utils import pluralise


def _build_parser():
    """
    Parser builder

    :returns: instanceof ArgumentParser
    :rtype: ```ArgumentParser```
    """
    parser = ArgumentParser(
        prog="python -m doctrans",
        description="Translate between docstrings, classes, methods, and argparse.",
    )
    parser.add_argument(
        "--version", action="version", version="%(prog)s {}".format(__version__)
    )

    subparsers = parser.add_subparsers()
    subparsers.required = True
    subparsers.dest = "command"
    property_parser = subparsers.add_parser(
        "property",
        help="Synchronise just one property on one class, method, or argparse",
    )
    property_parser.add_argument(
        "--prop",
        help="File where class `class` is declared.",
        action="append",
        dest="classes",
    )

    sync_parser = subparsers.add_parser(
        "sync", help="Force classes, methods, and/or argparse to be equivalent"
    )
    sync_parser.add_argument(
        "--class",
        help="File where class `class` is declared.",
        action="append",
        dest="classes",
    )
    sync_parser.add_argument(
        "--class-name",
        help="Name of `class`",
        default="ConfigClass",
        action="append",
        dest="class_names",
    )

    sync_parser.add_argument(
        "--function",
        help="File where function is `def`ined.",
        action="append",
        dest="functions",
    )
    sync_parser.add_argument(
        "--function-name",
        help="Name of Function. If method, use Python resolution syntax,"
        " i.e., ClassName.method_name",
        default="C.method_name",
        action="append",
        dest="function_names",
    )

    sync_parser.add_argument(
        "--argparse-function",
        help="File where argparse function is `def`ined.",
        action="append",
        dest="argparse_functions",
    )
    sync_parser.add_argument(
        "--argparse-function-name",
        help="Name of argparse function.",
        default="set_cli_args",
        action="append",
        dest="argparse_function_names",
    )

    sync_parser.add_argument(
        "--truth",
        help="Single source of truth. Others will be generated from this. Will run with first found choice.",
        choices=("argparse_function", "class", "function"),
        required=True,
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
    if command == "sync":
        args = Namespace(
            **{
                k: v if k == "truth" or isinstance(v, list) or v is None else [v]
                for k, v in vars(args).items()
                if k != "command"
            }
        )

        truth_file = getattr(args, pluralise(args.truth))
        if truth_file is None:
            _parser.error("--truth must be an existent file. Got: None")
        else:
            truth_file = truth_file[0]

        number_of_files = sum(
            len(val)
            for key, val in vars(args).items()
            if isinstance(val, list) and not key.endswith("_names")
        )

        if number_of_files < 2:
            _parser.error(
                "Two or more of `--argparse-function`, `--class`, and `--function` must be specified"
            )
        elif truth_file is None or not path.isfile(truth_file):
            _parser.error(
                "--truth must be an existent file. Got: {!r}".format(truth_file)
            )

        return args if return_args else ground_truth(args, truth_file)


if __name__ == "__main__":
    main()

__all__ = ["main"]
