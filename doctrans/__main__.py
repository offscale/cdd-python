# !/usr/bin/env python

"""
`__main__` implementation, can be run directly or with `python -m doctrans`
"""

from argparse import ArgumentParser
from os import path

from doctrans import __version__


def _build_parser():
    """
    Parser builder

    :returns: instanceof ArgumentParser
    :rtype: ```ArgumentParser```
    """
    parser = ArgumentParser(
        prog='python -m doctrans',
        description='Translate between docstrings, classes, and argparse'
    )
    parser.add_argument('--version', action='version', version='%(prog)s {}'.format(__version__))

    parser.add_argument('--config', help='File where config `class` is declared.')
    parser.add_argument('--config-name', help='Name of `class`', default='Config')

    parser.add_argument('--function', help='File where function is `def`ined.')
    parser.add_argument('--function-name', help='Name of Function. If method, use C++ syntax, '
                                                'i.e., ClassName::method_name',
                        default='train')

    parser.add_argument('--argparse-function', help='File where argparse function is `def`ined.')
    parser.add_argument('--argparse-function-name', help='Name of argparse function.',
                        default='set_cli_args')

    parser.add_argument('--truth', help='Single source of truth. Others will be generated from this.',
                        choices=('argparse_function', 'config', 'function'), required=True)

    return parser


def main(cli_argv=None):
    """
    Run the CLI parser

    :param cli_argv: CLI arguments. If None uses `sys.argv`.
    :type cli_argv: ```Optional[List[str]]```
    """
    _parser = _build_parser()
    args = _parser.parse_args(args=cli_argv)
    args.argparse_function = args.argparse_function or args.config or args.function
    args.config = args.config or args.argparse_function
    args.function = args.function or args.config

    if args.argparse_function is None:
        _parser.error('One or more of `--argparse-function`, `--config`, and `--function` must be specified.')
    elif not path.isfile(args.truth):
        _parser.error('--truth must be choose an existent file. Got: {!r}'.format(getattr(args, args.truth)))


if __name__ == '__main__':
    main()
