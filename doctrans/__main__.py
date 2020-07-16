# !/usr/bin/env python

from argparse import ArgumentParser
from os import path
from sys import stdout

from ml_params import __version__


def _build_parser():
    parser = ArgumentParser(
        prog='python -m doctrans',
        description='Translate between docstrings, classes, and argparse'
    )
    parser.add_argument('--version', action='version', version='%(prog)s {}'.format(__version__))

    parser.add_argument('--input', required=True,
                        help='Input file to extract docstrings from')
    parser.add_argument('--output', default=stdout,
                        help='Output file to send classes to')
    return parser


if __name__ == '__main__':
    _parser = _build_parser()
    args = _parser.parse_args()

    if not path.isfile(args.input):
        _parser.error('--input must be specify an existent file. Got: {!r}'.format(args.input))
