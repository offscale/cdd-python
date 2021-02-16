""" Tests for CLI (__main__.py) """
import os
from argparse import ArgumentParser
from importlib.machinery import SourceFileLoader
from importlib.util import module_from_spec, spec_from_loader
from unittest import TestCase
from unittest.mock import MagicMock, patch

from cdd import __version__
from cdd.__main__ import _build_parser
from cdd.pure_utils import PY3_8
from cdd.tests.utils_for_tests import run_cli_test, unittest_main


class TestCli(TestCase):
    """ Test class for __main__.py """

    def test_build_parser(self) -> None:
        """ Test that `_build_parser` produces a parser object """
        parser = _build_parser()
        self.assertIsInstance(parser, ArgumentParser)
        self.assertEqual(
            parser.description,
            "Translate between docstrings, classes, methods, and argparse.",
        )

    def test_version(self) -> None:
        """ Tests CLI interface gives version """
        run_cli_test(
            self,
            ["--version"],
            exit_code=0,
            output=__version__,
            output_checker=lambda output: output[output.rfind(" ") + 1 :][:-1],
        )

    def test_name_main(self) -> None:
        """ Test the `if __name__ == '__main___'` block """

        argparse_mock = MagicMock()

        loader = SourceFileLoader(
            "__main__",
            os.path.join(os.path.dirname(os.path.dirname(__file__)), "__main__.py"),
        )
        with patch("argparse.ArgumentParser._print_message", argparse_mock), patch(
            "sys.argv", []
        ), self.assertRaises(SystemExit) as e:
            loader.exec_module(module_from_spec(spec_from_loader(loader.name, loader)))
        self.assertEqual(e.exception.code, SystemExit(2).code)

        self.assertEqual(
            (lambda output: output[(output.rfind(" ") + 1) :][:-1])(
                (argparse_mock.call_args.args if PY3_8 else argparse_mock.call_args[0])[
                    0
                ]
            ),
            "command",
        )


unittest_main()
