""" Tests for CLI (__main__.py) """
import os
from argparse import ArgumentParser
from functools import partial
from unittest import TestCase, main as unittest_main
from unittest.mock import MagicMock, patch

from doctrans import __version__
from doctrans.__main__ import _build_parser, main
from doctrans.tests.mocks.classes import class_str


class TestCli(TestCase):
    """ Test class for __main__.py """

    def test_build_parser(self) -> None:
        """ Test that `_build_parser` produces a parser object """
        parser = _build_parser()
        self.assertIsInstance(parser, ArgumentParser)
        self.assertEqual(parser.description,
                         'Translate between docstrings, classes, and argparse')

    def run_cli_test(self, cli_argv,
                     exit_code, output,
                     output_checker=lambda output: (lambda q: output[output.find(q) + len(q):])('error: '),
                     return_args=False):
        """
        CLI test helper, wraps exit code and stdout/stderr output

        :param cli_argv: cli_argv, can be sys.argv or proxy
        :type cli_argv: ```List[str]```

        :param exit_code: exit code
        :type exit_code: ```Optional[int]```

        :param output: string representation (from stdout/stderr)
        :type output: ```Optional[str]```

        :param output_checker: Function to check the output with
        :type output_checker: ```Callable[[str], bool]```

        :param return_args: Primarily use is for tests. Returns the args rather than executing anything.
        :type return_args: ```bool```

        :return: output
        :rtype: ```Tuple[str, Optional[Namespace]]```
        """
        argparse_mock, args = MagicMock(), None
        with patch('argparse.ArgumentParser._print_message', argparse_mock), patch('sys.argv', cli_argv):
            main_f = partial(main, cli_argv=cli_argv, return_args=return_args)
            if exit_code is None:
                args = main_f()
            else:
                with self.assertRaises(SystemExit) as e:
                    args = main_f()
        if exit_code is not None:
            self.assertEqual(e.exception.code, SystemExit(exit_code).code)
        if argparse_mock.call_args is None:
            self.assertIsNone(output)
        else:
            self.assertEqual(output_checker(argparse_mock.call_args.args[0]),
                             output)
        return output, args

    def test_version(self) -> None:
        """ Tests CLI interface gives version """
        self.run_cli_test(['--version'],
                          exit_code=0,
                          output=__version__,
                          output_checker=lambda output: output[output.rfind(' ') + 1:][:-1])

    def test_args(self) -> None:
        """ Tests CLI interface sets namespace correctly """
        filename = os.path.join(os.path.dirname(__file__),
                                'delete_this_{}'.format(os.path.basename(__file__)))
        with open(filename, 'wt') as f:
            f.write(class_str)
        try:
            _, args = self.run_cli_test(
                ['--config', filename, '--truth', 'config'],
                exit_code=None,
                output=None,
                return_args=True
            )
        finally:
            if os.path.isfile(filename):
                os.remove(filename)

        self.assertEqual(args.argparse_function, filename)
        self.assertEqual(args.argparse_function_name, 'set_cli_args')
        self.assertEqual(args.config, filename)
        self.assertEqual(args.config_name, 'Config')
        self.assertEqual(args.function, filename)
        self.assertEqual(args.function_name, 'train')
        self.assertEqual(args.truth, filename)

    def test_non_existent_file_fails(self) -> None:
        """ Tests nonexistent file throws the right error """
        filename = os.path.join(os.path.dirname(__file__),
                                'delete_this_{}'.format(os.path.basename(__file__)))

        self.run_cli_test(
            ['--config', filename, '--truth', 'config'],
            exit_code=2,
            output='--truth must be choose an existent file. Got: \'{}\'\n'.format(filename),
        )

    def test_missing_argument_fails(self) -> None:
        """ Tests missing argument throws the right error """
        self.run_cli_test(
            ['--truth', 'config'],
            exit_code=2,
            output='One or more of `--argparse-function`, `--config`, and `--function` must be specified.\n'
        )

    def test_incorrect_arg_fails(self) -> None:
        """ Tests CLI interface failure cases """
        self.run_cli_test(
            ['--wrong'],
            exit_code=2,
            output='the following arguments are required: --truth\n'
        )


if __name__ == '__main__':
    unittest_main()
