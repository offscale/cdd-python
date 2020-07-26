""" Tests for CLI (__main__.py) """
from argparse import ArgumentParser
from unittest import TestCase, main as unittest_main
from unittest.mock import MagicMock, patch

from doctrans import __version__
from doctrans.__main__ import _build_parser, main


class TestCli(TestCase):
    """ Test class for __main__.py """

    def test_build_parser(self) -> None:
        """ Test that `_build_parser` produces a parser object """
        parser = _build_parser()
        self.assertIsInstance(parser, ArgumentParser)
        self.assertEqual(parser.description,
                         'Translate between docstrings, classes, and argparse')

    def run_cli_test(self, argv, exit_code, output, output_checker):
        """
        CLI test helper, wraps exit code and stdout/stderr output

        :param argv: argv, can be sys.argv or proxy
        :type argv: ```List[str]```

        :param exit_code: exit code
        :type exit_code: ```int```

        :param output: string representation (from stdout/stderr)
        :type output: ```str```

        :param output_checker: Function to check the output with
        :type output_checker: ```Callable[[str], bool]```

        :return: output
        :rtype: ```str```
        """
        argparse_mock = MagicMock()
        with patch('argparse.ArgumentParser._print_message', argparse_mock), patch('sys.argv', argv):
            with self.assertRaises(SystemExit) as e:
                main(cli_argv=argv)
        self.assertEqual(e.exception.code, SystemExit(exit_code).code)
        self.assertEqual(output_checker(argparse_mock.call_args.args[0]),
                         output)
        return output

    def test_version(self) -> None:
        """ Tests CLI interface gives version """
        self.run_cli_test(['--version'], exit_code=0, output=__version__,
                          output_checker=lambda output: output.rpartition(' ')[2][:-1])

    def test_incorrect_arg_fails(self) -> None:
        """ Tests CLI interface failure cases """
        self.run_cli_test(['--wrong'], exit_code=2, output='the following arguments are required: --truth\n',
                          output_checker=lambda output: output.partition('error: ')[2])


if __name__ == '__main__':
    unittest_main()
