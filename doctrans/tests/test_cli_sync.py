""" Tests for CLI sync subparser (__main__.py) """

import os
from tempfile import TemporaryDirectory
from unittest import TestCase

from doctrans import __version__
from doctrans.tests.mocks.classes import class_str
from doctrans.tests.utils_for_tests import unittest_main, run_cli_test


class TestCliSync(TestCase):
    """ Test class for __main__.py """

    def test_version(self) -> None:
        """ Tests CLI interface gives version """
        run_cli_test(
            self,
            ["--version"],
            exit_code=0,
            output=__version__,
            output_checker=lambda output: output[output.rfind(" ") + 1 :][:-1],
        )

    def test_args(self) -> None:
        """ Tests CLI interface sets namespace correctly """
        filename = os.path.join(
            os.path.dirname(__file__),
            "delete_this_0{}".format(os.path.basename(__file__)),
        )
        with open(filename, "wt") as f:
            f.write(class_str)
        try:
            _, args = run_cli_test(
                self,
                [
                    "sync",
                    "--class",
                    filename,
                    "--argparse-function",
                    filename,
                    "--truth",
                    "class",
                ],
                exit_code=None,
                output=None,
                return_args=True,
            )
        finally:
            if os.path.isfile(filename):
                os.remove(filename)

        self.assertListEqual(args.argparse_functions, [filename])
        self.assertListEqual(args.argparse_function_names, ["set_cli_args"])

        self.assertListEqual(args.classes, [filename])
        self.assertListEqual(args.class_names, ["ConfigClass"])

        self.assertEqual(args.truth, "class")

    def test_non_existent_file_fails(self) -> None:
        """ Tests nonexistent file throws the right error """
        filename = os.path.join(
            os.path.dirname(__file__),
            "delete_this_1{}".format(os.path.basename(__file__)),
        )

        run_cli_test(
            self,
            [
                "sync",
                "--argparse-function",
                filename,
                "--class",
                filename,
                "--truth",
                "class",
            ],
            exit_code=2,
            output="--truth must be an existent file. Got: {!r}\n".format(filename),
        )

    def test_missing_argument_fails(self) -> None:
        """ Tests missing argument throws the right error """
        run_cli_test(
            self,
            ["sync", "--truth", "class"],
            exit_code=2,
            output="--truth must be an existent file. Got: None\n",
        )

    def test_missing_argument_fails_insufficient_args(self) -> None:
        """ Tests missing argument throws the right error """
        with TemporaryDirectory() as tmpdir:
            filename = os.path.join(
                tmpdir, "delete_this_2{}".format(os.path.basename(__file__)),
            )
            with open(filename, "wt") as f:
                f.write(class_str)
            run_cli_test(
                self,
                ["sync", "--truth", "class", "--class", filename],
                exit_code=2,
                output="Two or more of `--argparse-function`, `--class`, and `--function` must be specified\n",
            )

    def test_incorrect_arg_fails(self) -> None:
        """ Tests CLI interface failure cases """
        run_cli_test(
            self,
            ["sync", "--wrong"],
            exit_code=2,
            output="the following arguments are required: --truth\n",
        )


unittest_main()
