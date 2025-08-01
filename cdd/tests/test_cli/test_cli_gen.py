"""Tests for CLI gen subparser (__main__.py)"""

import os
from os.path import extsep
from tempfile import TemporaryDirectory
from unittest import TestCase
from unittest.mock import patch

from cdd.tests.utils_for_tests import mock_function, run_cli_test, unittest_main


class TestCliGen(TestCase):
    """Test class for __main__.py"""

    def test_gen_fails(self) -> None:
        """Tests CLI interface failure cases"""
        run_cli_test(
            self,
            ["gen", "--wrong"],
            exit_code=2,
            output="the following arguments are required: --name-tpl, --input-mapping, --emit, -o/--output-filename\n",
        )

    def test_existent_file_fails(self) -> None:
        """Tests nonexistent file throws the right error"""
        with TemporaryDirectory() as tempdir:
            filename: str = os.path.join(
                tempdir,
                "delete_this_1{__file___basename}".format(
                    __file___basename=os.path.basename(__file__)
                ),
            )
            open(filename, "a").close()

            run_cli_test(
                self,
                [
                    "gen",
                    "--name-tpl",
                    "{name}Config",
                    "--input-mapping",
                    "cdd.pure_utils.simple_types",
                    "--emit",
                    "class",
                    "--output-filename",
                    filename,
                ],
                exception=OSError,
                exit_code=2,
                output="File exists and this is a destructive operation. Delete/move {filename!r} then rerun.".format(
                    filename=filename
                ),
            )

    def test_gen(self) -> None:
        """Tests CLI interface gets all the way to the gen call without error"""
        with TemporaryDirectory() as tempdir:
            output_filename: str = os.path.join(
                tempdir, "classes{extsep}py".format(extsep=extsep)
            )

            with patch("cdd.__main__.gen", mock_function):
                run_cli_test(
                    self,
                    [
                        "gen",
                        "--name-tpl",
                        "{name}Config",
                        "--input-mapping",
                        "cdd.pure_utils.simple_types",
                        "--emit",
                        "class",
                        "--output-filename",
                        output_filename,
                    ],
                    exit_code=None,
                    output=None,
                )


unittest_main()
