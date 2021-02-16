""" Tests for CLI sync_properties subparser (__main__.py) """
import os
from tempfile import TemporaryDirectory
from unittest import TestCase
from unittest.mock import patch

from cdd.tests.utils_for_tests import mock_function, run_cli_test, unittest_main


class TestCliSyncProperties(TestCase):
    """ Test class for __main__.py """

    def test_sync_properties_fails(self) -> None:
        """ Tests CLI interface failure cases """
        run_cli_test(
            self,
            ["sync_properties", "--wrong"],
            exit_code=2,
            output="the following arguments are required:"
            " --input-filename, --input-param, --output-filename, --output-param\n",
        )

    def test_non_existent_file_fails(self) -> None:
        """ Tests nonexistent file throws the right error """
        with TemporaryDirectory() as tempdir:
            filename = os.path.join(
                tempdir,
                "delete_this_1{}".format(os.path.basename(__file__)),
            )

            run_cli_test(
                self,
                [
                    "sync_properties",
                    "--input-file",
                    filename,
                    "--input-param",
                    "Foo.g.f",
                    "--output-file",
                    filename,
                    "--output-param",
                    "f.h",
                ],
                exit_code=2,
                output="--input-file must be an existent file. Got: {!r}\n".format(
                    filename
                ),
            )

        with TemporaryDirectory() as tempdir:
            input_filename = os.path.join(
                tempdir,
                "input_filename.py",
            )
            output_filename = os.path.join(
                tempdir,
                "output_filename.py",
            )
            open(input_filename, "wt").close()

            run_cli_test(
                self,
                [
                    "sync_properties",
                    "--input-file",
                    input_filename,
                    "--input-param",
                    "Foo.g.f",
                    "--output-file",
                    output_filename,
                    "--output-param",
                    "f.h",
                ],
                exit_code=2,
                output="--output-file must be an existent file. Got: {!r}\n".format(
                    output_filename
                ),
            )

    def test_sync_properties(self) -> None:
        """ Tests CLI interface gets all the way to the sync_properties call without error """
        with TemporaryDirectory() as tempdir:
            input_filename = os.path.join(tempdir, "class_.py")
            output_filename = os.path.join(tempdir, "method.py")
            open(input_filename, "wt").close()
            open(output_filename, "wt").close()

            with patch("cdd.__main__.sync_properties", mock_function):
                self.assertTrue(
                    run_cli_test(
                        self,
                        [
                            "sync_properties",
                            "--input-file",
                            input_filename,
                            "--input-param",
                            "Foo.g.f",
                            "--output-file",
                            output_filename,
                            "--output-param",
                            "f.h",
                        ],
                        exit_code=None,
                        output=None,
                    ),
                )


unittest_main()
