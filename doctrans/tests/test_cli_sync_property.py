""" Tests for CLI sync_property subparser (__main__.py) """
import os
from tempfile import TemporaryDirectory
from unittest import TestCase
from unittest.mock import patch

from doctrans.tests.utils_for_tests import unittest_main, run_cli_test


class TestCliSyncProperty(TestCase):
    """ Test class for __main__.py """

    def test_sync_property_fails(self) -> None:
        """ Tests CLI interface failure cases """
        run_cli_test(
            self,
            ["sync_property", "--wrong"],
            exit_code=2,
            output="the following arguments are required:"
            " --input-file, --input-param, --output-file, --output-param\n",
        )

    def test_non_existent_file_fails(self) -> None:
        """ Tests nonexistent file throws the right error """
        with TemporaryDirectory() as tempdir:
            filename = os.path.join(
                tempdir, "delete_this_1{}".format(os.path.basename(__file__)),
            )

            run_cli_test(
                self,
                [
                    "sync_property",
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
            input_file = os.path.join(tempdir, "input_file.py",)
            output_file = os.path.join(tempdir, "output_file.py",)
            open(input_file, "wt").close()

            run_cli_test(
                self,
                [
                    "sync_property",
                    "--input-file",
                    input_file,
                    "--input-param",
                    "Foo.g.f",
                    "--output-file",
                    output_file,
                    "--output-param",
                    "f.h",
                ],
                exit_code=2,
                output="--output-file must be an existent file. Got: {!r}\n".format(
                    output_file
                ),
            )

    def test_sync_property(self) -> None:
        """ Tests CLI interface gets all the way to the sync_property call without error """
        with TemporaryDirectory() as tempdir:
            class_py = os.path.join(tempdir, "class_.py")
            method_py = os.path.join(tempdir, "method.py")
            open(class_py, "wt").close()
            open(method_py, "wt").close()

            def _sync_property(*args, **kwargs):
                return True

            with patch("doctrans.__main__.sync_property", _sync_property):
                self.assertTrue(
                    run_cli_test(
                        self,
                        [
                            "sync_property",
                            "--input-file",
                            class_py,
                            "--input-param",
                            "Foo.g.f",
                            "--output-file",
                            method_py,
                            "--output-param",
                            "f.h",
                        ],
                        exit_code=None,
                        output=None,
                    ),
                )


unittest_main()
