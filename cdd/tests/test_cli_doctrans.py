""" Tests for CLI doctrans subparser (__main__.py) """
from os import path
from tempfile import TemporaryDirectory
from unittest import TestCase
from unittest.mock import patch

from cdd.tests.utils_for_tests import mock_function, run_cli_test, unittest_main


class TestCliDocTrans(TestCase):
    """ Test class for __main__.py """

    def test_doctrans_fails_with_wrong_args(self) -> None:
        """ Tests CLI interface wrong args failure case """

        run_cli_test(
            self,
            ["doctrans", "--wrong"],
            exit_code=2,
            output="the following arguments are required: --filename, --format\n",
        )

    def test_doctrans_fails_with_file_missing(self) -> None:
        """ Tests CLI interface file missing failure case """

        with patch("cdd.__main__.doctrans", mock_function):
            self.assertTrue(
                run_cli_test(
                    self,
                    [
                        "doctrans",
                        "--filename",
                        "foo",
                        "--format",
                        "google",
                        "--no-type-annotations",
                    ],
                    exit_code=2,
                    output="--filename must be an existent file. Got: 'foo'\n",
                ),
            )

    def test_doctrans_succeeds(self) -> None:
        """ Tests CLI interface gets all the way to the doctrans call without error """

        with TemporaryDirectory() as tempdir:
            filename = path.join(tempdir, "foo")
            open(filename, "a").close()
            with patch("cdd.__main__.doctrans", mock_function):
                self.assertTrue(
                    run_cli_test(
                        self,
                        [
                            "doctrans",
                            "--filename",
                            filename,
                            "--format",
                            "numpydoc",
                            "--type-annotations",
                        ],
                        exit_code=None,
                        output=None,
                    ),
                )


unittest_main()
