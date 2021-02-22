""" Tests for CLI openapi subparser (__main__.py) """

from unittest import TestCase
from unittest.mock import patch

from cdd.tests.utils_for_tests import mock_function, run_cli_test, unittest_main


class TestOpenApi(TestCase):
    """ Test class for __main__.py """

    def test_gen_routes_fails(self) -> None:
        """ Tests CLI interface failure cases """
        run_cli_test(
            self,
            ["openapi", "--wrong"],
            exit_code=2,
            output="the following arguments are required: --model-paths, --routes-paths\n",
        )

    def test_openapi(self) -> None:
        """ Tests CLI interface gets all the way to the `openapi` call without error """

        with patch("cdd.__main__.openapi_bulk", mock_function):
            self.assertTrue(
                run_cli_test(
                    self,
                    [
                        "openapi",
                        "--app-name",
                        "app",
                        "--model-paths",
                        "cdd.tests.mocks.sqlalchemy",
                        "--routes-paths",
                        "cdd.tests.mocks.routes",
                    ],
                    exit_code=None,
                    output=None,
                ),
            )


unittest_main()
