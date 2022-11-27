""" Tests for CLI gen_routes subparser (__main__.py) """

from unittest import TestCase
from unittest.mock import patch

from cdd.tests.utils_for_tests import mock_function, run_cli_test, unittest_main


class TestCliGenRoutes(TestCase):
    """Test class for __main__.py"""

    def test_gen_routes_fails(self) -> None:
        """Tests CLI interface failure cases"""
        run_cli_test(
            self,
            ["gen_routes", "--wrong"],
            exit_code=2,
            output="the following arguments are required: --crud, --model-path, --model-name, --routes-path\n",
        )

    def test_gen_routes(self) -> None:
        """Tests CLI interface gets all the way to the gen_routes call without error"""

        with patch(
            "cdd.__main__.gen_routes", lambda *args, **kwargs: (True,) * 2
        ), patch("cdd.__main__.upsert_routes", mock_function):
            self.assertTrue(
                run_cli_test(
                    self,
                    [
                        "gen_routes",
                        "--crud",
                        "CRD",
                        "--model-path",
                        "cdd.tests.mocks.sqlalchemy",
                        "--routes-path",
                        "/api/config",
                        "--model-name",
                        "Config",
                    ],
                    exit_code=None,
                    output=None,
                ),
            )


unittest_main()
