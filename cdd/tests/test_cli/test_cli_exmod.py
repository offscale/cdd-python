""" Tests for CLI exmod subparser (__main__.py) """

from unittest import TestCase
from unittest.mock import MagicMock, patch

from cdd.tests.utils_for_tests import run_cli_test, unittest_main


class TestCliExMod(TestCase):
    """Test class for __main__.py"""

    def test_exmod_fails(self) -> None:
        """Tests CLI interface exmod failure cases"""
        run_cli_test(
            self,
            ["exmod", "--wrong"],
            exit_code=2,
            output="the following arguments are required: --module/-m, --emit, --output-directory/-o\n",
        )

    def test_exmod_is_called(self) -> None:
        """Tests CLI interface exmod function gets called"""
        with patch("cdd.__main__.exmod", new_callable=MagicMock()):
            self.assertTrue(
                run_cli_test(
                    self,
                    [
                        "exmod",
                        "--module",
                        "foo",
                        "--emit",
                        "argparse",
                        "--output-directory",
                        "foo",
                    ],
                    exit_code=None,
                    output=None,
                ),
            )


unittest_main()
