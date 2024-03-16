""" Tests for pkg utils """

from operator import eq
from os import path
from site import getsitepackages
from unittest import TestCase

from cdd.shared.pkg_utils import get_python_lib, relative_filename
from cdd.shared.pure_utils import rpartial
from cdd.tests.utils_for_tests import unittest_main


class TestPkgUtils(TestCase):
    """Test class for pkg utils"""

    def test_relative_filename(self) -> None:
        """Tests relative_filename ident"""
        expect: str = "gaffe"
        self.assertEqual(relative_filename(expect), expect)

    def test_get_python_lib(self) -> None:
        """Tests that `get_python_lib` works"""
        python_lib: str = get_python_lib()
        site_packages: str = getsitepackages()[0]
        site_packages: str = next(
            filter(
                rpartial(eq, python_lib),
                (
                    lambda two_dir_above: (
                        site_packages,
                        two_dir_above,
                        path.join(site_packages, "Lib", "site-packages"),
                        path.join(two_dir_above, "python3", "dist-packages"),
                    )
                )(path.dirname(path.dirname(site_packages))),
            ),
            site_packages,
        )
        self.assertEqual(site_packages, python_lib)


unittest_main()
