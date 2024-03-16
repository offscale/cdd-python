""" Tests for pkg utils """

from operator import eq
from os import path
from site import getsitepackages
from unittest import TestCase

from cdd.shared.pkg_utils import get_python_lib, relative_filename
from cdd.shared.pure_utils import rpartial, pp
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
        two_dirs_above: str = path.dirname(path.dirname(site_packages))
        pp(
            {
                "python_lib": python_lib,
                "two_dirs_above": two_dirs_above,
                "site_packages": site_packages,
                'path.join(two_dir_above, "Lib", "site-packages")': path.join(
                    two_dirs_above, "Lib", "site-packages"
                ),
                'path.join(two_dir_above, "python3", "dist-packages")': path.join(
                    two_dirs_above, "python3", "dist-packages"
                ),
            }
        )
        site_packages: str = next(
            filter(
                rpartial(eq, python_lib),
                (
                    lambda two_dir_above: (
                        site_packages,
                        two_dir_above,
                        path.join(two_dir_above, "Lib", "site-packages"),
                        path.join(two_dir_above, "python3", "dist-packages"),
                    )
                )(path.dirname(path.dirname(site_packages))),
            ),
            site_packages,
        )
        self.assertEqual(site_packages, python_lib)


unittest_main()
