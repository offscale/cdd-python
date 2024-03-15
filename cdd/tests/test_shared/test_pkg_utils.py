""" Tests for pkg utils """

from os import path
from site import getsitepackages
from unittest import TestCase

from cdd.shared.pkg_utils import get_python_lib, relative_filename
from cdd.tests.utils_for_tests import unittest_main


class TestPkgUtils(TestCase):
    """Test class for pkg utils"""

    def test_relative_filename(self) -> None:
        """Tests relative_filename ident"""
        expect: str = "gaffe"
        self.assertEqual(relative_filename(expect), expect)

    def test_get_python_lib(self) -> None:
        """Tests that `get_python_lib` works"""
        site_packages = getsitepackages()[0]
        python_lib = get_python_lib()
        if site_packages != python_lib:
            site_packages = path.dirname(path.dirname(site_packages))
        self.assertEqual(
            (
                site_packages
                if site_packages == python_lib
                else path.join(site_packages, "python3", "dist-packages")
            ),
            python_lib,
        )


unittest_main()
