"""
Tests for setup.py
"""

from importlib.machinery import SourceFileLoader
from importlib.util import module_from_spec, spec_from_loader
from operator import methodcaller
from os import path
from os.path import extsep
from sys import modules
from unittest import TestCase
from unittest.mock import patch

from cdd.tests.utils_for_tests import mock_function, unittest_main


class TestSetupPy(TestCase):
    """
    Tests whether docstrings are parsed out—and emitted—correctly
    """

    @classmethod
    def setUpClass(cls) -> None:
        """Construct the setup_py module"""
        cls.mod = cls.import_setup_py()

    @staticmethod
    def import_setup_py():
        """
        Function which imports setup.py as a module

        :return: setup.py as a module
        :rtype: ```Union[module, ModuleSpec]```
        """
        modname: str = "setup_py"
        loader = SourceFileLoader(
            modname,
            path.join(
                path.dirname(path.dirname(path.dirname(__file__))),
                "setup{extsep}py".format(extsep=extsep),
            ),
        )
        modules[modname] = module_from_spec(spec_from_loader(loader.name, loader))
        loader.exec_module(modules[modname])
        return modules[modname]

    def test_properties(self) -> None:
        """
        Tests whether 'setup.py' has correct properties
        """
        self.assertEqual(getattr(self.mod, "package_name"), "cdd")
        self.assertEqual(self.mod.__name__, "setup_py")

    def test_main(self) -> None:
        """
        Tests that no errors occur in `main` function call (up to `setup()`, which is tested in setuptools)
        """
        with patch("setup_py.setup", mock_function):
            self.assertIsNone(self.mod.main())

    def test_setup_py_main(self) -> None:
        """
        Tests that `__name__ == __main__` calls the `main` function via `setup_py_main` call
        """

        with patch("setup_py.main", mock_function), patch(
            "setup_py.__name__", "__main__"
        ):
            self.assertIsNone(self.mod.setup_py_main())


unittest_main()
