"""
Tests for setup.py
"""
from importlib.machinery import SourceFileLoader
from importlib.util import module_from_spec, spec_from_loader
from operator import methodcaller
from os import path
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
        """ Construct the setup_py module """
        cls.mod = cls.import_setup_py()

    @staticmethod
    def import_setup_py():
        """
        Function which imports setup.py as a module

        :returns: setup.py as a module
        :rtype: ```Union[module, ModuleSpec]```
        """
        modname = "setup_py"
        loader = SourceFileLoader(
            modname,
            path.join(path.dirname(path.dirname(path.dirname(__file__))), "setup.py"),
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

    def test_to_funcs(self) -> None:
        """ Tests that `to_funcs` produces the right local and install dirs """
        to_funcs = getattr(self.mod, "to_funcs")
        args = "5", "6"
        local_dir_join_func_resp, install_dir_join_func_resp = map(
            methodcaller("__call__"), to_funcs(*args)
        )
        self.assertNotEqual(local_dir_join_func_resp, install_dir_join_func_resp)
        self.assertEqual(
            local_dir_join_func_resp,
            path.join(path.dirname(path.dirname(__file__)), *args),
        )

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
