"""
Tests for `cdd.emit.file`
"""

import ast
import os
from importlib.util import find_spec
from os.path import extsep
from tempfile import TemporaryDirectory
from unittest import TestCase

import cdd.argparse_function.emit
import cdd.argparse_function.parse
import cdd.class_.emit
import cdd.docstring.emit
import cdd.function.emit
import cdd.json_schema.emit
import cdd.shared.emit.file
import cdd.sqlalchemy.emit
from cdd.shared.ast_utils import get_value, set_value
from cdd.shared.pure_utils import emit_separating_tabs
from cdd.tests.mocks.classes import class_ast
from cdd.tests.utils_for_tests import run_ast_test, unittest_main


class TestEmitFile(TestCase):
    """Tests emission"""

    def test_to_file(self) -> None:
        """
        Tests whether `file` constructs a file, and fills it with the right content
        """

        with TemporaryDirectory() as tempdir:
            filename: str = os.path.join(
                tempdir, "delete_me{extsep}py".format(extsep=extsep)
            )
            try:
                cdd.shared.emit.file.file(class_ast, filename, skip_black=True)

                with open(filename, "rt") as f:
                    ugly = f.read()

                os.remove(filename)

                cdd.shared.emit.file.file(class_ast, filename, skip_black=False)

                with open(filename, "rt") as f:
                    blacked = f.read()

                self.assertNotEqual(
                    ugly + "" if find_spec("black") is not None else "\t", blacked
                )
                ugly_mod = ast.parse(ugly)
                black_mod = ast.parse(blacked)
                for mod in ugly_mod, black_mod:
                    mod.body[0].body[0].value = set_value(
                        emit_separating_tabs(get_value(mod.body[0].body[0].value))
                    )
                run_ast_test(self, ugly_mod, black_mod)

            finally:
                if os.path.isfile(filename):
                    os.remove(filename)


unittest_main()
