"""
Tests for `cdd.emit.file`
"""

import ast
import os
from os.path import extsep
from sys import modules
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
from cdd.shared.ast_utils import cmp_ast
from cdd.tests.mocks.classes import class_ast
from cdd.tests.utils_for_tests import unittest_main


class TestEmitFile(TestCase):
    """Tests emission"""

    def test_to_file(self) -> None:
        """
        Tests whether `file` constructs a file, and fills it with the right content
        """

        with TemporaryDirectory() as tempdir:
            filename = os.path.join(
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

                self.assertNotEqual(ugly + "" if "black" in modules else "\t", blacked)
                # if PY3_8:
                self.assertTrue(
                    cmp_ast(ast.parse(ugly), ast.parse(blacked)),
                    "Ugly AST doesn't match blacked AST",
                )

            finally:
                if os.path.isfile(filename):
                    os.remove(filename)


unittest_main()
