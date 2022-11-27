"""
Tests for `cdd.emit.file`
"""

import ast
import os
from os.path import extsep
from sys import modules
from tempfile import TemporaryDirectory
from unittest import TestCase

import cdd.emit.argparse_function
import cdd.emit.class_
import cdd.emit.docstring
import cdd.emit.file
import cdd.emit.function
import cdd.emit.json_schema
import cdd.emit.sqlalchemy
import cdd.parse.argparse_function
from cdd.ast_utils import cmp_ast
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
                cdd.emit.file.file(class_ast, filename, skip_black=True)

                with open(filename, "rt") as f:
                    ugly = f.read()

                os.remove(filename)

                cdd.emit.file.file(class_ast, filename, skip_black=False)

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
