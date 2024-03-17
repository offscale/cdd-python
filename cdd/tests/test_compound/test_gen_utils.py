""" Tests for gen_utils """

from ast import Assign, List, Load, Module, Name, Store
from os import path
from tempfile import TemporaryDirectory
from unittest import TestCase

import cdd.shared.source_transformer
from cdd.compound.gen_utils import (
    file_to_input_mapping,
    gen_module,
    get_input_mapping_from_path,
)
from cdd.tests.mocks.classes import class_ast
from cdd.tests.utils_for_tests import run_ast_test, unittest_main


def f(s):
    """
    :param s: str
    :type s: ```str```
    """
    return s


class TestGenUtils(TestCase):
    """Test class for cdd.gen_utils"""

    def test_file_to_input_mapping_else_condition(self) -> None:
        """Test that `file_to_input_mapping` else condition works"""
        with TemporaryDirectory() as temp_dir:
            filename: str = path.join(temp_dir, "foo{}py".format(path.extsep))
            with open(filename, "wt") as f:
                f.write(cdd.shared.source_transformer.to_code(class_ast))
            input_mapping = file_to_input_mapping(filename, "infer")
        self.assertEqual(len(input_mapping.keys()), 1)
        self.assertIn(class_ast.name, input_mapping)
        run_ast_test(self, input_mapping[class_ast.name], class_ast)

    def test_get_input_mapping_from_path(self) -> None:
        """test `get_input_mapping_from_path`"""
        self.assertEqual(f(""), "")
        name_to_node = get_input_mapping_from_path(
            "function", "cdd.tests.test_compound", "test_gen_utils"
        )
        self.assertEqual(len(name_to_node), 1)
        self.assertIn("f", name_to_node)
        self.assertIsInstance(name_to_node["f"], dict)

    def test_gen_module_when_emit_and_infer_imports(self) -> None:
        """
        Tests that `emit_and_infer_imports` works when `emit_and_infer_imports` is True
        """
        run_ast_test(
            self,
            gen_module(
                decorator_list=[],
                emit_and_infer_imports=True,
                emit_call=False,
                emit_default_doc=False,
                emit_name="class",
                functions_and_classes=None,
                imports=None,
                input_mapping_it={},
                name_tpl="{name}Foo",
                no_word_wrap=True,
                parse_name="class",
                prepend=None,
            ),
            Module(
                body=[
                    Assign(
                        targets=[
                            Name(
                                id="__all__", ctx=Store(), lineno=None, col_offset=None
                            )
                        ],
                        value=List(
                            elts=[],
                            ctx=Load(),
                            expr=None,
                        ),
                        type_comment=None,
                        expr=None,
                        lineno=None,
                    )
                ],
                stmt=None,
                type_ignores=[],
            ),
        )


unittest_main()
