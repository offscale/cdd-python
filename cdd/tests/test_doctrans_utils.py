""" Tests for doctrans_utils """

from ast import Expr, Load, Module, Name
from collections import deque
from copy import deepcopy
from unittest import TestCase

from cdd.ast_utils import annotate_ancestry, set_value
from cdd.doctrans_utils import DocTrans, clear_annotation, has_type_annotations
from cdd.source_transformer import ast_parse
from cdd.tests.mocks.doctrans import (
    ann_assign_with_annotation,
    assign_with_type_comment,
    class_with_internal_annotated,
    class_with_internal_type_commented_and_docstring_typed,
    function_type_annotated,
    function_type_in_docstring,
)
from cdd.tests.utils_for_tests import reindent_docstring, run_ast_test, unittest_main


class TestDocTransUtils(TestCase):
    """Test class for doctrans_utils.py"""

    def test_has_type_annotations(self) -> None:
        """Tests has_type_annotations"""

        self.assertTrue(has_type_annotations(ast_parse("a: int = 5")))
        self.assertFalse(has_type_annotations(ast_parse("a = 5")))
        self.assertTrue(has_type_annotations(ast_parse("def a() -> None: pass")))
        self.assertFalse(has_type_annotations(ast_parse("def a(): pass")))

    def test_doctrans_function_from_annotated_to_docstring(self) -> None:
        """Tests `DocTrans` converts type annotated function to docstring function"""

        original_node = annotate_ancestry(deepcopy(function_type_annotated))
        doc_trans = DocTrans(
            docstring_format="rest",
            type_annotations=False,
            existing_type_annotations=True,
            whole_ast=original_node,
        )
        gen_ast = doc_trans.visit(original_node)

        run_ast_test(self, reindent_docstring(gen_ast), gold=function_type_in_docstring)

    def test_doctrans_function_from_docstring_to_annotated(self) -> None:
        """Tests `DocTrans` converts docstring function to type annotated function"""
        original_node = annotate_ancestry(deepcopy(function_type_in_docstring))
        doc_trans = DocTrans(
            docstring_format="rest",
            type_annotations=True,
            existing_type_annotations=False,
            whole_ast=original_node,
        )
        gen_ast = doc_trans.visit(original_node)

        run_ast_test(self, gen_ast, gold=function_type_annotated)

    def test_doctrans_assign_to_annassign(self) -> None:
        """
        Tests that `Assign` converts to `AnnAssign`
        """
        original_node = annotate_ancestry(deepcopy(assign_with_type_comment))
        doc_trans = DocTrans(
            docstring_format="rest",
            type_annotations=True,
            existing_type_annotations=True,
            whole_ast=original_node,
        )
        run_ast_test(
            self,
            gen_ast=doc_trans.visit(original_node),
            gold=ann_assign_with_annotation,
        )

    def test_doctrans_annassign_to_assign(self) -> None:
        """
        Tests that `AnnAssign` converts to `Assign`
        """
        original_node = annotate_ancestry(deepcopy(ann_assign_with_annotation))
        doc_trans = DocTrans(
            docstring_format="rest",
            type_annotations=False,
            existing_type_annotations=True,
            whole_ast=original_node,
        )
        run_ast_test(
            self, gen_ast=doc_trans.visit(original_node), gold=assign_with_type_comment
        )

    def test_doctrans_annassign_to_assign_with_clearing_type_annotations(self) -> None:
        """
        Tests that `AnnAssign` converts to `Assign`
        """
        original_node = annotate_ancestry(deepcopy(ann_assign_with_annotation))
        original_node.type_comment = "NEVER SEE THIS"
        doc_trans = DocTrans(
            docstring_format="rest",
            type_annotations=True,
            existing_type_annotations=True,
            whole_ast=original_node,
        )
        gen_ast = doc_trans.visit(original_node)
        run_ast_test(self, gen_ast=gen_ast, gold=ann_assign_with_annotation)

    def test_doctrans_assign_to_assign(self) -> None:
        """
        Tests that `AnnAssign` converts to `Assign`
        """
        original_node = annotate_ancestry(deepcopy(assign_with_type_comment))
        doc_trans = DocTrans(
            docstring_format="rest",
            type_annotations=False,
            existing_type_annotations=False,
            whole_ast=original_node,
        )
        gen_ast = doc_trans.visit(original_node)
        run_ast_test(self, gen_ast=gen_ast, gold=assign_with_type_comment)

    def test_class_with_internal_annotated(self) -> None:
        """Tests that class, function, and class variable hierarchy is correctly annotated handles the ident case"""
        original_node = annotate_ancestry(deepcopy(class_with_internal_annotated))
        doc_trans = DocTrans(
            docstring_format="rest",
            type_annotations=True,
            existing_type_annotations=True,
            whole_ast=original_node,
        )
        gen_ast = doc_trans.visit(original_node)
        run_ast_test(self, gen_ast=gen_ast, gold=class_with_internal_annotated)

    def test__get_ass_typ(self) -> None:
        """Tests that _get_ass_typ returns when location isn't set"""
        original_node = annotate_ancestry(deepcopy(assign_with_type_comment))
        doc_trans = DocTrans(
            docstring_format="rest",
            type_annotations=True,
            existing_type_annotations=True,
            whole_ast=original_node,
        )
        del original_node._location
        run_ast_test(
            self,
            gen_ast=doc_trans._get_ass_typ(original_node),
            gold=Name("int", Load()),
        )

    def test_class_with_internal_converts_to_annotated(self) -> None:
        """Tests that class, function, and class variable hierarchy is correctly converts to annotated"""
        original_node = annotate_ancestry(
            deepcopy(class_with_internal_type_commented_and_docstring_typed)
        )
        doc_trans = DocTrans(
            docstring_format="rest",
            type_annotations=True,
            existing_type_annotations=False,
            whole_ast=original_node,
        )
        gen_ast = doc_trans.visit(original_node)
        run_ast_test(self, gen_ast=gen_ast, gold=class_with_internal_annotated)

    def test_class_annotated_converts_to_type_commented_and_docstring_typed(
        self,
    ) -> None:
        """Tests that class, function, and class variable hierarchy is correctly converted to annotated"""
        original_node = annotate_ancestry(deepcopy(class_with_internal_annotated))
        doc_trans = DocTrans(
            docstring_format="rest",
            type_annotations=False,
            existing_type_annotations=True,
            whole_ast=original_node,
        )
        gen_ast = doc_trans.visit(original_node)
        run_ast_test(
            self,
            gen_ast=gen_ast,
            gold=class_with_internal_type_commented_and_docstring_typed,
        )

    def test_module_docstring(self) -> None:
        """Tests that module gets the right new docstring"""
        module_node = Module(
            body=[Expr(set_value("\nModule\n"))], stmt=None, type_ignores=[]
        )
        original = deepcopy(module_node)
        doc_trans = DocTrans(
            docstring_format="rest",
            type_annotations=True,
            existing_type_annotations=True,
            whole_ast=module_node,
        )
        doc_trans.visit_Module(module_node)
        run_ast_test(self, gen_ast=module_node, gold=original)

    def test_clear_annotation(self) -> None:
        """Tests that `clear_annotation` clears correctly"""
        node_cls = type("Node", tuple(), {"annotation": None, "type_comment": None})
        node = node_cls()
        node.annotation = 5
        node.type_comment = 6
        clear_annotation(node)
        deque(
            (
                self.assertIsNone(getattr(node, attr))
                for attr in dir(node_cls)
                if not attr.startswith("_")
            ),
            maxlen=0,
        )


unittest_main()
