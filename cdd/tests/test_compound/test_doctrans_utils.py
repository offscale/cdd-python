"""Tests for doctrans_utils"""

from ast import (
    Expr,
    FunctionDef,
    Load,
    Module,
    Name,
    arguments,
    fix_missing_locations,
    get_docstring,
)
from collections import deque
from copy import deepcopy
from os import path
from os.path import extsep
from unittest import TestCase
from unittest.mock import MagicMock, patch

import cdd.shared.ast_utils
from cdd.compound.doctrans_utils import (
    DocTrans,
    clear_annotation,
    doctransify_cst,
    has_type_annotations,
)
from cdd.shared.ast_utils import annotate_ancestry, set_value
from cdd.shared.pure_utils import omit_whitespace
from cdd.shared.source_transformer import ast_parse
from cdd.tests.mocks.cst import cstify_cst
from cdd.tests.mocks.doctrans import (
    ann_assign_with_annotation,
    assign_with_type_comment,
    class_with_internal_annotated,
    class_with_internal_type_commented_and_docstring_typed,
    function_type_annotated,
    function_type_in_docstring,
)
from cdd.tests.mocks.methods import function_google_tf_ops_losses__safe_mean_ast
from cdd.tests.utils_for_tests import run_ast_test, unittest_main


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
            word_wrap=True,
            type_annotations=False,
            existing_type_annotations=True,
            whole_ast=original_node,
        )
        gen_ast = doc_trans.visit(original_node)

        run_ast_test(self, gen_ast, gold=function_type_in_docstring)

    def test_doctrans_function_from_docstring_to_annotated(self) -> None:
        """Tests `DocTrans` converts docstring function to type annotated function"""
        original_node = annotate_ancestry(deepcopy(function_type_in_docstring))
        doc_trans = DocTrans(
            docstring_format="rest",
            word_wrap=True,
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
            word_wrap=True,
            type_annotations=True,
            existing_type_annotations=False,
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
            word_wrap=True,
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
            word_wrap=True,
            type_annotations=True,
            existing_type_annotations=False,
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
            word_wrap=True,
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
            word_wrap=True,
            type_annotations=True,
            existing_type_annotations=False,
            whole_ast=original_node,
        )
        gen_ast = doc_trans.visit(original_node)
        run_ast_test(self, gen_ast=gen_ast, gold=class_with_internal_annotated)

    def test__get_ass_typ(self) -> None:
        """Tests that _get_ass_typ returns when location isn't set"""
        original_node = annotate_ancestry(deepcopy(assign_with_type_comment))
        doc_trans = DocTrans(
            docstring_format="rest",
            word_wrap=True,
            type_annotations=True,
            existing_type_annotations=False,
            whole_ast=original_node,
        )
        del original_node._location
        self.assertEqual(
            doc_trans._get_ass_typ(original_node),
            Name("int", Load(), lineno=None, col_offset=None).id,
        )

    def test_class_with_internal_converts_to_annotated(self) -> None:
        """Tests that class, function, and class variable hierarchy is correctly converts to annotated"""
        original_node = annotate_ancestry(
            deepcopy(class_with_internal_type_commented_and_docstring_typed)
        )
        doc_trans = DocTrans(
            docstring_format="rest",
            word_wrap=True,
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
            word_wrap=True,
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
        module_node: Module = Module(
            body=[Expr(set_value("\nModule\n"), lineno=None, col_offset=None)],
            stmt=None,
            type_ignores=[],
        )
        original: Module = deepcopy(module_node)
        DocTrans(
            docstring_format="rest",
            word_wrap=True,
            type_annotations=True,
            existing_type_annotations=False,
            whole_ast=module_node,
        )
        run_ast_test(self, gen_ast=module_node, gold=original)

    def test_empty_types(self) -> None:
        """Tests function_google_tf_ops_losses__safe_mean_ast (which has empty arg types)"""
        original: Module = Module(
            body=[
                fix_missing_locations(
                    deepcopy(function_google_tf_ops_losses__safe_mean_ast)
                )
            ],
            stmt=None,
            type_ignores=[],
        )
        doc_trans = DocTrans(
            docstring_format="google",
            word_wrap=True,
            type_annotations=True,
            existing_type_annotations=False,
            whole_ast=deepcopy(original),
        )

        # Reindent docstrings
        for body in original.body[0], doc_trans.whole_ast.body[0]:
            body.body[0] = Expr(
                set_value(omit_whitespace(get_docstring(body, clean=True))),
                lineno=None,
                col_offset=None,
            )

        run_ast_test(self, gen_ast=doc_trans.whole_ast, gold=original)

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

    def test_doctransify_cst(self) -> None:
        """
        Tests that `doctransify_cst` calls the node potentially-augmenting functions right number of times
        """
        cst_list = list(deepcopy(cstify_cst))
        with open(
            path.join(
                path.dirname(path.dirname(__file__)),
                "mocks",
                "cstify{extsep}py".format(extsep=extsep),
            ),
            "rt",
        ) as f:
            ast_mod = ast_parse(f.read(), skip_docstring_remit=True)

        fake_maybe_replace_doc_str_in_function_or_class = MagicMock()
        with patch(
            "cdd.shared.ast_cst_utils.maybe_replace_doc_str_in_function_or_class",
            fake_maybe_replace_doc_str_in_function_or_class,
        ):
            doctransify_cst(cst_list, ast_mod)
        self.assertTrue(fake_maybe_replace_doc_str_in_function_or_class.called)
        self.assertEqual(fake_maybe_replace_doc_str_in_function_or_class.call_count, 3)

    def test_doctransify_cst_early_exit(self) -> None:
        """
        Tests that `doctransify_cst` early exits
        """
        cst_list = list(deepcopy(cstify_cst))
        ast_mod = cdd.shared.ast_utils.annotate_ancestry(
            Module(
                body=[
                    FunctionDef(
                        args=arguments(
                            args=[],
                            defaults=[],
                            kw_defaults=[],
                            kwarg=None,
                            kwonlyargs=[],
                            posonlyargs=[],
                            vararg=None,
                            arg=None,
                        ),
                        body=[],
                        name="foo",
                        arguments_args=None,
                        identifier_name=None,
                        stmt=None,
                        lineno=1,
                    )
                ],
                type_ignores=[],
                stmt=None,
            )
        )
        doctransify_cst(cst_list, ast_mod)


unittest_main()
