""" Tests for ast_utils """
import ast
from ast import (
    FunctionDef,
    Module,
    ClassDef,
    Name,
    arguments,
    arg,
    Constant,
    NameConstant,
    Str,
    Tuple,
    AnnAssign,
    Load,
    Store,
    Assign,
)
from unittest import TestCase
from unittest.mock import patch

from meta.asttools import cmp_ast

from doctrans.ast_utils import (
    find_ast_type,
    get_function_type,
    get_value,
    find_in_ast,
    emit_ann_assign,
    annotate_ancestry,
    RewriteAtQuery,
    emit_arg,
)
from doctrans.source_transformer import ast_parse
from doctrans.tests.mocks.classes import class_ast, class_str
from doctrans.tests.mocks.methods import (
    class_with_optional_arg_method_ast,
    class_with_method_and_body_types_ast,
    class_with_method_and_body_types_str,
)
from doctrans.tests.utils_for_tests import run_ast_test, unittest_main


class TestAstUtils(TestCase):
    """ Test class for ast_utils """

    def test_annotate_ancestry(self) -> None:
        """ Tests that `annotate_ancestry` properly decorates """
        node = Module(
            body=[
                AnnAssign(
                    annotation=Name(ctx=Load(), id="str"),
                    simple=1,
                    target=Name(ctx=Store(), id="dataset_name"),
                    value=Constant(kind=None, value="~/tensorflow_datasets"),
                ),
                Assign(
                    annotation=None,
                    simple=1,
                    targets=[Name(ctx=Store(), id="epochs")],
                    value=Constant(kind=None, value="333"),
                ),
            ]
        )
        self.assertFalse(hasattr(node.body[0], "_location"))
        self.assertFalse(hasattr(node.body[1], "_location"))
        annotate_ancestry(node)
        self.assertEqual(node.body[0]._location, ["dataset_name"])
        self.assertEqual(node.body[1]._location, ["epochs"])

    def test_emit_ann_assign(self) -> None:
        """ Tests that AnnAssign is emitted from `emit_ann_assign` """
        self.assertIsInstance(class_ast.body[1], AnnAssign)
        self.assertIsInstance(emit_ann_assign(class_ast.body[1]), AnnAssign)
        self.assertIsInstance(emit_ann_assign(class_ast.body[1]), AnnAssign)
        gen_ast = emit_ann_assign(
            find_in_ast(
                "C.function_name.dataset_name".split("."),
                class_with_method_and_body_types_ast,
            )
        )
        self.assertIsInstance(gen_ast, AnnAssign)
        run_ast_test(
            self,
            gen_ast,
            AnnAssign(
                annotation=Name(ctx=Load(), id="str"),
                simple=1,
                target=Name(ctx=Store(), id="dataset_name"),
                value=Constant(kind=None, value="~/tensorflow_datasets"),
            ),
        )

    def test_emit_ann_assign_fails(self) -> None:
        """ Tests that `emit_ann_assign` raised the right error """
        for typ_ in FunctionDef, ClassDef, Module:
            self.assertRaises(NotImplementedError, lambda: emit_ann_assign(typ_))

    def test_emit_arg(self) -> None:
        """ Tests that `arg` is emitted from `emit_arg` """
        self.assertIsInstance(
            class_with_method_and_body_types_ast.body[1].args.args[1], arg
        )
        self.assertIsInstance(
            emit_arg(class_with_method_and_body_types_ast.body[1].args.args[1]), arg
        )
        assign = Assign(
            targets=[Name(ctx=Store(), id="yup")],
            value=Constant(kind=None, value="nup"),
        )
        gen_ast = emit_arg(assign)
        self.assertIsInstance(gen_ast, arg)
        run_ast_test(
            self,
            gen_ast=gen_ast,
            gold=arg(annotation=None, arg="yup", type_comment=None),
        )

    def test_emit_arg_fails(self) -> None:
        """ Tests that `emit_arg` raised the right error """
        for typ_ in FunctionDef, ClassDef, Module:
            self.assertRaises(NotImplementedError, lambda: emit_arg(typ_))

    def test_find_in_ast(self) -> None:
        """ Tests that `find_in_ast` successfully finds nodes in AST """
        run_ast_test(
            self,
            find_in_ast("ConfigClass.dataset_name".split("."), class_ast),
            AnnAssign(
                annotation=Name(ctx=Load(), id="str"),
                simple=1,
                target=Name(ctx=Store(), id="dataset_name"),
                value=Constant(kind=None, value="mnist"),
            ),
        )

    def test_find_in_ast_self(self) -> None:
        """ Tests that `find_in_ast` successfully finds itself in AST """
        run_ast_test(self, find_in_ast(["ConfigClass"], class_ast), class_ast)
        module = Module(body=[], type_ignores=[])
        run_ast_test(self, find_in_ast([], module), module)
        module_with_fun = Module(
            body=[
                FunctionDef(
                    name="call_peril",
                    args=arguments(
                        args=[],
                        defaults=[],
                        kw_defaults=[],
                        kwarg=None,
                        kwonlyargs=[],
                        posonlyargs=[],
                        vararg=None,
                    ),
                    body=[],
                    decorator_list=[],
                    lineno=None,
                )
            ]
        )
        annotate_ancestry(module_with_fun)
        run_ast_test(
            self, find_in_ast(["call_peril"], module_with_fun), module_with_fun.body[0]
        )

    def test_find_in_ast_None(self) -> None:
        """ Tests that `find_in_ast` fails correctly in AST """
        self.assertIsNone(find_in_ast(["John Galt"], class_ast))

    def test_find_in_ast_no_val(self) -> None:
        """ Tests that `find_in_ast` correctly gives AST node from
         `def class C(object): def function_name(self,dataset_name: str,…)`"""
        run_ast_test(
            self,
            find_in_ast(
                "C.function_name.dataset_name".split("."),
                class_with_optional_arg_method_ast,
            ),
            arg(
                annotation=Name(ctx=Load(), id="str"),
                arg="dataset_name",
                type_comment=None,
            ),
        )

    def test_find_in_ast_with_val(self) -> None:
        """ Tests that `find_in_ast` correctly gives AST node from
         `def class C(object): def function_name(self,dataset_name: str='foo',…)`"""
        gen_ast = find_in_ast(
            "C.function_name.dataset_name".split("."),
            class_with_method_and_body_types_ast,
        )
        self.assertTrue(
            cmp_ast(gen_ast.default, Constant(kind=None, value="~/tensorflow_datasets"))
        )
        run_ast_test(
            self,
            gen_ast,
            arg(
                annotation=Name(ctx=Load(), id="str"),
                arg="dataset_name",
                type_comment=None,
            ),
        )

    def test_replace_in_ast_with_val(self) -> None:
        """
        Tests that `RewriteAtQuery` can actually replace a node at given location
        """
        parsed_ast = ast_parse(class_with_method_and_body_types_str)
        rewrite_at_query = RewriteAtQuery(
            search="C.function_name.dataset_name".split("."),
            replacement_node=AnnAssign(
                annotation=Name(ctx=Load(), id="int"),
                simple=1,
                target=Name(ctx=Store(), id="dataset_name"),
                value=Constant(kind=None, value=15),
            ),
        )
        gen_ast = rewrite_at_query.visit(parsed_ast)
        self.assertTrue(rewrite_at_query.replaced, True)

        run_ast_test(
            self,
            gen_ast,
            ast.parse(
                class_with_method_and_body_types_str.replace(
                    'dataset_name: str = "mnist"', "dataset_name: int = 15"
                )
            ),
        )

    def test_replace_in_ast_with_val_on_non_function(self) -> None:
        """
        Tests that `RewriteAtQuery` can actually replace a node at given location
        """
        parsed_ast = ast_parse(class_str)
        rewrite_at_query = RewriteAtQuery(
            search="ConfigClass.dataset_name".split("."),
            replacement_node=AnnAssign(
                annotation=Name(ctx=Load(), id="int"),
                simple=1,
                target=Name(ctx=Store(), id="dataset_name"),
                value=Constant(kind=None, value=15),
            ),
        )
        gen_ast = rewrite_at_query.visit(parsed_ast)
        self.assertTrue(rewrite_at_query.replaced, True)

        run_ast_test(
            self,
            gen_ast,
            ast.parse(
                class_str.replace(
                    'dataset_name: str = "mnist"', "dataset_name: int = 15"
                )
            ),
        )

    def test_get_function_type(self) -> None:
        """ Test get_function_type returns the right type """
        self.assertIsNone(
            get_function_type(
                FunctionDef(
                    args=arguments(
                        args=[
                            arg(
                                annotation=None, arg="something else", type_comment=None
                            )
                        ]
                    )
                )
            )
        )
        self.assertIsNone(get_function_type(FunctionDef(args=arguments(args=[]))))
        self.assertEqual(
            get_function_type(
                FunctionDef(
                    args=arguments(
                        args=[arg(annotation=None, arg="self", type_comment=None)]
                    )
                )
            ),
            "self",
        )
        self.assertEqual(
            get_function_type(
                FunctionDef(
                    args=arguments(
                        args=[arg(annotation=None, arg="cls", type_comment=None)]
                    )
                )
            ),
            "cls",
        )

    def test_get_value(self) -> None:
        """ Tests get_value succeeds """
        val = "foo"
        self.assertEqual(get_value(Str(s=val)), val)
        self.assertEqual(get_value(Constant(value=val)), val)
        self.assertIsInstance(get_value(Tuple()), Tuple)
        self.assertIsInstance(get_value(Tuple()), Tuple)
        self.assertIsInstance(get_value(Name()), Name)

    def test_set_value(self) -> None:
        """ Tests that `set_value` returns the right type for the right Python version """
        with patch("doctrans.ast_utils.PY3_8", True):
            import doctrans.ast_utils

            self.assertIsInstance(doctrans.ast_utils.set_value(None, None), Constant)

        with patch("doctrans.ast_utils.PY3_8", False):
            import doctrans.ast_utils

            self.assertIsInstance(
                doctrans.ast_utils.set_value(None, None), NameConstant
            )

        with patch("doctrans.ast_utils.PY3_8", True):
            import doctrans.ast_utils

            self.assertIsInstance(doctrans.ast_utils.set_value("foo", None), Constant)

        with patch("doctrans.ast_utils.PY3_8", False):
            import doctrans.ast_utils

            self.assertIsInstance(doctrans.ast_utils.set_value("foo", None), Str)

    def test_find_ast_type(self) -> None:
        """ Test that `find_ast_type` gives the wrapped class back """

        class_def = ClassDef(
            name="", bases=tuple(), keywords=tuple(), decorator_list=[], body=[]
        )
        run_ast_test(self, find_ast_type(Module(body=[class_def])), class_def)

    def test_find_ast_type_fails(self) -> None:
        """ Test that `find_ast_type` throws the right errors """

        self.assertRaises(NotImplementedError, lambda: find_ast_type(None))
        self.assertRaises(NotImplementedError, lambda: find_ast_type(""))
        self.assertRaises(TypeError, lambda: find_ast_type(Module(body=[])))
        self.assertRaises(
            NotImplementedError,
            lambda: find_ast_type(Module(body=[ClassDef(), ClassDef()])),
        )

    def test_to_named_class_def(self) -> None:
        """ Test that find_ast_type gives the wrapped named class back """

        class_def = ClassDef(
            name="foo", bases=tuple(), keywords=tuple(), decorator_list=[], body=[]
        )
        run_ast_test(
            self,
            find_ast_type(
                Module(
                    body=[
                        ClassDef(
                            name="bar",
                            bases=tuple(),
                            keywords=tuple(),
                            decorator_list=[],
                            body=[],
                        ),
                        class_def,
                    ]
                ),
                node_name="foo",
            ),
            class_def,
        )


unittest_main()
