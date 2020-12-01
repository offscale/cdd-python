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
    ImportFrom,
    Import,
)
from os import path
from unittest import TestCase

from doctrans.ast_utils import (
    find_ast_type,
    get_function_type,
    get_value,
    find_in_ast,
    emit_ann_assign,
    annotate_ancestry,
    RewriteAtQuery,
    emit_arg,
    param2ast,
    set_value,
    get_at_root,
)
from doctrans.pure_utils import PY_GTE_3_9, PY3_8, PY_GTE_3_8
from doctrans.source_transformer import ast_parse
from doctrans.tests.mocks.classes import class_ast, class_str
from doctrans.tests.mocks.methods import (
    class_with_optional_arg_method_ast,
    class_with_method_and_body_types_ast,
    class_with_method_and_body_types_str,
)
from doctrans.tests.utils_for_tests import run_ast_test, unittest_main
from meta.asttools import cmp_ast


class TestAstUtils(TestCase):
    """ Test class for ast_utils """

    def test_annotate_ancestry(self) -> None:
        """ Tests that `annotate_ancestry` properly decorates """
        node = Module(
            body=[
                AnnAssign(
                    annotation=Name(
                        "str",
                        Load(),
                    ),
                    simple=1,
                    target=Name("dataset_name", Store()),
                    value=Constant(
                        kind=None,
                        value="~/tensorflow_datasets",
                        constant_value=None,
                        string=None,
                    ),
                    expr=None,
                    expr_target=None,
                    expr_annotation=None,
                ),
                Assign(
                    annotation=None,
                    simple=1,
                    targets=[Name("epochs", Store())],
                    value=Constant(
                        kind=None, value="333", constant_value=None, string=None
                    ),
                    expr=None,
                    expr_target=None,
                    expr_annotation=None,
                ),
            ],
            stmt=None,
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
                annotation=Name(
                    "str",
                    Load(),
                ),
                simple=1,
                target=Name("dataset_name", Store()),
                value=Constant(
                    kind=None,
                    value="~/tensorflow_datasets",
                    constant_value=None,
                    string=None,
                ),
                expr=None,
                expr_target=None,
                expr_annotation=None,
            ),
            run_cmp_ast=PY_GTE_3_9,
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
            targets=[Name("yup", Store())],
            value=Constant(kind=None, value="nup", constant_value=None, string=None),
            expr=None,
        )
        gen_ast = emit_arg(assign)
        self.assertIsInstance(gen_ast, arg)
        run_ast_test(
            self,
            gen_ast=gen_ast,
            gold=arg(
                annotation=None,
                arg="yup",
                type_comment=None,
                expr=None,
                identifier_arg=None,
            ),
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
                annotation=Name(
                    "str",
                    Load(),
                ),
                simple=1,
                target=Name("dataset_name", Store()),
                value=Constant(
                    kind=None, value="mnist", constant_value=None, string=None
                ),
                expr=None,
                expr_target=None,
                expr_annotation=None,
            ),
            run_cmp_ast=PY_GTE_3_8,
        )

    def test_find_in_ast_self(self) -> None:
        """ Tests that `find_in_ast` successfully finds itself in AST """
        run_ast_test(self, find_in_ast(["ConfigClass"], class_ast), class_ast)
        module = Module(body=[], type_ignores=[], stmt=None)
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
                        arg=None,
                    ),
                    body=[],
                    decorator_list=[],
                    lineno=None,
                    arguments_args=None,
                    identifier_name=None,
                    stmt=None,
                )
            ],
            stmt=None,
        )
        annotate_ancestry(module_with_fun)
        run_ast_test(
            self, find_in_ast(["call_peril"], module_with_fun), module_with_fun.body[0]
        )

    def test_find_in_ast_None(self) -> None:
        """ Tests that `find_in_ast` fails correctly in AST """
        self.assertIsNone(find_in_ast(["John Galt"], class_ast))

    def test_find_in_ast_no_val(self) -> None:
        """Tests that `find_in_ast` correctly gives AST node from
        `def class C(object): def function_name(self,dataset_name: str,…)`"""
        run_ast_test(
            self,
            find_in_ast(
                "C.function_name.dataset_name".split("."),
                class_with_optional_arg_method_ast,
            ),
            arg(
                annotation=Name(
                    "str",
                    Load(),
                ),
                arg="dataset_name",
                type_comment=None,
                expr=None,
                identifier_arg=None,
            ),
        )

    def test_find_in_ast_with_val(self) -> None:
        """Tests that `find_in_ast` correctly gives AST node from
        `def class C(object): def function_name(self,dataset_name: str='foo',…)`"""
        gen_ast = find_in_ast(
            "C.function_name.dataset_name".split("."),
            class_with_method_and_body_types_ast,
        )

        self.assertIsInstance(gen_ast.default, Constant if PY_GTE_3_8 else Str)

        self.assertEqual(get_value(gen_ast.default), "~/tensorflow_datasets")
        run_ast_test(
            self,
            gen_ast,
            arg(
                annotation=Name(
                    "str",
                    Load(),
                ),
                arg="dataset_name",
                type_comment=None,
                expr=None,
                identifier_arg=None,
            ),
            run_cmp_ast=PY_GTE_3_9,
        )

    def test_get_at_root(self) -> None:
        """ Tests that `get_at_root` successfully gets the imports """
        with open(path.join(path.dirname(__file__), "mocks", "eval.py")) as f:
            imports = get_at_root(ast.parse(f.read()), (Import, ImportFrom))
        self.assertIsInstance(imports, list)
        self.assertEqual(len(imports), 1)
        self.assertTrue(
            cmp_ast(
                imports[0],
                ast.Import(
                    names=[
                        ast.alias(
                            asname=None,
                            name="doctrans.tests.mocks",
                            identifier=None,
                            identifier_name=None,
                        )
                    ],
                    alias=None,
                ),
            )
        )

    def test_replace_in_ast_with_val(self) -> None:
        """
        Tests that `RewriteAtQuery` can actually replace a node at given location
        """
        parsed_ast = ast_parse(class_with_method_and_body_types_str)
        rewrite_at_query = RewriteAtQuery(
            search="C.function_name.dataset_name".split("."),
            replacement_node=AnnAssign(
                annotation=Name("int", Load()),
                simple=1,
                target=Name("dataset_name", Store()),
                value=Constant(kind=None, value=15, constant_value=None, string=None),
                expr=None,
                expr_annotation=None,
                expr_target=None,
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
            run_cmp_ast=PY_GTE_3_8,
        )

    def test_replace_in_ast_with_val_on_non_function(self) -> None:
        """
        Tests that `RewriteAtQuery` can actually replace a node at given location
        """
        parsed_ast = ast_parse(class_str)
        rewrite_at_query = RewriteAtQuery(
            search="ConfigClass.dataset_name".split("."),
            replacement_node=AnnAssign(
                annotation=Name("int", Load()),
                simple=1,
                target=Name("dataset_name", Store()),
                value=Constant(kind=None, value=15, constant_value=None, string=None),
                expr=None,
                expr_target=None,
                expr_annotation=None,
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
            run_cmp_ast=PY_GTE_3_8,
        )

    def test_get_function_type(self) -> None:
        """ Test get_function_type returns the right type """
        self.assertEqual(
            get_function_type(
                FunctionDef(
                    args=arguments(
                        args=[
                            arg(
                                annotation=None,
                                arg="something else",
                                type_comment=None,
                                expr=None,
                                identifier_arg=None,
                            )
                        ],
                        arg=None,
                    ),
                    arguments_args=None,
                    identifier_name=None,
                    stmt=None,
                )
            ),
            "static",
        )
        self.assertEqual(
            get_function_type(
                FunctionDef(
                    args=arguments(args=[], arg=None),
                    arguments_args=None,
                    identifier_name=None,
                    stmt=None,
                )
            ),
            "static",
        )
        self.assertEqual(
            get_function_type(
                FunctionDef(
                    args=arguments(
                        args=[
                            arg(
                                annotation=None,
                                arg="self",
                                type_comment=None,
                                expr=None,
                                identifier_arg=None,
                            )
                        ],
                        arg=None,
                    ),
                    arguments_args=None,
                    identifier_name=None,
                    stmt=None,
                )
            ),
            "self",
        )
        self.assertEqual(
            get_function_type(
                FunctionDef(
                    args=arguments(
                        args=[
                            arg(
                                annotation=None,
                                arg="cls",
                                type_comment=None,
                                expr=None,
                                identifier_arg=None,
                            )
                        ],
                        arg=None,
                    ),
                    arguments_args=None,
                    identifier_name=None,
                    stmt=None,
                )
            ),
            "cls",
        )

    def test_get_value(self) -> None:
        """ Tests get_value succeeds """
        val = "foo"
        self.assertEqual(get_value(Str(s=val, constant_value=None, string=None)), val)
        self.assertEqual(
            get_value(Constant(value=val, constant_value=None, string=None)), val
        )
        self.assertIsInstance(get_value(Tuple(expr=None)), Tuple)
        self.assertIsInstance(get_value(Tuple(expr=None)), Tuple)
        self.assertIsNone(get_value(Name(None, None)))

    def test_set_value(self) -> None:
        """ Tests that `set_value` returns the right type for the right Python version """
        import doctrans.ast_utils

        _doctrans_ast_utils_PY3_8_orig = PY3_8
        # patch stopped working, getattr failed also ^
        try:
            doctrans.ast_utils.PY3_8 = True

            self.assertIsInstance(
                doctrans.ast_utils.set_value(None, None),
                Constant if PY_GTE_3_8 else NameConstant,
            )

            doctrans.ast_utils.PY3_8 = False

            self.assertIsInstance(
                doctrans.ast_utils.set_value(None, None), NameConstant
            )

            doctrans.ast_utils.PY3_8 = True
            self.assertIsInstance(
                doctrans.ast_utils.set_value("foo", None),
                Constant if PY_GTE_3_8 else Str,
            )

            doctrans.ast_utils.PY3_8 = False
            self.assertIsInstance(doctrans.ast_utils.set_value("foo", None), Str)
        finally:
            doctrans.ast_utils = _doctrans_ast_utils_PY3_8_orig

    def test_find_ast_type(self) -> None:
        """ Test that `find_ast_type` gives the wrapped class back """

        class_def = ClassDef(
            name="",
            bases=tuple(),
            keywords=tuple(),
            decorator_list=[],
            body=[],
            expr=None,
            identifier_name=None,
        )
        run_ast_test(
            self, find_ast_type(Module(body=[class_def], stmt=None)), class_def
        )

    def test_find_ast_type_fails(self) -> None:
        """ Test that `find_ast_type` throws the right errors """

        self.assertRaises(NotImplementedError, lambda: find_ast_type(None))
        self.assertRaises(NotImplementedError, lambda: find_ast_type(""))
        self.assertRaises(TypeError, lambda: find_ast_type(Module(body=[], stmt=None)))
        self.assertRaises(
            NotImplementedError,
            lambda: find_ast_type(
                Module(
                    body=[
                        ClassDef(expr=None, identifier_name=None),
                        ClassDef(expr=None, identifier_name=None),
                    ],
                    stmt=None,
                )
            ),
        )

    def test_to_named_class_def(self) -> None:
        """ Test that find_ast_type gives the wrapped named class back """

        class_def = ClassDef(
            name="foo",
            bases=tuple(),
            keywords=tuple(),
            decorator_list=[],
            body=[],
            expr=None,
            identifier_name=None,
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
                            expr=None,
                            identifier_name=None,
                        ),
                        class_def,
                    ],
                    stmt=None,
                ),
                node_name="foo",
            ),
            class_def,
        )

    def test_param2ast_with_assign(self) -> None:
        """ Check that `param2ast` behaves correctly with a non annotated (typeless) input """

        gold = Assign(
            targets=[Name("zion", Store())],
            value=set_value(value=None),
            expr=None,
            lineno=None,
        )
        run_ast_test(self, param2ast({"typ": None, "name": "zion"}), gold=gold)


unittest_main()
