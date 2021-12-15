""" Tests for ast_utils """

import ast
import pickle
from ast import (
    AnnAssign,
    Assign,
    Attribute,
    BinOp,
    Call,
    ClassDef,
    Constant,
    Dict,
    Expr,
    FunctionDef,
    Import,
    ImportFrom,
    List,
    Load,
    Module,
    Mult,
    Name,
    NameConstant,
    Store,
    Str,
    Tuple,
    arg,
    arguments,
    keyword,
)
from copy import deepcopy
from itertools import repeat
from os import extsep, path
from unittest import TestCase

from cdd.ast_utils import (
    NoneStr,
    RewriteAtQuery,
    _parse_default_from_ast,
    annotate_ancestry,
    cmp_ast,
    del_ass_where_name,
    emit_ann_assign,
    emit_arg,
    find_ast_type,
    find_in_ast,
    get_ass_where_name,
    get_at_root,
    get_function_type,
    get_value,
    infer_type_and_default,
    maybe_type_comment,
    merge_assignment_lists,
    merge_modules,
    node_to_dict,
    param2argparse_param,
    param2ast,
    parse_to_scalar,
    set_arg,
    set_docstring,
    set_slice,
    set_value,
    to_annotation,
)
from cdd.pure_utils import PY3_8, PY_GTE_3_8, tab
from cdd.source_transformer import ast_parse
from cdd.tests.mocks.argparse import argparse_add_argument_expr
from cdd.tests.mocks.classes import class_ast, class_str
from cdd.tests.mocks.doctrans import function_type_annotated
from cdd.tests.mocks.methods import (
    class_with_method_and_body_types_ast,
    class_with_method_and_body_types_str,
    class_with_optional_arg_method_ast,
    function_adder_str,
)
from cdd.tests.utils_for_tests import inspectable_compile, run_ast_test, unittest_main


class TestAstUtils(TestCase):
    """Test class for ast_utils"""

    def test_annotate_ancestry(self) -> None:
        """Tests that `annotate_ancestry` properly decorates"""
        node = Module(
            body=[
                AnnAssign(
                    annotation=Name(
                        "str",
                        Load(),
                    ),
                    simple=1,
                    target=Name("dataset_name", Store()),
                    value=set_value("~/tensorflow_datasets"),
                    expr=None,
                    expr_target=None,
                    expr_annotation=None,
                ),
                Assign(
                    annotation=None,
                    simple=1,
                    targets=[Name("epochs", Store())],
                    value=set_value("333"),
                    expr=None,
                    expr_target=None,
                    expr_annotation=None,
                    **maybe_type_comment
                ),
            ],
            stmt=None,
        )
        self.assertFalse(hasattr(node.body[0], "_location"))
        self.assertFalse(hasattr(node.body[1], "_location"))
        annotate_ancestry(node)
        self.assertEqual(node.body[0]._location, ["dataset_name"])
        self.assertEqual(node.body[1]._location, ["epochs"])

    def test_cmp_ast(self) -> None:
        """Test `cmp_ast` branch that isn't tested anywhere else"""
        self.assertFalse(cmp_ast(None, 5))

    def test_emit_ann_assign(self) -> None:
        """Tests that AnnAssign is emitted from `emit_ann_assign`"""
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
                value=set_value("~/tensorflow_datasets"),
                expr=None,
                expr_target=None,
                expr_annotation=None,
            ),
        )

    def test_emit_ann_assign_fails(self) -> None:
        """Tests that `emit_ann_assign` raised the right error"""
        for typ_ in FunctionDef, ClassDef, Module:
            self.assertRaises(NotImplementedError, lambda: emit_ann_assign(typ_))

    def test_emit_arg(self) -> None:
        """Tests that `arg` is emitted from `emit_arg`"""
        self.assertIsInstance(
            class_with_method_and_body_types_ast.body[1].args.args[1], arg
        )
        self.assertIsInstance(
            emit_arg(class_with_method_and_body_types_ast.body[1].args.args[1]), arg
        )
        assign = Assign(
            targets=[Name("yup", Store())],
            value=set_value("nup"),
            expr=None,
            **maybe_type_comment
        )
        gen_ast = emit_arg(assign)
        self.assertIsInstance(gen_ast, arg)
        run_ast_test(
            self,
            gen_ast=gen_ast,
            gold=set_arg("yup"),
        )

    def test_emit_arg_fails(self) -> None:
        """Tests that `emit_arg` raised the right error"""
        for typ_ in FunctionDef, ClassDef, Module:
            self.assertRaises(NotImplementedError, lambda: emit_arg(typ_))

    def test_find_in_ast(self) -> None:
        """Tests that `find_in_ast` successfully finds nodes in AST"""

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
                value=set_value("mnist"),
                expr=None,
                expr_target=None,
                expr_annotation=None,
            ),
        )

    def test_find_in_ast_self(self) -> None:
        """Tests that `find_in_ast` successfully finds itself in AST"""
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
            self,
            find_in_ast(["call_peril"], module_with_fun),
            module_with_fun.body[0],
            skip_black=True,
        )

    def test_find_in_ast_None(self) -> None:
        """Tests that `find_in_ast` fails correctly in AST"""
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
            set_arg(
                annotation=Name(
                    "str",
                    Load(),
                ),
                arg="dataset_name",
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
            set_arg(
                annotation=Name(
                    "str",
                    Load(),
                ),
                arg="dataset_name",
            ),
        )

    def test_get_at_root(self) -> None:
        """Tests that `get_at_root` successfully gets the imports"""
        with open(
            path.join(
                path.dirname(__file__), "mocks", "eval{extsep}py".format(extsep=extsep)
            )
        ) as f:
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
                            name="cdd.tests.mocks",
                            identifier=None,
                            identifier_name=None,
                        )
                    ],
                    alias=None,
                ),
            )
        )

    def test_node_to_dict(self) -> None:
        """
        Tests `node_to_dict`
        """
        self.assertDictEqual(
            node_to_dict(set_arg(arg="a", annotation=Name("int", Load()))),
            dict(
                annotation="int",
                arg="a",
                identifier_arg=None,
                **{"expr": None, "type_comment": None} if PY_GTE_3_8 else {}
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
                annotation=Name("int", Load()),
                simple=1,
                target=Name("dataset_name", Store()),
                value=set_value(15),
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
                value=set_value(15),
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
        )

    def test_get_function_type(self) -> None:
        """Test get_function_type returns the right type"""
        self.assertEqual(
            get_function_type(
                FunctionDef(
                    args=arguments(
                        args=[set_arg("something else")],
                        arg=None,
                        defaults=[],
                        kw_defaults=[],
                        kwarg=None,
                        kwonlyargs=[],
                        posonlyargs=[],
                        vararg=None,
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
                    args=arguments(
                        args=[],
                        arg=None,
                        defaults=[],
                        kw_defaults=[],
                        kwarg=None,
                        kwonlyargs=[],
                        posonlyargs=[],
                        vararg=None,
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
                    args=arguments(
                        args=[set_arg("self")],
                        arg=None,
                        defaults=[],
                        kw_defaults=[],
                        kwarg=None,
                        kwonlyargs=[],
                        posonlyargs=[],
                        vararg=None,
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
                        args=[set_arg("cls")],
                        arg=None,
                        defaults=[],
                        kw_defaults=[],
                        kwarg=None,
                        kwonlyargs=[],
                        posonlyargs=[],
                        vararg=None,
                    ),
                    arguments_args=None,
                    identifier_name=None,
                    stmt=None,
                )
            ),
            "cls",
        )

    def test_get_value(self) -> None:
        """Tests get_value succeeds"""
        val = "foo"
        self.assertEqual(get_value(Str(s=val, constant_value=None, string=None)), val)
        self.assertEqual(
            get_value(Constant(value=val, constant_value=None, string=None)), val
        )
        self.assertIsInstance(get_value(Tuple(expr=None)), Tuple)
        self.assertIsInstance(get_value(Tuple(expr=None)), Tuple)
        self.assertIsNone(get_value(Name(None, None)))
        self.assertEqual(get_value(get_value(ast.parse("-5").body[0])), -5)

    def test_set_value(self) -> None:
        """Tests that `set_value` returns the right type for the right Python version"""
        import cdd.ast_utils

        _cdd_ast_utils_PY3_8_orig = PY3_8
        # patch stopped working, getattr failed also ^
        try:
            cdd.ast_utils.PY3_8 = True

            self.assertIsInstance(
                cdd.ast_utils.set_value(None),
                Constant if PY_GTE_3_8 else NameConstant,
            )

            cdd.ast_utils.PY3_8 = False

            self.assertIsInstance(cdd.ast_utils.set_value(None), NameConstant)
            self.assertIsInstance(cdd.ast_utils.set_value(True), NameConstant)

            cdd.ast_utils.PY3_8 = True
            self.assertIsInstance(
                cdd.ast_utils.set_value("foo"),
                Constant if PY_GTE_3_8 else Str,
            )

            cdd.ast_utils.PY3_8 = False
            self.assertIsInstance(cdd.ast_utils.set_value("foo"), Str)
        finally:
            cdd.ast_utils = _cdd_ast_utils_PY3_8_orig

    def test_set_docstring(self) -> None:
        """
        Tests that `set_docstring` sets the docstring
        """
        with_doc_str = deepcopy(function_type_annotated)
        doc_str = ast.get_docstring(ast.parse(function_adder_str), clean=True)
        set_docstring(doc_str, False, with_doc_str)
        self.assertIsNone(ast.get_docstring(function_type_annotated, clean=True))
        self.assertEqual(ast.get_docstring(with_doc_str, clean=True), doc_str)

        without_doc_str = deepcopy(function_type_annotated)
        doc_str = "\t\n"
        set_docstring(doc_str, False, without_doc_str)
        self.assertIsNone(ast.get_docstring(without_doc_str, clean=True), doc_str)

    def test_find_ast_type(self) -> None:
        """Test that `find_ast_type` gives the wrapped class back"""

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
            self,
            find_ast_type(Module(body=[class_def], stmt=None)),
            class_def,
            skip_black=True,
        )

    def test_find_ast_type_fails(self) -> None:
        """Test that `find_ast_type` throws the right errors"""

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
        """Test that find_ast_type gives the wrapped named class back"""

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
            skip_black=True,
        )

    def test_param2ast_with_assign(self) -> None:
        """Check that `param2ast` behaves correctly with a non annotated (typeless) input"""

        run_ast_test(
            self,
            param2ast(
                ("zion", {"typ": None}),
            ),
            gold=AnnAssign(
                annotation=Name("object", Load()),
                simple=1,
                target=Name("zion", Store()),
                expr=None,
                expr_target=None,
                expr_annotation=None,
            ),
        )

    def test_param2ast_with_assign_dict(self) -> None:
        """Check that `param2ast` behaves correctly with dict type"""

        run_ast_test(
            self,
            param2ast(
                ("menthol", {"typ": "dict"}),
            ),
            gold=AnnAssign(
                annotation=set_slice(Name("dict", Load())),
                simple=1,
                target=Name("menthol", Store()),
                value=Dict(keys=[], values=[], expr=None),
                expr=None,
                expr_target=None,
                expr_annotation=None,
            ),
        )

    def test_param2ast_with_bad_default(self) -> None:
        """Check that `param2ast` behaves correctly with a bad default"""

        run_ast_test(
            self,
            param2ast(
                (
                    "stateful_metrics",
                    {"typ": "NoneType", "default": "the `Model`'s metrics"},
                ),
            ),
            gold=AnnAssign(
                annotation=Name("NoneType", Load()),
                simple=1,
                target=Name("stateful_metrics", Store()),
                value=set_value("```the `Model`'s metrics```"),
                expr=None,
                expr_annotation=None,
                expr_target=None,
            ),
        )

    def test_param2ast_with_wrapped_default(self) -> None:
        """Check that `param2ast` behaves correctly with a wrapped default"""

        run_ast_test(
            self,
            param2ast(
                ("zion", {"typ": None, "default": set_value(NoneStr)}),
            ),
            gold=AnnAssign(
                annotation=Name("object", Load()),
                simple=1,
                target=Name("zion", Store()),
                value=set_value(None),
                expr=None,
                expr_target=None,
                expr_annotation=None,
            ),
        )

    def test_param2argparse_param_none_default(self) -> None:
        """
        Tests that param2argparse_param works to reparse the default
        """
        run_ast_test(
            gen_ast=param2argparse_param(("yup", {"default": NoneStr})),
            gold=Expr(
                Call(
                    args=[set_value("--yup")],
                    func=Attribute(
                        Name("argument_parser", Load()),
                        "add_argument",
                        Load(),
                    ),
                    keywords=[],
                    expr=None,
                    expr_func=None,
                )
            ),
            test_case_instance=self,
        )

    def test_param2argparse_param_default_simple_type(self) -> None:
        """
        Tests that param2argparse_param works to change the type based on the default
        """
        run_ast_test(
            gen_ast=param2argparse_param(
                ("byo", {"default": 5, "typ": "str"}),
            ),
            gold=Expr(
                Call(
                    args=[set_value("--byo")],
                    func=Attribute(
                        Name("argument_parser", Load()),
                        "add_argument",
                        Load(),
                    ),
                    keywords=[
                        keyword(arg="type", value=Name("int", Load()), identifier=None),
                        keyword(arg="required", value=set_value(True), identifier=None),
                        keyword(arg="default", value=set_value(5), identifier=None),
                    ],
                    expr=None,
                    expr_func=None,
                )
            ),
            test_case_instance=self,
        )

    def test_param2argparse_param_default_list(self) -> None:
        """
        Tests that param2argparse_param works to change the type based on the default
          whence said default is a list
        """
        run_ast_test(
            gen_ast=param2argparse_param(
                ("byo", {"default": [], "typ": "str"}),
            ),
            gold=argparse_add_argument_expr,
            test_case_instance=self,
        )

    def test_param2argparse_param_default_ast_tuple(self) -> None:
        """
        Tests that param2argparse_param works to change the type based on the default
          whence said default is an ast.Tuple
        """
        run_ast_test(
            gen_ast=param2argparse_param(
                (
                    "byo",
                    {
                        "default": Tuple(
                            elts=[],
                            ctx=Load(),
                            expr=None,
                        ),
                        "typ": "str",
                    },
                ),
            ),
            gold=Expr(
                Call(
                    args=[set_value("--byo")],
                    func=Attribute(
                        Name("argument_parser", Load()),
                        "add_argument",
                        Load(),
                    ),
                    keywords=[
                        keyword(
                            arg="type", value=Name("loads", Load()), identifier=None
                        ),
                        keyword(arg="required", value=set_value(True), identifier=None),
                        keyword(arg="default", value=set_value("()"), identifier=None),
                    ],
                    expr=None,
                    expr_func=None,
                )
            ),
            test_case_instance=self,
        )

    def test_param2argparse_param_default_ast_list(self) -> None:
        """
        Tests that param2argparse_param works to change the type based on the default
          whence said default is an ast.List
        """
        run_ast_test(
            gen_ast=param2argparse_param(
                (
                    "byo",
                    {
                        "default": List(
                            elts=[],
                            ctx=Load(),
                            expr=None,
                        ),
                        "typ": "str",
                    },
                ),
            ),
            gold=argparse_add_argument_expr,
            test_case_instance=self,
        )

    def test_param2argparse_param_default_ast_expr_with_list(self) -> None:
        """
        Tests that param2argparse_param works to change the type based on the default
          whence said default is an ast.List inside an ast.Expr
        """
        run_ast_test(
            gen_ast=param2argparse_param(
                (
                    "byo",
                    {
                        "default": Expr(
                            List(
                                elts=[],
                                ctx=Load(),
                                expr=None,
                            ),
                            expr_value=None,
                        ),
                        "typ": "str",
                    },
                ),
            ),
            gold=argparse_add_argument_expr,
            test_case_instance=self,
        )

    def test_param2argparse_param_default_ast_binop(self) -> None:
        """
        Tests that param2argparse_param works to change the type based on the default
          whence said default is a non specially handled ast.AST
        """
        run_ast_test(
            gen_ast=param2argparse_param(
                (
                    "byo",
                    {
                        "default": BinOp(
                            set_value(5),
                            Mult(),
                            set_value(5),
                        ),
                        "typ": "str",
                    },
                ),
            ),
            gold=Expr(
                Call(
                    args=[set_value("--byo")],
                    func=Attribute(
                        Name("argument_parser", Load()),
                        "add_argument",
                        Load(),
                    ),
                    keywords=[
                        keyword(arg="required", value=set_value(True), identifier=None),
                        keyword(
                            arg="default",
                            value=set_value("```(5 * 5)```"),
                            identifier=None,
                        ),
                    ],
                    expr=None,
                    expr_func=None,
                )
            ),
            test_case_instance=self,
        )

    def test_param2argparse_param_default_function(self) -> None:
        """
        Tests that param2argparse_param works to change the type based on the default
          whence said default is an in-memory function
        """

        function_str = (
            "from operator import add\n"
            "def adder(a, b):\n"
            "{tab}return add(a, b)".format(tab=tab)
        )
        adder = getattr(
            inspectable_compile(function_str),
            "adder",
        )
        pickled_adder = pickle.dumps(adder)  # eww

        run_ast_test(
            gen_ast=param2argparse_param(
                (
                    "byo",
                    {
                        "default": adder,
                        "typ": "str",
                    },
                ),
            ),
            gold=Expr(
                Call(
                    args=[set_value("--byo")],
                    func=Attribute(
                        Name("argument_parser", Load()),
                        "add_argument",
                        Load(),
                    ),
                    keywords=[
                        keyword(
                            arg="type",
                            value=Name("pickle.loads", Load()),
                            identifier=None,
                        ),
                        keyword(
                            arg="default",
                            value=set_value(pickled_adder),
                            identifier=None,
                        ),
                    ],
                    expr=None,
                    expr_func=None,
                )
            ),
            test_case_instance=self,
        )

    def test_param2argparse_param_default_code_quoted(self) -> None:
        """
        Tests that param2argparse_param works to change the type based on the default
          whence said default is a code quoted str
        """
        run_ast_test(
            gen_ast=(
                param2argparse_param(
                    (
                        "byo",
                        {
                            "default": "```(4)```",
                            "typ": "str",
                        },
                    ),
                )
            ),
            gold=Expr(
                Call(
                    args=[set_value("--byo")],
                    func=Attribute(
                        Name("argument_parser", Load()),
                        "add_argument",
                        Load(),
                    ),
                    keywords=[
                        keyword(arg="type", value=Name("int", Load()), identifier=None),
                        keyword(arg="required", value=set_value(True), identifier=None),
                        keyword(arg="default", value=set_value(4), identifier=None),
                    ],
                    expr=None,
                    expr_func=None,
                )
            ),
            test_case_instance=self,
        )

    def test_param2argparse_param_default_torch(self) -> None:
        """
        Tests that param2argparse_param works to change the type based on the default
          whence said default is a proxy for an internal PyTorch type
        """

        class FakeTorch(object):
            """Not a real torch"""

            def __str__(self):
                """But a real str

                :return: An actual str
                :rtype: ```Literal['<required parameter>']```
                """
                return "<required parameter>"

        # type("FakeTorch", tuple(), {"__str__": lambda _: "<required parameter>"})

        run_ast_test(
            gen_ast=param2argparse_param(
                (
                    "byo",
                    {
                        "default": FakeTorch(),
                    },
                ),
            ),
            gold=Expr(
                Call(
                    args=[set_value("--byo")],
                    func=Attribute(
                        Name("argument_parser", Load()),
                        "add_argument",
                        Load(),
                    ),
                    keywords=[
                        keyword(
                            arg="type",
                            value=Name(FakeTorch.__name__, Load()),
                            identifier=None,
                        ),
                        keyword(arg="required", value=set_value(True), identifier=None),
                    ],
                    expr=None,
                    expr_func=None,
                )
            ),
            test_case_instance=self,
        )

    def test_param2argparse_param_default_class_str(self) -> None:
        """
        Tests that param2argparse_param works to change the type based on the default
          whence said default is a proxy for an unexpected type
        """

        self.assertEqual(
            get_value(
                param2argparse_param(
                    (
                        "byo",
                        {"default": 5.5, "typ": "<class 'float'>"},
                    ),
                )
                .value.keywords[0]
                .value
            ),
            "float",
        )

    def test_param2argparse_param_default_notimplemented(self) -> None:
        """
        Tests that param2argparse_param works to change the type based on the default
          whence said default is a proxy for an unexpected type
        """

        with self.assertRaises(NotImplementedError) as cm:
            param2argparse_param(
                (
                    "byo",
                    {
                        "default": memoryview(b""),
                    },
                ),
            )
        self.assertEqual(
            *map(
                lambda s: s[: s.rfind("<") + 10],
                (
                    "Parsing type <class 'memoryview'>, which contains <memory at 0x10bb8c400>",
                    str(cm.exception),
                ),
            )
        )

    def test_parse_to_scalar(self) -> None:
        """Test various inputs and outputs for `parse_to_scalar`"""
        for fst, snd in (
            (5, 5),
            ("5", "5"),
            (set_value(5), 5),
            (ast.Expr(None), NoneStr),
        ):
            self.assertEqual(parse_to_scalar(fst), snd)

        self.assertEqual(
            get_value(parse_to_scalar(ast.parse("[5]").body[0]).elts[0]), 5
        )
        self.assertTrue(
            cmp_ast(
                parse_to_scalar(ast.parse("[5]").body[0]),
                List([set_value(5)], Load()),
            )
        )

        self.assertEqual(parse_to_scalar(ast.parse("[5]")), "[5]")

        parse_to_scalar(ast.parse("[5]").body[0])

        self.assertRaises(NotImplementedError, parse_to_scalar, memoryview(b""))
        self.assertRaises(NotImplementedError, parse_to_scalar, memoryview(b""))

    def test_infer_type_and_default(self) -> None:
        """Test edge cases for `infer_type_and_default`"""
        self.assertTupleEqual(
            infer_type_and_default(None, 5, "str", False), (None, 5, False, "int")
        )

        self.assertTupleEqual(
            infer_type_and_default(None, [5], "str", False), ("append", 5, False, "int")
        )

        self.assertTupleEqual(
            infer_type_and_default(None, tuple(range(5)), "str", False),
            (None, str(list(range(5))), False, "loads"),
        )

        self.assertTupleEqual(
            infer_type_and_default(None, 0j, "str", False), (None, 0j, False, "complex")
        )

        self.assertTupleEqual(
            infer_type_and_default(None, None, "str", False), (None, None, False, None)
        )

        self.assertTupleEqual(
            infer_type_and_default(None, "```{(5,6): (7,8)}```", "str", False),
            (None, "- 5\n- 6\n", False, "loads"),
        )

    def test__parse_default_from_ast(self) -> None:
        """
        Test `_parse_default_from_ast`
        """
        self.assertTupleEqual(
            _parse_default_from_ast(None, ast.parse("[1,2,56]").body[0], True, None),
            (None, "[1, 2, 56]", True, "loads"),
        )
        self.assertTupleEqual(
            _parse_default_from_ast(None, ast.parse("[5]").body[0], True, None),
            ("append", 5, True, "int"),
        )

    def test_get_ass_where_name(self) -> None:
        """
        Test `get_ass_where_name`
        """
        _mock = ast.parse("foo = 'bar';can = 5;haz: int = 5")
        self.assertTupleEqual(
            tuple(map(get_value, get_ass_where_name(_mock, "foo"))), ("bar",)
        )
        self.assertTupleEqual(
            tuple(map(get_value, get_ass_where_name(_mock, "haz"))), (5,)
        )

    def test_del_ass_where_name(self) -> None:
        """
        Test `del_ass_where_name`
        """
        _mock = ast.parse("foo = 'bar';can = 5;haz: int = 5")
        _mock.body.append(
            Assign(
                targets=[Name("yup", Store())],
                value=set_value("nup"),
                expr=None,
                **maybe_type_comment
            )
        )
        del_ass_where_name(_mock, "yup")
        self.assertTupleEqual(tuple(get_ass_where_name(_mock, "yup")), tuple())

    def test_to_annotation(self) -> None:
        """
        Test `to_annotation`
        """
        for res in "str", Name("str", Load()):
            self.assertTrue(cmp_ast(to_annotation(res), Name("str", Load())))

    def test_merge_assignment_lists(self) -> None:
        """
        Test `merge_assignment_lists`
        """
        src = "__all__ = [ 'a', 'b'];"
        node = ast.parse("__all__ = ['alpha', 'beta']\n".join(repeat(src, 2)))
        merge_assignment_lists(node, "__all__")
        all__ = tuple(get_ass_where_name(node, "__all__"))
        self.assertEqual(len(all__), 1)
        self.assertTupleEqual(
            tuple(map(get_value, all__[0].elts)), ("a", "alpha", "b", "beta")
        )

    def test_merge_modules(self) -> None:
        """
        Test `merge_modules`
        """
        import_line = "\nfrom string import ascii_uppercase"
        src = """\"\"\"\nCool mod\"\"\"{import_line}""".format(import_line=import_line)
        self.assertTrue(
            cmp_ast(
                ast.parse(src),
                merge_modules(
                    *map(ast.parse, (src, src)), remove_imports_from_second=True
                ),
            )
        )
        self.assertTrue(
            cmp_ast(
                ast.parse(src + import_line),
                merge_modules(
                    *map(ast.parse, (src, src)), remove_imports_from_second=False
                ),
            )
        )


unittest_main()
