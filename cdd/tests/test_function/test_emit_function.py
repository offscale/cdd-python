"""
Tests for `cdd.emit.function`
"""

import ast
from ast import FunctionDef
from collections import OrderedDict
from copy import deepcopy
from functools import partial
from textwrap import indent
from unittest import TestCase

import cdd.argparse_function.emit
import cdd.argparse_function.parse
import cdd.class_.emit
import cdd.class_.parse
import cdd.docstring.emit
import cdd.docstring.parse
import cdd.function.emit
import cdd.function.parse
import cdd.json_schema.emit
import cdd.shared.emit.file
import cdd.sqlalchemy.emit
from cdd.shared.ast_utils import get_function_type, set_value
from cdd.shared.pure_utils import none_types, rpartial, tab
from cdd.tests.mocks.classes import class_squared_hinge_config_ast
from cdd.tests.mocks.docstrings import (
    docstring_header_str,
    docstring_no_type_no_default_tpl_str,
    docstring_str,
)
from cdd.tests.mocks.methods import (
    class_with_method_ast,
    class_with_method_str,
    class_with_method_types_ast,
    class_with_method_types_str,
    function_google_tf_squared_hinge_str,
)
from cdd.tests.utils_for_tests import reindent_docstring, run_ast_test, unittest_main


class TestEmitFunction(TestCase):
    """Tests emission"""

    def test_to_function(self) -> None:
        """
        Tests whether `function` produces method from `class_with_method_types_ast` given `docstring_str`
        """

        function_def = reindent_docstring(
            deepcopy(
                next(
                    filter(
                        rpartial(isinstance, FunctionDef),
                        class_with_method_types_ast.body,
                    )
                )
            )
        )

        function_name = function_def.name
        function_type = get_function_type(function_def)

        gen_ast = cdd.function.emit.function(
            cdd.docstring.parse.docstring(docstring_str),
            function_name=function_name,
            function_type=function_type,
            emit_default_doc=False,
            type_annotations=True,
            emit_separating_tab=True,
            indent_level=1,
            emit_as_kwonlyargs=False,
        )

        run_ast_test(
            self,
            gen_ast=gen_ast,
            gold=function_def,
        )

    def test_to_function_with_docstring_types(self) -> None:
        """
        Tests that `function` can generate a function_def with types in docstring
        """

        # Sanity check
        run_ast_test(
            self,
            class_with_method_ast,
            gold=ast.parse(class_with_method_str).body[0],
        )

        function_def = reindent_docstring(
            deepcopy(
                next(
                    filter(
                        rpartial(isinstance, FunctionDef), class_with_method_ast.body
                    )
                )
            )
        )

        ir = cdd.function.parse.function(function_def)
        gen_ast = reindent_docstring(
            cdd.function.emit.function(
                ir,
                function_name=function_def.name,
                function_type=get_function_type(function_def),
                emit_default_doc=False,
                type_annotations=False,
                indent_level=1,
                emit_separating_tab=True,
                emit_as_kwonlyargs=False,
                word_wrap=False,
            )
        )

        run_ast_test(self, gen_ast=gen_ast, gold=function_def)

    def test_to_function_with_type_annotations(self) -> None:
        """
        Tests that `function` can generate a function_def with inline types
        """
        function_def = deepcopy(
            next(
                filter(
                    rpartial(isinstance, FunctionDef), class_with_method_types_ast.body
                )
            )
        )
        function_name = function_def.name
        function_type = get_function_type(function_def)
        function_def.body[0].value = set_value(
            "\n{tab}{ds}{tab}".format(
                tab=tab,
                ds=indent(
                    docstring_no_type_no_default_tpl_str.format(
                        header_doc_str=docstring_header_str
                    ),
                    prefix=tab,
                    predicate=lambda _: _,
                ).lstrip(),
            )
        )

        gen_ast = cdd.function.emit.function(
            cdd.function.parse.function(
                function_def,
                function_name=function_name,
                function_type=function_type,
            ),
            function_name=function_name,
            function_type=function_type,
            emit_default_doc=False,
            type_annotations=True,
            emit_separating_tab=True,
            indent_level=0,
            emit_as_kwonlyargs=False,
        )
        # emit.file(gen_ast, os.path.join(os.path.dirname(__file__), 'delme.py'), mode='wt')

        run_ast_test(
            self,
            gen_ast=gen_ast,
            gold=function_def,
        )

    def test_to_function_emit_as_kwonlyargs(self) -> None:
        """
        Tests whether `function` produces method with keyword only arguments
        """
        function_def = reindent_docstring(
            deepcopy(
                next(
                    filter(
                        rpartial(isinstance, FunctionDef),
                        ast.parse(
                            class_with_method_types_str.replace("self", "self, *")
                        )
                        .body[0]
                        .body,
                    )
                )
            )
        )
        function_name = function_def.name
        function_type = get_function_type(function_def)

        gen_ast = cdd.function.emit.function(
            cdd.docstring.parse.docstring(docstring_str),
            function_name=function_name,
            function_type=function_type,
            emit_default_doc=False,
            type_annotations=True,
            emit_separating_tab=True,
            indent_level=1,
            emit_as_kwonlyargs=True,
        )

        run_ast_test(
            self,
            gen_ast=gen_ast,
            gold=function_def,
        )

    def test_from_function_google_tf_squared_hinge_str_to_class(self) -> None:
        """
        Tests that `emit.function` produces correctly with:
        - __call__
        """

        gold_ir = cdd.class_.parse.class_(class_squared_hinge_config_ast)
        gold_ir.update(
            {
                key: OrderedDict(
                    (
                        (
                            name,
                            {
                                k: v
                                for k, v in _param.items()
                                if k != "typ"
                                and (k != "default" or v not in none_types)
                            },
                        )
                        for name, _param in gold_ir[key].items()
                    )
                )
                for key in ("params", "returns")
            }
        )

        gen_ir = cdd.function.parse.function(
            ast.parse(function_google_tf_squared_hinge_str).body[0],
            infer_type=True,
            word_wrap=False,
        )
        self.assertEqual(
            *map(lambda ir: ir["returns"]["return_type"]["doc"], (gen_ir, gold_ir))
        )
        # print('#gen_ir')
        # pp(gen_ir)
        # print('#gold_ir')
        # pp(gold_ir)

        run_ast_test(
            self,
            *map(
                partial(
                    cdd.class_.emit.class_,
                    class_name="SquaredHingeConfig",
                    emit_call=True,
                    emit_default_doc=True,
                    emit_original_whitespace=True,
                    word_wrap=False,
                ),
                (gen_ir, gold_ir),
            )
        )


unittest_main()
