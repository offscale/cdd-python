""" Tests for gen """

import ast
import os
import sys
from ast import (
    Assign,
    Attribute,
    ClassDef,
    Expr,
    FunctionDef,
    Import,
    ImportFrom,
    Load,
    Module,
    Name,
    Store,
    alias,
    arguments,
    List,
    Dict,
)
from copy import deepcopy
from io import StringIO
from shutil import rmtree
from tempfile import mkdtemp
from unittest import TestCase
from unittest.mock import patch

from doctrans.ast_utils import set_value, set_arg, maybe_type_comment
from doctrans.gen import gen
from doctrans.pure_utils import rpartial
from doctrans.source_transformer import to_code
from doctrans.tests.utils_for_tests import run_ast_test, unittest_main


def populate_files(tempdir, input_module_str=None):
    """
    Populate files in the tempdir

    :param tempdir: Temporary directory
    :type tempdir: ```str```

    :param input_module_str: Input string to write to the input_filename. If None, uses preset mock module.
    :type input_module_str: ```Optional[str]```

    :return: input filename, input str, expected_output
    :rtype: ```Tuple[str, str, str, Module]```
    """
    input_filename = os.path.join(tempdir, "input.py")
    input_class_name = "Foo"
    input_class_ast = ClassDef(
        bases=[Name("object", Load())],
        body=[
            Expr(
                set_value(
                    "\n    The amazing {input_class_name}\n\n"
                    "    :cvar a: An a\n"
                    "    :cvar b: A b\n"
                    "    ".format(input_class_name=input_class_name)
                )
            ),
            Assign(
                targets=[Name("a", Store())],
                value=set_value(5),
                expr=None,
                lineno=None,
                **maybe_type_comment
            ),
            Assign(
                targets=[Name("b", Store())],
                value=set_value(16),
                expr=None,
                lineno=None,
                **maybe_type_comment
            ),
        ],
        decorator_list=[],
        keywords=[],
        name=input_class_name,
        expr=None,
        identifier_name=None,
    )

    input_module_ast = Module(
        body=[
            input_class_ast,
            Assign(
                targets=[Name("input_map", Store())],
                value=Dict(
                    keys=[set_value(input_class_name)],
                    values=[Name(input_class_name, Load())],
                    expr=None,
                ),
                expr=None,
                lineno=None,
                **maybe_type_comment
            ),
            Assign(
                targets=[Name("__all__", Store())],
                value=List(
                    ctx=Load(),
                    elts=[set_value(input_class_name), set_value("input_map")],
                    expr=None,
                ),
                expr=None,
                lineno=None,
                **maybe_type_comment
            ),
        ],
        type_ignores=[],
        stmt=None,
    )

    input_module_str = input_module_str or to_code(input_module_ast)
    # expected_output_class_str = (
    #     "class FooConfig(object):\n"
    #     '    """\n'
    #     "    The amazing Foo\n\n"
    #     "    :cvar a: An a. Defaults to 5\n"
    #     '    :cvar b: A b. Defaults to 16"""\n'
    #     "    a = 5\n"
    #     "    b = 16\n\n"
    #     "    def __call__(self):\n"
    #     "        self.a = 5\n"
    #     "        self.b = 16\n"
    # )
    class_name = "{input_class_name}Config".format(input_class_name=input_class_name)
    expected_class_ast = ClassDef(
        name=class_name,
        bases=[Name("object", Load())],
        keywords=[],
        body=[
            Expr(
                set_value(
                    "\n    The amazing {input_class_name}\n\n"
                    "    :cvar a: An a. Defaults to 5\n"
                    "    :cvar b: A b. Defaults to 16".format(
                        input_class_name=input_class_name
                    )
                )
            ),
            Assign(
                targets=[Name("a", Store())],
                value=set_value(5),
                expr=None,
                lineno=None,
                **maybe_type_comment
            ),
            Assign(
                targets=[Name("b", Store())],
                value=set_value(16),
                expr=None,
                lineno=None,
                **maybe_type_comment
            ),
            FunctionDef(
                name="__call__",
                args=arguments(
                    posonlyargs=[],
                    args=[set_arg("self")],
                    kwonlyargs=[],
                    kw_defaults=[],
                    defaults=[],
                    arg=None,
                    vararg=None,
                    kwarg=None,
                ),
                body=[
                    Assign(
                        targets=[Attribute(Name("self", Load()), "a", Store())],
                        value=set_value(5),
                        expr=None,
                        lineno=None,
                        **maybe_type_comment
                    ),
                    Assign(
                        targets=[Attribute(Name("self", Load()), "b", Store())],
                        value=set_value(16),
                        expr=None,
                        lineno=None,
                        **maybe_type_comment
                    ),
                ],
                decorator_list=[],
                arguments_args=None,
                identifier_name=None,
                stmt=None,
                lineno=None,
                returns=None,
                **maybe_type_comment
            ),
        ],
        decorator_list=[],
        expr=None,
        identifier_name=None,
    )

    with open(input_filename, "wt") as f:
        f.write(input_module_str)

    return input_filename, input_module_ast, input_class_ast, expected_class_ast


_import_star_from_input_ast = ImportFrom(
    module="input",
    names=[
        alias(
            name="input_map",
            asname=None,
            identifier=None,
            identifier_name=None,
        ),
        alias(
            name="Foo",
            asname=None,
            identifier=None,
            identifier_name=None,
        ),
    ],
    level=1,
    identifier=None,
)
_import_star_from_input_str = to_code(_import_star_from_input_ast)

_import_gen_test_module_ast = Import(
    names=[
        alias(
            name="gen_test_module",
            asname=None,
            identifier=None,
            identifier_name=None,
        )
    ],
    alias=None,
)
_import_gen_test_module_str = "{}\n".format(
    to_code(_import_gen_test_module_ast).rstrip("\n")
)


class TestGen(TestCase):
    """ Test class for gen.py """

    sys_path = deepcopy(sys.path)
    tempdir = None

    @classmethod
    def setUpClass(cls) -> None:
        """ Construct temporary module for use by tests """
        cls.tempdir = mkdtemp()
        temp_module_dir = os.path.join(cls.tempdir, "gen_test_module")
        os.mkdir(temp_module_dir)
        (
            cls.input_filename,
            cls.input_module_ast,
            cls.input_class_ast,
            cls.expected_class_ast,
        ) = populate_files(temp_module_dir)
        with open(os.path.join(temp_module_dir, "__init__.py"), "w") as f:
            f.write(_import_star_from_input_str)

        sys.path.append(cls.tempdir)

    @classmethod
    def tearDownClass(cls) -> None:
        """ Drop the new module from the path and delete the temporary directory """
        sys.path = cls.sys_path
        # input("removing: {!r}".format(cls.tempdir))
        rmtree(cls.tempdir)

    def test_gen(self) -> None:
        """ Tests `gen` """

        output_filename = os.path.join(self.tempdir, "test_gen_output.py")
        with patch("sys.stdout", new_callable=StringIO), patch(
            "sys.stderr", new_callable=StringIO
        ):
            self.assertIsNone(
                gen(
                    name_tpl="{name}Config",
                    input_mapping="gen_test_module.input_map",
                    type_="class",
                    output_filename=output_filename,
                    prepend="PREPENDED\n",
                    emit_call=True,
                )
            )
        with open(output_filename, "rt") as f:
            gen_module_ast = ast.parse(f.read())
        run_ast_test(
            self,
            gen_ast=next(filter(rpartial(isinstance, ClassDef), gen_module_ast.body)),
            gold=self.expected_class_ast,
        )

    def test_gen_with_imports_from_file(self) -> None:
        """ Tests `gen` with `imports_from_file` """

        output_filename = os.path.join(
            self.tempdir, "test_gen_with_imports_from_file_output.py"
        )
        with patch("sys.stdout", new_callable=StringIO), patch(
            "sys.stderr", new_callable=StringIO
        ):
            self.assertIsNone(
                gen(
                    name_tpl="{name}Config",
                    input_mapping="gen_test_module.input_map",
                    imports_from_file="gen_test_module",
                    type_="class",
                    output_filename=output_filename,
                    emit_call=True,
                )
            )
        with open(output_filename, "rt") as f:
            gen_ast = ast.parse(f.read())
        run_ast_test(
            self,
            gen_ast=gen_ast,
            gold=Module(
                body=[
                    _import_star_from_input_ast,
                    self.expected_class_ast,
                    Assign(
                        targets=[Name("__all__", Store())],
                        value=List(
                            ctx=Load(),
                            elts=[set_value("FooConfig")],
                            expr=None,
                        ),
                        expr=None,
                        lineno=None,
                        **maybe_type_comment
                    ),
                ],
                type_ignores=[],
                stmt=None,
            ),
        )

    def test_gen_with_imports_from_file_and_prepended_import(self) -> None:
        """ Tests `gen` with `imports_from_file` and `prepend` """

        output_filename = os.path.join(
            self.tempdir,
            "test_gen_with_imports_from_file_and_prepended_import_output.py",
        )
        with patch("sys.stdout", new_callable=StringIO), patch(
            "sys.stderr", new_callable=StringIO
        ):
            self.assertIsNone(
                gen(
                    name_tpl="{name}Config",
                    input_mapping="gen_test_module.input_map",
                    imports_from_file="gen_test_module",
                    type_="class",
                    prepend=_import_gen_test_module_str,
                    output_filename=output_filename,
                    emit_call=True,
                )
            )

        with open(output_filename, "rt") as f:
            gen_ast = ast.parse(f.read())
        gold = Module(
            body=[
                _import_gen_test_module_ast,
                _import_star_from_input_ast,
                self.expected_class_ast,
                # self.input_module_ast.body[1],
                Assign(
                    targets=[Name("__all__", Store())],
                    value=List(
                        ctx=Load(),
                        elts=[set_value("FooConfig")],
                        expr=None,
                    ),
                    expr=None,
                    lineno=None,
                    **maybe_type_comment
                ),
            ],
            type_ignores=[],
            stmt=None,
        )
        run_ast_test(
            self,
            gen_ast=gen_ast,
            gold=gold,
        )


unittest_main()
