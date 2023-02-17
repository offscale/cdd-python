""" Tests for gen """

import ast
import os
import sys
from ast import (
    Assign,
    Attribute,
    Call,
    ClassDef,
    Dict,
    Expr,
    Import,
    ImportFrom,
    List,
    Load,
    Module,
    Name,
    Store,
    alias,
)
from copy import deepcopy
from io import StringIO
from json import dump
from os.path import extsep
from shutil import rmtree
from tempfile import TemporaryDirectory, mkdtemp
from unittest import TestCase
from unittest.mock import MagicMock, patch

import cdd.class_.emit
import cdd.function.parse
from cdd.compound.gen import gen
from cdd.shared.ast_utils import maybe_type_comment, set_value
from cdd.shared.pure_utils import rpartial
from cdd.shared.source_transformer import to_code
from cdd.tests.mocks.json_schema import server_error_schema
from cdd.tests.mocks.methods import function_adder_ast
from cdd.tests.mocks.sqlalchemy import (
    config_decl_base_str,
    config_tbl_with_comments_ast,
    sqlalchemy_imports_ast,
)
from cdd.tests.utils_for_tests import run_ast_test, unittest_main

method_adder_ast = deepcopy(function_adder_ast)
method_adder_ast.body[0] = Expr(set_value(" C class (mocked!) "))
method_adder_ast.decorator_list = [Name("staticmethod", Load())]
del function_adder_ast


def populate_files(tempdir, input_module_str=None):
    """
    Populate files in the tempdir

    :param tempdir: Temporary directory
    :type tempdir: ```str```

    :param input_module_str: Input string to write to the input_filename. If None, uses preset mock module.
    :type input_module_str: ```Optional[str]```

    :return: input filename, sqlalchemy_filename, input str, expected_output
    :rtype: ```Tuple[str, str, str, str, Module]```
    """
    input_filename = os.path.join(tempdir, "input{extsep}py".format(extsep=extsep))
    sqlalchemy_filename = os.path.join(
        tempdir, "config_tbl{extsep}py".format(extsep=extsep)
    )
    sqlalchemy_class_name = "Config"
    input_class_name = "Foo"
    input_class_ast = cdd.class_.emit.class_(
        cdd.function.parse.function(deepcopy(method_adder_ast)),
        emit_call=False,
        class_name=input_class_name,
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
                **maybe_type_comment,
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
                **maybe_type_comment,
            ),
        ],
        type_ignores=[],
        stmt=None,
    )
    config_decl_base_ast = ast.parse(config_decl_base_str)
    sqlalchemy_module_ast = Module(
        body=[
            sqlalchemy_imports_ast,
            ImportFrom(
                module="sqlalchemy.orm",
                names=[
                    alias(
                        name="declarative_base",
                        asname=None,
                        identifier=None,
                        identifier_name=None,
                    ),
                ],
                level=0,
                identifier=None,
            ),
            Assign(
                targets=[Name("Base", Store())],
                value=Call(
                    args=[],
                    func=Name("declarative_base", Load()),
                    keywords=[],
                ),
                expr=None,
                lineno=None,
                **maybe_type_comment,
            ),
            config_decl_base_ast,
            Assign(
                targets=[Name("sql_map", Store())],
                value=Dict(
                    keys=[set_value("config_tbl")],
                    values=[
                        Name(
                            next(
                                filter(
                                    rpartial(isinstance, ClassDef),
                                    config_decl_base_ast.body,
                                )
                            ).name,
                            Load(),
                        )
                    ],
                    expr=None,
                ),
                expr=None,
                lineno=None,
                **maybe_type_comment,
            ),
            Assign(
                targets=[Name("__all__", Store())],
                value=List(
                    ctx=Load(),
                    elts=[set_value(sqlalchemy_class_name), set_value("sql_map")],
                    expr=None,
                ),
                expr=None,
                lineno=None,
                **maybe_type_comment,
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
    expected_class_ast = cdd.class_.emit.class_(
        cdd.function.parse.function(deepcopy(method_adder_ast)),
        emit_call=True,
        class_name="{input_class_name}Config".format(input_class_name=input_class_name),
    )
    with open(input_filename, "wt") as f:
        f.write(input_module_str)
    with open(sqlalchemy_filename, "wt") as f:
        f.write(to_code(sqlalchemy_module_ast))
    return (
        input_filename,
        sqlalchemy_filename,
        input_module_ast,
        input_class_ast,
        expected_class_ast,
    )


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
_import_sqlalchemy_class_ast = ImportFrom(
    module="config_tbl",
    names=[
        alias(
            name="sql_map",
            asname=None,
            identifier=None,
            identifier_name=None,
        ),
        alias(
            name="Config",
            asname=None,
            identifier=None,
            identifier_name=None,
        ),
    ],
    level=1,
    identifier=None,
)
_import_star_from_input_str = to_code(_import_star_from_input_ast)
_import_sqlalchemy_class_str = to_code(_import_sqlalchemy_class_ast)

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
    """Test class for gen.py"""

    sys_path = deepcopy(sys.path)
    tempdir = None

    @classmethod
    def setUpClass(cls) -> None:
        """Construct temporary module for use by tests"""
        cls.tempdir = mkdtemp()
        temp_module_dir = os.path.join(cls.tempdir, "gen_test_module")
        os.mkdir(temp_module_dir)
        (
            cls.input_filename,
            cls.sqlalchemy_filename,
            cls.input_module_ast,
            cls.input_class_ast,
            cls.expected_class_ast,
        ) = populate_files(temp_module_dir)
        with open(
            os.path.join(temp_module_dir, "__init__{extsep}py".format(extsep=extsep)),
            "w",
        ) as f:
            f.write(f"{_import_star_from_input_str}{_import_sqlalchemy_class_str}")

        sys.path.append(cls.tempdir)

    @classmethod
    def tearDownClass(cls) -> None:
        """Drop the new module from the path and delete the temporary directory"""
        sys.path = cls.sys_path
        # input("removing: {tempdir!r}".format(tempdir=cls.tempdir))
        rmtree(cls.tempdir)

    def test_gen(self) -> None:
        """Tests `gen`"""

        output_filename = os.path.join(
            self.tempdir, "test_gen_output{extsep}py".format(extsep=extsep)
        )
        with patch("sys.stdout", new_callable=StringIO), patch(
            "sys.stderr", new_callable=StringIO
        ):
            self.assertIsNone(
                gen(
                    name_tpl="{name}Config",
                    input_mapping="gen_test_module.input_map",
                    emit_name="class",
                    parse_name="infer",
                    output_filename=output_filename,
                    prepend="PREPENDED\n",
                    emit_call=True,
                    emit_default_doc=False,
                )
            )
        with open(output_filename, "rt") as f:
            gen_module_str = f.read()
        gen_module_ast = ast.parse(gen_module_str)
        run_ast_test(
            self,
            gen_ast=next(filter(rpartial(isinstance, ClassDef), gen_module_ast.body)),
            gold=self.expected_class_ast,
        )

    def test_sqlalchemy_to_sqlalchemy_table(self) -> None:
        """Tests conversion of sqlalchemy -> sqlalchemy_table"""
        output_filename = os.path.join(
            self.tempdir, "test_sqlalchemy_table_output{extsep}py".format(extsep=extsep)
        )
        gen(
            name_tpl="{name}",
            input_mapping=f"gen_test_module.sql_map",
            imports_from_file="gen_test_module",
            emit_name="sqlalchemy_table",
            parse_name="sqlalchemy",
            output_filename=output_filename,
            emit_call=True,
            emit_default_doc=True,
        )
        with open(output_filename, "rt") as f:
            gen_module_str = f.read()
        gen_module_ast = ast.parse(gen_module_str)
        run_ast_test(
            self,
            gen_ast=next(filter(rpartial(isinstance, Assign), gen_module_ast.body)),
            gold=config_tbl_with_comments_ast,
        )

    def test_gen_with_imports_from_file(self) -> None:
        """Tests `gen` with `imports_from_file`"""

        output_filename = os.path.join(
            self.tempdir,
            "test_gen_with_imports_from_file_output{extsep}py".format(extsep=extsep),
        )
        with patch("sys.stdout", new_callable=StringIO), patch(
            "sys.stderr", new_callable=StringIO
        ):
            self.assertIsNone(
                gen(
                    name_tpl="{name}Config",
                    input_mapping="gen_test_module.input_map",
                    imports_from_file="gen_test_module",
                    emit_name="class",
                    parse_name="infer",
                    output_filename=output_filename,
                    emit_call=True,
                    emit_default_doc=False,
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
                        **maybe_type_comment,
                    ),
                ],
                type_ignores=[],
                stmt=None,
            ),
        )

    def test_gen_with_imports_from_file_and_prepended_import(self) -> None:
        """Tests `gen` with `imports_from_file` and `prepend`"""

        output_filename = os.path.join(
            self.tempdir,
            "test_gen_with_imports_from_file_and_prepended_import_output{extsep}py".format(
                extsep=extsep
            ),
        )
        with patch("sys.stdout", new_callable=StringIO), patch(
            "sys.stderr", new_callable=StringIO
        ):
            self.assertIsNone(
                gen(
                    name_tpl="{name}Config",
                    input_mapping="gen_test_module.input_map",
                    imports_from_file="gen_test_module",
                    emit_name="class",
                    parse_name="infer",
                    prepend=_import_gen_test_module_str,
                    output_filename=output_filename,
                    emit_call=True,
                    emit_default_doc=False,
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
                    **maybe_type_comment,
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

    def test_gen_phase_mocked(self) -> None:
        """Tests that different phases are branched to correctly (doesn't test their internals though)"""
        with patch(
            "cdd.compound.openapi.utils.emit_utils.update_with_imports_from_columns",
            new_callable=MagicMock(),
        ) as phase1_func:
            gen(
                name_tpl="{name}",
                input_mapping="",
                parse_name=None,
                emit_name="sqlalchemy",
                output_filename="",
                phase=1,
            )
        self.assertEqual(phase1_func.call_count, 1)

        with patch(
            "cdd.compound.openapi.utils.emit_utils.update_fk_for_file",
            new_callable=MagicMock(),
        ) as phase2_func:
            gen(
                name_tpl="{name}",
                input_mapping="",
                parse_name=None,
                emit_name="sqlalchemy",
                output_filename="",
                phase=2,
            )
        self.assertEqual(phase2_func.call_count, 1)

        self.assertRaises(
            NotImplementedError,
            gen,
            name_tpl="{name}",
            input_mapping="",
            parse_name=None,
            emit_name="sqlalchemy",
            output_filename="",
            phase=33,
        )

    def test_gen_json_schema_input_mapping(self) -> None:
        """Test `gen` with JSON schema parse target on file"""
        with patch(
            "cdd.json_schema.emit.json_schema_file", new_callable=MagicMock()
        ) as json_schema_file_mock, TemporaryDirectory() as tempdir:
            json_schema_file = os.path.join(tempdir, "foo.json")
            with open(json_schema_file, "wt") as f:
                dump(server_error_schema, f)
            gen(
                "{name}",
                input_mapping=json_schema_file,
                parse_name="json_schema",
                emit_name="json_schema",
                output_filename=os.path.join(tempdir, "foo.gen.json"),
            )
        self.assertEqual(json_schema_file_mock.call_count, 1)


unittest_main()
# mock_class = ClassDef(
#             name="ClassyB",
#             bases=tuple(),
#             decorator_list=[],
#             body=[FunctionDef(
#     name="add_6_5",
#     args=arguments(
#         posonlyargs=[],
#         args=list(map(set_arg, ("a", "b"))),
#         kwonlyargs=[],
#         kw_defaults=[],
#         vararg=None,
#         kwarg=None,
#         defaults=list(map(set_value, (6, 5))),
#         arg=None,
#     ),
#     body=[
#         Expr(
#             set_value(
#                 "\n    :param a: first param\n    "
#                 ":type a: ```int```\n\n    "
#                 ":param b: second param\n    "
#                 ":type b: ```int```\n\n    "
#                 ":return: Aggregated summation of `a` and `b`.\n    "
#                 ":rtype: ```int```\n    ",
#             )
#         ),
#         Return(
#             value=Call(
#                 func=Attribute(Name("operator", Load()), "add", Load()),
#                 args=[Name("a", Load()), Name("b", Load())],
#                 keywords=[],
#                 expr=None,
#                 expr_func=None,
#             ),
#             expr=None,
#         ),
#     ],
#     decorator_list=[],
#     arguments_args=None,
#     identifier_name=None,
#     stmt=None,
# )],
#             keywords=tuple(),
#             identifier_name=None,
#             expr=None,
#         )

# print("===============================================\n",
#       to_code(mock_class),
#      "===============================================",)
