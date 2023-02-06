"""
Tests for `cdd.emit.argparse`
"""

from copy import deepcopy
from unittest import TestCase

import cdd.argparse_function.emit
import cdd.argparse_function.parse
import cdd.class_.emit
import cdd.class_.parse
import cdd.docstring.emit
import cdd.function.emit
import cdd.function.parse
import cdd.json_schema.emit
import cdd.shared.emit.file
import cdd.sqlalchemy.emit
from cdd.tests.mocks.argparse import (
    argparse_func_action_append_ast,
    argparse_func_ast,
    argparse_func_torch_nn_l1loss_ast,
    argparse_func_with_body_ast,
    argparse_function_google_tf_tensorboard_ast,
)
from cdd.tests.mocks.classes import (
    class_ast,
    class_google_tf_tensorboard_ast,
    class_nargs_ast,
)
from cdd.tests.mocks.ir import class_torch_nn_l1loss_ir
from cdd.tests.utils_for_tests import reindent_docstring, run_ast_test, unittest_main


class TestEmitArgparse(TestCase):
    """Tests emission"""

    def test_to_argparse(self) -> None:
        """
        Tests whether `to_argparse` produces `argparse_func_ast` given `class_ast`
        """
        run_ast_test(
            self,
            gen_ast=cdd.argparse_function.emit.argparse_function(
                cdd.class_.parse.class_(class_ast),
                emit_default_doc=False,
            ),
            gold=argparse_func_ast,
        )

    def test_to_argparse_func_nargs(self) -> None:
        """
        Tests whether an argparse function is generated with `action="append"` set properly
        """
        run_ast_test(
            self,
            gen_ast=cdd.argparse_function.emit.argparse_function(
                cdd.class_.parse.class_(class_nargs_ast),
                emit_default_doc=False,
                function_name="set_cli_action_append",
            ),
            gold=argparse_func_action_append_ast,
        )

    def test_to_argparse_google_tf_tensorboard(self) -> None:
        """
        Tests whether `to_argparse` produces `argparse_function_google_tf_tensorboard_ast`
                                    given `class_google_tf_tensorboard_ast`
        """
        run_ast_test(
            self,
            gen_ast=cdd.argparse_function.emit.argparse_function(
                cdd.class_.parse.class_(
                    class_google_tf_tensorboard_ast, merge_inner_function="__init__"
                ),
                emit_default_doc=False,
                word_wrap=False,
            ),
            gold=argparse_function_google_tf_tensorboard_ast,
        )

    def test_from_argparse_with_extra_body_to_argparse_with_extra_body(self) -> None:
        """Tests if this can make the roundtrip from a full argparse function to a argparse full function"""

        ir = cdd.argparse_function.parse.argparse_ast(argparse_func_with_body_ast)
        func = cdd.argparse_function.emit.argparse_function(
            ir, emit_default_doc=False, word_wrap=True
        )
        run_ast_test(
            self, *map(reindent_docstring, (func, argparse_func_with_body_ast))
        )

    def test_from_torch_ir_to_argparse(self) -> None:
        """Tests if emission of class from torch IR is as expected"""

        func = cdd.argparse_function.emit.argparse_function(
            deepcopy(class_torch_nn_l1loss_ir),
            emit_default_doc=False,
            wrap_description=False,
            word_wrap=False,
        )
        run_ast_test(
            self,
            func,
            argparse_func_torch_nn_l1loss_ast,
        )


unittest_main()
