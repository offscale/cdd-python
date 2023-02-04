"""
Tests for the Intermediate Representation produced by the `class` and `function` parsers
"""

import ast
from collections import OrderedDict
from operator import itemgetter
from unittest import TestCase
from unittest.mock import patch

import cdd.argparse_function.emit
import cdd.class_.parse
import cdd.json_schema.parse
from cdd.shared.ast_utils import RewriteAtQuery
from cdd.tests.mocks.classes import (
    class_ast,
    class_google_tf_tensorboard_ast,
    class_google_tf_tensorboard_str,
    class_reduction_v2,
    class_torch_nn_l1loss_ast,
    class_torch_nn_l1loss_str,
    class_torch_nn_one_cycle_lr_ast,
    class_torch_nn_one_cycle_lr_str,
)
from cdd.tests.mocks.ir import (
    class_google_tf_tensorboard_ir,
    class_torch_nn_l1loss_ir,
    class_torch_nn_one_cycle_lr_ir,
    docstring_google_tf_adadelta_function_ir,
    intermediate_repr_no_default_doc,
)
from cdd.tests.mocks.methods import docstring_google_tf_adadelta_function_str
from cdd.tests.utils_for_tests import inspectable_compile, run_ast_test, unittest_main


class TestParseClass(TestCase):
    """
    Tests whether the intermediate representation is consistent when parsed from different inputs.

    IR is a dictionary of form:
        {  "name": Optional[str],
           "type": Optional[str],
           "doc": Optional[str],
           "params": OrderedDict[str, {'typ': str, 'doc': Optional[str], 'default': Any}]
           "returns": Optional[OrderedDict[Literal['return_type'],
                                           {'typ': str, 'doc': Optional[str], 'default': Any}),)]] }
    """

    def test_from_class(self) -> None:
        """
        Tests whether `class_` produces `intermediate_repr_no_default_doc`
              from `class_ast`
        """
        ir = cdd.class_.parse.class_(class_ast)
        del ir["_internal"]  # Not needed for this test
        ir["name"] = None
        self.assertDictEqual(ir, intermediate_repr_no_default_doc)

    def test_from_class_in_memory(self) -> None:
        """
        Tests that `parse.class_` produces properly from a `class` in memory of current interpreter
        """

        class A(object):
            """A is one boring class"""

        ir = cdd.class_.parse.class_(A)
        del ir["_internal"]  # Not needed for this test
        self.assertDictEqual(
            ir,
            {
                "doc": "A is one boring class",
                "name": "TestParseClass.test_from_class_in_memory.<locals>.A",
                "params": OrderedDict(),
                "returns": None,
            },
        )

    def test_from_adadelta_class_in_memory(self) -> None:
        """
        Tests that parse.class produces properly from a `class` in memory of current interpreter
        """
        Adadelta = getattr(
            inspectable_compile(docstring_google_tf_adadelta_function_str),
            "Adadelta",
        )
        ir = cdd.class_.parse.class_(Adadelta)
        del ir["_internal"]
        # self.assertDictEqual(ir, docstring_google_tf_adadelta_ir)
        self.assertDictEqual(
            ir,
            docstring_google_tf_adadelta_function_ir,
        )

    def test_from_class_and_function(self) -> None:
        """
        Tests that the parser can combine the outer class docstring + structure
        with the inner function parameter defaults
        """

        # Sanity check
        run_ast_test(
            self,
            class_google_tf_tensorboard_ast,
            gold=ast.parse(class_google_tf_tensorboard_str).body[0],
        )

        parsed_ir = cdd.class_.parse.class_(
            class_google_tf_tensorboard_ast,
            merge_inner_function="__init__",
            parse_original_whitespace=True,
            word_wrap=False,
            infer_type=True,
        )

        del parsed_ir["_internal"]  # Not needed for this test

        self.assertEqual(
            *map(itemgetter("doc"), (parsed_ir, class_google_tf_tensorboard_ir))
        )
        self.assertDictEqual(parsed_ir, class_google_tf_tensorboard_ir)

    def test_from_class_and_function_in_memory(self) -> None:
        """
        Tests that the parser can combine the outer class docstring + structure
        with the inner function parameter defaults
        """

        parsed_ir = cdd.class_.parse.class_(
            RewriteAtQuery,
            merge_inner_function="__init__",
            infer_type=True,
        )

        del parsed_ir["_internal"]  # Not needed for this test

        self.assertDictEqual(
            parsed_ir,
            {
                "doc": "Replace the node at query with given node",
                "name": "RewriteAtQuery",
                "params": OrderedDict(
                    (
                        (
                            "search",
                            {
                                "doc": "Search query, e.g., "
                                "['node_name', 'function_name', "
                                "'arg_name']",
                                "typ": "List[str]",
                            },
                        ),
                        (
                            "replacement_node",
                            {"doc": "Node to replace this search", "typ": "AST"},
                        ),
                        (
                            "replaced",
                            {
                                "doc": "Whether a node has been replaced "
                                "(only replaces first "
                                "occurrence)"
                            },
                        ),
                    )
                ),
                "returns": None,
            },
        )

    def test_from_class_and_function_torch(self) -> None:
        """
        Tests that the parser can combine the outer class docstring + structure
        with the inner function parameter defaults, given a PyTorch loss class
        """

        # Sanity check
        run_ast_test(
            self,
            class_torch_nn_l1loss_ast,
            gold=ast.parse(class_torch_nn_l1loss_str).body[0],
        )

        parsed_ir = cdd.class_.parse.class_(
            class_torch_nn_l1loss_ast,
            merge_inner_function="__init__",
            infer_type=True,
            parse_original_whitespace=True,
        )

        del parsed_ir["_internal"]  # Not needed for this test

        self.assertDictEqual(parsed_ir, class_torch_nn_l1loss_ir)

    def test_from_class_torch_nn_one_cycle_lr(self) -> None:
        """
        Tests that the parser can combine the outer class docstring + structure
        with the inner function parameter defaults, given a PyTorch loss LR scheduler class
        """

        # Sanity check
        run_ast_test(
            self,
            class_torch_nn_one_cycle_lr_ast,
            gold=ast.parse(class_torch_nn_one_cycle_lr_str).body[0],
        )

        parsed_ir = cdd.class_.parse.class_(
            class_torch_nn_one_cycle_lr_ast,
            merge_inner_function="__init__",
            infer_type=True,
            word_wrap=False,
        )

        del parsed_ir["_internal"]  # Not needed for this test

        self.assertEqual(parsed_ir["params"]["last_epoch"]["typ"], "int")
        self.assertEqual(parsed_ir["doc"], class_torch_nn_one_cycle_lr_ir["doc"])
        self.assertDictEqual(parsed_ir, class_torch_nn_one_cycle_lr_ir)

    def test__class_from_memory(self) -> None:
        """
        Tests that the parser can combine the outer class docstring + structure
        with the inner function parameter defaults, given a PyTorch loss LR scheduler class
        """

        class A(object):
            """A is one boring class"""

        with patch("inspect.getsourcefile", lambda _: None):
            ir = cdd.class_.parse._class_from_memory(
                A, A.__name__, False, False, False, False
            )
        self.assertDictEqual(
            ir,
            {
                "doc": "A is one boring class",
                "name": "A",
                "params": OrderedDict(),
                "returns": None,
            },
        )

    def test_from_class_class_reduction_v2(self) -> None:
        """
        Test class_reduction_v2 produces correct IR
        """
        ir = cdd.class_.parse.class_(class_reduction_v2)
        self.assertEqual(
            ir["params"],
            OrderedDict(
                (
                    ("AUTO", {"default": "auto", "typ": "str"}),
                    ("NONE", {"default": "none", "typ": "str"}),
                    ("SUM", {"default": "sum", "typ": "str"}),
                    (
                        "SUM_OVER_BATCH_SIZE",
                        {"default": "sum_over_batch_size", "typ": "str"},
                    ),
                )
            ),
        )
        self.assertEqual(ir["returns"], None)


unittest_main()
