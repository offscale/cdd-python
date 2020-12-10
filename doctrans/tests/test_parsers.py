"""
Tests for the Intermediate Representation produced by the parsers
"""
import ast
from ast import FunctionDef
from unittest import TestCase

from doctrans import parse, emit
from doctrans.ast_utils import get_value, RewriteAtQuery
from doctrans.pure_utils import PY_GTE_3_8, tab
from doctrans.tests.mocks.argparse import argparse_func_ast
from doctrans.tests.mocks.classes import (
    class_ast,
    class_google_tf_tensorboard_ast,
    class_google_tf_tensorboard_str,
    class_google_tf_tensorboard_ir,
)
from doctrans.tests.mocks.docstrings import (
    intermediate_repr_no_default_doc,
)
from doctrans.tests.mocks.ir import method_complex_args_variety_ir
from doctrans.tests.mocks.methods import (
    function_adder_ast,
    function_default_complex_default_arg_ast,
    method_complex_args_variety_ast,
    method_complex_args_variety_str,
    function_adder_str,
    function_adder_ir,
    docstring_google_tf_adadelta_function_str,
)
from doctrans.tests.utils_for_tests import (
    unittest_main,
    inspectable_compile,
    run_ast_test,
)


class TestParsers(TestCase):
    """
    Tests whether the intermediate representation is consistent when parsed from different inputs.

    IR is a dictionary of form:
              {
                  'name': ...,
                  'type': ...,
                  'doc': ...,
                  'params': [{'name': ..., 'typ': ..., 'doc': ..., 'default': ..., 'required': ... }, ...],
                  'returns': {'name': ..., 'typ': ..., 'doc': ..., 'default': ..., 'required': ... }
              }
    """

    def test_from_argparse_ast(self) -> None:
        """
        Tests whether `argparse_ast` produces `intermediate_repr_no_default_doc_or_prop`
              from `argparse_func_ast`"""
        self.assertDictEqual(
            parse.argparse_ast(argparse_func_ast), intermediate_repr_no_default_doc
        )

    def test_from_argparse_ast_empty(self) -> None:
        """
        Tests `argparse_ast` empty condition
        """
        self.assertEqual(
            emit.to_code(
                emit.argparse_function(
                    parse.argparse_ast(
                        FunctionDef(
                            body=[],
                            arguments_args=None,
                            identifier_name=None,
                            stmt=None,
                        )
                    ),
                    emit_default_doc=True,
                )
            ).rstrip("\n"),
            "def set_cli_args(argument_parser):\n"
            '{tab}"""\n'
            "{tab}Set CLI arguments\n\n"
            "{tab}:param argument_parser: argument parser\n"
            "{tab}:type argument_parser: ```ArgumentParser```\n\n"
            "{tab}:return: argument_parser\n"
            "{tab}:rtype: ```ArgumentParser```\n"
            '{tab}"""\n'
            "{tab}argument_parser.description = ''\n"
            "{tab}return argument_parser".format(tab=tab),
        )

    def test_from_class(self) -> None:
        """
        Tests whether `class_` produces `intermediate_repr_no_default_doc`
              from `class_ast`
        """
        ir = parse.class_(class_ast)
        del ir["_internal"]  # Not needed for this test
        self.assertDictEqual(ir, intermediate_repr_no_default_doc)

    def test_from_function(self) -> None:
        """
        Tests that parse.function produces properly
        """
        gen_ir = parse.function(function_default_complex_default_arg_ast)
        gold_ir = {
            "description": "",
            "name": "call_peril",
            "params": [
                {
                    "default": "mnist",
                    "name": "dataset_name",
                    "typ": "str",
                    "doc": None,
                },
                {
                    "default": get_value(
                        function_default_complex_default_arg_ast.args.defaults[1]
                    ),
                    "name": "writer",
                    "typ": None,
                    "doc": None,
                },
            ],
            "returns": None,
            "type": "static",
        }
        self.assertDictEqual(
            gen_ir,
            gold_ir,
        )

    def test_from_function_kw_only(self) -> None:
        """
        Tests that parse.function produces properly from function with only keyword arguments
        """
        self.assertDictEqual(
            parse.function(function_adder_ast),
            function_adder_ir,
        )

    def test_from_function_in_memory(self) -> None:
        """
        Tests that parse.function produces properly from a function in memory of current interpreter
        """

        def foo(a=5, b=6):
            """
            the foo function

            :param a: the a value
            :param b: the b value

            """

        self.assertIsNone(foo(5, 6))

        ir = parse.function(foo)
        del ir["_internal"]  # Not needed for this test
        self.assertDictEqual(
            ir,
            {
                "doc": "the foo function",
                "name": "TestParsers.test_from_function_in_memory.<locals>.foo",
                "params": [
                    {"default": 5, "doc": "the a value", "name": "a", "typ": "int"},
                    {"default": 6, "doc": "the b value", "name": "b", "typ": "int"},
                ],
                "returns": None,
                "type": "static",
            },
        )

    maxDiff = None

    def test_from_method_in_memory(self) -> None:
        """
        Tests that `parse.function` produces properly from a function in memory of current interpreter with:
        - kw only args;
        - default args;
        - annotations
        - required;
        - unannotated;
        - splat
        """

        method_complex_args_variety_with_imports_str = (
            "from sys import stdout\n"
            "from {} import Literal\n"
            "{}".format(
                "typing" if PY_GTE_3_8 else "typing_extensions",
                method_complex_args_variety_str,
            )
        )
        call_cliff = getattr(
            inspectable_compile(method_complex_args_variety_with_imports_str),
            "call_cliff",
        )

        ir = parse.function(call_cliff)
        del ir["_internal"]  # Not needed for this test

        # This is a hack because JetBrains wraps stdout
        self.assertIn(
            type(ir["params"][-2]["default"]).__name__,
            frozenset(("FlushingStringIO", "TextIOWrapper")),
        )
        ir["params"][-2]["default"] = "stdout"

        # This extra typ is removed, for now. TODO: Update AST-level parser to set types when defaults are given.
        for i in -2, 3:
            del ir["params"][i]["typ"]

        self.assertDictEqual(
            ir,
            method_complex_args_variety_ir,
        )

    def test_from_method_in_memory_return_complex(self) -> None:
        """
        Tests that `parse.function` produces properly from a function in memory of current interpreter with:
        - complex return type
        - kwonly args
        """

        method_complex_args_variety_with_imports_str = (
            "from sys import stdout\n"
            "from {} import Literal\n"
            "{}".format(
                "typing" if PY_GTE_3_8 else "typing_extensions",
                function_adder_str,
            )
        )
        add_6_5 = getattr(
            inspectable_compile(method_complex_args_variety_with_imports_str),
            "add_6_5",
        )

        ir = parse.function(add_6_5)
        del ir["_internal"]  # Not needed for this test

        self.assertDictEqual(
            ir,
            function_adder_ir,
        )

    def test_from_method_complex_args_variety(self) -> None:
        """
        Tests that `parse.function` produces correctly with:
        - kw only args;
        - default args;
        - annotations
        - required;
        - unannotated;
        - splat
        """
        self.assertDictEqual(
            parse.function(method_complex_args_variety_ast),
            method_complex_args_variety_ir,
        )

    def test_from_class_in_memory(self) -> None:
        """
        Tests that parse.class produces properly from a `class` in memory of current interpreter
        """

        class A(object):
            """A is one boring class"""

        ir = parse.class_(A)
        del ir["_internal"]  # Not needed for this test
        self.assertDictEqual(
            ir,
            {
                "doc": "A is one boring class",
                "name": "TestParsers.test_from_class_in_memory.<locals>.A",
                "params": [],
                "returns": None,
            },
        )

    def test_from_adadelta_class_in_memory(self) -> None:
        """
        Tests that parse.class produces properly from a `class` in memory of current interpreter
        """
        Adadelta = getattr(
            inspectable_compile(
                "{!s}".format(docstring_google_tf_adadelta_function_str)
            ),
            "Adadelta",
        )
        ir = parse.class_(Adadelta)
        del ir["_internal"]
        # self.assertDictEqual(ir, docstring_google_tf_adadelta_ir)
        self.assertDictEqual(
            ir,
            {
                "doc": "Optimizer that implements the Adadelta algorithm.\n"
                "\n"
                "Adadelta optimization is a stochastic gradient descent method that "
                "is based on\n"
                "adaptive learning rate per dimension to address two drawbacks:\n"
                "\n"
                "- The continual decay of learning rates throughout training\n"
                "- The need for a manually selected global learning rate\n"
                "\n"
                "Adadelta is a more robust extension of Adagrad that adapts "
                "learning rates\n"
                "based on a moving window of gradient updates, instead of "
                "accumulating all\n"
                "past gradients. This way, Adadelta continues learning even when "
                "many updates\n"
                "have been done. Compared to Adagrad, in the original version of "
                "Adadelta you\n"
                "don't have to set an initial learning rate. In this version, "
                "initial\n"
                "learning rate can be set, as in most other Keras optimizers.\n"
                "\n"
                'According to section 4.3 ("Effective Learning rates"), near the '
                "end of\n"
                "training step sizes converge to 1 which is effectively a high "
                "learning\n"
                "rate which would cause divergence. This occurs only near the end "
                "of the\n"
                "training as gradients and step sizes are small, and the epsilon "
                "constant\n"
                "in the numerator and denominator dominate past gradients and "
                "parameter\n"
                "updates which converge the learning rate to 1.\n"
                "\n"
                'According to section 4.4("Speech Data"),where a large neural '
                "network with\n"
                "4 hidden layers was trained on a corpus of US English data, "
                "ADADELTA was\n"
                "used with 100 network replicas.The epsilon used is 1e-6 with "
                "rho=0.95\n"
                "which converged faster than ADAGRAD, by the following "
                "construction:\n"
                "def __init__(self, lr=1.0, rho=0.95, epsilon=1e-6, decay=0., "
                "**kwargs):\n"
                "\n"
                "\n"
                "Reference:\n"
                "    - [Zeiler, 2012](http://arxiv.org/abs/1212.5701)",
                "name": "Adadelta",
                "params": [
                    {
                        "default": 0.001,
                        "doc": "A `Tensor`, floating point value, or a schedule "
                        "that is a\n"
                        "  "
                        "`tf.keras.optimizers.schedules.LearningRateSchedule`. "
                        "The learning rate.\n"
                        "  To match the exact form in the original paper "
                        "use 1.0.",
                        "name": "learning_rate",
                        "typ": "float",
                    },
                    {
                        "default": 0.95,
                        "doc": "A `Tensor` or a floating point value. The decay "
                        "rate.",
                        "name": "rho",
                        "typ": "float",
                    },
                    {
                        "default": 1e-07,
                        "doc": "A `Tensor` or a floating point value.  A "
                        "constant epsilon used\n"
                        "         to better conditioning the grad update.",
                        "name": "epsilon",
                        "typ": "float",
                    },
                    {
                        "default": '"Adadelta"',
                        "doc": "Optional name prefix for the operations created "
                        "when applying\n"
                        "  gradients. ",
                        "name": "name",
                        "typ": "str",
                    },
                    {
                        "doc": "Keyword arguments. Allowed to be one of\n"
                        '  `"clipnorm"` or `"clipvalue"`.\n'
                        '  `"clipnorm"` (float) clips gradients by norm; '
                        '`"clipvalue"` (float) clips\n'
                        "  gradients by value.",
                        "name": "kwargs",
                        "typ": "dict",
                    },
                    {"default": True, "name": "_HAS_AGGREGATE_GRAD"},
                ],
                "returns": None,
            },
        )

    def test_from_class_and_function(self) -> None:
        """Tests that the parser can combine the outer class docstring + structure
        with the inner function parameter defaults"""

        # Sanity check
        run_ast_test(
            self,
            class_google_tf_tensorboard_ast,
            gold=ast.parse(class_google_tf_tensorboard_str).body[0],
        )

        parsed_ir = parse.class_(
            class_google_tf_tensorboard_ast,
            merge_inner_function="__init__",
            infer_type=True,
        )

        del parsed_ir["_internal"]  # Not needed for this test

        self.assertDictEqual(parsed_ir, class_google_tf_tensorboard_ir)

    def test_from_class_and_function_in_memory(self) -> None:
        """Tests that the parser can combine the outer class docstring + structure
        with the inner function parameter defaults"""

        parsed_ir = parse.class_(
            RewriteAtQuery,
            merge_inner_function="__init__",
            infer_type=True,
        )

        # del parsed_ir["_internal"]  # Not needed for this test

        self.assertDictEqual(
            parsed_ir,
            {
                "doc": "Replace the node at query with given node",
                "name": "RewriteAtQuery",
                "params": [
                    {
                        "doc": "Search query, e.g., ['node_name', "
                        "'function_name', 'arg_name']",
                        "name": "search",
                        "typ": "List[str]",
                    },
                    {
                        "doc": "Node to replace this search",
                        "name": "replacement_node",
                        "typ": "AST",
                    },
                    {
                        "doc": "whether a node has been replaced (only replaces "
                        "first occurrence)",
                        "name": "replaced",
                    },
                ],
                "returns": None,
            },
        )


unittest_main()
