"""
Shared utility functions for `cdd.class_`
"""

from ast import ClassDef, Module
from typing import List, Optional, Tuple, Union

from cdd.shared.pure_utils import PY_GTE_3_8
from cdd.shared.types import IntermediateRepr

if PY_GTE_3_8:
    from typing import Literal, Protocol
else:
    from typing_extensions import Literal, Protocol


class ClassEmitProtocol(Protocol):
    """
    Protocol for class emitter
    """

    def __call__(
        self,
        intermediate_repr: IntermediateRepr,
        emit_call: bool = False,
        class_name: Optional[str] = None,
        class_bases: Tuple[str] = ("object",),
        decorator_list: Optional[List[str]] = None,
        word_wrap: bool = True,
        docstring_format: Literal["rest", "numpydoc", "google"] = "rest",
        emit_original_whitespace: bool = False,
        emit_default_doc: bool = False,
    ) -> ClassDef:
        """
        Construct a class

        :param intermediate_repr: a dictionary consistent with `IntermediateRepr`, defined as:
            ParamVal = TypedDict("ParamVal", {"typ": str, "doc": Optional[str], "default": Any})
            IntermediateRepr = TypedDict("IntermediateRepr", {
                "name": Optional[str],
                "type": Optional[str],
                "doc": Optional[str],
                "params": OrderedDict[str, ParamVal],
                "returns": Optional[OrderedDict[Literal["return_type"], ParamVal]],
            })

        :param emit_call: Whether to emit a `__call__` method from the `_internal` IR subdict

        :param class_name: name of class

        :param class_bases: bases of class (the generated class will inherit these)

        :param decorator_list: List of decorators

        :param word_wrap: Whether to word-wrap. Set `DOCTRANS_LINE_LENGTH` to configure length.

        :param docstring_format: Format of docstring

        :param emit_original_whitespace: Whether to emit original whitespace or strip it out (in docstring)

        :param emit_default_doc: Whether help/docstring should include 'With default' text

        :return: Class AST
        """


class ClassParserProtocol(Protocol):
    """
    Protocol for class parser
    """

    def __call__(
        self,
        class_def: Union[Module, ClassDef],
        class_name: Optional[str] = None,
        merge_inner_function: Optional[str] = None,
        infer_type: bool = False,
        parse_original_whitespace: bool = False,
        word_wrap: bool = True,
    ) -> IntermediateRepr:
        """
        Converts an AST to our IR

        :param class_def: Class AST or Module AST with a ClassDef inside

        :param class_name: Name of `class`. If None, gives first found.

        :param merge_inner_function: Name of inner function to merge. If None, merge nothing.

        :param infer_type: Whether to try inferring the typ (from the default)

        :param parse_original_whitespace: Whether to parse original whitespace or strip it out

        :param word_wrap: Whether to word-wrap. Set `DOCTRANS_LINE_LENGTH` to configure length.

        :return: a dictionary consistent with `IntermediateRepr`, defined as:
            ParamVal = TypedDict("ParamVal", {"typ": str, "doc": Optional[str], "default": Any})
            IntermediateRepr = TypedDict("IntermediateRepr", {
                "name": Optional[str],
                "type": Optional[str],
                "doc": Optional[str],
                "params": OrderedDict[str, ParamVal],
                "returns": Optional[OrderedDict[Literal["return_type"], ParamVal]],
            })
        """


__all__ = ["ClassEmitProtocol", "ClassParserProtocol"]  # type: list[str]
