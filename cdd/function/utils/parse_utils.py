"""
Utility functions for `cdd.parse.function`
"""

from ast import Constant, Return, Tuple
from collections import OrderedDict
from typing import Optional

import cdd.shared.ast_utils
import cdd.shared.source_transformer
from cdd.shared.pure_utils import PY_GTE_3_8, rpartial

if PY_GTE_3_8:
    from cdd.shared.pure_utils import FakeConstant as Str

    Num = Str
else:
    from ast import Num, Str


def _interpolate_return(function_def, intermediate_repr):
    """
    Interpolate the return value into the IR.

    :param function_def: function definition
    :type function_def: ```FunctionDef```

    :param intermediate_repr: a dictionary consistent with `IntermediateRepr`, defined as:
        ParamVal = TypedDict("ParamVal", {"typ": str, "doc": Optional[str], "default": Any})
        IntermediateRepr = TypedDict("IntermediateRepr", {
            "name": Optional[str],
            "type": Optional[str],
            "doc": Optional[str],
            "params": OrderedDict[str, ParamVal],
            "returns": Optional[OrderedDict[Literal["return_type"], ParamVal]],
        })
    :type intermediate_repr: ```dict```

    :return: a dictionary consistent with `IntermediateRepr`, defined as:
        ParamVal = TypedDict("ParamVal", {"typ": str, "doc": Optional[str], "default": Any})
        IntermediateRepr = TypedDict("IntermediateRepr", {
            "name": Optional[str],
            "type": Optional[str],
            "doc": Optional[str],
            "params": OrderedDict[str, ParamVal],
            "returns": Optional[OrderedDict[Literal["return_type"], ParamVal]],
        })
    :rtype: ```dict```
    """
    return_ast: Optional[Return] = next(
        filter(rpartial(isinstance, Return), function_def.body[::-1]), None
    )
    if return_ast is not None and return_ast.value is not None:
        if intermediate_repr.get("returns") is None:
            intermediate_repr["returns"] = OrderedDict((("return_type", {}),))

        if (
            "typ" in intermediate_repr["returns"]["return_type"]
            and "[" not in intermediate_repr["returns"]["return_type"]["typ"]
        ):
            del intermediate_repr["returns"]["return_type"]["typ"]
        intermediate_repr["returns"]["return_type"]["default"] = (
            lambda default: (
                "({})".format(default)
                if isinstance(return_ast.value, Tuple)
                and (not default.startswith("(") or not default.endswith(")"))
                else (
                    lambda default_: (
                        default_
                        if isinstance(
                            default_,
                            (str, int, float, complex, Num, Str, Constant),
                        )
                        else "```{}```".format(default)
                    )
                )(
                    cdd.shared.ast_utils.get_value(
                        cdd.shared.ast_utils.get_value(return_ast)
                    )
                )
            )
        )(cdd.shared.source_transformer.to_code(return_ast.value).rstrip("\n"))
    if hasattr(function_def, "returns") and function_def.returns is not None:
        intermediate_repr["returns"] = intermediate_repr.get("returns") or OrderedDict(
            (("return_type", {}),)
        )
        intermediate_repr["returns"]["return_type"]["typ"] = (
            cdd.shared.source_transformer.to_code(function_def.returns).rstrip("\n")
        )

    return intermediate_repr


__all__ = ["_interpolate_return"]  # type: list[str]
