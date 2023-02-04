"""
Utility functions for `cdd.parse.function`
"""

import ast
from ast import Return, Tuple
from collections import OrderedDict

from cdd.shared.ast_utils import get_value
from cdd.shared.pure_utils import rpartial
from cdd.shared.source_transformer import to_code


def _interpolate_return(function_def, intermediate_repr):
    """
    Interpolate the return value into the IR.

    :param function_def: function definition
    :type function_def: ```FunctionDef```

    :param intermediate_repr: a dictionary of form
        {  "name": Optional[str],
           "type": Optional[str],
           "doc": Optional[str],
           "params": OrderedDict[str, {'typ': str, 'doc': Optional[str], 'default': Any}]
           "returns": Optional[OrderedDict[Literal['return_type'],
                                           {'typ': str, 'doc': Optional[str], 'default': Any}),)]] }
    :type intermediate_repr: ```dict```

    :return: a dictionary of form
        {  "name": Optional[str],
           "type": Optional[str],
           "doc": Optional[str],
           "params": OrderedDict[str, {'typ': str, 'doc': Optional[str], 'default': Any}]
           "returns": Optional[OrderedDict[Literal['return_type'],
                                           {'typ': str, 'doc': Optional[str], 'default': Any}),)]] }
    :rtype: ```dict```
    """
    return_ast = next(
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
            lambda default: "({})".format(default)
            if isinstance(return_ast.value, Tuple)
            and (not default.startswith("(") or not default.endswith(")"))
            else (
                lambda default_: default_
                if isinstance(
                    default_, (str, int, float, complex, ast.Num, ast.Str, ast.Constant)
                )
                else "```{}```".format(default)
            )(get_value(get_value(return_ast)))
        )(to_code(return_ast.value).rstrip("\n"))
    if hasattr(function_def, "returns") and function_def.returns is not None:
        intermediate_repr["returns"] = intermediate_repr.get("returns") or OrderedDict(
            (("return_type", {}),)
        )
        intermediate_repr["returns"]["return_type"]["typ"] = to_code(
            function_def.returns
        ).rstrip("\n")

    return intermediate_repr


__all__ = ["_interpolate_return"]
