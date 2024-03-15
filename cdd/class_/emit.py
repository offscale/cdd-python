"""
`class` emitter
"""

import ast
from ast import ClassDef, Constant, Expr, FunctionDef, Load, Name
from collections import OrderedDict
from functools import partial
from itertools import chain
from typing import Optional

import cdd.shared.ast_utils
from cdd.class_.utils.emit_utils import RewriteName
from cdd.docstring.emit import docstring
from cdd.function.utils.emit_utils import make_call_meth
from cdd.shared.pure_utils import PY_GTE_3_8, PY_GTE_3_9, rpartial

if PY_GTE_3_9:
    FrozenSet = frozenset
else:
    from typing import FrozenSet


def class_(
    intermediate_repr,
    emit_call=False,
    class_name=None,
    class_bases=("object",),
    decorator_list=None,
    word_wrap=True,
    docstring_format="rest",
    emit_original_whitespace=False,
    emit_default_doc=False,
):
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
    :type intermediate_repr: ```dict```

    :param emit_call: Whether to emit a `__call__` method from the `_internal` IR subdict
    :type emit_call: ```bool```

    :param class_name: name of class
    :type class_name: ```str```

    :param class_bases: bases of class (the generated class will inherit these)
    :type class_bases: ```Iterable[str]```

    :param decorator_list: List of decorators
    :type decorator_list: ```Optional[List[str]]```

    :param word_wrap: Whether to word-wrap. Set `DOCTRANS_LINE_LENGTH` to configure length.
    :type word_wrap: ```bool```

    :param docstring_format: Format of docstring
    :type docstring_format: ```Literal['rest', 'numpydoc', 'google']```

    :param emit_original_whitespace: Whether to emit original whitespace or strip it out (in docstring)
    :type emit_original_whitespace: ```bool```

    :param emit_default_doc: Whether help/docstring should include 'With default' text
    :type emit_default_doc: ```bool```

    :return: Class AST
    :rtype: ```ClassDef```
    """
    assert isinstance(
        intermediate_repr, dict
    ), "Expected `dict` got `{type_name}`".format(
        type_name=type(intermediate_repr).__name__
    )
    assert class_name or intermediate_repr["name"], "Class has no name"

    returns: OrderedDict = (
        intermediate_repr["returns"]
        if "return_type" in ((intermediate_repr or {}).get("returns") or iter(()))
        else OrderedDict()
    )
    if returns:
        intermediate_repr["params"].update(returns)
        del intermediate_repr["returns"]

    internal_body: ClassDef.body = intermediate_repr.get("_internal", {}).get(
        "body", []
    )
    # TODO: Add correct classmethod/staticmethod to decorate function using `annotate_ancestry` and first-field checks
    # Such that the `self.` or `cls.` rewrite only applies to non-staticmethods
    # assert internal_body, "Expected `internal_body` to have contents"
    param_names: Optional[FrozenSet[str]] = (
        frozenset(intermediate_repr["params"].keys())
        if "params" in intermediate_repr
        else None
    )
    if param_names:
        if internal_body:
            internal_body: ClassDef.body = list(
                map(
                    ast.fix_missing_locations,
                    map(RewriteName(param_names).visit, internal_body),
                )
            )
        elif (returns or {"return_type": None}).get("return_type") is not None:
            internal_body = returns["return_type"]

    indent_level: int = 1

    _emit_docstring = partial(
        docstring,
        docstring_format=docstring_format,
        indent_level=indent_level,
        emit_default_doc=emit_default_doc,
        emit_separating_tab=True,
        emit_types=False,
        word_wrap=word_wrap,
    )
    return ClassDef(
        bases=list(
            map(
                rpartial(partial(Name, lineno=None, col_offset=None), Load()),
                class_bases,
            )
        ),
        body=list(
            filter(
                None,
                chain.from_iterable(
                    (
                        (
                            (
                                lambda ds: (
                                    None
                                    if ds is None
                                    else Expr(
                                        cdd.shared.ast_utils.set_value(ds),
                                        lineno=None,
                                        col_offset=None,
                                    )
                                )
                            )(
                                _emit_docstring(
                                    {
                                        k: intermediate_repr[k]
                                        for k in intermediate_repr
                                        if k != "_internal"
                                    },
                                    emit_original_whitespace=emit_original_whitespace,
                                    purpose="class",
                                ).rstrip()
                                or None
                            ),
                        ),
                        map(
                            cdd.shared.ast_utils.param2ast,
                            (intermediate_repr.get("params") or OrderedDict()).items(),
                        ),
                        iter(
                            (
                                (
                                    (
                                        internal_body[0]
                                        if len(internal_body) == 1
                                        and isinstance(internal_body[0], FunctionDef)
                                        and internal_body[0].name == "__call__"
                                        else make_call_meth(
                                            internal_body,
                                            (
                                                returns["return_type"]["default"]
                                                if "default"
                                                in (
                                                    (
                                                        returns
                                                        or {"return_type": iter(())}
                                                    ).get("return_type")
                                                    or iter(())
                                                )
                                                else None
                                            ),
                                            param_names,
                                            docstring_format=docstring_format,
                                            word_wrap=word_wrap,
                                        )
                                    ),
                                )
                                or iter(())
                            )
                            if emit_call and internal_body
                            else iter(())
                        ),
                    ),
                ),
            )
        )
        or [
            Expr(
                Constant(Ellipsis) if PY_GTE_3_8 else Ellipsis,
                lineno=None,
                col_offset=None,
            )
        ],  # empty body will cause syntax error
        decorator_list=(
            list(
                map(
                    rpartial(partial(Name, lineno=None, col_offset=None), Load()),
                    decorator_list,
                )
            )
            if decorator_list
            else []
        ),
        type_params=[],
        keywords=[],
        name=class_name or intermediate_repr["name"],
        expr=None,
        identifier_name=None,
        lineno=None,
        col_offset=None,
    )


__all__ = ["class_"]  # type: list[str]
