"""
Utility functions for `cdd.emit.argparse_function`
"""

import ast
from ast import Name, Return
from typing import Any, Callable, Dict, Optional, Union, cast

from cdd.shared.ast_utils import NoneStr, get_value, set_value
from cdd.shared.defaults_utils import extract_default, set_default_doc
from cdd.shared.pure_utils import identity, simple_types
from cdd.shared.source_transformer import to_code


def _parse_return(e, intermediate_repr, function_def, emit_default_doc):
    """
    Parse return into a param dict

    :param e: Return AST node
    :type e: Return

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

    :param function_def: AST node for function definition
    :type function_def: ```FunctionDef```

    :param emit_default_doc: Whether help/docstring should include 'With default' text
    :type emit_default_doc: ```bool```

    :return: Name, dict with keys: 'typ', 'doc', 'default'
    :rtype: ```tuple[str, dict]```
    """
    assert isinstance(e, Return), "Expected `Return` got `{type_name}`".format(
        type_name=type(e).__name__
    )

    typ: str = intermediate_repr["returns"]["return_type"]["typ"]
    if "[" in intermediate_repr["returns"]["return_type"]["typ"]:
        typ: str = to_code(
            get_value(ast.parse(typ).body[0].value.slice).elts[1]
        ).rstrip("\n")

    return set_default_doc(
        (
            "return_type",
            {
                "doc": extract_default(
                    next(
                        line.partition(",")[2].lstrip()
                        for line in get_value(function_def.body[0].value).split("\n")
                        if line.lstrip().startswith(":return")
                    ),
                    emit_default_doc=emit_default_doc,
                )[0],
                "default": to_code(e.value.elts[1]).rstrip("\n"),
                "typ": typ,
                # 'Tuple[ArgumentParser, {typ}]'.format(typ=intermediate_repr['returns']['typ'])
            },
        ),
        emit_default_doc=emit_default_doc,
    )


def parse_out_param(expr, require_default=False, emit_default_doc=True):
    """
    Turns the class_def repr of '--dataset_name', type=str, help='name of dataset.', required=True, default='mnist'
      into
           Tuple[Literal['dataset_name'], {"typ": Literal["str"], "doc": Literal["name of dataset."],
                                           "default": Literal["mnist"]}]

    :param expr: Expr
    :type expr: ```Expr```

    :param require_default: Whether a default is required, if not found in doc, infer the proper default from type
    :type require_default: ```bool```

    :param emit_default_doc: Whether help/docstring should include 'With default' text
    :type emit_default_doc: ```bool```

    :return: Name, dict with keys: 'typ', 'doc', 'default'
    :rtype: ```tuple[str, dict]```
    """
    required: bool = get_value(
        get_value(
            next(
                (
                    keyword
                    for keyword in expr.value.keywords
                    if keyword.arg == "required"
                ),
                set_value(False),
            )
        )
    )

    typ: str = next(
        (
            _handle_value(get_value(key_word))
            for key_word in expr.value.keywords
            if key_word.arg == "type"
        ),
        "str",
    )
    name: str = get_value(expr.value.args[0])[len("--") :]
    default: Optional[Any] = next(
        (
            get_value(key_word.value)
            for key_word in expr.value.keywords
            if key_word.arg == "default"
        ),
        None,
    )
    doc: Optional[str] = (
        lambda help_: (
            help_
            if help_ is None
            else (
                help_
                if default is None
                or emit_default_doc is False
                or (hasattr(default, "__len__") and len(default) == 0)
                or "defaults to" in help_
                or "Defaults to" in help_
                else "{help} Defaults to {default}".format(
                    help=help_ if help_.endswith(".") else "{}.".format(help_),
                    default=default,
                )
            )
        )
    )(
        next(
            (
                get_value(key_word.value)
                for key_word in expr.value.keywords
                if key_word.arg == "help" and key_word.value
            ),
            None,
        )
    )
    if default is None:
        doc, default = extract_default(doc, emit_default_doc=emit_default_doc)
    if default is None:
        if required:
            # if name.endswith("kwargs"):
            #    default = NoneStr
            # else:
            default: Optional[
                Dict[Optional[str], Union[int, float, complex, str, bool, None]]
            ] = (simple_types[typ] if typ in simple_types else NoneStr)

        elif require_default:  # or typ.startswith("Optional"):
            default: Optional[
                Dict[Optional[str], Union[int, float, complex, str, bool, None]]
            ] = NoneStr

    action: Optional[Any] = next(
        (
            get_value(key_word.value)
            for key_word in expr.value.keywords
            if key_word.arg == "action"
        ),
        None,
    )

    typ: Optional[Any] = next(
        (
            _handle_keyword(keyword, typ)
            for keyword in expr.value.keywords
            if keyword.arg == "choices"
        ),
        typ,
    )
    if action == "append":
        typ: str = "List[{typ}]".format(typ=typ)

    if not required and "Optional" not in typ:
        typ: str = "Optional[{typ}]".format(typ=typ)

    return name, dict(
        doc=doc, typ=typ, **({} if default is None else {"default": default})
    )


def _handle_keyword(keyword, typ):
    """
    Decide which type to wrap the keyword tuples in

    :param keyword: AST keyword
    :type keyword: ```ast.keyword```

    :param typ: string representation of type
    :type typ: ```str```

    :return: string representation of type
    :rtype: ```str```
    """
    quote_f: Callable[[str], str] = cast(Callable[[str], str], identity)

    type_: str = "Union"
    if typ == Any or typ in simple_types:
        if typ in ("str", Any):

            def quote_f(s):
                """
                Wrap the input in quotes

                :param s: Any value
                :type s: ```Any```

                :return: the input value
                :rtype: ```Any```
                """
                return "'{}'".format(s)

        type_: str = "Literal"

    return "{type}[{types}]".format(
        type=type_,
        types=", ".join(quote_f(get_value(elt)) for elt in keyword.value.elts),
    )


def _handle_value(node):
    """
    Handle `keyword.value` types, returning the correct one as a `str` or `Any`

    :param node: AST node from `keyword.value`
    :type node: ```Name```

    :return: `str` or `Any`, representing the type for argparse
    :rtype: ```Union[str, Any]```
    """
    # if isinstance(node, Attribute): return Any
    if isinstance(node, Name):
        return "Optional[dict]" if node.id == "loads" else node.id
    raise NotImplementedError(type(node).__name__)


__all__ = ["_parse_return", "parse_out_param", "_handle_keyword", "_handle_value"]
