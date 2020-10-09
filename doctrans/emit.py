"""
Transform from string or AST representations of input, to AST, file, or str input_str.
"""
import ast
from ast import (
    ClassDef,
    Name,
    Load,
    Expr,
    Module,
    FunctionDef,
    arguments,
    Assign,
    Attribute,
    Store,
    Tuple,
    Return,
    arg,
)
from functools import partial

from black import format_str, Mode

from doctrans.ast_utils import param2argparse_param, param2ast, set_value
from doctrans.defaults_utils import set_default_doc
from doctrans.emitter_utils import get_internal_body, to_docstring
from doctrans.pure_utils import tab, simple_types, PY3_8
from doctrans.source_transformer import to_code


def argparse_function(
    intermediate_repr,
    emit_default_doc=False,
    emit_default_doc_in_return=False,
    function_name="set_cli_args",
    function_type="static",
):
    """
    Convert to an argparse FunctionDef

    :param intermediate_repr: a dictionary of form
          {
              'short_description': ...,
              'long_description': ...,
              'params': [{'name': ..., 'typ': ..., 'doc': ..., 'default': ..., 'required': ... }, ...],
              "returns': {'name': ..., 'typ': ..., 'doc': ..., 'default': ..., 'required': ... }
          }
    :type intermediate_repr: ```dict```

    :param emit_default_doc: Whether help/docstring should include 'With default' text
    :type emit_default_doc: ```bool``

    :param emit_default_doc_in_return: Whether help/docstring in return should include 'With default' text
    :type emit_default_doc_in_return: ```bool```

    :param function_name: name of function_def
    :type function_name: ```str```

    :param function_type: Type of function, static is static or global method, others just become first arg
    :type function_type: ```Literal['self', 'cls', 'static']```

    :return:  AST node for function definition which constructs argparse
    :rtype: ```FunctionDef``
    """
    function_name = function_name or intermediate_repr["name"]
    function_type = function_type or intermediate_repr["type"]
    return FunctionDef(
        args=arguments(
            args=list(
                filter(
                    None,
                    (
                        # None
                        # if function_type in frozenset((None, "static"))
                        # else arg(
                        #     annotation=None,
                        #     arg=function_type,
                        #     type_comment=None,
                        #     expr=None,
                        #     identifier_arg=None,
                        # ),
                        arg(
                            annotation=None,
                            arg="argument_parser",
                            type_comment=None,
                            expr=None,
                            identifier_arg=None,
                        ),
                    ),
                )
            ),
            defaults=[],
            kw_defaults=[],
            kwarg=None,
            kwonlyargs=[],
            posonlyargs=[],
            vararg=None,
            arg=None,
        ),
        body=list(
            filter(
                None,
                (
                    Expr(
                        set_value(
                            kind=None,
                            value="\n    Set CLI arguments\n\n    "
                            ":param argument_parser: argument parser\n    "
                            ":type argument_parser: ```ArgumentParser```\n\n    "
                            "{return_params}".format(
                                return_params=":return: argument_parser, {returns[doc]}\n    "
                                ":rtype: ```Tuple[ArgumentParser, {returns[typ]}]```\n    ".format(
                                    returns=set_default_doc(
                                        intermediate_repr["returns"],
                                        emit_default_doc=emit_default_doc_in_return,
                                    )
                                )
                                if intermediate_repr.get("returns")
                                else (
                                    ":return: argument_parser\n    "
                                    ":rtype: ```ArgumentParser```\n    "
                                )
                            ),
                        )
                    ),
                    Assign(
                        targets=[
                            Attribute(
                                Name("argument_parser", Load()),
                                "description",
                                Store(),
                            )
                        ],
                        type_comment=None,
                        value=set_value(
                            kind=None,
                            value=intermediate_repr["long_description"]
                            or intermediate_repr["short_description"],
                        ),
                        lineno=None,
                        expr=None,
                    ),
                    *(
                        list(
                            map(
                                partial(
                                    param2argparse_param,
                                    emit_default_doc=emit_default_doc,
                                ),
                                intermediate_repr["params"],
                            )
                        )
                        if "params" in intermediate_repr
                        else tuple()
                    ),
                    *get_internal_body(
                        target_name=function_name,
                        target_type=function_type,
                        intermediate_repr=intermediate_repr,
                    ),
                    Return(
                        value=Tuple(
                            ctx=Load(),
                            elts=[
                                Name("argument_parser", Load()),
                                ast.parse(intermediate_repr["returns"]["default"])
                                .body[0]
                                .value,
                            ],
                            expr=None,
                        ),
                        expr=None,
                    )
                    if intermediate_repr.get("returns")
                    else None,
                ),
            )
        ),
        decorator_list=[],
        name=function_name,
        returns=None,
        type_comment=None,
        lineno=None,
        arguments_args=None,
        identifier_name=None,
        stmt=None,
    )


def class_(intermediate_repr, class_name="ConfigClass", class_bases=("object",)):
    """
    Construct a class

    :param intermediate_repr: a dictionary of form
          {
              'short_description': ...,
              'long_description': ...,
              'params': [{'name': ..., 'typ': ..., 'doc': ..., 'default': ..., 'required': ... }, ...],
              "returns': {'name': ..., 'typ': ..., 'doc': ..., 'default': ..., 'required': ... }
          }
    :type intermediate_repr: ```dict```

    :param class_name: name of class
    :type class_name: ```str```

    :param class_bases: bases of class (the generated class will inherit these)
    :type class_bases: ```Iterable[str]```

    :return: Class AST of the docstring
    :rtype: ```ClassDef```
    """
    returns = [intermediate_repr["returns"]] if intermediate_repr.get("returns") else []

    intermediate_repr["params"] = intermediate_repr["params"] + returns
    del intermediate_repr["returns"]

    return ClassDef(
        bases=[Name(base_class, Load()) for base_class in class_bases],
        body=[
            Expr(
                set_value(
                    kind=None,
                    value=to_docstring(
                        intermediate_repr, indent_level=0, emit_separating_tab=False
                    )
                    .replace("\n:param ", "{tab}:cvar ".format(tab=tab))
                    .replace(
                        "{tab}:cvar ".format(tab=tab),
                        "\n{tab}:cvar ".format(tab=tab),
                        1,
                    )
                    .rstrip(),
                )
            )
        ]
        + list(map(param2ast, intermediate_repr["params"])),
        decorator_list=[],
        keywords=[],
        name=class_name,
        expr=None,
        identifier_name=None,
    )


def docstring(intermediate_repr, docstring_format="rest", emit_default_doc=True):
    """
    Converts an AST to a docstring

    :param intermediate_repr: a dictionary of form
          {
              'short_description': ...,
              'long_description': ...,
              'params': [{'name': ..., 'typ': ..., 'doc': ..., 'default': ..., 'required': ... }, ...],
              "returns': {'name': ..., 'typ': ..., 'doc': ..., 'default': ..., 'required': ... }
          }
    :type intermediate_repr: ```dict```

    :param docstring_format: Format of docstring
    :type docstring_format: ```Literal['rest', 'numpy', 'google']```

    :param emit_default_doc: Whether help/docstring should include 'With default' text
    :type emit_default_doc: ```bool``

    :return: docstring
    :rtype: ```str``
    """
    if docstring_format != "rest":
        raise NotImplementedError()

    return """\n{description}\n\n{params}\n{returns}\n""".format(
        description=intermediate_repr["long_description"]
        or intermediate_repr["short_description"],
        params="\n".join(
            ":param {param[name]}: {param[doc]}\n"
            ":type {param[name]}: ```{typ}```\n".format(
                param=set_default_doc(param, emit_default_doc=emit_default_doc),
                typ=(
                    "**{name}".format(name=param["name"])
                    if "kwargs" in param["name"]
                    else param["typ"]
                ),
            )
            for param in intermediate_repr["params"]
        ),
        returns=":return: {param[doc]}\n"
        ":rtype: ```{param[typ]}```".format(
            param=set_default_doc(
                intermediate_repr["returns"], emit_default_doc=emit_default_doc
            )
        ),
    )


def file(node, filename, mode="a", skip_black=False):
    """
    Convert AST to a file

    :param node: AST node
    :type node: ```Union[Module, ClassDef, FunctionDef]```

    :param filename: emit to this file
    :type filename: ```str```

    :param mode: Mode to open the file in, defaults to append
    :type mode: ```str```

    :param skip_black: Skip formatting with black
    :type skip_black: ```bool```

    :return: None
    :rtype: ```NoneType```
    """
    if isinstance(node, (ClassDef, FunctionDef)):
        node = Module(body=[node], type_ignores=[], stmt=None)
    src = to_code(node)
    if not skip_black:
        src = format_str(
            src,
            mode=Mode(
                target_versions=set(),
                line_length=119,
                is_pyi=False,
                string_normalization=False,
            ),
        )
    with open(filename, mode) as f:
        f.write(src)


def function(
    intermediate_repr,
    function_name,
    function_type,
    emit_default_doc=False,
    docstring_format="rest",
    indent_level=2,
    emit_separating_tab=PY3_8,
    inline_types=True,
    emit_as_kwonlyargs=True,
):
    """
    Construct a function from our IR

    :param intermediate_repr: a dictionary of form
          {
              'short_description': ...,
              'long_description': ...,
              'params': [{'name': ..., 'typ': ..., 'doc': ..., 'default': ..., 'required': ... }, ...],
              "returns': {'name': ..., 'typ': ..., 'doc': ..., 'default': ..., 'required': ... }
          }
    :type intermediate_repr: ```dict```

    :param function_name: name of function_def
    :type function_name: ```Optional[str]```

    :param function_type: Type of function, static is static or global method, others just become first arg
    :type function_type: ```Optional[Literal['self', 'cls', 'static']]```

    :param docstring_format: Format of docstring
    :type docstring_format: ```Literal['rest', 'numpy', 'google']```

    :param emit_default_doc: Whether help/docstring should include 'With default' text
    :type emit_default_doc: ```bool``

    :param indent_level: docstring indentation level whence: 0=no_tabs, 1=one tab; 2=two tabs
    :type indent_level: ```int```

    :param emit_separating_tab: docstring decider for whether to put a tab between :param and return and desc
    :type emit_separating_tab: ```bool```

    :param inline_types: Whether the type should be inline or in docstring
    :type inline_types: ```bool```

    :param emit_as_kwonlyargs: Whether argument(s) emitted must be keyword only
    :type emit_as_kwonlyargs: ```bool```

    :return: AST node for function definition
    :rtype: ```FunctionDef``
    """
    params_no_kwargs = tuple(
        filter(
            lambda param: not param["name"].endswith("kwargs"),
            intermediate_repr["params"],
        )
    )

    function_name = function_name or intermediate_repr["name"]
    function_type = function_type or intermediate_repr["type"]

    args = (
        []
        if function_type in frozenset((None, "static"))
        else [
            arg(
                annotation=None,
                arg=function_type,
                type_comment=None,
                expr=None,
                identifier_arg=None,
            )
        ]
    )
    args_from_params = list(
        map(
            lambda param: arg(
                annotation=(
                    Name(param["typ"], Load())
                    if param["typ"] in simple_types
                    else ast.parse(param["typ"]).body[0].value
                )
                if inline_types and "typ" in param
                else None,
                arg=param["name"],
                type_comment=None,
                expr=None,
                identifier_arg=None,
            ),
            params_no_kwargs,
        ),
    )
    defaults_from_params = list(
        map(
            lambda param: set_value(kind=None, value=param.get("default")),
            params_no_kwargs,
        )
    )
    if emit_as_kwonlyargs:
        kwonlyargs, kw_defaults, defaults = args_from_params, defaults_from_params, []
    else:
        kwonlyargs, kw_defaults, defaults = [], [], defaults_from_params
        args += args_from_params

    return FunctionDef(
        args=arguments(
            args=args,
            defaults=defaults,
            kw_defaults=kw_defaults,
            kwarg=next(
                map(
                    lambda param: arg(
                        annotation=None,
                        arg=param["name"],
                        type_comment=None,
                        expr=None,
                        identifier_arg=None,
                    ),
                    filter(
                        lambda param: param["name"].endswith("kwargs"),
                        intermediate_repr["params"],
                    ),
                ),
                None,
            ),
            kwonlyargs=kwonlyargs,
            posonlyargs=[],
            vararg=None,
            arg=None,
        ),
        body=list(
            filter(
                None,
                (
                    Expr(
                        set_value(
                            kind=None,
                            value=to_docstring(
                                intermediate_repr,
                                emit_default_doc=emit_default_doc,
                                docstring_format=docstring_format,
                                emit_types=not inline_types,
                                indent_level=indent_level,
                                emit_separating_tab=emit_separating_tab,
                            ),
                        )
                    ),
                    *(
                        get_internal_body(
                            target_name=function_name,
                            target_type=function_type,
                            intermediate_repr=intermediate_repr,
                        )
                    ),
                    Return(
                        value=ast.parse(intermediate_repr["returns"]["default"])
                        .body[0]
                        .value,
                        expr=None,
                    )
                    if (intermediate_repr.get("returns") or {}).get("default")
                    else None,
                ),
            )
        ),
        decorator_list=[],
        name=function_name,
        returns=(
            ast.parse(intermediate_repr["returns"]["typ"]).body[0].value
            if inline_types and (intermediate_repr.get("returns") or {}).get("typ")
            else None
        ),
        type_comment=None,
        lineno=None,
        arguments_args=None,
        identifier_name=None,
        stmt=None,
    )


__all__ = ["argparse_function", "class_", "docstring", "file", "function"]
