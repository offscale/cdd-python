"""
Transform from string or AST representations of input, to AST, file, or str output.
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

from black import format_str, FileMode

from doctrans import parse
from doctrans.ast_utils import param2argparse_param, param2ast, set_value
from doctrans.defaults_utils import set_default_doc
from doctrans.emitter_utils import get_internal_body
from doctrans.pure_utils import tab, simple_types, PY_GTE_3_9, PY3_8
from doctrans.source_transformer import to_code


def argparse_function(
    intermediate_repr,
    emit_default_doc=False,
    emit_default_doc_in_return=False,
    function_name="set_cli_args",
    function_type=None,
):
    """
    Convert to an argparse function definition

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

    :param function_name: name of function
    :type function_name: ```str```

    :param function_type: None is a loose function (def f()`), others self-explanatory
    :type function_type: ```Optional[Literal['self', 'cls']]```

    :returns: function which constructs argparse
    :rtype: ```FunctionDef``
    """
    return FunctionDef(
        args=arguments(
            args=list(
                filter(
                    None,
                    (
                        function_type
                        if function_type is None
                        else arg(annotation=None, arg=function_type, type_comment=None),
                        arg(annotation=None, arg="argument_parser", type_comment=None),
                    ),
                )
            ),
            defaults=[],
            kw_defaults=[],
            kwarg=None,
            kwonlyargs=[],
            posonlyargs=[],
            vararg=None,
        ),
        body=list(
            filter(
                None,
                (
                    Expr(
                        value=set_value(
                            kind=None,
                            value="\n    Set CLI arguments\n\n    "
                            ":param argument_parser: argument parser\n    "
                            ":type argument_parser: ```ArgumentParser```\n\n    "
                            ":return: argument_parser, {returns[doc]}\n    "
                            ":rtype: ```Tuple[ArgumentParser,"
                            " {returns[typ]}]```\n    "
                            "".format(
                                returns=set_default_doc(
                                    intermediate_repr["returns"],
                                    emit_default_doc=emit_default_doc_in_return,
                                )
                            ),
                        )
                    )
                    if "returns" in intermediate_repr
                    else None,
                    Assign(
                        targets=[
                            Attribute(
                                attr="description",
                                ctx=Store(),
                                value=Name(ctx=Load(), id="argument_parser"),
                            )
                        ],
                        type_comment=None,
                        value=set_value(
                            kind=None,
                            value=intermediate_repr["long_description"]
                            or intermediate_repr["short_description"],
                        ),
                        lineno=None,
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
                    *get_internal_body(intermediate_repr),
                    Return(
                        value=Tuple(
                            ctx=Load(),
                            elts=[
                                Name(ctx=Load(), id="argument_parser"),
                                ast.parse(intermediate_repr["returns"]["default"])
                                .body[0]
                                .value,
                            ],
                        )
                    )
                    if "returns" in intermediate_repr
                    else None,
                ),
            )
        ),
        decorator_list=[],
        name=function_name,
        returns=None,
        type_comment=None,
        lineno=None,
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
    return ClassDef(
        bases=[Name(ctx=Load(), id=base_class) for base_class in class_bases],
        body=[
            Expr(
                value=set_value(
                    kind=None,
                    value="\n    {description}\n\n{cvars}".format(
                        description=intermediate_repr["long_description"]
                        or intermediate_repr["short_description"],
                        cvars="\n".join(
                            "{tab}:cvar {param[name]}: {param[doc]}".format(
                                tab=tab, param=set_default_doc(param)
                            )
                            for param in intermediate_repr["params"] + returns
                        ),
                    ),
                )
            )
        ]
        + list(map(param2ast, intermediate_repr["params"] + returns)),
        decorator_list=[],
        keywords=[],
        name=class_name,
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

    :returns: docstring
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


def file(node, filename, mode="a", skip_black=PY_GTE_3_9):
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
        node = Module(body=[node], type_ignores=[])
    src = to_code(node)
    if not skip_black:
        src = format_str(
            src,
            mode=FileMode(
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
):
    """
    Construct a function

    :param intermediate_repr: a dictionary of form
          {
              'short_description': ...,
              'long_description': ...,
              'params': [{'name': ..., 'typ': ..., 'doc': ..., 'default': ..., 'required': ... }, ...],
              "returns': {'name': ..., 'typ': ..., 'doc': ..., 'default': ..., 'required': ... }
          }
    :type intermediate_repr: ```dict```

    :param function_name: name of function
    :type function_name: ```str```

    :param function_type: None is a loose function (def f()`), others self-explanatory
    :type function_type: ```Optional[Literal['self', 'cls']]```

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

    :returns: function (could be a method on a class)
    :rtype: ```FunctionDef``
    """
    params_no_kwargs = tuple(
        filter(
            lambda param: not param["name"].endswith("kwargs"),
            intermediate_repr["params"],
        )
    )

    return FunctionDef(
        args=arguments(
            args=list(
                filter(
                    None,
                    (
                        function_type
                        if function_type is None
                        else arg(annotation=None, arg=function_type, type_comment=None),
                        *map(
                            lambda param: arg(
                                annotation=(
                                    Name(ctx=Load(), id=param["typ"])
                                    if param["typ"] in simple_types
                                    else ast.parse(param["typ"]).body[0].value
                                )
                                if inline_types and "typ" in param
                                else None,
                                arg=param["name"],
                                type_comment=None,
                            ),
                            params_no_kwargs,
                        ),
                    ),
                )
            ),
            defaults=list(
                map(
                    lambda param: set_value(kind=None, value=param["default"]),
                    filter(lambda param: "default" in param, params_no_kwargs),
                )
            )
            + [set_value(kind=None, value=None)],
            kw_defaults=[],
            kwarg=next(
                map(
                    lambda param: arg(
                        annotation=None, arg=param["name"], type_comment=None
                    ),
                    filter(
                        lambda param: param["name"].endswith("kwargs"),
                        intermediate_repr["params"],
                    ),
                )
            ),
            kwonlyargs=[],
            posonlyargs=[],
            vararg=None,
        ),
        body=list(
            filter(
                None,
                (
                    Expr(
                        value=set_value(
                            kind=None,
                            value=parse.to_docstring(
                                intermediate_repr,
                                emit_default_doc=emit_default_doc,
                                docstring_format=docstring_format,
                                emit_types=not inline_types,
                                indent_level=indent_level,
                                emit_separating_tab=emit_separating_tab,
                            ),
                        )
                    ),
                    *(get_internal_body(intermediate_repr)),
                    Return(
                        value=ast.parse(intermediate_repr["returns"]["default"])
                        .body[0]
                        .value
                    )
                    if "returns" in intermediate_repr
                    and intermediate_repr["returns"].get("default")
                    else None,
                ),
            )
        ),
        decorator_list=[],
        name=function_name,
        returns=(
            ast.parse(intermediate_repr["returns"]["typ"]).body[0].value
            if "returns" in intermediate_repr and "typ" in intermediate_repr["returns"]
            else None
        )
        if inline_types
        else None,
        type_comment=None,
        lineno=None,
    )


__all__ = ["argparse_function", "class_", "docstring", "file", "function"]
