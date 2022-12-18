"""
Utility functions for `cdd.emit.sqlalchemy`
"""

import ast
from ast import AST, Attribute, Call, Expr, FunctionDef, Load, Name, Return, arguments
from collections import OrderedDict, deque
from operator import attrgetter, methodcaller
from platform import system

from cdd.ast_utils import (
    get_value,
    maybe_type_comment,
    set_arg,
    set_value,
    typ2column_type,
)
from cdd.pure_utils import none_types, rpartial, tab
from cdd.tests.mocks.docstrings import docstring_repr_google_str, docstring_repr_str


def param_to_sqlalchemy_column_call(name_param, include_name):
    """
    Turn a param into a `Column(…)`

    :param name_param: Name, dict with keys: 'typ', 'doc', 'default'
    :type name_param: ```Tuple[str, dict]```

    :param include_name: Whether to include the name (exclude in declarative base)
    :type include_name: ```bool```

    :return: Form of: `Column(…)`
    :rtype: ```Call```
    """
    if system() == "Darwin":
        print("param_to_sqlalchemy_column_call::include_name:", include_name, ";")
    name, _param = name_param
    del name_param

    args, keywords, nullable = [], [], None

    if _param["typ"].startswith("Optional["):
        _param["typ"] = _param["typ"][len("Optional[") : -1]
        nullable = True

    if include_name:
        args.append(set_value(name))

    x_typ_sql = _param.get("x_typ", {}).get("sql", {})

    if "Literal[" in _param["typ"]:
        parsed_typ = get_value(ast.parse(_param["typ"]).body[0])
        assert (
            parsed_typ.value.id == "Literal"
        ), "Only basic Literal support is implemented, not {}".format(
            parsed_typ.value.id
        )
        args.append(
            Call(
                func=Name("Enum", Load()),
                args=get_value(parsed_typ.slice).elts,
                keywords=[
                    ast.keyword(arg="name", value=set_value(name), identifier=None)
                ],
                expr=None,
                expr_func=None,
            )
        )
    else:
        args.append(
            Name(
                x_typ_sql["type"]
                if "type" in x_typ_sql
                else typ2column_type[_param["typ"]],
                Load(),
            )
        )

    default = x_typ_sql.get("default", _param.get("default", ast))
    has_default = default is not ast
    pk = _param.get("doc", "").startswith("[PK]")
    if pk:
        _param["doc"] = _param["doc"][4:].lstrip()
        keywords.append(
            ast.keyword(arg="primary_key", value=set_value(True), identifier=None),
        )
    elif has_default and default not in none_types:
        nullable = False

    rstripped_dot_doc = _param["doc"].rstrip(".")
    doc_added_at = None
    if rstripped_dot_doc:
        doc_added_at = len(keywords)
        keywords.append(
            ast.keyword(arg="doc", value=set_value(rstripped_dot_doc), identifier=None)
        )

    if x_typ_sql.get("constraints"):
        keywords += [
            ast.keyword(
                arg=k, value=v if isinstance(v, AST) else set_value(v), identifier=None
            )
            for k, v in _param["x_typ"]["sql"]["constraints"].items()
        ]

    if has_default:
        # if default == NoneStr: default = None
        keywords.append(
            ast.keyword(
                arg="default",
                value=default if isinstance(default, AST) else set_value(default),
                identifier=None,
            )
        )

    if isinstance(nullable, bool):
        keywords.append(
            ast.keyword(arg="nullable", value=set_value(nullable), identifier=None)
        )

    # if include_name is True and _param.get("doc") and _param["doc"] != "[PK]":
    if doc_added_at is not None:
        keywords[doc_added_at].arg = "comment"
    # elif _param["doc"]:
    #     keywords.append(
    #         ast.keyword(arg="comment", value=set_value(_param["doc"]), identifier=None)
    #     )

    return Call(
        func=Name("Column", Load()),
        args=args,
        keywords=sorted(keywords, key=attrgetter("arg")),
        expr=None,
        expr_func=None,
    )


def generate_repr_method(params, cls_name, docstring_format):
    """
    Generate a `__repr__` method with all params, using `str.format` syntax

    :param params: an `OrderedDict` of form
        OrderedDict[str, {'typ': str, 'doc': Optional[str], 'default': Any}]
    :type params: ```OrderedDict```

    :param cls_name: Name of class
    :type cls_name: ```str```

    :param docstring_format: Format of docstring
    :type docstring_format: ```Literal['rest', 'numpydoc', 'google']```

    :return: `__repr__` method
    :rtype: ```FunctionDef```
    """
    keys = tuple(params.keys())
    return FunctionDef(
        name="__repr__",
        args=arguments(
            posonlyargs=[],
            arg=None,
            args=[set_arg("self")],
            kwonlyargs=[],
            kw_defaults=[],
            defaults=[],
            vararg=None,
            kwarg=None,
        ),
        body=[
            Expr(
                set_value(
                    """\n{sep}{_repr_docstring}""".format(
                        sep=tab * 2,
                        _repr_docstring=(
                            docstring_repr_str
                            if docstring_format == "rest"
                            else docstring_repr_google_str
                        ).lstrip(),
                    )
                )
            ),
            Return(
                value=Call(
                    func=Attribute(
                        set_value(
                            "{cls_name}({format_args})".format(
                                cls_name=cls_name,
                                format_args=", ".join(
                                    map("{0}={{{0}!r}}".format, keys)
                                ),
                            )
                        ),
                        "format",
                        Load(),
                    ),
                    args=[],
                    keywords=list(
                        map(
                            lambda key: ast.keyword(
                                arg=key,
                                value=Attribute(Name("self", Load()), key, Load()),
                                identifier=None,
                            ),
                            keys,
                        )
                    ),
                    expr=None,
                    expr_func=None,
                ),
                expr=None,
            ),
        ],
        decorator_list=[],
        arguments_args=None,
        identifier_name=None,
        stmt=None,
        lineno=None,
        returns=None,
        **maybe_type_comment
    )


def ensure_has_primary_key(intermediate_repr):
    """
    Add a primary key to the input (if nonexistent) then return the input.

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
    params = (
        intermediate_repr
        if isinstance(intermediate_repr, OrderedDict)
        else intermediate_repr["params"]
    )
    if not any(
        filter(
            rpartial(str.startswith, "[PK]"),
            map(
                methodcaller("get", "doc", ""),
                params.values(),
            ),
        )
    ):
        candidate_pks = []
        deque(
            map(
                candidate_pks.append,
                filter(
                    lambda k: "_name" in k or "_id" in k or "id_" in k or k == "id",
                    params.keys(),
                ),
            ),
            maxlen=0,
        )
        if len(candidate_pks) == 1:
            params[candidate_pks[0]]["doc"] = (
                "[PK] {}".format(params["dataset_name"]["doc"])
                if params[candidate_pks[0]].get("doc")
                else "[PK]"
            )
        else:
            assert (
                "id" not in intermediate_repr
            ), "Primary key unable to infer and column `id` already taken"
            params["id"] = {
                "doc": "[PK]",
                "typ": "int",
                "x_typ": {
                    "sql": {
                        "constraints": {
                            "server_default": Call(
                                args=[],
                                func=Name(ctx=Load(), id="Identity"),
                                keywords=[],
                            )
                        }
                    }
                },
            }
    return intermediate_repr


__all__ = [
    "ensure_has_primary_key",
    "generate_repr_method",
    "param_to_sqlalchemy_column_call",
]
