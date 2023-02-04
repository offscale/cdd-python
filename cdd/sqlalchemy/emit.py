"""
SQLalchemy emitters
"""

from ast import Assign, Call, ClassDef, Expr, Load, Name, Store, keyword
from collections import OrderedDict
from functools import partial
from itertools import chain
from operator import add
from os import environ

from cdd.compound.openapi.utils.emit_utils import (
    ensure_has_primary_key,
    generate_repr_method,
    param_to_sqlalchemy_column_call,
)
from cdd.docstring.emit import docstring
from cdd.shared.ast_utils import maybe_type_comment, set_value
from cdd.shared.pure_utils import deindent, indent_all_but_first, tab

FORCE_PK_ID = environ.get("FORCE_PK_ID", False) in (False, 0, "0", "false")


def sqlalchemy_table(
    intermediate_repr,
    name="config_tbl",
    table_name=None,
    force_pk_id=FORCE_PK_ID,
    docstring_format="rest",
    word_wrap=True,
    emit_original_whitespace=False,
    emit_default_doc=True,
):
    """
    Construct an `name = sqlalchemy.Table(name, metadata, Column(…), …)`

    :param intermediate_repr: a dictionary of form
        {  "name": Optional[str],
           "type": Optional[str],
           "doc": Optional[str],
           "params": OrderedDict[str, {'typ': str, 'doc': Optional[str], 'default': Any}]
           "returns": Optional[OrderedDict[Literal['return_type'],
                                           {'typ': str, 'doc': Optional[str], 'default': Any}),)]] }
    :type intermediate_repr: ```dict```

    :param name: name of binding + table
    :type name: ```str```

    :param table_name: Table name, defaults to `name`
    :type table_name: ```str```

    :param force_pk_id: Whether to force primary_key to be named `id` (if there isn't already a primary_key)
    :type force_pk_id: ```bool```

    :param docstring_format: Format of docstring
    :type docstring_format: ```Literal['rest', 'numpydoc', 'google']```

    :param word_wrap: Whether to word-wrap. Set `DOCTRANS_LINE_LENGTH` to configure length.
    :type word_wrap: ```bool```

    :param docstring_format: Format of docstring
    :type docstring_format: ```Literal['rest', 'numpydoc', 'google']```

    :param emit_original_whitespace: Whether to emit an original whitespace (in docstring) or strip it out
    :type emit_original_whitespace: ```bool```

    :param emit_original_whitespace: Whether to emit original whitespace or strip it out
    :type emit_original_whitespace: ```bool```

    :param emit_default_doc: Whether help/docstring should include 'With default' text
    :type emit_default_doc: ```bool```

    :return: AST of the Table expression + assignment
    :rtype: ```ClassDef```
    """
    return Assign(
        targets=[Name(name, Store())],
        value=Call(
            func=Name("Table", Load()),
            args=list(
                chain.from_iterable(
                    (
                        iter(
                            (
                                set_value(name if table_name is None else table_name),
                                Name("metadata", Load()),
                            )
                        ),
                        map(
                            partial(param_to_sqlalchemy_column_call, include_name=True),
                            ensure_has_primary_key(
                                intermediate_repr["params"], force_pk_id
                            ).items(),
                        ),
                    )
                )
            ),
            keywords=[
                keyword(
                    arg="comment",
                    value=set_value(
                        deindent(
                            add(
                                *map(
                                    partial(
                                        docstring,
                                        emit_default_doc=emit_default_doc,
                                        docstring_format=docstring_format,
                                        word_wrap=word_wrap,
                                        emit_original_whitespace=emit_original_whitespace,
                                        emit_types=True,
                                    ),
                                    (
                                        {
                                            "doc": intermediate_repr["doc"].lstrip()
                                            + "\n\n"
                                            if intermediate_repr["returns"]
                                            else "",
                                            "params": OrderedDict(),
                                            "returns": None,
                                        },
                                        {
                                            "doc": "",
                                            "params": OrderedDict(),
                                            "returns": intermediate_repr["returns"],
                                        },
                                    ),
                                )
                            ).strip()
                        )
                    ),
                    identifier=None,
                    expr=None,
                    lineno=None,
                    **maybe_type_comment,
                )
            ]
            if intermediate_repr.get("doc")
            else [],
            expr=None,
            expr_func=None,
        ),
        lineno=None,
        expr=None,
        **maybe_type_comment,
    )


def sqlalchemy(
    intermediate_repr,
    emit_repr=True,
    class_name=None,
    class_bases=("Base",),
    decorator_list=None,
    table_name=None,
    force_pk_id=FORCE_PK_ID,
    docstring_format="rest",
    word_wrap=True,
    emit_original_whitespace=False,
    emit_default_doc=True,
):
    """
    Construct an SQLAlchemy declarative class

    :param intermediate_repr: a dictionary of form
        {  "name": Optional[str],
           "type": Optional[str],
           "doc": Optional[str],
           "params": OrderedDict[str, {'typ': str, 'doc': Optional[str], 'default': Any}]
           "returns": Optional[OrderedDict[Literal['return_type'],
                                           {'typ': str, 'doc': Optional[str], 'default': Any}),)]] }
    :type intermediate_repr: ```dict```

    :param emit_repr: Whether to generate a `__repr__` method
    :type emit_repr: ```bool```

    :param class_name: name of class
    :type class_name: ```str```

    :param class_bases: bases of class (the generated class will inherit these)
    :type class_bases: ```Iterable[str]```

    :param decorator_list: List of decorators
    :type decorator_list: ```Optional[Union[List[Str], List[]]]```

    :param table_name: Table name, defaults to `class_name`
    :type table_name: ```str```

    :param force_pk_id: Whether to force primary_key to be named `id` (if there isn't already a primary_key)
    :type force_pk_id: ```bool```

    :param docstring_format: Format of docstring
    :type docstring_format: ```Literal['rest', 'numpydoc', 'google']```

    :param word_wrap: Whether to word-wrap. Set `DOCTRANS_LINE_LENGTH` to configure length.
    :type word_wrap: ```bool```

    :param docstring_format: Format of docstring
    :type docstring_format: ```Literal['rest', 'numpydoc', 'google']```

    :param emit_original_whitespace: Whether to emit an original whitespace (in docstring) or strip it out
    :type emit_original_whitespace: ```bool```

    :param emit_default_doc: Whether help/docstring should include 'With default' text
    :type emit_default_doc: ```bool```

    :return: SQLalchemy declarative class AST
    :rtype: ```ClassDef```
    """

    def _add(a, b):
        """
        Concatenate a with b with correct whitespace around and within each

        :param a: first string
        :type a: ```str```

        :param b: second string
        :type b: ```str```

        :return: combined strings with correct whitespace around and within
        :rtype: ```str```
        """
        b_splits = b.split("\n{tab}".format(tab=tab))
        res = "{a}{snd}\n{tab}{end}{tab}".format(
            a=a, tab=tab, snd=b_splits[0], end="\n".join(b_splits[1:])
        )
        return indent_all_but_first(res, indent_level=1, sep=tab)

    if class_name is None and intermediate_repr["name"]:
        class_name = intermediate_repr["name"]
    assert class_name is not None

    return ClassDef(
        name=class_name,
        bases=list(map(lambda class_base: Name(class_base, Load()), class_bases)),
        decorator_list=decorator_list or [],
        keywords=[],
        body=list(
            filter(
                None,
                (
                    Expr(
                        set_value(
                            _add(
                                *map(
                                    partial(
                                        docstring,
                                        docstring_format=docstring_format,
                                        emit_default_doc=emit_default_doc,
                                        emit_original_whitespace=emit_original_whitespace,
                                        emit_separating_tab=True,
                                        emit_types=True,
                                        indent_level=1,
                                        word_wrap=word_wrap,
                                    ),
                                    (
                                        {
                                            "doc": intermediate_repr["doc"],
                                            "params": OrderedDict(),
                                            "returns": None,
                                        },
                                        {
                                            "doc": "",
                                            "params": OrderedDict(),
                                            "returns": intermediate_repr["returns"],
                                        },
                                    ),
                                )
                            )
                        )
                    )
                    if intermediate_repr.get("doc")
                    or (intermediate_repr["returns"] or {})
                    .get("return_type", {})
                    .get("doc")
                    else None,
                    Assign(
                        targets=[Name("__tablename__", Store())],
                        value=set_value(table_name or class_name),
                        expr=None,
                        lineno=None,
                        **maybe_type_comment,
                    ),
                    *map(
                        lambda name_param: Assign(
                            targets=[Name(name_param[0], Store())],
                            value=param_to_sqlalchemy_column_call(
                                name_param, include_name=False
                            ),
                            expr=None,
                            lineno=None,
                            **maybe_type_comment,
                        ),
                        ensure_has_primary_key(
                            intermediate_repr["params"], force_pk_id
                        ).items(),
                    ),
                    generate_repr_method(
                        intermediate_repr["params"], class_name, docstring_format
                    )
                    if emit_repr
                    else None,
                ),
            )
        ),
        expr=None,
        identifier_name=None,
    )


__all__ = ["sqlalchemy", "sqlalchemy_table"]
