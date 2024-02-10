"""
SQLalchemy emitters
"""

from ast import Assign, Call, ClassDef, Expr, Load, Name, Store, keyword
from collections import OrderedDict
from functools import partial
from itertools import chain
from operator import add
from os import environ
from typing import Optional

import cdd.compound.openapi.utils.emit_utils
import cdd.sqlalchemy.utils.emit_utils
from cdd.docstring.emit import docstring
from cdd.shared.ast_utils import maybe_type_comment, set_value
from cdd.shared.pure_utils import deindent, ensure_valid_identifier
from cdd.sqlalchemy.utils.parse_utils import concat_with_whitespace

FORCE_PK_ID = environ.get("FORCE_PK_ID", False) not in (False, 0, "0", "false")


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
        targets=[
            Name(
                ensure_valid_identifier(
                    name
                    if name not in (None, "config_tbl") or not intermediate_repr["name"]
                    else intermediate_repr["name"]
                ),
                Store(),
            )
        ],
        value=Call(
            func=Name("Table", Load(), lineno=None, col_offset=None),
            args=list(
                chain.from_iterable(
                    (
                        iter(
                            (
                                set_value(name if table_name is None else table_name),
                                Name("metadata", Load(), lineno=None, col_offset=None),
                            )
                        ),
                        *map(
                            partial(
                                cdd.sqlalchemy.utils.emit_utils.param_to_sqlalchemy_column_calls,
                                include_name=True,
                            ),
                            cdd.sqlalchemy.utils.emit_utils.ensure_has_primary_key(
                                intermediate_repr["params"], force_pk_id
                            ).items(),
                        ),
                    )
                )
            ),
            keywords=list(
                chain.from_iterable(
                    (
                        (
                            (
                                (
                                    lambda val: (
                                        (
                                            keyword(
                                                arg="comment",
                                                value=set_value(val),
                                                identifier=None,
                                                expr=None,
                                                lineno=None,
                                                **maybe_type_comment,
                                            ),
                                        )
                                        if val
                                        else iter(())
                                    )
                                )(
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
                                                        "doc": (
                                                            intermediate_repr[
                                                                "doc"
                                                            ].lstrip()
                                                            + "\n\n"
                                                            if intermediate_repr[
                                                                "returns"
                                                            ]
                                                            else ""
                                                        ),
                                                        "params": OrderedDict(),
                                                        "returns": None,
                                                    },
                                                    {
                                                        "doc": "",
                                                        "params": OrderedDict(),
                                                        "returns": intermediate_repr[
                                                            "returns"
                                                        ],
                                                    },
                                                ),
                                            )
                                        ).strip()
                                    )
                                )
                                if intermediate_repr.get("doc")
                                else iter(())
                            )
                        ),
                        (
                            keyword(
                                arg="keep_existing",
                                value=set_value(True),
                                identifier=None,
                                expr=None,
                                lineno=None,
                                **maybe_type_comment,
                            ),
                        ),
                    )
                )
            ),
            expr=None,
            expr_func=None,
            lineno=None,
            col_offset=None,
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

    :param emit_repr: Whether to generate a `__repr__` method
    :type emit_repr: ```bool```

    :param class_name: name of class
    :type class_name: ```str```

    :param class_bases: bases of class (the generated class will inherit these)
    :type class_bases: ```Iterable[str]```

    :param decorator_list: List of decorators
    :type decorator_list: ```Optional[Union[List[str], List[]]]```

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

    if class_name is None and intermediate_repr["name"]:
        class_name: Optional[str] = intermediate_repr["name"]
    assert class_name is not None, "`class_name` is `None`"

    return ClassDef(
        name=class_name,
        bases=list(
            map(
                lambda class_base: Name(
                    class_base, Load(), lineno=None, col_offset=None
                ),
                class_bases,
            )
        ),
        decorator_list=decorator_list or [],
        type_params=[],
        keywords=[],
        body=list(
            filter(
                None,
                chain.from_iterable(
                    (
                        (
                            (
                                Expr(
                                    set_value(
                                        concat_with_whitespace(
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
                                                        "returns": intermediate_repr[
                                                            "returns"
                                                        ],
                                                    },
                                                ),
                                            )
                                        )
                                    ),
                                    lineno=None,
                                    col_offset=None,
                                )
                                if intermediate_repr.get("doc")
                                or (intermediate_repr["returns"] or {})
                                .get("return_type", {})
                                .get("doc")
                                else None
                            ),
                            Assign(
                                targets=[
                                    Name(
                                        "__tablename__",
                                        Store(),
                                        lineno=None,
                                        col_offset=None,
                                    )
                                ],
                                value=set_value(table_name or class_name),
                                expr=None,
                                lineno=None,
                                **maybe_type_comment,
                            ),
                        ),
                        *map(
                            lambda name_param: map(
                                lambda column: Assign(
                                    targets=[
                                        Name(
                                            name_param[0],
                                            Store(),
                                            lineno=None,
                                            col_offset=None,
                                        )
                                    ],
                                    value=column,
                                    expr=None,
                                    lineno=None,
                                    **maybe_type_comment,
                                ),
                                cdd.sqlalchemy.utils.emit_utils.param_to_sqlalchemy_column_calls(
                                    name_param, include_name=False
                                ),
                            ),
                            cdd.sqlalchemy.utils.emit_utils.ensure_has_primary_key(
                                intermediate_repr["params"], force_pk_id
                            ).items(),
                        ),
                        (
                            (
                                cdd.sqlalchemy.utils.emit_utils.generate_repr_method(
                                    intermediate_repr["params"],
                                    class_name,
                                    docstring_format,
                                )
                                if emit_repr
                                else None
                            ),
                        ),
                    )
                ),
            )
        ),
        expr=None,
        identifier_name=None,
        lineno=None,
        col_offset=None,
    )


def sqlalchemy_hybrid(
    intermediate_repr,
    emit_repr=True,
    emit_create_from_attr=True,
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
    Construct an `class TableName(Base): __table__ = sqlalchemy.Table(name, metadata, Column(…), …)`

    Valid in SQLalchemy 2.0 and 1.4:
    - docs.sqlalchemy.org/en/14/orm/declarative_tables.html#declarative-with-imperative-table-a-k-a-hybrid-declarative
    - docs.sqlalchemy.org/en/20/orm/declarative_tables.html#declarative-with-imperative-table-a-k-a-hybrid-declarative

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

    :param emit_repr: Whether to generate a `__repr__` method
    :type emit_repr: ```bool```

    :param emit_create_from_attr: Whether to generate a `create_from_attr` staticmethod
    :type emit_create_from_attr: ```bool```

    :param class_name: name of class
    :type class_name: ```str```

    :param class_bases: bases of class (the generated class will inherit these)
    :type class_bases: ```Iterable[str]```

    :param decorator_list: List of decorators
    :type decorator_list: ```Optional[Union[List[str], List[]]]```

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

    :return: SQLalchemy hybrids declarative class AST
    :rtype: ```ClassDef```
    """

    if class_name is None and intermediate_repr["name"]:
        class_name: str = intermediate_repr["name"]
    assert class_name is not None, "`class_name` is `None`"

    return ClassDef(
        name=class_name,
        bases=list(
            map(
                lambda class_base: Name(
                    class_base, Load(), lineno=None, col_offset=None
                ),
                class_bases,
            )
        ),
        decorator_list=decorator_list or [],
        type_params=[],
        keywords=[],
        body=list(
            filter(
                None,
                (
                    (
                        Expr(
                            set_value(
                                concat_with_whitespace(
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
                            ),
                            lineno=None,
                            col_offset=None,
                        )
                        if intermediate_repr.get("doc")
                        or (intermediate_repr["returns"] or {})
                        .get("return_type", {})
                        .get("doc")
                        else None
                    ),
                    Assign(
                        targets=[
                            Name("__tablename__", Store(), lineno=None, col_offset=None)
                        ],
                        value=set_value(table_name or class_name),
                        expr=None,
                        lineno=None,
                        **maybe_type_comment,
                    ),
                    sqlalchemy_table(
                        intermediate_repr=intermediate_repr,
                        name="__table__",
                        table_name=table_name or intermediate_repr["name"],
                        force_pk_id=force_pk_id,
                        docstring_format=docstring_format,
                        word_wrap=word_wrap,
                        emit_original_whitespace=emit_original_whitespace,
                        emit_default_doc=emit_default_doc,
                    ),
                    (
                        cdd.sqlalchemy.utils.emit_utils.generate_repr_method(
                            intermediate_repr["params"],
                            class_name,
                            docstring_format,
                            hybrid=True,
                        )
                        if emit_repr
                        else None
                    ),
                    (
                        cdd.sqlalchemy.utils.emit_utils.generate_create_from_attr_staticmethod(
                            intermediate_repr["params"], class_name, docstring_format
                        )
                        if emit_create_from_attr
                        else None
                    ),
                ),
            )
        ),
        expr=None,
        identifier_name=None,
        lineno=None,
        col_offset=None,
    )


__all__ = ["sqlalchemy", "sqlalchemy_hybrid", "sqlalchemy_table"]
