"""
Utility functions for `cdd.parse.sqlalchemy`
"""

from ast import Assign, Call, ClassDef, Constant, Load, Name
from itertools import chain

import cdd.shared.ast_utils
import cdd.shared.source_transformer
from cdd.shared.pure_utils import (
    PY_GTE_3_8,
    PY_GTE_3_9,
    append_to_dict,
    indent_all_but_first,
    rpartial,
    tab,
)

if PY_GTE_3_8:
    from cdd.shared.pure_utils import FakeConstant as Str
else:
    from ast import Str

if PY_GTE_3_9:
    FrozenSet = frozenset
else:
    from typing import FrozenSet

# SQLalchemy 1.14
# `from sqlalchemy import __all__; sorted(filter(lambda s: any(filter(str.isupper, s)), __all__))`
sqlalchemy_top_level_imports: FrozenSet[str] = frozenset(
    (
        "ARRAY",
        "BIGINT",
        "BINARY",
        "BLANK_SCHEMA",
        "BLOB",
        "BOOLEAN",
        "BigInteger",
        "Boolean",
        "CHAR",
        "CLOB",
        "CheckConstraint",
        "Column",
        "ColumnDefault",
        "Computed",
        "Constraint",
        "DATE",
        "DATETIME",
        "DDL",
        "DECIMAL",
        "Date",
        "DateTime",
        "DefaultClause",
        "Enum",
        "FLOAT",
        "FetchedValue",
        "Float",
        "ForeignKey",
        "ForeignKeyConstraint",
        "INT",
        "INTEGER",
        "Identity",
        "Index",
        "Integer",
        "Interval",
        "JSON",
        "LABEL_STYLE_DEFAULT",
        "LABEL_STYLE_DISAMBIGUATE_ONLY",
        "LABEL_STYLE_NONE",
        "LABEL_STYLE_TABLENAME_PLUS_COL",
        "LargeBinary",
        "MetaData",
        "NCHAR",
        "NUMERIC",
        "NVARCHAR",
        "Numeric",
        "PickleType",
        "PrimaryKeyConstraint",
        "REAL",
        "SMALLINT",
        "Sequence",
        "SmallInteger",
        "String",
        "TEXT",
        "TIME",
        "TIMESTAMP",
        "Table",
        "Text",
        "ThreadLocalMetaData",
        "Time",
        "TupleType",
        "TypeDecorator",
        "Unicode",
        "UnicodeText",
        "UniqueConstraint",
        "VARBINARY",
        "VARCHAR",
    )
)


def column_parse_extra_sql(idx_arg):
    """
    Parse Column arg into extra sql type information

    :param idx_arg: argument number, node
    :type idx_arg: ```tuple[int, AST]```

    :rtype: ```Optional[Tuple[str, dict]]```
    """
    idx, arg = idx_arg
    if idx < 2 and isinstance(arg, Name) and arg.id in column_type2typ:
        return "x_typ", {"sql": {"type": arg.id}}
    return None


def column_parse_arg(idx_arg):
    """
    Parse Column arg

    :param idx_arg: argument number, node
    :type idx_arg: ```tuple[int, AST]```

    :rtype: ```Optional[Tuple[str, AST]]```
    """
    idx, arg = idx_arg
    if idx < 2 and isinstance(arg, Name):
        return "typ", column_type2typ.get(arg.id, arg.id)
    elif isinstance(arg, Call):
        func_id = arg.func.id.rpartition(".")[2]
        if func_id == "Enum":
            return "typ", "Literal{}".format(
                list(map(cdd.shared.ast_utils.get_value, arg.args))
            )
        elif func_id == "ForeignKey":
            return "foreign_key", ",".join(
                map(cdd.shared.ast_utils.get_value, arg.args)
            )
        else:
            return "typ", cdd.shared.source_transformer.to_code(idx_arg[1]).replace(
                "\n", ""
            )

    val = cdd.shared.ast_utils.get_value(arg)
    assert val != arg, "Unable to parse {!r}".format(arg)
    return None if idx == 0 else (None, val)


def column_parse_kwarg(key_word):
    """
    Parse Column kwarg

    :param key_word: The keyword argument
    :type key_word: ```ast.keyword```

    :rtype: ```tuple[str, Any]```
    """
    val = cdd.shared.ast_utils.get_value(key_word.value)

    # Checking that the keyword.value has a value OR is a function call.
    assert val != key_word.value or isinstance(
        key_word.value, Call
    ), "Unable to parse {!r} of {}".format(
        key_word.arg, cdd.shared.source_transformer.to_code(key_word.value)
    )
    return key_word.arg, val


def column_call_to_param(call):
    """
    Parse column call `Call(func=Name("Column", Load(), …)` into param

    :param call: Column call from SQLAlchemy `Table` construction
    :type call: ```Call```

    :return: Name, dict with keys: 'typ', 'doc', 'default'
    :rtype: ```tuple[str, dict]```
    """
    assert call.func.id == "Column", "{} != Column".format(call.func.id)
    assert (
        len(call.args) < 4
    ), "Complex column parsing not implemented for: Column({})".format(
        ", ".join(map(repr, map(cdd.shared.ast_utils.get_value, call.args)))
    )

    _param = dict(
        filter(
            None,
            chain.from_iterable(
                (
                    map(column_parse_arg, enumerate(call.args)),
                    map(column_parse_extra_sql, enumerate(call.args)),
                    map(column_parse_kwarg, call.keywords),
                )
            ),
        )
    )
    if "comment" in _param and "doc" not in _param:
        _param["doc"] = _param.pop("comment")

    if "server_default" in _param:
        append_to_dict(
            _param,
            ["x_typ", "sql", "constraints", "server_default"],
            _param["server_default"],
        )

    for shortname, longname in ("PK", "primary_key"), (
        "FK({})".format(_param.get("foreign_key")),
        "foreign_key",
    ):
        if longname in _param:
            _param["doc"] = (
                "[{}] {}".format(shortname, _param["doc"])
                if _param.get("doc")
                else "[{}]".format(shortname)
            )
            del _param[longname]

    def _handle_null():
        """
        Properly handle null condition
        """
        if not _param["typ"].startswith("Optional["):
            _param["typ"] = "Optional[{}]".format(_param["typ"])

    if "nullable" in _param:
        not _param["nullable"] or _handle_null()
        del _param["nullable"]

    if (
        "default" in _param
        and not cdd.shared.ast_utils.get_value(call.args[0]).endswith("kwargs")
        and "doc" in _param
    ):
        _param["doc"] += "."

    return cdd.shared.ast_utils.get_value(call.args[0]), _param


def column_call_name_manipulator(call, operation="remove", name=None):
    """
    :param call: `Column` function call within SQLalchemy
    :type call: ```Call``

    :param operation:
    :type operation: ```Literal["remove", "add"]```

    :param name:
    :type name: ```str```

    :return: Column call
    :rtype: ```Call```
    """
    assert (
        isinstance(call, Call)
        and isinstance(call.func, Name)
        and call.func.id == "Column"
    )
    if isinstance(call.args[0], (Constant, Str)) and operation == "remove":
        del call.args[0]
    elif operation == "add" and name is not None:
        call.args.insert(0, Name(name, Load()))
    return call


def concat_with_whitespace(a, b):
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


def get_pk_and_type(sqlalchemy_class):
    """
    Get the primary key and its type from an SQLalchemy class

    :param sqlalchemy_class: SQLalchemy class
    :type sqlalchemy_class: ```ClassDef```

    :return: Primary key and its type
    :rtype: ```tuple[str, str]```
    """
    assert isinstance(
        sqlalchemy_class, ClassDef
    ), "Expected `ClassDef` got `{type_name}`".format(
        type_name=type(sqlalchemy_class).__name__
    )
    return (
        lambda assign: (
            assign
            if assign is None
            else (
                assign.targets[0].id,
                assign.value.args[0].id,  # First arg is type
            )
        )
    )(
        next(
            filter(
                lambda assign: any(
                    filter(
                        lambda key_word: key_word.arg == "primary_key"
                        and cdd.shared.ast_utils.get_value(key_word.value) is True,
                        assign.value.keywords,
                    )
                ),
                filter(
                    lambda assign: isinstance(assign.value, Call)
                    and isinstance(assign.value.func, Name)
                    and assign.value.func.id == "Column",
                    filter(rpartial(isinstance, Assign), sqlalchemy_class.body),
                ),
            ),
            None,
        )
    )


def get_table_name(sqlalchemy_class):
    """
    Get the primary key and its type from an SQLalchemy class

    :param sqlalchemy_class: SQLalchemy class
    :type sqlalchemy_class: ```ClassDef```

    :return: Primary key and its type
    :rtype: ```str```
    """
    return next(
        map(
            lambda assign: cdd.shared.ast_utils.get_value(assign.value),
            filter(
                lambda node: next(
                    filter(lambda target: target.id == "__tablename__", node.targets),
                    None,
                )
                and node,
                filter(
                    lambda node: isinstance(node, Assign)
                    and isinstance(node.value, (Str, Constant)),
                    sqlalchemy_class.body,
                ),
            ),
        ),
        sqlalchemy_class.name,
    )


# Construct from https://docs.sqlalchemy.org/en/13/core/type_basics.html#generic-types
column_type2typ = {
    "BigInteger": "int",
    "Boolean": "bool",
    "DateTime": "datetime",
    "Float": "float",
    "Integer": "int",
    "JSON": "Optional[dict]",
    "LargeBinary": "BlobProperty",
    "String": "str",
    "Text": "str",
    "Unicode": "str",
    "UnicodeText": "str",
    "boolean": "bool",
    "dict": "dict",
    "float": "float",
    "int": "int",
    "str": "str",
}

__all__ = [
    "column_call_name_manipulator",
    "column_call_to_param",
    "column_type2typ",
    "concat_with_whitespace",
    "get_pk_and_type",
    "get_table_name",
    "sqlalchemy_top_level_imports",
]
