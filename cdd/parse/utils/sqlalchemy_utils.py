"""
Utility functions for `cdd.parse.sqlalchemy`
"""

import ast
from ast import (
    Assign,
    Call,
    ClassDef,
    Constant,
    ImportFrom,
    Load,
    Module,
    Name,
    Str,
    alias,
)
from itertools import chain, filterfalse
from operator import attrgetter

from cdd.ast_utils import get_value
from cdd.pure_utils import rpartial
from cdd.source_transformer import to_code

# SQLalchemy 1.14
# `from sqlalchemy import __all__; sorted(filter(lambda s: any(filter(str.isupper, s)), __all__))`
sqlalchemy_top_level_imports = frozenset(
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


def column_parse_arg(idx_arg):
    """
    Parse Column arg

    :param idx_arg: argument number, node
    :type idx_arg: ```Tuple[int, AST]```

    :rtype: ```Optional[Tuple[str, AST]]```
    """
    idx, arg = idx_arg
    if idx < 2 and isinstance(arg, Name):
        return "typ", column_type2typ.get(arg.id, arg.id)
    elif isinstance(arg, Call):
        func_id = arg.func.id.rpartition(".")[2]
        if func_id == "Enum":
            return "typ", "Literal{}".format(list(map(get_value, arg.args)))
        elif func_id == "ForeignKey":
            return "foreign_key", ",".join(map(get_value, arg.args))
        else:
            raise NotImplementedError(func_id)

    val = get_value(arg)
    assert val != arg, "Unable to parse {!r}".format(arg)
    if idx == 0:
        return None  # Column name
    return None, val


def column_parse_kwarg(key_word):
    """
    Parse Column kwarg

    :param key_word: The keyword argument
    :type key_word: ```ast.keyword```

    :rtype: ```Tuple[str, Any]```
    """
    val = get_value(key_word.value)
    assert val != key_word.value, "Unable to parse {!r} of {}".format(
        key_word.arg, to_code(key_word.value)
    )
    return key_word.arg, val


def column_call_to_param(call):
    """
    Parse column call `Call(func=Name("Column", Load(), …)` into param

    :param call: Column call from SQLAlchemy `Table` construction
    :type call: ```Call```

    :return: Name, dict with keys: 'typ', 'doc', 'default'
    :rtype: ```Tuple[str, dict]```
    """
    assert call.func.id == "Column", "{} != Column".format(call.func.id)
    assert (
        len(call.args) < 4
    ), "Complex column parsing not implemented for: Column({})".format(
        ", ".join(map(repr, map(get_value, call.args)))
    )

    _param = dict(
        filter(
            None,
            chain.from_iterable(
                (
                    map(column_parse_arg, enumerate(call.args)),
                    map(column_parse_kwarg, call.keywords),
                )
            ),
        )
    )
    if "comment" in _param and "doc" not in _param:
        _param["doc"] = _param.pop("comment")

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
        and not get_value(call.args[0]).endswith("kwargs")
        and "doc" in _param
    ):
        _param["doc"] += "."

    return get_value(call.args[0]), _param


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


def infer_imports_from_sqlalchemy(sqlalchemy_class_def):
    """
    Infer imports from SQLalchemy class

    :param sqlalchemy_class_def: SQLalchemy class
    :type sqlalchemy_class_def: ```ClassDef```

    :return: filter of imports (can be considered ```Iterable[str]```)
    :rtype: ```filter```
    """
    candidates = frozenset(
        map(
            attrgetter("id"),
            filter(
                rpartial(isinstance, Name),
                ast.walk(
                    Module(
                        body=list(
                            filter(
                                rpartial(isinstance, Call),
                                ast.walk(sqlalchemy_class_def),
                            )
                        ),
                        type_ignores=[],
                        stmt=None,
                    )
                ),
            ),
        )
    )

    candidates_not_in_valid_types = frozenset(
        filterfalse(
            frozenset(
                ("list", "string", "int", "float", "complex", "long")
            ).__contains__,
            filterfalse(sqlalchemy_top_level_imports.__contains__, candidates),
        )
    )
    return candidates_not_in_valid_types ^ candidates


def imports_from(sqlalchemy_classes):
    """
    Generate `from sqlalchemy import <>` from the body of SQLalchemy `class`es

    :param sqlalchemy_classes: SQLalchemy `class`es with base class of `Base`
    :type sqlalchemy_classes: ```ClassDef```

    :return: `from sqlalchemy import <>` where <> is what was inferred from `sqlalchemy_classes`
    :rtype: ```ImportFrom```
    """
    return ImportFrom(
        module="sqlalchemy",
        names=list(
            map(
                lambda names: alias(
                    names,
                    None,
                    identifier=None,
                    identifier_name=None,
                ),
                sorted(
                    frozenset(
                        filter(
                            None,
                            chain.from_iterable(
                                map(
                                    infer_imports_from_sqlalchemy,
                                    sqlalchemy_classes,
                                )
                            ),
                        )
                    )
                ),
            )
        ),
        level=0,
    )


def get_pk_and_type(sqlalchemy_class):
    """
    Get the primary key and its type from an SQLalchemy class

    :param sqlalchemy_class: SQLalchemy class
    :type sqlalchemy_class: ```ClassDef```

    :return: Primary key and its type
    :rtype: ```Tuple[str, str]```
    """
    assert isinstance(
        sqlalchemy_class, ClassDef
    ), "Expected `ClassDef` got `{type_name}`".format(
        type_name=type(sqlalchemy_class).__name__
    )
    return (
        lambda assign: assign
        if assign is None
        else (
            assign.targets[0].id,
            assign.value.args[0].id,  # First arg is type
        )
    )(
        next(
            filter(
                lambda assign: any(
                    filter(
                        lambda key_word: key_word.arg == "primary_key"
                        and get_value(key_word.value) is True,
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
            lambda assign: get_value(assign.value),
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
    "get_pk_and_type",
    "get_table_name",
    "imports_from",
    "sqlalchemy_top_level_imports",
]
