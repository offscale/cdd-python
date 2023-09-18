"""
SQLalchemy parsers

TODO
====

  - Implement update (see https://github.com/sqlalchemy/sqlalchemy/discussions/5940)
  - Implement batch CRUD

"""

import ast
from ast import AnnAssign, Assign, Call, ClassDef, Module
from collections import OrderedDict
from inspect import getsource

import cdd.shared.parse.utils.parser_utils
from cdd.compound.openapi.utils.emit_utils import sqlalchemy_class_to_table
from cdd.docstring.parse import docstring
from cdd.shared.ast_utils import get_value
from cdd.shared.defaults_utils import extract_default
from cdd.shared.pure_utils import assert_equal, rpartial
from cdd.sqlalchemy.utils.parse_utils import column_call_to_param


def sqlalchemy_table(call_or_name, parse_original_whitespace=False):
    """
    Parse out a `sqlalchemy.Table`, or a `name = sqlalchemy.Table`, into the IR

    :param call_or_name: The call to `sqlalchemy.Table` or an assignment followed by the call
    :type call_or_name: ```Union[AnnAssign, Assign, Call]```

    :param parse_original_whitespace: Whether to parse original whitespace or strip it out
    :type parse_original_whitespace: ```bool```

    :return: a dictionary of form
        {  "name": Optional[str],
           "type": Optional[str],
           "doc": Optional[str],
           "params": OrderedDict[str, {'typ': str, 'doc': Optional[str], 'default': Any}]
           "returns": Optional[OrderedDict[Literal['return_type'],
                                           {'typ': str, 'doc': Optional[str], 'default': Any}),)]] }
    :rtype: ```dict```
    """
    if isinstance(call_or_name, Assign):
        name, call_or_name = call_or_name.targets[0].id, call_or_name.value
    elif isinstance(call_or_name, AnnAssign):
        name, call_or_name = call_or_name.target.id, call_or_name.value
    else:
        if not isinstance(call_or_name, Call):
            call_or_name = get_value(call_or_name)
        name = get_value(call_or_name.args[0])

    # Binding should be same name as table… I guess?
    assert_equal(get_value(call_or_name.args[0]), name)

    comment = next(
        map(
            get_value,
            map(
                get_value, filter(lambda kw: kw.arg == "comment", call_or_name.keywords)
            ),
        ),
        None,
    )
    doc = next(
        map(
            get_value,
            map(get_value, filter(lambda kw: kw.arg == "doc", call_or_name.keywords)),
        ),
        None,
    )
    intermediate_repr = (
        {"type": None, "doc": "", "params": OrderedDict()}
        if comment is None and doc is None
        else docstring(
            doc or comment, parse_original_whitespace=parse_original_whitespace
        )
    )
    intermediate_repr["name"] = name
    assert isinstance(call_or_name, Call), "Expected `all` got `{node_name!r}`".format(
        node_name=type(call_or_name).__name__
    )
    assert_equal(call_or_name.func.id.rpartition(".")[2], "Table")
    assert len(call_or_name.args) > 2

    merge_ir = {
        "params": OrderedDict(map(column_call_to_param, call_or_name.args[2:])),
        "returns": None,
    }
    cdd.shared.parse.utils.parser_utils.ir_merge(
        target=intermediate_repr, other=merge_ir
    )
    if intermediate_repr["returns"] and intermediate_repr["returns"].get(
        "return_type", {}
    ).get("doc"):
        intermediate_repr["returns"]["return_type"]["doc"] = extract_default(
            intermediate_repr["returns"]["return_type"]["doc"], emit_default_doc=False
        )[0]

    return intermediate_repr


def sqlalchemy(class_def, parse_original_whitespace=False):
    """
    Parse out a `class C(Base): __tablename__=  'tbl'; dataset_name = Column(String, doc="p", primary_key=True)`,
        as constructed on an SQLalchemy declarative `Base`.

    Also supports hybrid syntax: `class TableName(Base): __table__ = sqlalchemy.Table(name, metadata, Column(…), …)`

    :param class_def: A class inheriting from declarative `Base`, where `Base = sqlalchemy.orm.declarative_base()`
    :type class_def: ```Union[ClassDef]```

    :param parse_original_whitespace: Whether to parse original whitespace or strip it out
    :type parse_original_whitespace: ```bool```

    :return: a dictionary of form
        {  "name": Optional[str],
           "type": Optional[str],
           "doc": Optional[str],
           "params": OrderedDict[str, {'typ': str, 'doc': Optional[str], 'default': Any}]
           "returns": Optional[OrderedDict[Literal['return_type'],
                                           {'typ': str, 'doc': Optional[str], 'default': Any}),)]] }
    :rtype: ```dict```
    """

    if not isinstance(class_def, ClassDef):
        class_def = (
            next(filter(rpartial(isinstance, ClassDef), class_def.body))
            if isinstance(class_def, Module)
            else ast.parse(getsource(class_def)).body[0]
        )
    assert isinstance(class_def, ClassDef), "Expected `ClassDef` got `{!r}`".format(
        type(class_def).__name__
    )

    return sqlalchemy_table(
        sqlalchemy_class_to_table(class_def, parse_original_whitespace)
    )


# Separate function to get a new docstring + as mirror to `emit.sqlalchemy_hybrid`


def sqlalchemy_hybrid(class_def, parse_original_whitespace=False):
    """
    Parse out a `class TableName(Base): __table__ = sqlalchemy.Table(name, metadata, Column(…), …)`,
        as constructed on an SQLalchemy declarative `Base`.

    :param class_def: A class inheriting from declarative `Base`, where `Base = sqlalchemy.orm.declarative_base()`
    :type class_def: ```Union[ClassDef]```

    :param parse_original_whitespace: Whether to parse original whitespace or strip it out
    :type parse_original_whitespace: ```bool```

    :return: a dictionary of form
        {  "name": Optional[str],
           "type": Optional[str],
           "doc": Optional[str],
           "params": OrderedDict[str, {'typ': str, 'doc': Optional[str], 'default': Any}]
           "returns": Optional[OrderedDict[Literal['return_type'],
                                           {'typ': str, 'doc': Optional[str], 'default': Any}),)]] }
    :rtype: ```dict```
    """
    return sqlalchemy(
        class_def=class_def, parse_original_whitespace=parse_original_whitespace
    )


__all__ = ["sqlalchemy", "sqlalchemy_hybrid", "sqlalchemy_table"]
