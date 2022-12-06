"""
SQLalchemy parsers

TODO
====

  - Implement update (see https://github.com/sqlalchemy/sqlalchemy/discussions/5940)
  - Implement batch CRUD

"""

from ast import AnnAssign, Assign, Call, ClassDef, Load, Name, get_docstring, keyword
from collections import OrderedDict
from functools import partial
from itertools import chain, filterfalse
from operator import attrgetter, eq

import cdd.parse.utils.parser_utils
from cdd.ast_utils import get_value, set_value
from cdd.defaults_utils import extract_default
from cdd.parse.docstring import docstring
from cdd.parse.utils.sqlalchemy_utils import column_call_to_param
from cdd.pure_utils import assert_equal, rpartial


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
    intermediate_repr = (
        {"type": None, "doc": "", "params": OrderedDict()}
        if comment is None
        else docstring(comment, parse_original_whitespace=parse_original_whitespace)
    )
    intermediate_repr["name"] = name
    assert isinstance(call_or_name, Call)
    assert_equal(call_or_name.func.id.rpartition(".")[2], "Table")
    assert len(call_or_name.args) > 2

    merge_ir = {
        "params": OrderedDict(map(column_call_to_param, call_or_name.args[2:])),
        "returns": None,
    }
    cdd.parse.utils.parser_utils.ir_merge(target=intermediate_repr, other=merge_ir)
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
    assert isinstance(class_def, ClassDef), "Expected `ClassDef` got `{}`".format(
        type(class_def).__name__
    )

    # Parse into the same format that `sqlalchemy_table` can read, then return with a call to it

    name = get_value(
        next(
            filter(
                lambda assign: any(
                    filter(
                        partial(eq, "__tablename__"),
                        map(attrgetter("id"), assign.targets),
                    )
                ),
                filter(rpartial(isinstance, Assign), class_def.body),
            )
        ).value
    )
    doc_string = get_docstring(class_def, clean=parse_original_whitespace)

    def _merge_name_to_column(assign):
        """
        Merge `a = Column()` into `Column("a")`

        :param assign: Of form `a = Column()`
        :type assign: ```Assign```

        :return: Unwrapped Call with name prepended
        :rtype: ```Call```
        """
        assign.value.args.insert(0, set_value(assign.targets[0].id))
        return assign.value

    return sqlalchemy_table(
        Call(
            func=Name("Table", Load()),
            args=list(
                chain.from_iterable(
                    (
                        iter((set_value(name), Name("metadata", Load()))),
                        map(
                            _merge_name_to_column,
                            filterfalse(
                                lambda assign: any(
                                    map(
                                        lambda target: target.id == "__tablename__"
                                        or hasattr(target, "value")
                                        and isinstance(target.value, Call)
                                        and target.func.rpartition(".")[2] == "Column",
                                        assign.targets,
                                    ),
                                ),
                                filter(rpartial(isinstance, Assign), class_def.body),
                            ),
                        ),
                    )
                )
            ),
            keywords=[]
            if doc_string is None
            else [keyword(arg="comment", value=set_value(doc_string), identifier=None)],
            expr=None,
            expr_func=None,
        )
    )


__all__ = ["sqlalchemy_table", "sqlalchemy"]
