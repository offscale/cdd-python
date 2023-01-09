"""
Utility functions for `cdd.emit.sqlalchemy`
"""

import ast
from ast import (
    AST,
    Assign,
    Attribute,
    Call,
    ClassDef,
    Expr,
    FunctionDef,
    ImportFrom,
    Index,
    Load,
    Module,
    Name,
    Return,
    Subscript,
    Tuple,
    alias,
    arguments,
)
from collections import OrderedDict, deque
from itertools import chain, filterfalse
from operator import attrgetter, methodcaller
from os import path
from platform import system

from cdd.ast_utils import get_value, maybe_type_comment, set_arg, set_value
from cdd.parse.utils.sqlalchemy_utils import (
    column_type2typ,
    get_pk_and_type,
    get_table_name,
    sqlalchemy_top_level_imports,
)
from cdd.pure_utils import (
    find_module_filepath,
    none_types,
    rpartial,
    tab,
    upper_camelcase_to_pascal,
)
from cdd.source_transformer import to_code
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

    if include_name:
        args.append(set_value(name))

    x_typ_sql = _param.get("x_typ", {}).get("sql", {})

    if "typ" in _param:
        nullable = update_args_infer_typ_sqlalchemy(
            _param, args, name, nullable, x_typ_sql
        )

    default = x_typ_sql.get("default", _param.get("default", ast))
    has_default = default is not ast
    pk = _param.get("doc", "").startswith("[PK]")
    fk = _param.get("doc", "").startswith("[FK")
    if pk:
        _param["doc"] = _param["doc"][4:].lstrip()
        keywords.append(
            ast.keyword(arg="primary_key", value=set_value(True), identifier=None),
        )
    elif fk:
        end = _param["doc"].find("]") + 1
        fk_val = _param["doc"][len("[FK(") : end - len(")]")]
        _param["doc"] = _param["doc"][end:].lstrip()
        args.append(
            Call(
                func=Name(id="ForeignKey", ctx=Load()),
                args=[set_value(fk_val)],
                keywords=[],
            )
        )
    elif has_default and default not in none_types:
        nullable = False

    rstripped_dot_doc = _param.get("doc", "").rstrip(".")
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


def update_args_infer_typ_sqlalchemy(_param, args, name, nullable, x_typ_sql):
    """
    :param _param: Param with typ
    :type _param: ```dict```

    :param args:
    :type args: ```List```

    :param name:
    :type name: ```str```

    :param nullable:
    :type nullable: ```Optional[bool]```

    :param x_typ_sql:
    :type x_typ_sql: ```dict```

    :rtype: ```bool```
    """
    if _param["typ"].startswith("Optional["):
        _param["typ"] = _param["typ"][len("Optional[") : -1]
        nullable = True
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
    elif "items" in _param and _param["items"].get("type", False) in typ2column_type:
        args.append(
            Call(
                func=Name(id="ARRAY", ctx=Load()),
                args=[Name(id=typ2column_type[_param["items"]["type"]], ctx=Load())],
                keywords=[],
                expr=None,
                expr_func=None,
            )
        )
    elif _param.get("typ").startswith("Union["):
        # Hack to remove the union type. Enum parse seems to be incorrect?
        union_typ = ast.parse(_param["typ"]).body[0]
        assert (
            isinstance(union_typ.value, Subscript)
            and isinstance(union_typ.value.slice, Index)
            and isinstance(union_typ.value.slice.value, Tuple)
        )
        assert len(union_typ.value.slice.value.elts) == 2
        left, right = map(attrgetter("id"), union_typ.value.slice.value.elts)
        args.append(
            Name(
                typ2column_type.get(right, right)
                if left in typ2column_type
                else typ2column_type.get(left, left),
                Load(),
            )
        )
    else:
        args.append(
            Name(
                x_typ_sql["type"]
                if "type" in x_typ_sql
                else typ2column_type.get(_param["typ"], _param["typ"]),
                Load(),
            )
        )
    return nullable


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
                "[PK] {}".format(params[candidate_pks[0]]["doc"])
                if params[candidate_pks[0]].get("doc")
                else "[PK]"
            )
        elif "id" in intermediate_repr:
            params["id"]["doc"] = (
                "[PK] {}".format(params["id"]["doc"])
                if params["id"].get("doc")
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


def update_with_imports_from_columns(filename):
    """
    Given an existing filename, figure out its relative imports

    This is subsequent phase process, and must be preceded by:
    - All SQLalchemy models being in the same directory as filename

    It will take:
    ```py
    Column(TableName0,
           ForeignKey("TableName0"),
           nullable=True)
    ```
    …and add this import:
    ```py
    from `basename(filename)`.table_name import TableName0
    ```

    :param filename: Python filename containing SQLalchemy `class`(es)
    :type filename: ```str```
    """
    with open(filename, "rt") as f:
        mod = ast.parse(f.read())

    candidates = sorted(
        frozenset(
            filterfalse(
                str.istitle,
                filterfalse(
                    frozenset(
                        ("list", "string", "int", "float", "complex", "long", "self")
                    ).__contains__,
                    filterfalse(
                        sqlalchemy_top_level_imports.__contains__,
                        map(
                            attrgetter("id"),
                            filter(
                                rpartial(isinstance, Name),
                                ast.walk(
                                    Module(
                                        body=list(
                                            filter(
                                                rpartial(isinstance, Call),
                                                ast.walk(mod),
                                            )
                                        ),
                                        type_ignores=[],
                                        stmt=None,
                                    )
                                ),
                            ),
                        ),
                    ),
                ),
            )
        )
    )

    module = path.dirname(filename)
    mod.body = list(
        chain.from_iterable(
            (
                map(
                    lambda class_name: ImportFrom(
                        module=".".join(
                            (module, upper_camelcase_to_pascal(class_name))
                        ),
                        names=[
                            alias(
                                class_name,
                                None,
                                identifier=None,
                                identifier_name=None,
                            )
                        ],
                        level=0,
                    ),
                    candidates,
                ),
                mod.body,
            )
        )
    )

    with open(filename, "wt") as f:
        f.write(to_code(mod))


def update_fk_for_file(filename):
    """
    Given an existing filename, use its imports and to replace its foreign keys with the correct values

    This is subsequent phase process, and must be preceded by:
    - All SQLalchemy models being in the same directory as filename
    - Correct imports being added

    Then it can transform classes with members like:
    ```py
    Column(
            TableName0,
            ForeignKey("TableName0"),
            nullable=True,
        )
    ```
    To the following, inferring that the primary key field is `id` by resolving the symbol and `ast.parse`ing it:
    ```py
    Column(Integer, ForeignKey("table_name0.id"))
    ```

    :param filename: Filename
    :type filename: ```str```
    """
    with open(filename, "rt") as f:
        mod = ast.parse(f.read())

    def handle_sqlalchemy_cls(symbol_to_module, sqlalchemy_class_def):
        """
        Ensure the SQLalchemy classes have their foreign keys resolved properly

        :param symbol_to_module: Dictionary of symbol to module, like `{"join": "os.path"}`
        :type symbol_to_module: ```Dict[str,str]````

        :param sqlalchemy_class_def: SQLalchemy `class`
        :type sqlalchemy_class_def: ```ClassDef```

        :return: SQLalchemy with foreign keys resolved properly
        :rtype: ```ClassDef```
        """
        sqlalchemy_class_def.body = list(
            map(
                lambda outer_node: rewrite_fk(symbol_to_module, outer_node)
                if isinstance(outer_node, Assign)
                and isinstance(outer_node.value, Call)
                and isinstance(outer_node.value.func, Name)
                and outer_node.value.func.id == "Column"
                and any(
                    filter(
                        lambda node: isinstance(node, Call)
                        and isinstance(node.func, Name)
                        and node.func.id == "ForeignKey",
                        outer_node.value.args,
                    )
                )
                else outer_node,
                sqlalchemy_class_def.body,
            )
        )
        return sqlalchemy_class_def

    symbol2module = dict(
        chain.from_iterable(
            map(
                lambda import_from: map(
                    lambda _alias: (_alias.name, import_from.module), import_from.names
                ),
                filterfalse(
                    lambda import_from: import_from.module == "sqlalchemy",
                    filter(
                        rpartial(isinstance, ImportFrom),
                        ast.walk(mod),
                    ),
                ),
            )
        )
    )

    mod.body = list(
        map(
            lambda node: handle_sqlalchemy_cls(symbol2module, node)
            if isinstance(node, ClassDef)
            and any(
                filter(
                    lambda base: isinstance(base, Name) and base.id == "Base",
                    node.bases,
                )
            )
            else node,
            mod.body,
        )
    )

    with open(filename, "wt") as f:
        f.write(to_code(mod))


def rewrite_fk(symbol_to_module, column_assign):
    """
    Rewrite of the form:
    ```py
    column_name = Column(
            TableName0,
            ForeignKey("TableName0"),
            nullable=True,
        )
    ```
    To the following, inferring that the primary key field is `id` by resolving the symbol and `ast.parse`ing it:
    ```py
    column_name = Column(Integer, ForeignKey("table_name0.id"))
    ```

    :param symbol_to_module: Dictionary of symbol to module, like `{"join": "os.path"}`
    :type symbol_to_module: ```Dict[str,str]````

    :param column_assign: `column_name = Column()` in SQLalchemy with unresolved foreign key
    :type column_assign: ```Assign```d

    :return: `Assign()` in SQLalchemy with resolved foreign key
    :rtype: ```Assign```
    """
    assert (
        isinstance(column_assign.value, Call)
        and isinstance(column_assign.value.func, Name)
        and column_assign.value.func.id == "Column"
    )

    def rewrite_fk_from_import(column_name, foreign_key_call):
        """
        :param column_name: Field name
        :type column_name: ```Name```

        :param foreign_key_call: `ForeignKey` function call
        :type foreign_key_call: ```Call```

        :return:
        :rtype: ```Tuple[Name, Call]```
        """
        assert isinstance(column_name, Name)
        assert (
            isinstance(foreign_key_call, Call)
            and isinstance(foreign_key_call.func, Name)
            and foreign_key_call.func.id == "ForeignKey"
        )
        if column_name.id in symbol_to_module:
            with open(
                find_module_filepath(symbol_to_module[column_name.id], column_name.id),
                "rt",
            ) as f:
                mod = ast.parse(f.read())
            matching_class = next(
                filter(
                    lambda node: isinstance(node, ClassDef)
                    and node.name == column_name.id,
                    mod.body,
                )
            )
            pk_typ = get_pk_and_type(matching_class)
            assert pk_typ is not None
            pk, typ = pk_typ
            del pk_typ
            return Name(id=typ, ctx=Load()), Call(
                func=Name(id="ForeignKey", ctx=Load()),
                args=[set_value(".".join((get_table_name(matching_class), pk)))],
                keywords=[],
            )
        return column_name, foreign_key_call

    column_assign.value.args = list(
        chain.from_iterable(
            (
                rewrite_fk_from_import(*column_assign.value.args[:2]),
                column_assign.value.args[2:],
            )
        )
    )

    return column_assign


typ2column_type = {v: k for k, v in column_type2typ.items()}
typ2column_type.update(
    {
        "bool": "Boolean",
        "dict": "JSON",
        "float": "Float",
        "int": "Integer",
        "str": "String",
        "string": "String",
    }
)

__all__ = [
    "ensure_has_primary_key",
    "generate_repr_method",
    "param_to_sqlalchemy_column_call",
]
