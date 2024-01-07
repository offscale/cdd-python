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
    Compare,
    DictComp,
    Expr,
    FunctionDef,
    ImportFrom,
    IsNot,
    Load,
    Module,
    Name,
    Return,
    Store,
    Tuple,
    alias,
    arguments,
    comprehension,
    keyword,
)
from collections import OrderedDict, deque
from functools import partial
from itertools import chain, filterfalse
from json import dumps
from operator import attrgetter, eq, methodcaller
from os import path
from platform import system
from typing import Any, Dict, List, Optional

import cdd.sqlalchemy.utils.shared_utils
from cdd.shared.ast_utils import (
    NoneStr,
    get_value,
    maybe_type_comment,
    set_arg,
    set_value,
)
from cdd.shared.pure_utils import (
    find_module_filepath,
    namespaced_upper_camelcase_to_pascal,
    none_types,
    rpartial,
    tab,
)
from cdd.shared.source_transformer import to_code
from cdd.shared.types import ParamVal
from cdd.sqlalchemy.utils.parse_utils import (
    column_type2typ,
    get_pk_and_type,
    get_table_name,
    sqlalchemy_top_level_imports,
)
from cdd.tests.mocks.docstrings import (
    docstring_create_from_attr_google_str,
    docstring_create_from_attr_str,
    docstring_repr_google_str,
    docstring_repr_str,
)


def param_to_sqlalchemy_column_call(name_param, include_name):
    """
    Turn a param into a `Column(…)`

    :param name_param: Name, dict with keys: 'typ', 'doc', 'default'
    :type name_param: ```tuple[str, dict]```

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

    x_typ_sql = _param.get("x_typ", {}).get("sql", {})  # type: dict

    if "typ" in _param:
        nullable: bool = (
            cdd.sqlalchemy.utils.shared_utils.update_args_infer_typ_sqlalchemy(
                _param, args, name, nullable, x_typ_sql
            )
        )

    default = x_typ_sql.get("default", _param.get("default", ast))
    has_default: bool = default is not ast
    pk: bool = _param.get("doc", "").startswith("[PK]")
    fk: bool = _param.get("doc", "").startswith("[FK")
    if pk:
        _param["doc"] = _param["doc"][4:].lstrip()
        keywords.append(
            ast.keyword(arg="primary_key", value=set_value(True), identifier=None),
        )
    elif fk:
        end: int = _param["doc"].find("]") + 1
        fk_val: str = _param["doc"][len("[FK(") : end - len(")]")]
        _param["doc"] = _param["doc"][end:].lstrip()
        args.append(
            Call(
                func=Name(id="ForeignKey", ctx=Load(), lineno=None, col_offset=None),
                args=[set_value(fk_val)],
                keywords=[],
                lineno=None,
                col_offset=None,
            )
        )
    elif has_default and default not in none_types:
        nullable: bool = False

    rstripped_dot_doc: str = _param.get("doc", "").rstrip(".")
    doc_added_at: Optional[int] = None
    if rstripped_dot_doc:
        doc_added_at: int = len(keywords)
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

    # TODO: Maybe `CREATE TYPE` and use that?
    if _param.get("typ") == "dict" and "ir" in _param:
        keywords.append(
            ast.keyword(
                arg="comment",
                value=set_value("[schema={}]".format(dumps(_param["ir"]))),
                identifier=None,
            )
        )

    if has_default:
        # if default == NoneStr: default = None
        keywords.append(
            ast.keyword(
                arg="default",
                value=(
                    default
                    if isinstance(default, AST)
                    else set_value(None if default == NoneStr else default)
                ),
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
        func=Name("Column", Load(), lineno=None, col_offset=None),
        args=args,
        keywords=sorted(keywords, key=attrgetter("arg")),
        expr=None,
        expr_func=None,
        lineno=None,
        col_offset=None,
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
    keys = tuple(params.keys())  # type: tuple[str, ...]
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
                ),
                lineno=None,
                col_offset=None,
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
                        lineno=None,
                        col_offset=None,
                    ),
                    args=[],
                    keywords=list(
                        map(
                            lambda key: ast.keyword(
                                arg=key,
                                value=Attribute(
                                    Name("self", Load(), lineno=None, col_offset=None),
                                    key,
                                    Load(),
                                    lineno=None,
                                    col_offset=None,
                                ),
                                identifier=None,
                            ),
                            keys,
                        )
                    ),
                    expr=None,
                    expr_func=None,
                    lineno=None,
                    col_offset=None,
                ),
                expr=None,
            ),
        ],
        decorator_list=[],
        type_params=[],
        arguments_args=None,
        identifier_name=None,
        stmt=None,
        lineno=None,
        returns=None,
        **maybe_type_comment,
    )


def generate_create_from_attr_staticmethod(params, cls_name, docstring_format):
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
    keys = tuple(params.keys())  # type: tuple[str, ...]
    return FunctionDef(
        name="create_from_attr",
        args=arguments(
            posonlyargs=[],
            arg=None,
            args=[set_arg("record")],
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
                            docstring_create_from_attr_str
                            if docstring_format == "rest"
                            else docstring_create_from_attr_google_str
                        )
                        .replace("self", cls_name)
                        .lstrip(),
                    )
                ),
                lineno=None,
                col_offset=None,
            ),
            Return(
                value=Call(
                    func=Name(id=cls_name, ctx=Load(), lineno=None, col_offset=None),
                    args=[],
                    keywords=[
                        keyword(
                            arg=None,
                            value=DictComp(
                                key=Name(
                                    id="attr", ctx=Load(), lineno=None, col_offset=None
                                ),
                                value=Call(
                                    func=Name(
                                        id="getattr",
                                        ctx=Load(),
                                        lineno=None,
                                        col_offset=None,
                                    ),
                                    args=[
                                        Name(
                                            id="node",
                                            ctx=Load(),
                                            lineno=None,
                                            col_offset=None,
                                        ),
                                        Name(
                                            id="attr",
                                            ctx=Load(),
                                            lineno=None,
                                            col_offset=None,
                                        ),
                                    ],
                                    keywords=[],
                                    lineno=None,
                                    col_offset=None,
                                ),
                                generators=[
                                    comprehension(
                                        target=Name(
                                            id="attr",
                                            ctx=Store(),
                                            lineno=None,
                                            col_offset=None,
                                        ),
                                        iter=Tuple(
                                            elts=list(map(set_value, keys)),
                                            ctx=Load(),
                                            lineno=None,
                                            col_offset=None,
                                        ),
                                        ifs=[
                                            Compare(
                                                left=Call(
                                                    func=Name(
                                                        id="getattr",
                                                        ctx=Load(),
                                                        lineno=None,
                                                        col_offset=None,
                                                    ),
                                                    args=[
                                                        Name(
                                                            id="node",
                                                            ctx=Load(),
                                                            lineno=None,
                                                            col_offset=None,
                                                        ),
                                                        Name(
                                                            id="attr",
                                                            ctx=Load(),
                                                            lineno=None,
                                                            col_offset=None,
                                                        ),
                                                        set_value(None),
                                                    ],
                                                    keywords=[],
                                                    lineno=None,
                                                    col_offset=None,
                                                ),
                                                ops=[IsNot()],
                                                comparators=[set_value(None)],
                                                lineno=None,
                                                col_offset=None,
                                            )
                                        ],
                                        is_async=0,
                                    )
                                ],
                                lineno=None,
                                col_offset=None,
                            ),
                            identifier=None,
                        )
                    ],
                    expr=None,
                    expr_func=None,
                    lineno=None,
                    col_offset=None,
                ),
                expr=None,
            ),
        ],
        decorator_list=[
            Name(id="staticmethod", ctx=Load(), lineno=None, col_offset=None)
        ],
        arguments_args=None,
        identifier_name=None,
        stmt=None,
        lineno=None,
        returns=None,
        **maybe_type_comment,
    )


def ensure_has_primary_key(intermediate_repr, force_pk_id=False):
    """
    Add a primary key to the input (if nonexistent) then return the input.

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

    :param force_pk_id: Whether to force primary_key to be named `id` (if there isn't already a primary_key)
    :type force_pk_id: ```bool```

    :return: a dictionary consistent with `IntermediateRepr`, defined as:
        ParamVal = TypedDict("ParamVal", {"typ": str, "doc": Optional[str], "default": Any})
        IntermediateRepr = TypedDict("IntermediateRepr", {
            "name": Optional[str],
            "type": Optional[str],
            "doc": Optional[str],
            "params": OrderedDict[str, ParamVal],
            "returns": Optional[OrderedDict[Literal["return_type"], ParamVal]],
        })
    :rtype: ```dict```
    """
    params: OrderedDict[str, ParamVal] = (
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
        candidate_pks: List[str] = []
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
        if not force_pk_id and len(candidate_pks) == 1:
            params[candidate_pks[0]]["doc"] = (
                "[PK] {}".format(params[candidate_pks[0]]["doc"])
                if params[candidate_pks[0]].get("doc")
                else "[PK]"
            )
        elif "id" in intermediate_repr.get("params", iter(())):
            params["id"]["doc"] = (
                "[PK] {}".format(params["id"]["doc"])
                if params["id"].get("doc")
                else "[PK]"
            )
        else:
            assert "id" not in intermediate_repr.get(
                "params", iter(())
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
                                lineno=None,
                                col_offset=None,
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
        mod: Module = ast.parse(f.read())

    candidates = sorted(
        frozenset(
            filter(
                str.istitle,
                filterfalse(
                    frozenset(
                        ("complex", "float", "int", "list", "long", "self", "string")
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

    module: str = path.basename(path.dirname(filename))
    mod.body = list(
        chain.from_iterable(
            (
                map(
                    lambda class_name: ImportFrom(
                        module=".".join(
                            (module, namespaced_upper_camelcase_to_pascal(class_name))
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
        mod: Module = ast.parse(f.read())

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
                lambda outer_node: (
                    rewrite_fk(symbol_to_module, outer_node)
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
                    else outer_node
                ),
                sqlalchemy_class_def.body,
            )
        )
        return sqlalchemy_class_def

    symbol2module: Dict[str, Any] = dict(
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
            lambda node: (
                handle_sqlalchemy_cls(symbol2module, node)
                if isinstance(node, ClassDef)
                and any(
                    filter(
                        lambda base: isinstance(base, Name) and base.id == "Base",
                        node.bases,
                    )
                )
                else node
            ),
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
    :type column_assign: ```Assign```

    :return: `Assign()` in SQLalchemy with resolved foreign key
    :rtype: ```Assign```
    """
    assert (
        isinstance(column_assign.value, Call)
        and isinstance(column_assign.value.func, Name)
        and column_assign.value.func.id == "Column"
    ), 'Expected `Call.func.Name.id` of "<var> = Column" eval to `<var> = Column(...)` got `{code}`'.format(
        code=to_code(column_assign).rstrip()
    )

    def rewrite_fk_from_import(column_name, foreign_key_call):
        """
        :param column_name: Field name
        :type column_name: ```Name```

        :param foreign_key_call: `ForeignKey` function call
        :type foreign_key_call: ```Call```

        :return:
        :rtype: ```tuple[Name, Call]```
        """
        assert isinstance(
            column_name, Name
        ), "Expected `Name` got `{type_name}`".format(
            type_name=type(column_name).__name__
        )
        assert (
            isinstance(foreign_key_call, Call)
            and isinstance(foreign_key_call.func, Name)
            and foreign_key_call.func.id == "ForeignKey"
        ), 'Expected `Call.func.Name.id` of "ForeignKey" eval to `ForeignKey(...)` got `{code}`'.format(
            code=to_code(foreign_key_call).rstrip()
        )
        if column_name.id in symbol_to_module:
            with open(
                find_module_filepath(symbol_to_module[column_name.id], column_name.id),
                "rt",
            ) as f:
                mod: Module = ast.parse(f.read())
            matching_class: ClassDef = next(
                filter(
                    lambda node: isinstance(node, ClassDef)
                    and node.name == column_name.id,
                    mod.body,
                )
            )
            pk_typ = get_pk_and_type(matching_class)  # type: tuple[str, str]
            assert pk_typ is not None
            pk, typ = pk_typ
            del pk_typ
            return Name(id=typ, ctx=Load(), lineno=None, col_offset=None), Call(
                func=Name(id="ForeignKey", ctx=Load(), lineno=None, col_offset=None),
                args=[set_value(".".join((get_table_name(matching_class), pk)))],
                keywords=[],
                lineno=None,
                col_offset=None,
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


def sqlalchemy_class_to_table(class_def, parse_original_whitespace):
    """
    Convert SQLalchemy class to SQLalchemy Table expression

    :param class_def: A class inheriting from declarative `Base`, where `Base = sqlalchemy.orm.declarative_base()`
    :type class_def: ```ClassDef```

    :param parse_original_whitespace: Whether to parse original whitespace or strip it out
    :type parse_original_whitespace: ```bool```

    :return: SQLalchemy `Table` expression
    :rtype: ```Call```
    """
    assert isinstance(
        class_def, ClassDef
    ), "Expected `ClassDef` got `{type_name}`".format(
        type_name=type(class_def).__name__
    )

    # Hybrid SQLalchemy class/table handler
    table_dunder: Optional[Call] = next(
        filter(
            lambda assign: any(
                filter(
                    partial(eq, "__table__"),
                    map(attrgetter("id"), assign.targets),
                )
            ),
            filter(rpartial(isinstance, Assign), class_def.body),
        ),
        None,
    )
    if table_dunder is not None:
        return table_dunder

    # Parse into the same format that `sqlalchemy_table` can read, then return with a call to it

    name: str = get_value(
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
    doc_string: Optional[str] = ast.get_docstring(
        class_def, clean=parse_original_whitespace
    )

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

    return Call(
        func=Name("Table", Load(), lineno=None, col_offset=None),
        args=list(
            chain.from_iterable(
                (
                    (set_value(name), Name("metadata_obj", Load())),
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
        keywords=(
            []
            if doc_string is None
            else [keyword(arg="comment", value=set_value(doc_string), identifier=None)]
        ),
        expr=None,
        expr_func=None,
        lineno=None,
        col_offset=None,
    )


def sqlalchemy_table_to_class(table_expr_ass):
    """Convert `table_name = Table(column_name)` to `class table_name(Base): column_name"""
    assert isinstance(
        table_expr_ass, Assign
    ), "Expected `Assign` got `{type_name}`".format(
        type_name=type(table_expr_ass).__name__
    )
    assert len(table_expr_ass.targets) == 1 and isinstance(
        table_expr_ass.targets[0], Name
    )
    assert len(table_expr_ass.value.args) > 1

    return ClassDef(
        name=table_expr_ass.targets[0].id,
        bases=[Name("Base", Load(), lineno=None, col_offset=None)],
        keywords=[],
        body=list(
            chain.from_iterable(
                (
                    (
                        Assign(
                            targets=[
                                Name(
                                    "__tablename__",
                                    Store(),
                                    lineno=None,
                                    col_offset=None,
                                )
                            ],
                            value=set_value(get_value(table_expr_ass.value.args[0])),
                            expr=None,
                            lineno=None,
                            **maybe_type_comment,
                        ),
                    ),
                    map(
                        lambda column_call: Assign(
                            targets=[
                                Name(
                                    get_value(column_call.args[0]),
                                    Store(),
                                    lineno=None,
                                    col_offset=None,
                                )
                            ],
                            value=Call(
                                func=column_call.func,
                                args=(
                                    column_call.args[1:]
                                    if len(column_call.args) > 1
                                    else []
                                ),
                                keywords=column_call.keywords,
                                expr=None,
                                expr_func=None,
                            ),
                            expr=None,
                            lineno=None,
                            **maybe_type_comment,
                        ),
                        filter(
                            lambda node: isinstance(node, Call)
                            and isinstance(node.func, Name)
                            and node.func.id == "Column",
                            table_expr_ass.value.args[2:],
                        ),
                    ),
                )
            )
        ),
        decorator_list=[],
        type_params=[],
        expr=None,
        lineno=None,
        col_offset=None,
        end_lineno=None,
        end_col_offset=None,
        identifier_name=None,
    )


typ2column_type: Dict[str, str] = {v: k for k, v in column_type2typ.items()}
typ2column_type.update(
    {
        "bool": "Boolean",
        "dict": "JSON",
        "float": "Float",
        "int": "Integer",
        "str": "String",
        "string": "String",
        "int64": "BigInteger",
        "Optional[dict]": "JSON",
    }
)

__all__ = [
    "ensure_has_primary_key",
    "generate_create_from_attr_staticmethod",
    "generate_repr_method",
    "param_to_sqlalchemy_column_call",
    "rewrite_fk",
    "sqlalchemy_class_to_table",
    "sqlalchemy_table_to_class",
    "typ2column_type",
    "update_fk_for_file",
    "update_with_imports_from_columns",
]  # type: list[str]
