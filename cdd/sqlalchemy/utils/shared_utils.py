"""
Shared utility functions for SQLalchemy
"""

import ast
from ast import Call, Expr, Load, Name, Subscript, Tuple, expr, keyword
from operator import attrgetter
from typing import Optional, cast

import cdd.compound.openapi.utils.emit_utils
import cdd.shared.ast_utils
import cdd.shared.source_transformer
import cdd.sqlalchemy.utils.emit_utils
from cdd.shared.pure_utils import PY_GTE_3_9, rpartial


def _update_args_infer_typ_sqlalchemy_for_scalar(_param, args, x_typ_sql):
    """
    Modify `args` list with the inferred SQLalchemy type for a single scalar

    :param _param: Param with typ
    :type _param: ```dict```

    :param args:
    :type args: ```list```

    :param x_typ_sql:
    :type x_typ_sql: ```dict```
    """
    type_name: str = (
        x_typ_sql["type"]
        if "type" in x_typ_sql
        else cdd.sqlalchemy.utils.emit_utils.typ2column_type.get(
            _param["typ"], _param["typ"]
        )
    )
    args.append(
        Call(
            func=Name(type_name, Load(), lineno=None, col_offset=None),
            args=list(
                map(
                    cdd.shared.ast_utils.set_value, x_typ_sql.get("type_args", iter(()))
                )
            ),
            keywords=[
                keyword(
                    arg=arg, value=cdd.shared.ast_utils.set_value(val), identifier=None
                )
                for arg, val in x_typ_sql.get("type_kwargs", dict()).items()
            ],
            expr=None,
            expr_func=None,
            lineno=None,
            col_offset=None,
        )
        if "type_args" in x_typ_sql or "type_kwargs" in x_typ_sql
        else Name(type_name, Load(), lineno=None, col_offset=None)
    )


def update_args_infer_typ_sqlalchemy(_param, args, name, nullable, x_typ_sql):
    """
    :param _param: Param with typ
    :type _param: ```dict```

    :param args:
    :type args: ```list```

    :param name:
    :type name: ```str```

    :param nullable: Whether it is NULL-able
    :type nullable: ```Optional[bool]```

    :param x_typ_sql:
    :type x_typ_sql: ```dict```

    :return: Whether the type is nullable, possibly a list/tuple of types to generate columns for
    :rtype: ```Tuple[bool, Optional[Union[List[AST], Tuple[AST]]]]```
    """
    if _param["typ"] is None:
        return _param.get("default") == cdd.shared.ast_utils.NoneStr, None
    elif _param["typ"].startswith("Optional["):
        _param["typ"] = _param["typ"][len("Optional[") : -1]
        nullable: bool = True
    if "Literal[" in _param["typ"]:
        parsed_typ: Call = cast(
            Call, cdd.shared.ast_utils.get_value(ast.parse(_param["typ"]).body[0])
        )
        assert parsed_typ.value.id == "Literal", "Expected `Literal` got: {!r}".format(
            parsed_typ.value.id
        )
        val = cdd.shared.ast_utils.get_value(parsed_typ.slice)
        (
            args.append(
                Call(
                    func=Name("Enum", Load(), lineno=None, col_offset=None),
                    args=val.elts,
                    keywords=[
                        ast.keyword(
                            arg="name",
                            value=cdd.shared.ast_utils.set_value(name),
                            identifier=None,
                        )
                    ],
                    expr=None,
                    expr_func=None,
                    lineno=None,
                    col_offset=None,
                )
            )
            if hasattr(val, "elts")
            else _update_args_infer_typ_sqlalchemy_for_scalar(_param, args, x_typ_sql)
        )
    elif _param["typ"].startswith("List["):
        after_generic: str = _param["typ"][len("List[") :]
        if "struct" in after_generic:  # "," in after_generic or
            name: Name = Name(id="JSON", ctx=Load(), lineno=None, col_offset=None)
        else:
            list_typ: Expr = cast(Expr, ast.parse(_param["typ"]).body[0])
            assert isinstance(
                list_typ, Expr
            ), "Expected `Expr` got `{type_name}`".format(
                type_name=type(list_typ).__name__
            )
            assert isinstance(
                list_typ.value, Subscript
            ), "Expected `Subscript` got `{type_name}`".format(
                type_name=type(list_typ.value).__name__
            )
            name: Optional[Name] = next(
                filter(rpartial(isinstance, Name), ast.walk(list_typ.value.slice)), None
            )
            assert name is not None, "Could not find a type in {!r}".format(
                cdd.shared.source_transformer.to_code(list_typ.value.slice)
            )
        args.append(
            Call(
                func=Name(id="ARRAY", ctx=Load(), lineno=None, col_offset=None),
                args=[
                    Name(
                        id=cdd.sqlalchemy.utils.emit_utils.typ2column_type.get(
                            name.id, name.id
                        ),
                        ctx=Load(),
                    )
                ],
                keywords=[],
                expr=None,
                expr_func=None,
                lineno=None,
                col_offset=None,
            )
        )
    elif (
        "items" in _param
        and _param["items"].get("type", False)
        in cdd.sqlalchemy.utils.emit_utils.typ2column_type
    ):
        args.append(
            Call(
                func=Name(id="ARRAY", ctx=Load(), lineno=None, col_offset=None),
                args=[
                    Name(
                        id=cdd.sqlalchemy.utils.emit_utils.typ2column_type[
                            _param["items"]["type"]
                        ],
                        ctx=Load(),
                    )
                ],
                keywords=[],
                expr=None,
                expr_func=None,
                lineno=None,
                col_offset=None,
            )
        )
    elif _param.get("typ").startswith("Union["):
        args.append(_handle_union_of_length_2(_param["typ"]))
    else:
        _update_args_infer_typ_sqlalchemy_for_scalar(_param, args, x_typ_sql)
    return nullable, None


def _handle_union_of_length_2(typ):
    """
    Internal function to turn `str` to `Name`

    :param typ: `str` which evaluates to `ast.Subscript`
    :type typ: ```str```

    :return: Parsed out name
    :rtype: ```Name```
    """
    # Hack to remove the union type. Enum parse seems to be incorrect?
    union_typ: Subscript = cast(Subscript, ast.parse(typ).body[0])
    assert isinstance(
        union_typ.value, Subscript
    ), "Expected `Subscript` got `{type_name}`".format(
        type_name=type(union_typ.value).__name__
    )
    union_typ_tuple: expr = (
        union_typ.value.slice if PY_GTE_3_9 else union_typ.value.slice.value
    )
    assert isinstance(
        union_typ_tuple, Tuple
    ), "Expected `Tuple` got `{type_name}`".format(
        type_name=type(union_typ_tuple).__name__
    )
    assert (
        len(union_typ_tuple.elts) == 2
    ), "Expected length of 2 got `{tuple_len}`".format(
        tuple_len=len(union_typ_tuple.elts)
    )
    left, right = map(attrgetter("id"), union_typ_tuple.elts)
    return Name(
        (
            cdd.sqlalchemy.utils.emit_utils.typ2column_type[right]
            if right in cdd.sqlalchemy.utils.emit_utils.typ2column_type
            else cdd.sqlalchemy.utils.emit_utils.typ2column_type.get(left, left)
        ),
        Load(),
        lineno=None,
        col_offset=None,
    )


__all__ = ["update_args_infer_typ_sqlalchemy"]
