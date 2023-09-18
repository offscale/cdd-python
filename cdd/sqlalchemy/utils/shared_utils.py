"""
Shared utility functions for SQLalchemy
"""

import ast
from ast import Call, Expr, Load, Name, Subscript, Tuple, keyword
from operator import attrgetter

import cdd.compound.openapi.utils.emit_utils
from cdd.shared.ast_utils import NoneStr, get_value, set_value
from cdd.shared.pure_utils import PY_GTE_3_9, rpartial
from cdd.shared.source_transformer import to_code


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
    if _param["typ"] is None:
        return _param.get("default") == NoneStr
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
    elif _param["typ"].startswith("List["):
        after_generic = _param["typ"][len("List[") :]
        if "struct" in after_generic:  # "," in after_generic or
            name = Name(id="JSON", ctx=Load())
        else:
            list_typ = ast.parse(_param["typ"]).body[0]
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
            name = next(
                filter(rpartial(isinstance, Name), ast.walk(list_typ.value.slice)), None
            )
            assert name is not None, "Could not find a type in {!r}".format(
                to_code(list_typ.value.slice)
            )
        args.append(
            Call(
                func=Name(id="ARRAY", ctx=Load()),
                args=[
                    Name(
                        id=cdd.compound.openapi.utils.emit_utils.typ2column_type.get(
                            name.id, name.id
                        ),
                        ctx=Load(),
                    )
                ],
                keywords=[],
                expr=None,
                expr_func=None,
            )
        )
    elif (
        "items" in _param
        and _param["items"].get("type", False)
        in cdd.compound.openapi.utils.emit_utils.typ2column_type
    ):
        args.append(
            Call(
                func=Name(id="ARRAY", ctx=Load()),
                args=[
                    Name(
                        id=cdd.compound.openapi.utils.emit_utils.typ2column_type[
                            _param["items"]["type"]
                        ],
                        ctx=Load(),
                    )
                ],
                keywords=[],
                expr=None,
                expr_func=None,
            )
        )
    elif _param.get("typ").startswith("Union["):
        # Hack to remove the union type. Enum parse seems to be incorrect?
        union_typ = ast.parse(_param["typ"]).body[0]
        assert isinstance(
            union_typ.value, Subscript
        ), "Expected `Subscript` got `{type_name}`".format(
            type_name=type(union_typ.value).__name__
        )
        union_typ_tuple = (
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
        args.append(
            Name(
                cdd.compound.openapi.utils.emit_utils.typ2column_type.get(right, right)
                if left in cdd.compound.openapi.utils.emit_utils.typ2column_type
                else cdd.compound.openapi.utils.emit_utils.typ2column_type.get(
                    left, left
                ),
                Load(),
            )
        )
    else:
        type_name = (
            x_typ_sql["type"]
            if "type" in x_typ_sql
            else cdd.compound.openapi.utils.emit_utils.typ2column_type.get(
                _param["typ"], _param["typ"]
            )
        )
        args.append(
            Call(
                func=Name(type_name, Load()),
                args=list(map(set_value, x_typ_sql.get("type_args", iter(())))),
                keywords=[
                    keyword(arg=arg, value=set_value(val), identifier=None)
                    for arg, val in x_typ_sql.get("type_kwargs", dict()).items()
                ],
                expr=None,
                expr_func=None,
            )
            if "type_args" in x_typ_sql or "type_kwargs" in x_typ_sql
            else Name(type_name, Load())
        )
    return nullable


__all__ = ["update_args_infer_typ_sqlalchemy"]
