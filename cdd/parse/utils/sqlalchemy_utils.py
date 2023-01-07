"""
Utility functions for `cdd.parse.sqlalchemy`
"""

from ast import Call, Constant, Load, Name, Str
from itertools import chain

from cdd.ast_utils import column_type2typ, get_value
from cdd.source_transformer import to_code


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


__all__ = ["column_call_name_manipulator", "column_call_to_param"]
