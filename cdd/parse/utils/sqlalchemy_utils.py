"""
Utility functions for `cdd.parse.sqlalchemy`
"""

from ast import Call, Name
from itertools import chain

from cdd.ast_utils import column_type2typ, get_value


def column_call_to_param(call):
    """
    Parse column call `Call(func=Name("Column", Load(), …)` into param

    :param call: Column call from SQLAlchemy `Table` construction
    :type call: ```Call```

    :return: Name, dict with keys: 'typ', 'doc', 'default'
    :rtype: ```Tuple[str, dict]```
    """
    assert call.func.id == "Column"
    assert len(call.args) == 2

    _param = dict(
        chain.from_iterable(
            filter(
                None,
                (
                    map(
                        lambda key_word: (key_word.arg, get_value(key_word.value)),
                        call.keywords,
                    ),
                    (("typ", column_type2typ[call.args[1].id]),)
                    if isinstance(call.args[1], Name)
                    else None,
                ),
            )
        )
    )

    if (
        isinstance(call.args[1], Call)
        and call.args[1].func.id.rpartition(".")[2] == "Enum"
    ):
        _param["typ"] = "Literal{}".format(list(map(get_value, call.args[1].args)))

    pk = "primary_key" in _param
    if pk:
        _param["doc"] = "[PK] {}".format(_param["doc"])
        del _param["primary_key"]

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


__all__ = ["column_call_to_param"]
