"""
Functions which produce intermediate_repr from various different inputs
"""

import ast
from importlib import import_module


def get_internal_body(target_name, target_type, intermediate_repr):
    """
    Get the internal body from our IR

    :param target_name: name of target. If both `target_name` and `target_type` match internal body extract, then emit
    :type target_name: ```str```

    :param target_type: Type of target, static is static or global method, others just become first arg
    :type target_type: ```Literal['self', 'cls', 'static']```

    :param intermediate_repr: a dictionary of form
        {  "name": Optional[str],
           "type": Optional[str],
           "_internal": {'body': List[ast.AST]},
           "doc": Optional[str],
           "params": OrderedDict[str, {'typ': str, 'doc': Optional[str], 'default': Any}]
           "returns": Optional[OrderedDict[Literal['return_type'],
                                           {'typ': str, 'doc': Optional[str], 'default': Any}),)]] }
    :type intermediate_repr: ```dict```

    :return: Internal body or an empty tuple
    :rtype: ```Union[list, tuple]```
    """
    return (
        intermediate_repr["_internal"]["body"]
        if intermediate_repr.get("_internal", {}).get("body")
        and intermediate_repr["_internal"]["from_name"] == target_name
        and intermediate_repr["_internal"]["from_type"] == target_type
        else tuple()
    )


def ast_parse_fix(s):
    """
    Hack to resolve unbalanced parentheses SyntaxError acquired from PyTorch parsing
    TODO: remove

    :param s: String to parse
    :type s: ```str```

    :return: Value
    """
    balanced = (s.count("[") + s.count("]")) & 1 == 0
    return ast.parse(s if balanced else "{}]".format(s)).body[0].value


# def normalise_intermediate_representation(intermediate_repr):
#     """
#     Normalise the intermediate representation. Performs:
#     - Move header and footer of docstring to same place—and with same whitespace—as original docstring
#
#     :param intermediate_repr: a dictionary of form
#         {  "name": Optional[str],
#            "type": Optional[str],
#            "doc": Optional[str],
#            "params": OrderedDict[str, {'typ': str, 'doc': Optional[str], 'default': Any}]
#            "returns": Optional[OrderedDict[Literal['return_type'],
#                                            {'typ': str, 'doc': Optional[str], 'default': Any}),)]] }
#     :type intermediate_repr: ```dict```
#
#     :return: a dictionary of form
#         {  "name": Optional[str],
#            "type": Optional[str],
#            "doc": Optional[str],
#            "params": OrderedDict[str, {'typ': str, 'doc': Optional[str], 'default': Any}]
#            "returns": Optional[OrderedDict[Literal['return_type'],
#                                            {'typ': str, 'doc': Optional[str], 'default': Any}),)]] }
#     :rtype: ```dict```
#     """
#     current_doc_str = intermediate_repr["doc"]
#     original_doc_str = intermediate_repr.get("_internal", {"original_doc_str": None})[
#         "original_doc_str"
#     ]
#     intermediate_repr["doc"] = ensure_doc_args_whence_original(
#         current_doc_str=current_doc_str, original_doc_str=original_doc_str
#     )
#     return intermediate_repr


def get_emitter(emit_name):
    """
    Get emiter function specialised for output `node`

    :param emit_name: Which type to emit.
    :type emit_name: ```Literal["argparse", "class", "function", "json_schema",
                                 "pydantic", "sqlalchemy", "sqlalchemy_table", "sqlalchemy_hybrid"]```

    :return: Function which returns intermediate_repr
    :rtype: ```Callable[[...], dict]````
    """
    emit_name = {"class": "class_"}.get(emit_name, emit_name)
    return getattr(
        import_module(
            ".".join(
                (
                    "cdd",
                    "sqlalchemy"
                    if emit_name in frozenset(("sqlalchemy_hybrid", "sqlalchemy_table"))
                    else emit_name,
                    "emit",
                )
            )
        ),
        emit_name,
    )


__all__ = [
    "ast_parse_fix",
    "get_internal_body",
    "get_emitter"
    # "normalise_intermediate_representation",
]
