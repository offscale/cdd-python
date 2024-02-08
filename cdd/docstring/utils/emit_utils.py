"""
Utility functions for `cdd.emit.docstring`
"""

import cdd.shared.ast_utils
from cdd.shared.defaults_utils import extract_default
from cdd.shared.pure_utils import simple_types, unquote


def interpolate_defaults(
    param, default_search_announce=None, require_default=False, emit_default_doc=True
):
    """
    Correctly set the 'default' and 'doc' parameters

    :param param: Name, dict with keys: 'typ', 'doc', 'default'
    :type param: ```tuple[str, dict]```

    :param default_search_announce: Default text(s) to look for. If None, uses default specified in default_utils.
    :type default_search_announce: ```Optional[Union[str, Iterable[str]]]```

    :param require_default: Whether a default is required, if not found in doc, infer the proper default from type
    :type require_default: ```bool```

    :param emit_default_doc: Whether help/docstring should include 'With default' text
    :type emit_default_doc: ```bool```

    :return: Name, dict with keys: 'typ', 'doc', 'default'
    :rtype: ```tuple[str, dict]```
    """
    name, _param = param
    del param
    if "doc" in _param:
        doc, default = extract_default(
            _param["doc"],
            typ=_param.get("typ"),
            default_search_announce=default_search_announce,
            emit_default_doc=emit_default_doc,
        )
        _param["doc"] = doc
        if default is not None:
            _param["default"] = unquote(default)
    if require_default and _param.get("default") is None:
        # if (
        #     "typ" in _param
        #     and _param["typ"] not in frozenset(("Any", "object"))
        #     and not _param["typ"].startswith("Optional")
        # ):
        #     _param["typ"] = "Optional[{}]".format(_param["typ"])
        _param["default"] = (
            simple_types[_param["typ"]]
            if _param.get("typ", memoryview) in simple_types
            else cdd.shared.ast_utils.NoneStr
        )

    return name, _param


__all__ = ["interpolate_defaults"]  # type: list[str]
