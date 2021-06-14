"""
Functions which produce docstring portions from various inputs
"""

from collections import namedtuple
from textwrap import indent

from cdd.defaults_utils import set_default_doc
from cdd.pure_utils import fill, identity, indent_all_but_first, tab


def emit_param_str(
    param,
    style,
    purpose,
    emit_doc=True,
    emit_type=True,
    word_wrap=True,
    emit_default_doc=True,
):
    """
    Produce the docstring param/return lines

    :param param: Name, dict with keys: 'typ', 'doc', 'default'
    :type param: ```Tuple[str, dict]```

    :param style: the style of docstring
    :type style: ```Literal['rest', 'numpydoc', 'google']```

    :param purpose: Emit `:param` if purpose == 'function' elif purpose == 'class' then `:cvar` (ReST only)
    :type purpose: ```Literal['class', 'function']```

    :param emit_doc: Whether to emit the doc
    :type emit_doc: ```bool```

    :param emit_type: Whether to emit the type
    :type emit_type: ```bool```

    :param word_wrap: Whether to word-wrap. Set `DOCTRANS_LINE_LENGTH` to configure length.
    :type word_wrap: ```bool```

    :param emit_default_doc: Whether help/docstring should include 'With default' text
    :type emit_default_doc: ```bool```

    :returns: Newline joined pair of param, type
    :rtype: ```str```
    """
    name, _param = param
    del param

    _fill = fill if word_wrap else identity

    if style == "rest":
        emit_type &= purpose == "function"
        key, key_typ = (
            ("returns", "rtype")
            if name == "return_type"
            else (
                "{var} {name}".format(
                    var="param" if purpose == "function" else "cvar", name=name
                ),
                "type {name}".format(name=name),
            )
        )

        return "\n".join(
            map(
                indent_all_but_first,
                map(
                    _fill,
                    filter(
                        None,
                        (
                            ":{key}: {doc}".format(
                                key=key,
                                doc=set_default_doc(
                                    (name, _param), emit_default_doc=emit_default_doc
                                )[1]["doc"].lstrip(),
                            )
                            if emit_doc and _param.get("doc")
                            else None,
                            ":{key_typ}: ```{typ}```".format(
                                key_typ=key_typ, typ=_param["typ"]
                            )
                            if emit_type and _param.get("typ")
                            else None,
                        ),
                    ),
                ),
            )
        )
    elif style == "numpydoc":
        return "\n".join(
            filter(
                None,
                (
                    _fill(
                        (_param["typ"] if _param.get("typ") else None)
                        if name == "return_type"
                        else "{name} :{typ}".format(
                            name=name,
                            typ=" {typ}".format(typ=_param["typ"])
                            if _param.get("typ")
                            else "",
                        )
                    )
                    if emit_type and _param.get("typ")
                    else None,
                    _fill(
                        indent(
                            set_default_doc(
                                (name, _param), emit_default_doc=emit_default_doc
                            )[1]["doc"],
                            tab,
                        )
                    )
                    if emit_doc and _param.get("doc")
                    else None,
                ),
            )
        )
    else:
        return "".join(
            filter(
                None,
                (
                    (
                        "  {typ}:".format(typ=_param["typ"])
                        if _param.get("typ")
                        else None
                    )
                    if name == "return_type"
                    else "  {name} ({typ}): ".format(
                        name=name,
                        typ="{typ!s}".format(typ=_param["typ"])
                        if _param.get("typ")
                        else "",
                    )
                    if _param.get("typ")
                    else None,
                    "{nl}{tab}{doc}".format(
                        doc=set_default_doc(
                            (name, _param), emit_default_doc=emit_default_doc
                        )[1]["doc"],
                        **{"nl": "\n", "tab": " " * 3}
                        if name == "return_type"
                        else {"nl": "", "tab": ""}
                    )
                    if emit_doc and _param.get("doc")
                    else None,
                ),
            )
        )


Tokens = namedtuple("Tokens", ("rest", "google", "numpydoc"))
TOKENS = Tokens(
    (":param", ":cvar", ":ivar", ":var", ":type", ":return", ":rtype"),
    ("Args:", "Kwargs:", "Raises:", "Returns:"),
    ("Parameters\n" "----------", "Returns\n" "-------"),
)
ARG_TOKENS = Tokens(
    TOKENS.rest[:-2],
    (TOKENS.google[0],),
    (TOKENS.numpydoc[0],),
)
RETURN_TOKENS = Tokens(TOKENS.rest[-2:], (TOKENS.google[-1],), (TOKENS.numpydoc[-1],))

__all__ = ["emit_param_str", "ARG_TOKENS", "TOKENS"]
