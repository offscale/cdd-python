"""
Functions which produce docstring portions from various inputs
"""

from doctrans.defaults_utils import set_default_doc
from doctrans.pure_utils import fill, identity


def emit_param_str(
    param, style, emit_doc=True, emit_type=True, word_wrap=True, emit_default_doc=True
):
    """
    Produce the docstring param/return lines

    :param param: Name, dict with keys: 'typ', 'doc', 'default'
    :type param: ```Tuple[str, dict]```

    :param emit_doc: Whether to emit the doc
    :type emit_doc: ```bool```

    :param emit_type: Whether to emit the type
    :type emit_type: ```bool```

    :param style: the style of docstring
    :type style: ```Literal['rest', 'numpy', 'google']```

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
        key, key_typ = (
            ("returns", "rtype")
            if name == "return_type"
            else ("param {name}".format(name=name), "type {name}".format(name=name))
        )

        return "\n".join(
            filter(
                None,
                (
                    _fill(
                        ":{key}: {doc}".format(
                            key=key,
                            doc=set_default_doc(
                                (name, _param), emit_default_doc=emit_default_doc
                            )[1]["doc"],
                        )
                    )
                    if emit_doc and _param.get("doc")
                    else None,
                    _fill(
                        ":{key_typ}: ```{typ}```".format(
                            key_typ=key_typ, typ=_param["typ"]
                        )
                    )
                    if emit_type and _param.get("typ")
                    else None,
                ),
            )
        )
    # else:
    #     raise NotImplementedError(style)


__all__ = ["emit_param_str"]
