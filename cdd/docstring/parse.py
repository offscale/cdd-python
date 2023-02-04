"""
Docstring parser
"""

from cdd.shared.docstring_parsers import parse_docstring


def docstring(
    doc_string,
    infer_type=False,
    return_tuple=False,
    parse_original_whitespace=False,
    emit_default_prop=True,
    emit_default_doc=True,
):
    """
    Converts a docstring to an AST

    :param doc_string: docstring portion
    :type doc_string: ```Union[str, Dict]```

    :param infer_type: Whether to try inferring the typ (from the default)
    :type infer_type: ```bool```

    :param return_tuple: Whether to return a tuple, or just the intermediate_repr
    :type return_tuple: ```bool```

    :param parse_original_whitespace: Whether to parse original whitespace or strip it out
    :type parse_original_whitespace: ```bool```

    :param emit_default_prop: Whether to include the default dictionary property.
    :type emit_default_prop: ```bool```

    :param emit_default_doc: Whether help/docstring should include 'With default' text
    :type emit_default_doc: ```bool```

    :return: intermediate_repr, whether it returns or not
    :rtype: ```Optional[Union[dict, Tuple[dict, bool]]]```
    """
    assert isinstance(
        doc_string, str
    ), "Expected `str` got `{doc_string_type!r}`".format(
        doc_string_type=type(doc_string).__name__
    )
    parsed = (
        doc_string
        if isinstance(doc_string, dict)
        else parse_docstring(
            doc_string,
            infer_type=infer_type,
            emit_default_prop=emit_default_prop,
            emit_default_doc=emit_default_doc,
            parse_original_whitespace=parse_original_whitespace,
        )
    )

    if return_tuple:
        return parsed, (
            "returns" in parsed
            and parsed["returns"] is not None
            and "return_type" in (parsed.get("returns") or iter(()))
        )

    return parsed


__all__ = ["docstring"]
