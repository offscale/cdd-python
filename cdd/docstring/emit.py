"""
Docstring emitter.

Emits into these formats from the cdd_python common IR format:
 - [ReST docstring format (Sphinx)](
  https://sphinx-rtd-tutorial.readthedocs.io/en/latest/docstrings.html#the-sphinx-docstring-format)
 - [numpydoc docstring format](https://numpydoc.readthedocs.io/en/latest/format.html)
 - [Google's docstring format](https://google.github.io/styleguide/pyguide.html)
"""

from collections import OrderedDict
from functools import partial

from cdd.shared.docstring_utils import (
    ARG_TOKENS,
    RETURN_TOKENS,
    emit_param_str,
    header_args_footer_to_str,
    parse_docstring_into_header_args_footer,
)
from cdd.shared.pure_utils import num_of_nls, tab


def docstring(
    intermediate_repr,
    docstring_format="rest",
    purpose="function",
    word_wrap=True,
    indent_level=0,
    emit_separating_tab=True,
    emit_types=True,
    emit_original_whitespace=False,
    emit_default_doc=True,
):
    """
    Converts an IR to a docstring

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

    :param docstring_format: Format of docstring
    :type docstring_format: ```Literal['rest', 'numpydoc', 'google']```

    :param purpose: Emit `:param` if purpose == 'function' elif purpose == 'class' then `:cvar`
    :type purpose: ```Literal['class', 'function']```

    :param word_wrap: Whether to word-wrap. Set `DOCTRANS_LINE_LENGTH` to configure length.
    :type word_wrap: ```bool```

    :param indent_level: indentation level whence: 0=no_tabs, 1=one tab; 2=two tabs
    :type indent_level: ```int```

    :param emit_separating_tab: Whether to put a tab between :param and return and desc
    :type emit_separating_tab: ```bool```

    :param emit_types: Whether to show `:type` lines
    :type emit_types: ```bool```

    :param emit_original_whitespace: Whether to emit original whitespace or strip it out
    :type emit_original_whitespace: ```bool```

    :param emit_default_doc: Whether help/docstring should include 'With default' text
    :type emit_default_doc: ```bool```

    :return: docstring
    :rtype: ```str```
    """
    # _sep = tab * indent_level
    params = "\n{maybe_nl}".format(
        maybe_nl="\n" if docstring_format == "rest" and purpose != "class" else ""
    ).join(
        (
            lambda param_lines: (
                [getattr(ARG_TOKENS, docstring_format)[0]] + param_lines
                if param_lines and docstring_format != "rest"
                else param_lines
            )
        )(
            list(
                map(
                    partial(
                        emit_param_str,
                        style=docstring_format,
                        purpose=purpose,
                        emit_type=emit_types,
                        emit_default_doc=emit_default_doc,
                        word_wrap=word_wrap,
                    ),
                    (intermediate_repr["params"] or OrderedDict()).items(),
                ),
            )
        )
    )

    returns = (
        (
            lambda line_: (
                "".join(
                    "{maybe_nl0_and_token}{maybe_nl1}{returns_doc}".format(
                        maybe_nl0_and_token=(
                            ""
                            if docstring_format == "rest"
                            else "\n{return_token}".format(
                                return_token=getattr(RETURN_TOKENS, docstring_format)[0]
                            )
                        ),
                        maybe_nl1="" if not params or params[-1] == "\n" else "\n",
                        returns_doc=line_,
                    )
                )
                if line_
                else ""
            )
        )(
            next(
                map(
                    partial(
                        emit_param_str,
                        style=docstring_format,
                        purpose=purpose,
                        emit_type=emit_types,
                        emit_default_doc=emit_default_doc,
                        word_wrap=word_wrap,
                    ),
                    intermediate_repr["returns"].items(),
                ),
                None,
            )
        )
        if "return_type" in (intermediate_repr.get("returns") or iter(()))
        else ""
    )

    params_end_nls = num_of_nls(params, end=True)
    returns_end_nls = num_of_nls(returns, end=True)

    candidate_args_returns = "{params}{maybe_nl0}{returns}{maybe_nl1}".format(
        params=params,
        maybe_nl0="\n" if params_end_nls < 2 and returns else "",
        returns=returns,
        maybe_nl1=(
            "\n"
            if not returns and params_end_nls > 0 or returns and returns_end_nls == 0
            else ""
        ),
    )

    original_doc_str: str = intermediate_repr.get("_internal", {}).get(
        "original_doc_str", ""
    )
    if original_doc_str:
        header, _, footer = parse_docstring_into_header_args_footer(
            candidate_args_returns, original_doc_str
        )
        header = (
            intermediate_repr.get("doc", "") if not header and not footer else header
        )
    else:
        header, footer = intermediate_repr.get("doc", ""), ""

    candidate_doc_str: str = header_args_footer_to_str(
        header=header,
        args_returns="" if candidate_args_returns.isspace() else candidate_args_returns,
        footer=footer,
    )

    if not candidate_doc_str or candidate_doc_str.isspace():
        return ""

    prev_nl, next_nl = 0, candidate_doc_str.find("\n")
    current_indent, line = 0, None

    # One line only
    if next_nl == -1:
        # current_indent:int = count_iter_items(takewhile(str.isspace, candidate_doc_str))
        # _sep = (indent_level - current_indent) * tab
        return (
            candidate_doc_str
            if candidate_doc_str[0] == "\n"
            else "\n{_sep}{candidate_doc_str}".format(
                _sep="", candidate_doc_str=candidate_doc_str
            )
        )
    else:
        # Ignore starting newlines/whitespace only lines, keep munching until last line
        while next_nl > -1:
            line = candidate_doc_str[prev_nl:next_nl]
            if not line.isspace():
                break
            # prev_nl = next_nl
            # current_indent:int = count_iter_items(takewhile(str.isspace, line))

    if indent_level > current_indent:
        _tab = (indent_level - current_indent) * tab
        lines = ([line] if line else []) + candidate_doc_str[
            (
                next_nl
                if len(candidate_doc_str) == next_nl
                or next_nl + 1 < len(candidate_doc_str)
                and candidate_doc_str[next_nl + 1] != "\n"
                else next_nl + 1
            ) :
        ].splitlines()
        candidate_doc_str: str = "\n".join(
            map(
                lambda _line: (
                    "{_tab}{_line}".format(_tab=_tab, _line=_line)
                    if _line or emit_separating_tab
                    # and not _line.startswith(_tab)
                    else _line
                ),
                lines,
            )
        )
        if len(lines) > 1:
            candidate_doc_str: str = (
                "{maybe_nl}{candidate_doc_str}{maybe_nl_tab}".format(
                    maybe_nl="\n" if candidate_doc_str.startswith(_tab) else "",
                    candidate_doc_str=candidate_doc_str,
                    maybe_nl_tab=(
                        ""
                        if candidate_doc_str[-1] == "\n"
                        else "\n{_tab}".format(_tab=_tab)
                    ),
                )
            )

    return candidate_doc_str


__all__ = ["docstring"]  # type: list[str]
