"""
Functions which produce docstring portions from various inputs
"""

from collections import namedtuple
from enum import Enum
from functools import partial
from itertools import chain, takewhile
from operator import itemgetter, contains
from textwrap import indent

from cdd.defaults_utils import set_default_doc
from cdd.pure_utils import (
    fill,
    identity,
    indent_all_but_first,
    tab,
    rpartial,
    count_iter_items,
    pp,
    previous_line_range,
)


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

    :return: Newline joined pair of param, type
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
                    else "  {name}: ".format(name=name),
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


####################################################################
# internal functions for `parse_docstring_into_header_args_footer` #
####################################################################


def _get_token_start_idx(doc_str):
    """
    Get the start index of the token, minus starting whitespace for that line

    :param doc_str: The docstring
    :type doc_str: ```str```

    :return: index of token start or -1 if not found
    :rtype: ```int```
    """
    stack = []
    for idx, ch in enumerate(doc_str):
        if ch.isspace():
            if stack:
                if "".join(stack) in TOKENS_SET:
                    token_start_idx = idx - len(stack)
                    for i in range(token_start_idx, 0, -1):
                        if doc_str[i] == "\n":
                            return i + 1
                stack.clear()
        else:
            stack.append(ch)
    return -1


def _last_doc_str_token(doc_str):
    """
    Get start index of last docstring token (e.g., the index of ':' in ':rtype')

    :param doc_str: The docstring
    :type doc_str: ```str```

    :return: last_found index
    :rtype: ```Optional[int]```
    """
    last_found, penultimate_stack, stack, line0, line1 = None, [], [], None, None
    for i, ch in enumerate(doc_str):
        if ch.isspace():
            if stack:
                if stack.count("-") == len(stack):
                    if "".join(penultimate_stack) in NUMPYDOC_TOKENS_SET:
                        last_found = i - len(stack) + len(penultimate_stack)
                elif "".join(stack) in TOKENS_SET:
                    last_found = i - len(stack)
                penultimate_stack = stack.copy()
                stack.clear()
        else:
            stack.append(ch)

    return last_found


def _get_start_of_last_found(last_found, doc_str):
    """
    Get last token found in doc_str…
    '  :params foo' returns index to first space ' '

    :param last_found: index of token start or None if not found
    :type last_found: ```Optional[int]```

    :param doc_str: The docstring
    :type doc_str: ```str```

    :return: _last_found index
    :rtype: ```Optional[int]```
    """
    last_found_starts = None

    for last_found_starts in range(last_found - 1, 0, -1):
        if doc_str[last_found_starts] == "\n":
            last_found_starts += 1
            break

    return last_found_starts


def _get_end_of_last_found(last_found, last_found_starts, doc_str, docstring_format):
    """
    '  :params foo' returns index to second 'o' + nl

    :param last_found: index of token start or None if not found
    :type last_found: ```Optional[int]```

    :param last_found_starts: index of start of line containing token start
    :type last_found_starts: ```int```

    :param doc_str: The docstring
    :type doc_str: ```str```

    :param docstring_format: Docstring format
    :type docstring_format: ```Style```

    :return: _last_found index
    :rtype: ```Optional[int]```
    """
    last_found_ends, previous_line = None, None
    for last_found_ends in range(last_found, len(doc_str), 1):
        if doc_str[last_found_ends] == "\n":
            break
    last_found_ends += 1

    if docstring_format is Style.numpydoc and doc_str[
        last_found_starts:last_found
    ].count("-") == len(doc_str[last_found_starts:last_found]):
        countdown_from = last_found_starts - 1
        new_last_found_starts = None
        for new_last_found_starts in range(countdown_from - 1, 0, -1):
            if doc_str[new_last_found_starts] == "\n":
                new_last_found_starts += 1
                break

        if doc_str[new_last_found_starts:countdown_from] in NUMPYDOC_TOKENS_SET:
            # Find index of 'a' from last string in this format:
            # "arg_name : single/multi line desc"
            last_token_appearance = None
            stack = []
            for idx in range(last_found, len(doc_str)):
                if doc_str[idx] == "\n":
                    for ch in stack:
                        if ch.isalpha() or ch.isspace() or ch.isnumeric():
                            pass
                        elif ch == ":":
                            last_token_appearance = idx
                            break
                    stack.clear()
                else:
                    stack.append(doc_str[idx])
            if last_token_appearance is not None:
                last_found_starts = last_token_appearance
                for i in range(last_token_appearance, 0, -1):
                    if doc_str[i] == "\n":
                        last_found_starts = i + 1
                        break

                return last_found_starts

    return last_found_ends


def _find_end_of_args_returns(last_found, last_found_starts, last_found_ends, doc_str):
    """
    Handle multiline indented after keyword, e.g.,
      Returns:
         Foo
           more foo

    :param last_found: index of token start
    :type last_found: ```int```

    :param last_found_starts: index of start of line containing token start
    :type last_found_starts: ```int```

    :param last_found_ends: index of end of line containing token start
    :type last_found_ends: ```int```

    :param doc_str: The docstring
    :type doc_str: ```str```

    :return: end of args_returns index
    :rtype: ```int```
    """
    # prev_line = doc_str[slice(*previous_line_range(doc_str, last_found_ends - 1))]
    # zeroth_indent = count_iter_items(takewhile(str.isspace, prev_line))
    smallest_indent = count_iter_items(takewhile(str.isspace, doc_str[last_found_ends:]))
    last_nl, nls, i = 0, 0, None

    # Early exit… if current line isn't indented then it can't be a multiline arg/return descriptor
    if smallest_indent == 0:
        return last_found_ends - 1

    for i in range(last_found_ends, len(doc_str), 1):
        if doc_str[i] == "\n":
            # Two nl in a row
            if last_nl - 1 == i:
                break

            last_nl = i
            nls += 1

            # munch whitespace
            indent_size = 0
            while i < len(doc_str) - 1 and doc_str[i].isspace() and doc_str[i] != "\n":
                i += 1
                indent_size += 1

            if indent_size > smallest_indent:
                if smallest_indent == 0:
                    smallest_indent = indent_size
                else:
                    break
            elif indent_size == smallest_indent == 0:
                start_previous_line, end_previous_line = previous_line_range(doc_str, i)
                # immediate_previous_line = doc_str[start_previous_line:end_previous_line]
                return start_previous_line - 1
    return i


def _get_token_last_idx(doc_str):
    """
    Get the last index of the token (to end of line including newline)

    :param doc_str: The docstring
    :type doc_str: ```str```

    :return: index of token last or `len(doc_str) - 1` if not found
    :rtype: ```int```
    """

    last_found = _last_doc_str_token(doc_str)
    if last_found is None:
        return -1

    dostring_format = derive_docstring_format(doc_str)

    last_found_starts = _get_start_of_last_found(last_found, doc_str)
    last_found_ends = _get_end_of_last_found(
        last_found, last_found_starts, doc_str, dostring_format
    )

    idx = _find_end_of_args_returns(
        last_found, last_found_starts, last_found_ends, doc_str
    )

    # Munch until nl
    while idx < len(doc_str) and doc_str[idx] != "\n":
        idx += 1

    return idx + 1

    # afterward_idx, within_args_returns, idx = -1, False, None
    # last_indent, ante_penultimate_stack, nls = None, [], 0
    # for idx in range(last_found, len(doc_str) - 1, 1):
    #     if doc_str[idx] == "\n":
    #         nls += 1
    #         if (
    #             "".join(stack) in TOKENS_SET
    #             or "".join(penultimate_stack + stack) in TOKENS_SET
    #         ):
    #             print("#found: {!r} ;".format("".join(stack)))
    #             within_args_returns = True
    #         else:
    #             last_indent = count_iter_items(takewhile(str.isspace, stack))
    #         afterward_idx = last_found + idx - len(stack)
    #         ante_penultimate_stack = deepcopy(penultimate_stack)
    #         penultimate_stack = deepcopy(stack)
    #         stack.clear()
    #     else:
    #         stack.append(doc_str[idx])

    # afterward_idx = last_found + idx - len(stack)

    # startswith_token = any(map(doc_str[afterward_idx:].startswith, TOKENS_SET))
    # if startswith_token and stack:
    #     last_line = "".join(stack)
    #     startswith_token = any(map(last_line.startswith, TOKENS_SET))
    #     afterward_idx = -len(last_line) - 1
    #
    # ante_penultimate_indent, penultimate_indent, stack_indent = map(
    #     lambda arr: count_iter_items(takewhile(str.isspace, arr)),
    #     (ante_penultimate_stack, penultimate_stack, stack),
    # )
    #
    # return (
    #     -1
    #     if stack_indent != 0
    #     and (stack_indent == last_indent or penultimate_indent == stack_indent)
    #     or penultimate_indent > ante_penultimate_indent
    #     or startswith_token
    #     else afterward_idx
    # )


########################################################################
# end internal functions for `parse_docstring_into_header_args_footer` #
########################################################################


def parse_docstring_into_header_args_footer(current_doc_str, original_doc_str):
    """
    Parse docstring into three parts: header; args|returns; footer

    :param current_doc_str: The current doc_str
    :type current_doc_str: ```str```

    :param original_doc_str: The original doc_str
    :type original_doc_str: ```str```

    :return: Header, args|returns, footer
    :rtype: ```Tuple[Optional[str], Optional[str], Optional[str]]```
    """
    if original_doc_str == current_doc_str:
        return current_doc_str

    start_idx_current, start_idx_original = map(
        _get_token_start_idx, (current_doc_str, original_doc_str)
    )

    last_idx_current, last_idx_original = map(
        _get_token_last_idx, (current_doc_str, original_doc_str)
    )

    header_original = (
        original_doc_str[:start_idx_original] if start_idx_original > -1 else ""
    )

    footer_current, footer_original = map(
        lambda doc_idx: doc_idx[0][doc_idx[1] :] if doc_idx[1] != -1 else None,
        ((current_doc_str, last_idx_current), (original_doc_str, last_idx_original)),
    )

    # Now we know where the args/returns were, and where they are now
    # To avoid whitespace issues, only copy across the args/returns portion, keep rest as original

    args_returns_original = original_doc_str[
        slice(
            (start_idx_original if start_idx_original > -1 else None),
            (last_idx_original if last_idx_original > -1 else None),
        )
    ]
    args_returns_current = current_doc_str[
        slice(start_idx_current, len(footer_current) if footer_current else None)
    ]
    indent_args_returns_original = count_iter_items(
        takewhile(str.isspace, args_returns_original or iter(()))
    )
    if indent_args_returns_original > 1 and args_returns_current:
        args_returns_current = indent(
            args_returns_current,
            prefix=" " * indent_args_returns_original,
            predicate=lambda _: _,
        )

    return header_original, args_returns_current, footer_original


def ensure_doc_args_whence_original(current_doc_str, original_doc_str):
    """
    Ensure doc args appear where they appeared originally

    :param current_doc_str: The current doc_str
    :type current_doc_str: ```str```

    :param original_doc_str: The original doc_str
    :type original_doc_str: ```str```

    :return: reshuffled doc_str with args/returns in same place as original (same header, footer, and whitespace)
    :rtype: ```str```
    """
    (
        original_header,
        current_args_returns,
        original_footer,
    ) = parse_docstring_into_header_args_footer(current_doc_str, original_doc_str)
    return "{original_header}{current_args_returns}{original_footer}".format(
        original_header=original_header or "",
        current_args_returns=current_args_returns or "",
        original_footer=original_footer or "",
    )


class Style(Enum):
    """
    Simple enum taken from the docstring_parser codebase
    """

    rest = 1
    google = 2
    numpydoc = 3
    auto = 255


def derive_docstring_format(docstring):
    """
    Infer the docstring format of the provided docstring

    :param docstring: the docstring
    :type docstring: ```Optional[str]```

    :return: the style of docstring
    :rtype: ```Literal['rest', 'numpydoc', 'google']```
    """
    if docstring is None or any(map(partial(contains, docstring), TOKENS.rest)):
        style = Style.rest
    elif any(map(partial(contains, docstring), TOKENS.google)):
        style = Style.google
    else:
        style = Style.numpydoc
    return style


DOCSTRING_FORMATS = "rest", "google", "numpydoc"
Tokens = namedtuple("Tokens", DOCSTRING_FORMATS)
TOKENS = Tokens(
    (":param", ":cvar", ":ivar", ":var", ":type", ":raises", ":return", ":rtype"),
    ("Args:", "Kwargs:", "Raises:", "Returns:"),
    ("Parameters\n" "----------", "Returns\n" "-------"),
)
# Note: FPs possible for `"Parameters"` and `"Returns"` randomly thrown into normal doc_str
TOKENS_SET = frozenset(
    map(
        itemgetter(0),
        map(
            rpartial(str.partition, "\n"),
            chain.from_iterable(TOKENS._asdict().values()),
        ),
    )
)
NUMPYDOC_TOKENS_SET = frozenset(
    map(
        itemgetter(0),
        map(
            rpartial(str.partition, "\n"),
            TOKENS.numpydoc,
        ),
    )
)
ARG_TOKENS = Tokens(
    TOKENS.rest[:-2],
    (TOKENS.google[0],),
    (TOKENS.numpydoc[0],),
)
RETURN_TOKENS = Tokens(TOKENS.rest[-2:], (TOKENS.google[-1],), (TOKENS.numpydoc[-1],))


__all__ = [
    "derive_docstring_format",
    "ensure_doc_args_whence_original",
    "emit_param_str",
    "Style",
    "ARG_TOKENS",
    "RETURN_TOKENS",
    "TOKENS",
]
