"""
Functions which produce docstring portions from various inputs
"""

from collections import namedtuple
from enum import Enum
from functools import partial
from itertools import chain, takewhile
from operator import contains, eq, itemgetter, ne
from textwrap import indent

from cdd.defaults_utils import set_default_doc
from cdd.pure_utils import (
    count_iter_items,
    fill,
    has_nl,
    identity,
    indent_all_but_first,
    omit_whitespace,
    rpartial,
    tab,
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
            ("return", "rtype")
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
        if ch == "\n":
            indent_amount = count_iter_items(takewhile(str.isspace, stack))
            line = "".join(stack[indent_amount:])
            if line in NUMPYDOC_TOKENS_SET:
                i = indent_amount + idx + 1
                next_line = doc_str[i : doc_str.find("\n", i)]
                if next_line.count("-") == len(next_line):
                    return idx - len(stack)
            elif any(filter(line.startswith, TOKENS_SET)):
                return idx - len(stack)
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
    last_found, penultimate_stack, stack = None, [], []
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


def _get_end_of_last_found_numpydoc(last_found, last_found_starts, doc_str):
    """
    Numpydoc specialisation of `_get_end_of_last_found` function

    :param last_found: index of token start or None if not found
    :type last_found: ```Optional[int]```

    :param last_found_starts: index of start of line containing token start
    :type last_found_starts: ```int```

    :param doc_str: The docstring
    :type doc_str: ```str```

    :return: _last_found index
    :rtype: ```Optional[int]```
    """
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
                    if ch == ":":
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
    last_found_ends = None
    for last_found_ends in range(last_found, len(doc_str), 1):
        if doc_str[last_found_ends] == "\n":
            break
    last_found_ends += 1

    if docstring_format is Style.numpydoc and doc_str[
        last_found_starts:last_found
    ].count("-") == len(doc_str[last_found_starts:last_found]):
        return _get_end_of_last_found_numpydoc(
            last_found, 0 if last_found_starts is None else last_found_starts, doc_str
        )

    return last_found_ends


def _find_end_of_args_returns(last_found_ends, doc_str):
    """
    Handle multiline indented after keyword, e.g.,
      Returns:
         Foo
           more foo

    :param last_found_ends: index of end of line containing token start
    :type last_found_ends: ```int```

    :param doc_str: The docstring
    :type doc_str: ```str```

    :return: end of args_returns index
    :rtype: ```int```
    """
    smallest_indent = count_iter_items(
        takewhile(str.isspace, doc_str[last_found_ends:])
    )

    # Early exit… if current line isn't indented then it can't be a multiline arg/return descriptor
    if smallest_indent == 0 and last_found_ends is not None:
        return last_found_ends - 1

    return len(doc_str) - 1


def _get_token_last_idx_if_no_next_token(doc_str, last_found_starts):
    """
    Get the last index of the token (to end of line including newline)
    if no next token is detected (internal, only use is in `_get_token_last_idx`)

    :param doc_str: The docstring
    :type doc_str: ```str```

    :param last_found_starts: Index of where the last found token starts in `doc_str`
    :type last_found_starts: ```int```

    :return: index of token last or `None` if not found
    :rtype: ```int```
    """
    next_nl = last_found_starts + count_iter_items(
        takewhile(partial(ne, "\n"), doc_str[last_found_starts:])
    )

    next_line = doc_str[last_found_starts:next_nl]
    if frozenset(next_line) == frozenset(("-",)):
        line_start = line_end = next_nl + 1
        line_no = 0
        # last_space = line_no, line_start, line_end
        PrevParam = namedtuple(
            "PrevParam", ("line_no", "indent", "line_start", "line_end")
        )
        prev_param = PrevParam(
            line_no=None, indent=None, line_start=line_start, line_end=line_end
        )

        while line_end < len(doc_str):
            line_end += count_iter_items(
                takewhile(partial(ne, "\n"), doc_str[line_start:])
            )
            line = doc_str[line_start:line_end]
            line_no += 1

            if line.isspace():
                pass
            elif line_no - 2 == prev_param[0]:
                starting_whitespace = count_iter_items(takewhile(str.isspace, line))
                if starting_whitespace >= prev_param[1]:
                    # Handle multiline description of param
                    prev_param = PrevParam(
                        line_no=line_no,
                        indent=starting_whitespace,
                        line_start=line_start,
                        line_end=line_end,
                    )
            elif ":" in line:
                # This is a param
                prev_param = PrevParam(
                    line_no=line_no,
                    indent=count_iter_items(takewhile(str.isspace, line)),
                    line_start=line_start,
                    line_end=line_end,
                )

            line_start = line_end
            line_end += 1
        return prev_param.line_end + 1

        # if i_started_at == line_end:

    if doc_str[last_found_starts:next_nl] == "Raises:":
        # Don't treat this as anything special… short circuit
        return last_found_starts - 1


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

    docstring_format = derive_docstring_format(doc_str)

    last_found_starts = _get_start_of_last_found(last_found, doc_str)
    last_found_ends = _get_end_of_last_found(
        last_found, last_found_starts, doc_str, docstring_format
    )

    idx = _find_end_of_args_returns(last_found_ends, doc_str)

    # Munch until previous nl
    while idx != 0 and doc_str[idx] != "\n":
        idx -= 1

    indent_amount = count_iter_items(takewhile(str.isspace, doc_str[idx + 1 :]))

    i_started_at = i = indent_amount + idx + 1
    if any(filter(doc_str[i:].startswith, TOKENS_SET)):
        # Munch until nl
        i += 1
        while i < len(doc_str) and doc_str[i] != "\n":
            i += 1

    if i_started_at == i:
        last_idx = _get_token_last_idx_if_no_next_token(
            doc_str, 0 if last_found_starts is None else last_found_starts
        )
        if last_idx is not None:
            return last_idx

    return i


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
    # if not current_doc_str and not original_doc_str: return None, None, None

    # To quieten linter
    header_original = footer_original = None
    start_idx_current = last_idx_current = start_idx_original = last_idx_original = None

    if current_doc_str:
        start_idx_current = _get_token_start_idx(current_doc_str)
        last_idx_current = _get_token_last_idx(current_doc_str)
        # footer_current = (
        #     current_doc_str[last_idx_current:] if last_idx_current != -1 else None
        # )
        # header_current = (
        #     original_doc_str[:start_idx_current] if start_idx_current > -1 else None
        # )
        #
        # args_returns_current = current_doc_str[
        #     slice(
        #         start_idx_current if start_idx_current > -1 else None,
        #         last_idx_current if last_idx_current > -1 else None,
        #     )
        # ]

    if original_doc_str:
        start_idx_original = _get_token_start_idx(original_doc_str)
        last_idx_original = _get_token_last_idx(original_doc_str)
        footer_original = (
            original_doc_str[last_idx_original:] if last_idx_original != -1 else None
        )
        header_original = (
            original_doc_str[:start_idx_original] if start_idx_original > -1 else None
        )

        args_returns_original = original_doc_str[
            slice(
                start_idx_original if start_idx_original > -1 else None,
                last_idx_original if last_idx_original > -1 else None,
            )
        ]

    # Now we know where the args/returns were, and where they are now
    # To avoid whitespace issues, only copy across the args/returns portion, keep rest as original

    args_returns_current, args_returns_original = map(
        lambda doc_start_end: doc_start_end[0][
            slice(
                doc_start_end[1] if doc_start_end[1] > -1 else None,
                doc_start_end[2] if doc_start_end[2] > -1 else None,
            )
        ],
        (
            (current_doc_str, start_idx_current, last_idx_current),
            (original_doc_str, start_idx_original, last_idx_original),
        ),
    )

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
    if original_doc_str and eq(
        *map(omit_whitespace, (original_doc_str, current_doc_str))
    ):
        return original_doc_str
    (
        original_header,
        current_args_returns,
        original_footer,
    ) = parse_docstring_into_header_args_footer(current_doc_str, original_doc_str)

    return header_args_footer_to_str(
        header=original_header or "",
        args_returns=current_args_returns or "",
        footer=original_footer or "",
    )


def header_args_footer_to_str(header, args_returns, footer):
    """
    Ensure there is always a newline between each of: header; args_returns; and footer

    :param header: Header section
    :type header: ```str```

    :param args_returns: args|returns section
    :type args_returns: ```str```

    :param footer: Footer section
    :type footer: ```str```

    :return: One string with these three section combined with a minimum of one nl betwixt each
    :rtype: ```str```
    """
    if args_returns:
        args_returns_start_has_nl = has_nl(args_returns, str.partition)
        args_returns_ends_has_nl = has_nl(args_returns, str.rpartition)
        args_returns = "{nl0}{args_returns}{nl1}".format(
            nl0="\n" if args_returns_start_has_nl else "",
            args_returns=args_returns,
            nl1="\n" if args_returns_ends_has_nl else "",
        )
    else:
        args_returns_start_has_nl = args_returns_ends_has_nl = True
    if footer:
        footer_start_has_nl = has_nl(footer, str.partition) or args_returns_ends_has_nl
        # foot_end_has_nl = footer[-1] == "\n"
    else:
        footer_start_has_nl = True  # foot_end_has_nl
    header_end_has_nl = not header or header[-1] == "\n" or args_returns_start_has_nl

    # Match indent of args_returns to header or footer
    if args_returns:
        header_or_footer = header if header else footer
        indent_amount = count_iter_items(takewhile(str.isspace, header_or_footer))
        newlines = (
            header_or_footer[:indent_amount].count("\n") if header_or_footer else 0
        )
        indent_amount = indent_amount - newlines
        current_indent_amount = count_iter_items(takewhile(str.isspace, args_returns))
        if current_indent_amount != indent_amount:
            _indent = indent_amount * " "
            len_args_returns = len(args_returns)
            args_returns = indent(args_returns, _indent, predicate=lambda _: _)
            if args_returns[-1] == "\n" and len_args_returns > 1:
                args_returns += _indent

    return "{header}{args_returns}{footer}".format(
        header=(header if header_end_has_nl else "{header}\n".format(header=header)),
        args_returns=args_returns,
        footer=(
            "{nl0}{footer}{nl1}".format(
                nl0="" if footer_start_has_nl else "\n",
                footer=footer,
                nl1="",  # nl1="" if foot_end_has_nl else "\n"
            )
        ),
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
    "ARG_TOKENS",
    "RETURN_TOKENS",
    "Style",
    "TOKENS",
    "derive_docstring_format",
    "emit_param_str",
    "ensure_doc_args_whence_original",
    "parse_docstring_into_header_args_footer",
]
