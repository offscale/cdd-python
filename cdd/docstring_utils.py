"""
Functions which produce docstring portions from various inputs
"""

from collections import namedtuple
from copy import deepcopy
from itertools import chain, takewhile
from operator import itemgetter, eq
from textwrap import indent

from cdd.defaults_utils import set_default_doc
from cdd.pure_utils import (
    fill,
    identity,
    indent_all_but_first,
    tab,
    rpartial,
    count_iter_items,
    omit_whitespace,
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


def ensure_doc_args_whence_original(current_doc_str, original_doc_str):
    """
    Ensure doc args appear where they appeared originally

    :param current_doc_str: The current doc_str
    :type current_doc_str: ```str```

    :param original_doc_str: The original doc_str
    :type original_doc_str: ```Optional[str]```

    :returns: reshuffled doc_str with args/returns in same place as original (same header, footer, and whitespace)
    :rtype: ```str```
    """
    if eq(*map(omit_whitespace, (original_doc_str or "", original_doc_str or ""))):
        return original_doc_str
    elif not original_doc_str:
        return current_doc_str

    def get_token_start_idx(doc_str):
        """
        Get the start index of the token

        :param doc_str: The docstring
        :type doc_str: ```str```

        :returns: index of token start or -1 if not found
        :rtype: ```int```
        """
        stack = []
        for idx, ch in enumerate(doc_str):
            if ch.isspace():
                if stack:
                    if "".join(stack) in TOKENS_SET:
                        return idx - len(stack)
                    stack.clear()
            else:
                stack.append(ch)
        return -1

    def get_token_last_idx(doc_str):
        """
        Get the last index of the token

        :param doc_str: The docstring
        :type doc_str: ```str```

        :returns: index of token last or `len(doc_str) - 1` if not found
        :rtype: ```int```
        """
        last_found, penultimate_stack, stack = None, [], []
        for idx, ch in enumerate(doc_str):
            if ch.isspace():
                if stack:
                    if (
                        "".join(stack) in TOKENS_SET
                        or "".join(penultimate_stack + stack) in TOKENS_SET
                    ):
                        last_found = idx - len(stack)
                    penultimate_stack = stack.copy()
                    stack.clear()
            else:
                stack.append(ch)
        if last_found is None:
            return -1
        penultimate_stack.clear()
        stack.clear()

        afterward_idx, idx, last_indent, ante_penultimate_stack = -1, None, None, []
        for idx, ch in enumerate(doc_str[last_found:]):
            if ch == "\n":
                if (
                    "".join(stack) not in TOKENS_SET
                    and "".join(penultimate_stack + stack) not in TOKENS_SET
                ):
                    last_indent = count_iter_items(takewhile(str.isspace, stack))
                afterward_idx = last_found + idx - len(stack)
                ante_penultimate_stack = deepcopy(penultimate_stack)
                penultimate_stack = deepcopy(stack)
                stack.clear()
            else:
                stack.append(ch)

        startswith_token = any(map(doc_str[afterward_idx:].startswith, TOKENS_SET))
        if startswith_token and stack:
            last_line = "".join(stack)
            startswith_token = any(map(last_line.startswith, TOKENS_SET))
            afterward_idx = -len(last_line) - 1

        ante_penultimate_indent, penultimate_indent, stack_indent = map(
            lambda arr: count_iter_items(takewhile(str.isspace, arr)),
            (ante_penultimate_stack, penultimate_stack, stack),
        )

        return (
            -1
            if stack_indent != 0
            and (stack_indent == last_indent or penultimate_indent == stack_indent)
            or penultimate_indent > ante_penultimate_indent
            or startswith_token
            else afterward_idx
        )

    start_idx_current, start_idx_original = map(
        get_token_start_idx, (current_doc_str, original_doc_str)
    )
    last_idx_current, last_idx_original = map(
        get_token_last_idx, (current_doc_str, original_doc_str)
    )

    afterward_current, afterward_original = map(
        lambda doc_idx: doc_idx[0][slice(doc_idx[1], None)]
        if doc_idx[1] != -1
        else None,
        ((current_doc_str, last_idx_current), (original_doc_str, last_idx_original)),
    )

    # Now we know where the args/returns were, and where they are now
    # To avoid whitespace issues, only copy across the args/returns portion, keep rest as original

    original_header = (
        original_doc_str[:start_idx_original] if start_idx_original > -1 else ""
    )
    original_footer = afterward_original
    current_args_returns = current_doc_str[
        slice(start_idx_current, len(afterward_current) if afterward_current else None)
    ]

    return "{original_header}{current_args_returns}{original_footer}".format(
        original_header=original_header or "",
        current_args_returns=current_args_returns or "",
        original_footer=original_footer or "",
    )


Tokens = namedtuple("Tokens", ("rest", "google", "numpydoc"))
TOKENS = Tokens(
    (":param", ":cvar", ":ivar", ":var", ":type", ":return", ":rtype"),
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
ARG_TOKENS = Tokens(
    TOKENS.rest[:-2],
    (TOKENS.google[0],),
    (TOKENS.numpydoc[0],),
)
RETURN_TOKENS = Tokens(TOKENS.rest[-2:], (TOKENS.google[-1],), (TOKENS.numpydoc[-1],))

__all__ = ["ensure_doc_args_whence_original", "emit_param_str", "ARG_TOKENS", "TOKENS"]
