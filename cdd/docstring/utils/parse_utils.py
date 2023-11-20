"""
Docstring parse utils
"""

import string
from functools import partial
from itertools import filterfalse, takewhile
from operator import contains, itemgetter

from cdd.shared.pure_utils import count_iter_items, pp, sliding_window

adhoc_type_to_type = {
    "bool": "bool",
    "boolean": "bool",
    "dict": "dict",
    "dictionary": "dict",
    "false": "bool",
    "filename": "str",
    "float": "float",
    "frequency": "int",  # or float?
    "integer": "int",
    "int64": "int",
    "`int64`castable": "int",
    "list": "list",
    "number": "int",
    "path": "str",
    "quantity": "int",
    "str": "str",
    "string": "str",
    "true": "bool",
    "tuple": "Tuple",
    "whether": "bool",
}

adhoc_3_tuple_to_type = {
    ("False", " ", "if"): "bool",
    ("False", " ", "on"): "bool",
    ("Filename", " ", "of"): "str",
    ("True", " ", "if"): "bool",
    ("True", " ", "on"): "bool",
    ("called", " ", "at"): "collections.abc.Callable",
    ("directory", " ", "where"): "str",
}

adhoc_3_tuple_to_collection = {
    ("List", " ", "of"): "List",
    ("Tuple", " ", "of"): "Tuple",
}


def _union_literal_from_sentence(sentence):
    """
    Extract the Union and/or Literal from a given sentence

    :param sentence: Input sentence with 'or' or 'of'
    :type sentence: ```str```

    :return: Union and/or Literal from a given sentence (or None if not found)
    :rtype: ```Optional[str]```
    """
    union = [[]]
    i = 0
    quotes = {"'": 0, '"': 0}
    while i < len(sentence):
        ch = sentence[i]
        is_space = ch.isspace()
        if not is_space and not ch == "`":
            union[-1].append(ch)
        elif is_space:
            if union[-1]:
                union[-1] = "".join(union[-1])
                if union[-1] in frozenset(
                    ("or", "or,", "or;", "or:", "of", "of,", "of;", "of:")
                ):
                    union[-1] = []
                else:
                    union.append([])
            # eat until next non-space
            j = i
            i += count_iter_items(takewhile(str.isspace, sentence[i:])) - 1
            union[-1] = sentence[j : i + 1]

            union.append([])
        if ch in frozenset(("'", '"')):
            if i == 0 or sentence[i - 1] != "\\":
                quotes[ch] += 1
            if (
                (i + 2) < len(sentence)
                and sum(quotes.values()) & 1 == 0
                and sentence[i + 1] == ","
            ):
                i += 1
        i += 1

    if not union[-1]:
        del union[-1]
    else:
        union[-1] = "".join(
            union[-1][:-1] if union[-1][-1] in frozenset((".", ",")) else union[-1]
        )
    # pp({"union": union})
    if len(union) > 1:
        candidate_type = next(
            map(
                adhoc_3_tuple_to_type.__getitem__,
                filter(
                    partial(contains, adhoc_3_tuple_to_type),
                    sliding_window(union, 3),
                ),
            ),
            None,
        )
        if candidate_type is not None:
            return candidate_type

    union = sorted(
        map(
            lambda k: adhoc_type_to_type.get(k.lower(), k),
            filterfalse(str.isspace, union),
        )
    )
    pp({"union": union})
    literals = count_iter_items(
        takewhile(
            frozenset(string.digits + "'\"").__contains__,
            map(itemgetter(0), union),
        )
    )
    if literals and len(union) > literals:
        return "Union[{}, {}]".format(
            "Literal[{}]".format(", ".join(union[:literals])),
            ", ".join(union[literals:]),
        )
    elif literals:
        return "Literal[{}]".format(", ".join(union[:literals]))
    elif union:
        return "Union[{}]".format(", ".join(union)) if len(union) > 1 else union[0]
    else:
        return None


def parse_adhoc_doc_for_typ(doc):
    """
    Google's Keras and other frameworks have an adhoc syntax.

    Call this function after the first-pass; i.e., after the arg {name, doc, typ, default} are 'known'.

    :param doc: Possibly ambiguous docstring for argument, that *might* hint as to the type
    :type doc: ```str```

    :return: The type (if determined) else `None`
    :rtype: ```Optional[str]```
    """

    if not doc:
        return None

    words = [[]]  # type: List[List[str]]
    word_chars = "{0}{1}`'\"/|".format(string.digits, string.ascii_letters)
    sentence_ends = -1
    for i, ch in enumerate(doc):
        if (
            ch in word_chars
            or ch == "."
            and len(doc) > (i + 1)
            and doc[i + 1] in word_chars
        ):
            words[-1].append(ch)
        elif ch in frozenset((".", ";", ",")) or ch.isspace():
            words[-1] = "".join(words[-1])
            words.append(ch)
            if ch == "." and sentence_ends == -1:
                sentence_ends = len(words)
            words.append([])
    words[-1] = "".join(words[-1])
    fst_sentence = "".join(words[:sentence_ends])
    sentence = None

    if words[0] == "Whether":
        return "bool"

    if " or " in fst_sentence or " of " in fst_sentence:
        sentence = fst_sentence
    else:
        sentence_starts = sentence_ends
        for a, b in sliding_window(words[sentence_starts:], 2):
            sentence_ends += 1
            if a == "." and not b.isidentifier():
                break
        snd_sentence = "".join(words[sentence_starts:sentence_ends])
        if " or " in snd_sentence or " of " in snd_sentence:
            sentence = snd_sentence

    if sentence is not None:
        wrap_type_with = "{}"
        if sentence.count("`") == 2:
            fst_tick = sentence.find("`")
            candidate_collection = next(
                map(
                    adhoc_3_tuple_to_collection.__getitem__,
                    filter(
                        partial(contains, adhoc_3_tuple_to_collection),
                        sliding_window(sentence[:fst_tick], 3),
                    ),
                ),
                None,
            )
            if candidate_collection is not None:
                wrap_type_with = candidate_collection + "[{}]"
            sentence = sentence[fst_tick : sentence.rfind("`")]

        candidate_type = _union_literal_from_sentence(sentence)
        if candidate_type is not None:
            return wrap_type_with.format(candidate_type)

    candidate_type = next(
        map(
            adhoc_type_to_type.__getitem__,
            filter(partial(contains, adhoc_type_to_type), words),
        ),
        None,
    )
    if candidate_type is not None:
        return candidate_type
    elif "/" in words[2]:
        return "Union[{}]".format(",".join(sorted(words[2].split("/"))))

    return None


__all__ = ["parse_adhoc_doc_for_typ"]
