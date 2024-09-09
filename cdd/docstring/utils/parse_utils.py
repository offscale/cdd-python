"""
Docstring parse utils
"""

import string
from collections import Counter
from functools import partial
from itertools import chain, filterfalse, takewhile
from keyword import iskeyword
from operator import contains, itemgetter
from typing import List, Optional, Tuple, Union, cast

from cdd.shared.ast_utils import deduplicate
from cdd.shared.pure_utils import (
    count_iter_items,
    simple_types,
    sliding_window,
    type_to_name,
)

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
    ("floating", " ", "point"): "float",
}

adhoc_3_tuple_to_collection = {
    ("List", " ", "of"): "List",
    ("Tuple", " ", "of"): "Tuple",
    ("Dictionary", " ", "of"): "Mapping",
}


def _union_literal_from_sentence(sentence, wrap_with="Union[{}]"):
    """
    Extract the Union and/or Literal from a given sentence

    :param sentence: Input sentence with 'or' or 'of'
    :type sentence: ```str```

    :return: Union and/or Literal from a given sentence (or None if not found)
    :rtype: ```Optional[str]```
    """
    union: Union[List[List[str]], List[str], Tuple[str]] = [[]]
    _union_literal_from_sentence_phase0(sentence, union)

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
        else:
            candidate_collection = next(
                map(
                    adhoc_3_tuple_to_collection.__getitem__,
                    filter(
                        partial(contains, adhoc_3_tuple_to_collection),
                        sliding_window(union, 3),
                    ),
                ),
                None,
            )
            if candidate_collection is not None:
                return None
    union = list(
        deduplicate(
            map(
                lambda k: adhoc_type_to_type.get(k.lower(), k),
                filterfalse(str.isspace, union),
            )
        )
    )
    # Sanity check, if the vars are not legit then exit now
    # checks if each var is keyword or digit or quoted
    if any(
        filter(
            lambda e: e not in type_to_name
            and (
                iskeyword(e)
                or e.isdigit()
                or (
                    # could take care and use a customer scanner to handle escaped quotes; but this hack for now
                    lambda counter: counter["'"] & 1 == 1
                    and counter["'"] > 0
                    or counter['"'] & 1 == 1
                    and counter['"'] > 0
                )(Counter(e))
            ),
            union,
        )
    ):
        return None

    valid = frozenset(string.digits + "'\"`")
    literals: int = (
        count_iter_items(
            takewhile(
                valid.__contains__,
                map(itemgetter(0), union),
            )
        )
        if union and union[0] and union[0][0] in valid
        else 0
    )
    # binary-search or even interpolation search can be done? is it sorted here?
    idx = next(
        map(
            itemgetter(0),
            filter(lambda idx_elem: idx_elem[1] == "None", enumerate(union)),
        ),
        None,
    )
    if idx is not None:
        del union[idx]
        wrap = "Optional[{}]"
    else:
        wrap = "{}"

    union = cast(Tuple[str], tuple(map(lambda typ: type_to_name.get(typ, typ), union)))

    if literals and len(union) > literals:
        return wrap.format(
            "Union[{}, {}]".format(
                "Literal[{}]".format(", ".join(union[:literals])),
                ", ".join(union[literals:]),
            )
        )
    elif literals:
        return wrap.format("Literal[{}]".format(", ".join(union[:literals])))
    elif union:
        return wrap.format(
            "Union[{}]".format(", ".join(union)) if len(union) > 1 else union[0]
        )
    else:
        return None


def _union_literal_from_sentence_phase0(sentence, union):
    """
    Internal function for `_union_literal_from_sentence`; does the first n=O(n) iteration through the sentence

    :param sentence: Input sentence with 'or' or 'of'
    :type sentence: ```str```

    :type union: ```Union[List[List[str]], List[str], Tuple[str]]```
    """
    i: int = 0
    quotes = {"'": 0, '"': 0}
    while i < len(sentence):
        ch = sentence[i]
        is_space = ch.isspace()
        if not is_space and not ch == "`":
            union[-1].append(ch)
        elif is_space:
            if union[-1]:
                union[-1] = "".join(
                    union[-1][:-1]
                    if union[-1][-1] in frozenset((",", ";"))
                    and (
                        union[-1][0] in frozenset(string.digits + "'\"`")
                        or union[-1][0].isidentifier()
                    )
                    else union[-1]
                )
                if union[-1] in frozenset(("or", "or,", "or;", "or:")):
                    union[-1] = []
                elif union[-1] in frozenset(("of", "of,", "of;", "of:")):
                    collection_type = adhoc_3_tuple_to_collection.get(tuple(union))
                    if collection_type is None:
                        union[-1] = []
                    else:
                        union = []
                else:
                    union.append([])
            # eat until next non-space
            j = i
            i += count_iter_items(takewhile(str.isspace, sentence[i:])) - 1

            union[slice(*((-1, None) if union else (None, None)))] = sentence[j : i + 1]

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


def parse_adhoc_doc_for_typ(doc, name, default_is_none):
    """
    Google's Keras and other frameworks have an adhoc syntax.

    Call this function after the first-pass; i.e., after the arg {name, doc, typ, default} are 'known'.

    :param doc: Possibly ambiguous docstring for argument, that *might* hint as to the type
    :type doc: ```str```

    :param name: Name of argument; useful for debugging and if the name hints as to the type
    :type name: ```str```

    :param default_is_none: Whether the default is `NoneStr`
    :type default_is_none: ```bool```

    :return: The type (if determined) else `None`
    :rtype: ```Optional[str]```
    """
    if not doc:
        return None

    wrap: str = "Optional[{}]" if default_is_none else "{}"

    words: List[Union[List[str], str]] = [[]]
    candidate_type, fst_sentence, sentence = _parse_adhoc_doc_for_typ_phase0(doc, words)

    if sentence is not None:
        sentence, wrap_type_with = _parse_adhoc_doc_for_typ_phase1(sentence, words)

        new_candidate_type: Optional[str] = cast(
            Optional[str], _union_literal_from_sentence(sentence)
        )
        if new_candidate_type is not None:
            if (
                new_candidate_type.startswith("Literal[")
                and candidate_type in simple_types
                and candidate_type is not None
            ):
                wrap_type_with = "Union[{}, " + "{}]".format(candidate_type)
            candidate_type: str = (
                new_candidate_type[len("Union[") : -len("]")]
                if wrap_type_with == "Mapping[{}]"
                else new_candidate_type
            )
        if candidate_type is not None:
            return wrap_type_with.format(candidate_type)

    if fst_sentence is not None:
        whole_sentence_as_type: Optional[str] = type_to_name.get(
            fst_sentence.rstrip(".")
        )
        if whole_sentence_as_type is not None:
            return whole_sentence_as_type
    if candidate_type is not None:
        return candidate_type
    elif len(words) > 2:
        if "/" in words[2]:
            return "Union[{}]".format(",".join(deduplicate(words[2].split("/"))))
        candidate_type: Optional[str] = next(
            map(
                adhoc_3_tuple_to_type.__getitem__,
                filter(
                    partial(contains, adhoc_3_tuple_to_type),
                    sliding_window(words, 3),
                ),
            ),
            None,
        )

    return candidate_type if candidate_type is None else wrap.format(candidate_type)


def _parse_adhoc_doc_for_typ_phase1(sentence, words):
    """
    Internal function for `parse_adhoc_doc_for_typ`.

    :param sentence: Input sentence
    :type sentence: ```str```

    :param words: Words
    :type words: ```List[Union[List[str], str]]```

    :return: sentence, wrap_type_with
    :rtype: ```Tuple[str, str]```
    """
    wrap_type_with: str = "{}"
    defaults_idx: int = sentence.rfind(", default")
    if defaults_idx != -1:
        sentence: str = sentence[:defaults_idx]
    if (sentence.count("`") & 1) == 0:
        fst_tick: int = (lambda idx: idx if idx > -1 else None)(sentence.find("`"))
        candidate_collection: Optional[str] = next(
            chain.from_iterable(
                (
                    map(
                        adhoc_3_tuple_to_collection.__getitem__,
                        filter(
                            partial(contains, adhoc_3_tuple_to_collection),
                            sliding_window(sentence[:fst_tick].split(), 3),
                        ),
                    ),
                    map(
                        adhoc_3_tuple_to_collection.__getitem__,
                        filter(
                            partial(contains, adhoc_3_tuple_to_collection),
                            sliding_window(words, 3),
                        ),
                    ),
                )
            ),
            None,
        )
        if candidate_collection is not None:
            wrap_type_with: str = candidate_collection + "[{}]"
        if fst_tick is not None:
            sentence: str = sentence[fst_tick : sentence.rfind("`")]
    return sentence, wrap_type_with


def _parse_adhoc_doc_for_typ_phase0(doc, words):
    """
    Internal function for `_parse_adhoc_doc_for_typ_`; does the few iterations through the sentence

    :param doc: Possibly ambiguous docstring for argument, that *might* hint as to the type
    :type doc: ```str```

    :param words: Words
    :type words: ```List[Union[List[str], str]]```

    :return: candidate_type, fst_sentence, sentence
    :rtype: ```Tuple[Optional[Any], Optional[str], Optional[str]]```
    """
    word_chars: str = "{0}{1}`'\"/|".format(string.digits, string.ascii_letters)
    sentence_ends: int = -1
    break_the_union: bool = False  # lincoln
    for i, ch in enumerate(doc):
        if (
            ch in word_chars
            or ch == "."
            and len(doc) > (i + 1)
            and doc[i + 1] in word_chars
            # Make "bar" start the next sentence:    `foo`.bar
            and (i - 1 == 0 or doc[i - 1] != "`")
        ):
            words[-1].append(ch)
        elif ch in frozenset((".", ";", ",")) or ch.isspace():
            words[-1] = "".join(words[-1])
            words.append(ch)
            if ch == "." and sentence_ends == -1:
                sentence_ends: int = len(words)
            elif ch == ";":
                break_the_union = True
            words.append([])
    words[-1] = "".join(words[-1])
    candidate_type: Optional[str] = next(
        map(
            adhoc_type_to_type.__getitem__,
            filter(partial(contains, adhoc_type_to_type), words),
        ),
        None,
    )
    fst_sentence: str = "".join(
        words[2 if break_the_union and len(words) > 2 else 0 : sentence_ends]
    )
    sentence: Optional[str] = None
    # type_in_fst_sentence = adhoc_type_to_type.get(next(filterfalse(str.isspace, words)))
    if " or " in fst_sentence or " of " in fst_sentence:
        sentence = fst_sentence
    else:
        sentence_starts: int = sentence_ends
        for a, b in sliding_window(words[sentence_starts:], 2):
            sentence_ends += 1
            if a == "." and not b.isidentifier():
                break
        snd_sentence: str = "".join(words[sentence_starts:sentence_ends])
        if " or " in snd_sentence or " of " in snd_sentence:
            sentence: str = snd_sentence
    return candidate_type, fst_sentence, sentence


__all__ = ["parse_adhoc_doc_for_typ"]  # type: list[str]
