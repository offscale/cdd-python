"""
Pure utils for pure functions. For the same input will always produce the same input_str.
"""
from keyword import iskeyword
from platform import python_version_tuple
from pprint import PrettyPrinter
from sys import version

pp = PrettyPrinter(indent=4).pprint
tab = " " * 4
simple_types = {"int": 0, float: 0.0, "str": "", "bool": False}


# From https://github.com/Suor/funcy/blob/0ee7ae8/funcy/funcs.py#L34-L36
def rpartial(func, *args):
    """Partially applies last arguments."""
    return lambda *a: func(*(a + args))


def identity(s):
    """
    Identity function

    :param s: Any value
    :type s: ```Any```

    :returns: the input value
    :rtype: ```Any```
    """
    return s


PY3_8 = version.startswith("3.8")
PY_GTE_3_9 = python_version_tuple() >= ("3", "9")

_ABERRANT_PLURAL_MAP = {
    "appendix": "appendices",
    "barracks": "barracks",
    "cactus": "cacti",
    "child": "children",
    "criterion": "criteria",
    "deer": "deer",
    "echo": "echoes",
    "elf": "elves",
    "embargo": "embargoes",
    "focus": "foci",
    "fungus": "fungi",
    "goose": "geese",
    "hero": "heroes",
    "hoof": "hooves",
    "index": "indices",
    "knife": "knives",
    "leaf": "leaves",
    "life": "lives",
    "man": "men",
    "mouse": "mice",
    "nucleus": "nuclei",
    "person": "people",
    "phenomenon": "phenomena",
    "potato": "potatoes",
    "self": "selves",
    "syllabus": "syllabi",
    "tomato": "tomatoes",
    "torpedo": "torpedoes",
    "veto": "vetoes",
    "woman": "women",
}

VOWELS = frozenset("aeiou")


def pluralise(singular):
    """Return plural form of given lowercase singular word (English only). Based on
    ActiveState recipe http://code.activestate.com/recipes/413172/ and 577781

    Note: For production you'd probably want to use nltk or an NLP AI model

    :param singular: Non plural
    :type singular: ```str```

    :returns: Plural version
    :rtype: ```str```
    """
    if not singular:
        return ""
    plural = _ABERRANT_PLURAL_MAP.get(singular) or singular.endswith("es") and singular
    if plural:
        return plural
    root = singular
    try:
        if singular[-1] == "y" and singular[-2] not in VOWELS:
            root = singular[:-1]
            suffix = "ies"
        elif singular[-1] == "s":
            if singular[-2] in VOWELS:
                if singular[-3:] == "ius":
                    root = singular[:-2]
                    suffix = "i"
                else:
                    root = singular[:-1]
                    suffix = "ses"
            else:
                suffix = "es"
        elif singular[-2:] in ("ch", "sh"):
            suffix = "es"
        else:
            suffix = "s"
    except IndexError:
        suffix = "s"

    return root + suffix


def deindent(s):
    """
    Remove all indentation from the input string

    :param s: Input string
    :type s: ```AnyStr```

    :returns: Deindented string
    :rtype: ```AnyStr```
    """
    return "\n".join(
        map(
            str.lstrip,
            s.split("\n"),
        )
    )


def reindent(s, indent_level=1, join_on="\n"):
    """
    Reindent the input string

    :param s: Input string
    :type s: ```AnyStr```

    :param indent_level: docstring indentation level whence: 0=no_tabs, 1=one tab; 2=two tabs
    :type indent_level: ```int```

    :param join_on: What to join on, e.g., '\n'
    :type join_on: ```str```

    :returns: Reindented string
    :rtype: ```AnyStr```
    """
    return join_on.join(
        map(
            lambda line: "{tab}{line}".format(
                tab=indent_level * tab, line=line.lstrip()
            ),
            s.split("\n"),
        )
    ).replace(tab, "", 1)


def sanitise(s):
    """
    Sanitise the input string, appending an `_` if it's a keyword

    :param s: Input string
    :type s: ```str```

    :returns: input string with '_' append if it's a keyword else input string
    :rtype: ```str```
    """
    return "{}_".format(s) if iskeyword(s) else s


def strip_split(param, sep):
    """
    Split and strip the input string on given separator

    :param param: Module/ClassDef/FunctionDef/AnnAssign/Assign resolver with a dot syntax.
    :type param: ```str```

    :param sep: Separator
    :type sep: ```str```

    :returns: Iterator of each element of the hierarchy
    :rtype: ```Iterator[str, ...]```
    """
    return (loc.strip() for loc in param.split(sep))


def quote(s, mark='"'):
    """
    Quote the input string if it's not already quoted

    :param s: Input string
    :type s: ```str```

    :param mark: Quote mark to wrap with
    :type mark: ```str```

    :returns: Quoted string
    :rtype: ```str```
    """
    if s is None or len(s) == 0 or s[0] == s[-1] and s[0] in frozenset(("'", '"')):
        return s
    return "{mark}{s}{mark}".format(mark=mark, s=s)


__all__ = [
    "pp",
    "tab",
    "quote",
    "simple_types",
    "rpartial",
    "identity",
    "PY3_8",
    "PY_GTE_3_9",
    "pluralise",
    "reindent",
    "sanitise",
    "strip_split",
]
