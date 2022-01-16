"""
Pure utils for pure functions. For the same input will always produce the same input_str.
"""

import typing
from ast import Name, Str
from collections import deque
from functools import partial
from importlib import import_module
from importlib.util import find_spec
from inspect import getmodule
from itertools import chain, count, zip_longest
from keyword import iskeyword
from operator import attrgetter, eq
from os import environ, extsep, path
from pprint import PrettyPrinter
from sys import version_info
from textwrap import fill as _fill
from textwrap import indent
from typing import Callable, Dict, FrozenSet, Optional, Tuple, Union

pp: Callable[[object], None] = PrettyPrinter(indent=4, width=100).pprint
tab: str = environ.get("DOCTRANS_TAB", " " * 4)
simple_types: Dict[Optional[str], Union[int, float, complex, str, bool, None]] = {
    "int": 0,
    "float": 0.0,
    "complex": 0j,
    "str": "",
    "bool": False,
    None: None,
}

line_length = environ.get("DOCTRANS_LINE_LENGTH", 100)
fill = partial(_fill, width=line_length)


# From https://github.com/Suor/funcy/blob/0ee7ae8/funcy/funcs.py#L34-L36
def rpartial(func, *args):
    """Partially applies last arguments."""
    return lambda *a: func(*(a + args))


def identity(*args, **kwargs):
    """
    Identity function

    :param args: Any values
    :type args: ```Tuple[Any]```

    :return: the input value
    :rtype: ```Any```
    """
    return args[0] if len(args) == 1 else args


_python_major_minor: Tuple[int, int] = version_info[:2]
PY3_8: bool = _python_major_minor == (3, 8)
PY_GTE_3_8: bool = _python_major_minor >= (3, 8)
PY_GTE_3_9: bool = _python_major_minor >= (3, 9)

ENCODING = "# -*- coding: utf-8 -*-"

none_types = None, "None", "```(None)```" if PY_GTE_3_9 else "```None```"

_ABERRANT_PLURAL_MAP: Dict[str, str] = {
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

VOWELS: FrozenSet[str] = frozenset("aeiou")


def pluralise(singular):
    """Return plural form of given lowercase singular word (English only). Based on
    ActiveState recipe http://code.activestate.com/recipes/413172/ and 577781

    Note: For production you'd probably want to use nltk or an NLP AI model

    :param singular: Non plural
    :type singular: ```str```

    :return: Plural version
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


# def previous_line_range(s, countdown_from=None):
#     """
#     Get previous line from a multiline string
#
#     :param s: Multiline string
#     :type s: ```str```
#
#     :param countdown_from: Index to start looking down from; len(s) otherwise
#     :type countdown_from: ```Optional[int]```
#
#     :return: Previous line range if found else None
#     :rtype: ```Optional[Tuple[int, int]]```
#     """
#     if countdown_from is None:
#         countdown_from = len(s)
#     for i in range(countdown_from - 1, 0, -1):
#         if s[i] == "\n":
#             return i + 1, countdown_from
#     return None


def deindent(s, level=None, sep=tab):
    """
    Remove all indentation from the input string, or `level`(s) of indent if specified

    :param s: Input string
    :type s: ```AnyStr```

    :param level: Number of tabs to remove from input string or if None: remove all
    :type level: ```Optional[int]```

    :param sep: Separator (usually `tab`)
    :type sep: ```str```

    :return: Deindented string
    :rtype: ```AnyStr```
    """
    if level is None:
        process_line = str.lstrip
    else:
        sep *= level

        def process_line(line):
            """
            :param line: The line to dedent
            :type line: ```AnyStr```

            :return: Dedented line
            :rtype: ```AnyStr```
            """
            return line[len(sep) :] if line.startswith(sep) else line

    return "\n".join(map(process_line, s.splitlines()))


def reindent(s, indent_level=1, join_on="\n"):
    """
    Reindent the input string

    :param s: Input string
    :type s: ```AnyStr```

    :param indent_level: docstring indentation level whence: 0=no_tabs, 1=one tab; 2=two tabs
    :type indent_level: ```int```

    :param join_on: What to join on, e.g., '\n'
    :type join_on: ```str```

    :return: Reindented string
    :rtype: ```AnyStr```
    """
    return join_on.join(
        map(
            lambda line: "{tab}{line}".format(
                tab=abs(indent_level) * tab, line=line.lstrip()
            ),
            s.split("\n"),
        )
    ).replace(tab, "", 1)


def strip_starting(line, str_to_strip=tab):
    """
    :param line: Input string
    :type line: ```AnyStr```

    :param str_to_strip: Removes only this (once… so not `str.lstrip`) from the start
    :type str_to_strip: ```str```
    """
    return line[len(str_to_strip) :] if line.startswith(str_to_strip) else line


def indent_all_but_first(s, indent_level=1, wipe_indents=False, sep=tab):
    """
    Indent all lines except the first one

    :param s: Input string
    :type s: ```str```

    :param indent_level: indentation level whence: 0=no_tabs, 1=one tab; 2=two tabs
    :type indent_level: ```int```

    :param wipe_indents: Whether to clean the `s` of indents first
    :type wipe_indents: ```bool```

    :param sep: Separator (usually `tab`)
    :type sep: ```str```

    :return: input string indented (except first line)
    :rtype: ```str```
    """
    lines = indent(deindent(s) if wipe_indents else s, sep * abs(indent_level)).split(
        "\n"
    )
    return "\n".join([lines[0].lstrip()] + lines[1:])


def multiline(s, quote_with=("'", "'")):
    """
    For readability and linting, it's useful to turn a long line, like:
    >>> '''123456789_\n123456789_\n123456789_\n123456789'''

    Into:
    >>> '''123456789_\n''' \
        '''123456789_\n''' \
        '''123456789_\n''' \
        '''123456789'''

    :param s: Input string
    :type s: ```str```

    :param quote_with: What to quote with
    :type quote_with: ```Tuple[str, str]```

    :return: multine input string
    :rtype: ```str```
    """
    return "{}{}".format(
        "",
        tab.join(
            map(
                lambda _s: "{quote_with[0]}{_s}{quote_with[1]} \\\n".format(
                    quote_with=quote_with, _s=_s
                ),
                s.splitlines(),
            )
        ).rstrip(" \n\\"),
    )


def sanitise(s):
    """
    Sanitise the input string, appending an `_` if it's a keyword

    :param s: Input string
    :type s: ```str```

    :return: input string with '_' append if it's a keyword else input string
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

    :return: Iterator of each element of the hierarchy
    :rtype: ```Iterator[str, ...]```
    """
    return map(str.strip, param.split(sep))


def unquote(input_str):
    """
    Unquote a string. Removes one set of leading quotes (' or ")

    :param input_str: Input string
    :type input_str: ```Optional[str]```

    :return: Unquoted string
    :rtype: ```Optional[str]```
    """
    if (
        isinstance(input_str, str)
        and len(input_str) > 1
        and (
            input_str.startswith('"')
            and input_str.endswith('"')
            or input_str.startswith("'")
            and input_str.endswith("'")
        )
    ):
        return input_str[1:-1]
    return input_str


def quote(s, mark='"'):
    """
    Quote the input string if it's not already quoted

    :param s: Input string or literal or None
    :type s: ```Union[str, float, complex, int, None]```

    :param mark: Quote mark to wrap with
    :type mark: ```str```

    :return: Quoted string or input (if input is not str)
    :rtype: ```Union[str, float, complex, int, None]```
    """
    very_simple_types = type(None), int, float, complex
    s = (
        s
        if isinstance(s, (str, *very_simple_types))
        else s.s
        if isinstance(s, Str)
        else s.id
        if isinstance(s, Name)
        else getattr(s, "value", s)
    )
    # ^ Poor man's `get_value`
    if (
        isinstance(s, very_simple_types)
        or len(s) == 0
        or len(s) > 1
        and s[0] == s[-1]
        and s[0] in frozenset(("'", '"'))
    ):
        return s
    return "{mark}{s}{mark}".format(mark=mark, s=s)


def assert_equal(a, b, cmp=eq):
    """
    assert a and b are equal

    :param a: anything
    :type a: ```Any```

    :param b: anything else
    :type b: ```Any```

    :param cmp: comparator function
    :type cmp: ```Callable[[a, b], bool]```

    :return: True if equal, otherwise raises `AssertionError`
    :rtype: ```Literal[True]```
    """
    if not cmp(a, b):
        raise AssertionError("{a!r} != {b!r}".format(a=a, b=b))
    return True


def update_d(d, arg=None, **kwargs):
    """
    Update d inplace

    :param d: dict to update
    :type d: ```dict```

    :param arg: dict to update with
    :type arg: ```Optional[dict]```

    :param kwargs: keyword args to update with
    :type kwargs: ```**kwargs```

    :return: Updated dict
    :rtype: ```dict```
    """
    if arg:
        d.update(arg)
    if kwargs:
        d.update(kwargs)
    return d


def lstrip_namespace(s, namespaces):
    """
    Remove starting namespace

    :param s: input string
    :type s: ```AnyStr```

    :param namespaces: namespaces to strip
    :type namespaces: ```Union[List[str], Tuple[str], Generator[str], Iterator[str]]```

    :return: `.lstrip`ped input (potentially just the original!)
    :rtype: ```AnyStr```
    """
    for namespace in namespaces:
        s = s.lstrip(namespace)
    return s


def diff(input_obj, op):
    """
    Given an input with `__len__` defined and an op which takes the input and produces one output
      with `__len__` defined, compute the difference and return (diff_len, output)

    :param input_obj: The input
    :type input_obj: ```Any```

    :param op: The operation to run
    :type op: ```Callable[[Any], Any]```

    :return: length of difference, response of operated input
    :rtype: ```Tuple[int, Any]```
    """
    input_len = len(
        input_obj
    )  # Separate line and binding, as `op` could mutate the `input`
    result = op(input_obj)
    return input_len - len(result), result


strip_diff = partial(diff, op=str.strip)
lstrip_diff = partial(diff, op=str.lstrip)
rstrip_diff = partial(diff, op=str.rstrip)


def balanced_parentheses(s):
    """
    Checks if parentheses are balanced, ignoring whatever is inside quotes

    :param s: Input string
    :type s: ```str```

    :return: Whether the parens are balanced
    :rtype: ```bool```
    """
    open_parens, closed_parens = "([{", ")]}"
    counter = {paren: 0 for paren in open_parens + closed_parens}
    quote_mark = None
    for idx, ch in enumerate(s):
        if (
            quote_mark is not None
            and ch == quote_mark
            and (idx == 0 or s[idx - 1] != "\\")
        ):
            quote_mark = None
        elif ch in frozenset(("'", '"')):
            if quote_mark is None:
                quote_mark = ch
            elif quote_mark == ch:
                quote_mark = None
        elif quote_mark is None and ch in counter:
            counter[ch] += 1
    return all(
        counter[open_parens[i]] == counter[closed_parens[i]]
        for i in range(len(open_parens))
    )


def blockwise(t, size=2, fillvalue=None):
    """
    Blockwise, like pairwise but with a `size` parameter
    From: https://stackoverflow.com/a/4628446

    :param t: iterator
    :type t: ```Iterator```

    :param size: size of each block
    :type size: ```int```

    :param fillvalue: What to use to "pair" with if uneven
    :type fillvalue: ```Any```

    :return: iterator with iterators inside of block size
    :rtype: ```Iterator```
    """
    return zip_longest(*[iter(t)] * abs(size), fillvalue=fillvalue)


def location_within(container, iterable, cmp=eq):
    """
    Finds element within iterable within container

    :param container: The container, e.g., a str, or list.
      We are looking for the subset which matches an element in `iterable`.
    :type container: ```Any```

    :param iterable: The iterable, can be constructed
    :type iterable: ```Any```

    :param cmp: Comparator to check input against
    :type cmp: ```Callable[[str, str], bool]```

    :return: (Start index iff found else -1, End index iff found else -1, subset iff found else None)
    :rtype: ```Tuple[int, int, Optional[Any]]```
    """
    if not hasattr(container, "__len__"):
        container = tuple(container)
    container_len = len(container)

    for elem in iterable:
        elem_len = len(elem)
        if elem_len > container_len:
            continue
        elif cmp(elem, container):
            return 0, elem_len, elem
        else:
            for i in range(container_len):
                end = i + elem_len
                if cmp(container[i:end], elem):
                    return i, end, elem
                elif i + elem_len + 1 > container_len:
                    break
    return -1, -1, None


BUILTIN_TYPES = frozenset(
    chain.from_iterable(
        (
            chain.from_iterable(
                map(
                    lambda s: (s, "typing.{}".format(s), "_extensions.{}".format(s)),
                    filter(lambda s: s[0].isupper() and not s.isupper(), dir(typing)),
                )
            ),
            (
                "int",
                "float",
                "complex",
                "list",
                "tuple",
                "str",
                "bytes",
                "bytearray",
                "memoryview",
                "set",
                "frozenset",
                "dict",
            ),
        )
    )
)

DUNDERS = frozenset(
    filter(
        rpartial(str.startswith, "__"),
        frozenset(
            chain.from_iterable(
                (
                    # https://docs.python.org/3/library/stdtypes.html
                    chain.from_iterable(
                        map(
                            dir,
                            (
                                int,
                                float,
                                complex,
                                list,
                                tuple,
                                range,
                                str,
                                bytes,
                                bytearray,
                                memoryview,
                                set,
                                frozenset,
                                dict,
                                type,
                                None,
                                Ellipsis,
                                NotImplemented,
                                object,
                            ),
                        )
                    ),
                    # https://docs.python.org/3/library/functions.html#dir
                    dir(),
                    # https://docs.python.org/3/library/stdtypes.html#special-attributes
                    (
                        "__dict__",
                        "__class__",
                        "__bases__",
                        "__name__",
                        "__qualname__",
                        "__mro__",
                        "__subclasses__",
                        # https://docs.python.org/3/reference/datamodel.html#slots
                        "__slots__",
                    ),
                )
            )
        ),
    )
)

INIT_FILENAME = "__init__{extsep}py".format(extsep=extsep)


def code_quoted(s):
    """
    Internally user-provided `None` and non `literal_eval`uatable input is quoted with ```

    This function checks if the input is quoted such

    :param s: The input
    :type s: ```Any```

    :return: Whether the input is code quoted
    :rtype: ```bool```
    """
    return (
        isinstance(s, str) and len(s) > 6 and s.startswith("```") and s.endswith("```")
    )


# From https://stackoverflow.com/a/15112059
def count_iter_items(iterable):
    """
    Consume an iterable not reading it into memory; return the number of items.

    :param iterable: An iterable
    :type iterable: ```Iterable```

    :return: Number of items in iterable
    :rtype: ```int```
    """
    counter = count()
    deque(zip(iterable, counter), maxlen=0)
    return next(counter)


def get_module(name, package=None, extra_symbols=None):
    """
    Import a module.

    The 'package' argument is required when performing a relative import. It
    specifies the package to use as the anchor point from which to resolve the
    relative import to an absolute import.

    Wraps `importlib.import_module` to return the module if it's available in interpreter on ModuleNotFoundError error

    :param name: Module name
    :type name: ```str```

    :param package: Package name
    :type package: ```Optional[str]```

    :param extra_symbols: Dictionary of extra symbols to use if `importlib.import_module` fails
    :type extra_symbols: ```Optional[dict]```

    :return: Module
    :rtype: ```Module```
    """
    try:
        return import_module(name, package)
    except ModuleNotFoundError:
        if name in globals():
            return globals()[name]
        else:
            pkg, _, rest_path = name.partition(".")
            if pkg in extra_symbols:
                return getmodule(
                    (attrgetter(rest_path) if rest_path else identity)(
                        extra_symbols[pkg]
                    )
                )
            raise


def has_nl(s, func):
    """
    Check if str endswith a newline (possibly separated by other whitespace)

    :param s: Input string
    :type s: ```str```

    :param func: `str.partition` or `str.rpartition` function (or something similar)
    :type func: ```Callable[[str], Tuple[str,str,str]]```

    :return: Whether s endswith a newline (possibly separated by other whitespace)
    :rtype: ```bool```
    """
    start, maybe_nl, end = partial(func, "\n")(s)
    return maybe_nl == "\n" and start.isspace()


def is_triple_quoted(s):
    """
    Whether the str is triple quoted

    :param s: Input string
    :type s: ```str```

    :return: Whether it has balanced triple quotes (either variety)
    :rtype: ```bool```
    """
    return len(s) > 5 and (
        s.startswith("'''")
        and s.endswith("'''")
        or s.startswith('"""')
        and s.endswith('"""')
    )


def is_ir_empty(intermediate_repr):
    """
    Checks whether the IR is empty, i.e., might have name and params but will generate a docstr without types or argdoc

    :param intermediate_repr: a dictionary of form
        {  "name": Optional[str],
           "type": Optional[str],
           "doc": Optional[str],
           "params": OrderedDict[str, {'typ': str, 'doc': Optional[str], 'default': Any}]
           "returns": Optional[OrderedDict[Literal['return_type'],
                                           {'typ': str, 'doc': Optional[str], 'default': Any}),)]] }
    :type intermediate_repr: ```dict```

    :return: Whether IR is empty
    :rtype: ```bool```
    """
    return not intermediate_repr.get("doc") and not any(
        param_d is not None and (param_d.get("typ") or param_d.get("doc"))
        for key in ("params", "returns")
        for param_d in (intermediate_repr.get(key) or {}).values()
    )


def paren_wrap_code(code):
    """
    The new builtin AST unparser adds extra parentheses, so match that behaviour on older versions

    :param code: Source code string
    :type code: ```str```

    :return: Potentially parenthetically wrapped input
    :rtype: ```str```
    """
    return (
        "({code})".format(code=code)
        if PY_GTE_3_9 and code[0] + code[-1] not in frozenset(("()", "[]", "{}"))
        else code
    )


def filename_from_mod_or_filename(mod_or_filename):
    """
    Resolve filename from module name or filename

    :param mod_or_filename: Module name or filename
    :type mod_or_filename: ```str```

    :return: Filename
    :rtype: ```str```
    """
    filename = type("", tuple(), {"origin": mod_or_filename})
    return (
        filename
        if path.sep in mod_or_filename or path.isfile(mod_or_filename)
        else find_spec(mod_or_filename) or filename
    ).origin


def emit_separating_tabs(s, indent_level=1, run_per_line=str.lstrip):
    """
    Emit a separating tab between paragraphs

    :param s: Input string (probably a docstring)
    :type s: ```str```

    :param indent_level: docstring indentation level whence: 0=no_tabs, 1=one tab; 2=two tabs
    :type indent_level: ```int```

    :param run_per_line: Run this function per line
    :type run_per_line: ```Callable[[str], str]```

    :return: Original string with a separating tab between paragraphs, & possibly addition indentation on other lines
    :rtype: ```str```
    """
    sep = tab * indent_level
    return "\n{sep}{}\n{sep}".format(
        run_per_line(
            "\n".join(map(lambda line: sep if len(line) == 0 else line, s.splitlines()))
        ),
        sep=sep,
    )


def set_attr(obj, key, val):
    """
    Sets the named attribute on the given object to the specified value.

    set_attr(x, 'y', v) is equivalent to ``x.y = v; return x''

    :param obj: An object
    :type obj: ```Any```

    :param key: A key
    :type key: ```str```

    :param val: A value
    :type val: ```Any```

    :return: The modified `obj`
    :rtype: ```Any```
    """
    setattr(obj, key, val)
    return obj


def set_item(obj, key, val):
    """
    Sets the item on the given object to the specified value.

    set_item(x, 'y', v) is equivalent to ``x[y] = v; return x''

    :param obj: An object
    :type obj: ```Any```

    :param key: A key
    :type key: ```Union[str, int]```

    :param val: A value
    :type val: ```Any```

    :return: The modified `obj`
    :rtype: ```Any```
    """
    obj[key] = val
    return obj


def no_magic_dir2attr(p_object):
    """
    Dictionary of `dir` without the __ prefix magics (also without _ prefix)
    return the names comprising (some of) the attributes
    of the given object, and of attributes reachable from it.

    :param p_object: Object
    :type p_object: ```Any```

    :return: Dict of name to attribute value
    :rtype: ```dict```
    """
    return {
        attr: getattr(p_object, attr)
        for attr in dir(p_object)
        if not attr.startswith("_")
    }


omit_whitespace = rpartial(str.translate, str.maketrans({" ": "", "\n": "", "\t": ""}))

sanitise_emit_name = dict(
    **{typ: typ for typ in ("function", "sqlalchemy", "sqlalchemy_table")},
    **{"class": "class_", "argparse": "argparse_function"}
).__getitem__

__all__ = [
    "BUILTIN_TYPES",
    "DUNDERS",
    "ENCODING",
    "INIT_FILENAME",
    "PY3_8",
    "PY_GTE_3_8",
    "PY_GTE_3_9",
    "assert_equal",
    "balanced_parentheses",
    "blockwise",
    "code_quoted",
    "count_iter_items",
    "deindent",
    "diff",
    "emit_separating_tabs",
    "filename_from_mod_or_filename",
    "fill",
    "get_module",
    "has_nl",
    "is_triple_quoted",
    "identity",
    "indent_all_but_first",
    "location_within",
    "lstrip_namespace",
    "multiline",
    "no_magic_dir2attr",
    "none_types",
    "omit_whitespace",
    "paren_wrap_code",
    "pluralise",
    "pp",
    # "previous_line_range",
    "quote",
    "reindent",
    "rpartial",
    "sanitise",
    "sanitise_emit_name",
    "set_attr",
    "set_item",
    "simple_types",
    "strip_split",
    "strip_starting",
    "tab",
    "unquote",
    "update_d",
]
