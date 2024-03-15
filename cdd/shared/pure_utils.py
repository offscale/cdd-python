"""
Pure utils for pure functions. For the same input will always produce the same input_str.
"""

import string
import typing
from ast import Name
from collections import deque
from functools import partial
from importlib import import_module
from importlib.machinery import ModuleSpec
from importlib.util import find_spec
from inspect import getmodule
from itertools import chain, count, filterfalse, islice, takewhile, tee, zip_longest
from json import JSONEncoder
from keyword import iskeyword
from operator import attrgetter, eq, itemgetter
from os import environ, extsep, listdir, path
from pprint import PrettyPrinter
from sys import stderr, version_info
from textwrap import fill as _fill
from textwrap import indent
from typing import Any, Callable, Dict, Optional, Sized, Tuple, Union, cast

_python_major_minor: Tuple[int, int] = version_info[:2]
PY3_8: bool = _python_major_minor == (3, 8)
PY_GTE_3_8: bool = _python_major_minor >= (3, 8)
PY_GTE_3_9: bool = _python_major_minor >= (3, 9)
PY_GTE_3_10: bool = _python_major_minor >= (3, 10)
PY_GTE_3_11: bool = _python_major_minor >= (3, 11)
PY_GTE_3_12: bool = _python_major_minor >= (3, 12)

if PY_GTE_3_8:
    from typing import Literal, Protocol
else:
    from ast import Str

    from typing_extensions import Literal, Protocol

if PY_GTE_3_9:
    FrozenSet = frozenset
else:
    from typing import FrozenSet

pp: Callable[[Any], None] = PrettyPrinter(indent=4, width=100, stream=stderr).pprint
tab: str = environ.get("DOCTRANS_TAB", " " * 4)
simple_types: Dict[Optional[str], Union[int, float, complex, str, bool, None]] = {
    "int": 0,
    "float": 0.0,
    "complex": 0j,
    "str": "",
    "bool": False,
    None: None,
}
type_to_name: Dict[str, str] = {
    "Int": "int",
    "int": "int",
    "Float": "float",
    "float": "float",
    "complex": "complex",
    "str": "str",
    "String": "str",
    "Bool": "bool",
    "bool": "bool",
    "None": "None",
}

line_length: int = int(environ.get("DOCTRANS_LINE_LENGTH", 100))
fill: Callable[[str], str] = partial(_fill, width=line_length)

ENCODING: str = "# -*- coding: utf-8 -*-"

none_types: Tuple[None, Literal["None"], str] = (
    None,
    "None",
    "```(None)```" if PY_GTE_3_9 else "```None```",
)

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


def read_file_to_str(filename, mode="rt"):
    """
    Read filename into a str, closing the file afterwards

    :param filename: Input filename
    :type filename: ```str```

    :param mode: File mode
    :type mode: ```str```

    :return: Filename content as str
    :rtype: ```str```
    """
    with open(filename, mode) as f:
        return f.read()


# From https://github.com/Suor/funcy/blob/0ee7ae8/funcy/funcs.py#L34-L36
def rpartial(func, *args):
    """Partially applies last arguments."""
    return lambda *a: func(*(a + args))


def remove_whitespace_comments(source):
    """
    Remove all insignificant whitespace and comments from source

    :param source: Python source string
    :type source: ```str```

    :return: `source` without significant whitespace and comments
    :rtype: ```str```
    """
    return "\n".join(
        filter(
            None,
            filterfalse(str.isspace, map(parse_comment_from_line, source.splitlines())),
        )
    )


def append_to_dict(d, keys, value):
    """
    Append keys to a dictionary in a hierarchical manner and set the last key to a value.
    For example, given an empty dictionary and keys = [a, b, c] and value = d then the new
    dictionary would look like the following: {a: {b: {c: d}}}

    :param d: dictionary to append keys
    :type d: ```dict```

    :param keys: keys to append to d
    :type keys: ```list[str]```

    :param value: value to set keys[-1]
    :type value: ```any```

    :return: `dict` with new keys and value added
    :rtype: ```dict```
    """
    pointer = d
    for i, key in enumerate(keys):
        if i == len(keys) - 1 and isinstance(pointer, dict):
            pointer[key] = value
            return d
        if isinstance(pointer, dict):
            pointer.setdefault(key, {})
            pointer = pointer[key]
        else:
            return d
    return d


def parse_comment_from_line(line):
    """
    Remove from comment onwards in line

    :param line: Python source line
    :type line: ```str```

    :return: `line` without comments
    :rtype: ```str```
    """
    double: int = 0
    single: int = 0
    for col, ch in enumerate(line):
        if col > 3 and line[col - 1] == "\\" and ch in frozenset(('"', "'", "#")):
            pass  # Ignore the char
        elif ch == '"':
            double += 1
        elif ch == "'":
            single += 1
        elif ch == "#" and single & 1 == 0 and double & 1 == 0:
            col_offset = (
                col
                if col == 0
                else (col - count_iter_items(takewhile(str.isspace, line[:col][::-1])))
            )
            return "".join(islice(line, 0, col_offset))
    return line


def identity(*args, **kwargs):
    """
    Identity function

    :param args: Any values
    :type args: ```tuple[Any]```

    :return: the input value
    :rtype: ```Any```
    """
    return args[0] if len(args) == 1 else args


def pluralise(singular):
    """Return plural form of given lowercase singular word (English only). Based on
    ActiveState recipe http://code.activestate.com/recipes/413172/ and 577781

    Note: For production you'd probably want to use nltk or an NLP AI model

    :param singular: Non-plural
    :type singular: ```str```

    :return: Plural version
    :rtype: ```str```
    """
    if not singular:
        return ""
    plural: str = (
        _ABERRANT_PLURAL_MAP.get(singular) or singular.endswith("es") and singular
    )
    if plural:
        return plural
    root: str = singular
    try:
        if singular[-1] == "y" and singular[-2] not in VOWELS:
            root: str = singular[:-1]
            suffix: str = "ies"
        elif singular[-1] == "s":
            if singular[-2] in VOWELS:
                if singular[-3:] == "ius":
                    root: str = singular[:-2]
                    suffix: str = "i"
                else:
                    root: str = singular[:-1]
                    suffix: str = "ses"
            else:
                suffix: str = "es"
        elif singular[-2:] in ("ch", "sh"):
            suffix: str = "es"
        else:
            suffix: str = "s"
    except IndexError:
        suffix: str = "s"

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
#     :rtype: ```Optional[tuple[int, int]]```
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
        process_line: Callable[[str], str] = str.lstrip
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
    lines: typing.List[str] = indent(
        deindent(s) if wipe_indents else s, sep * abs(indent_level)
    ).split("\n")
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
    :type quote_with: ```tuple[str, str]```

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
    Unquote a string. Removes one set of leading quotes `'\''` or `'"'`

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
    very_simple_types = (
        type(None),
        int,
        float,
        complex,
    )  # type: tuple[Type[None], Type[int], Type[float], Type[complex]]
    s: str = (
        s
        if isinstance(s, (str, *very_simple_types))
        else (
            s.s
            if isinstance(s, Str)
            else s.id if isinstance(s, Name) else getattr(s, "value", s)
        )
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


def all_dunder_for_module(
    module_directory,
    include,
    exclude=frozenset(("compound", "shared", "tests")),
    path_validator=path.isdir,
):
    """
    Generate `__all__` for a given module using single-level filename hierarchy exclusively

    :param module_directory: Module path
    :type module_directory: ```str```

    :param include: Additional strings to include
    :type include: ```Iterable[str]```

    :param exclude: base filenames to ignore
    :type exclude: ```frozenset```

    :param path_validator: Path validation function
    :type path_validator: ```Callable[[str], bool]```

    :return: list of strings matching the expected `__all__`
    :rtype: ```list[str]```
    """
    return sorted(
        chain.from_iterable(
            (
                include,
                map(
                    itemgetter(0),
                    map(
                        path.splitext,
                        filter(
                            lambda base: path_validator(
                                path.join(module_directory, base)
                            )
                            and not base.startswith("_")
                            and not base.endswith("_utils{}py".format(path.extsep))
                            and base not in exclude,
                            listdir(module_directory),
                        ),
                    ),
                ),
            )
        )
    )


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
        d.update(cast(dict, arg))
    if kwargs:
        d.update(kwargs)
    return d


def lstrip_namespace(s, namespaces):
    """
    Remove starting namespace

    :param s: input string
    :type s: ```AnyStr```

    :param namespaces: namespaces to strip
    :type namespaces: ```Union[list[str], tuple[str], Generator[str], Iterator[str]]```

    :return: `str.lstrip`ped input (potentially just the original!)
    :rtype: ```AnyStr```
    """
    for namespace in namespaces:
        s: str = s.lstrip(cast(str, namespace))
    return s


def diff(input_obj, op):
    """
    Given an input with `__len__` defined and an op which takes the input and produces one output
      with `__len__` defined, compute the difference and return (diff_len, output)

    :param input_obj: The input
    :type input_obj: ```Any```

    :param op: The operation to run
    :type op: ```Callable[[Any], Sized]```

    :return: length of difference, response of operated input
    :rtype: ```tuple[int, Any]```
    """
    input_len: int = len(
        input_obj
    )  # Separate line and binding, as `op` could mutate the `input`
    result: Sized = op(input_obj)
    return input_len - len(result), result


strip_diff: Callable[[str], Tuple[int, Any]] = partial(diff, op=str.strip)
lstrip_diff: Callable[[str], Tuple[int, Any]] = partial(diff, op=str.lstrip)
rstrip_diff: Callable[[str], Tuple[int, Any]] = partial(diff, op=str.rstrip)


def balanced_parentheses(s):
    """
    Checks if parentheses are balanced, ignoring whatever is inside quotes

    :param s: Input string
    :type s: ```str```

    :return: Whether the parens are balanced
    :rtype: ```bool```
    """
    open_parens: str = "([{"
    closed_parens: str = ")]}"
    counter: Dict[str, int] = {paren: 0 for paren in open_parens + closed_parens}
    quote_mark: Optional[Literal["'", '"']] = None
    for idx, ch in enumerate(s):
        if (
            quote_mark is not None
            and ch == quote_mark
            and (idx == 0 or s[idx - 1] != "\\")
        ):
            quote_mark: Optional[Literal["'", '"']] = None
        elif quote_mark is None:
            if ch in frozenset(("'", '"')):
                quote_mark: Optional[Literal["'", '"']] = cast(Literal["'", '"'], ch)
            elif ch in counter:
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
    :rtype: ```tuple[int, int, Optional[Any]]```
    """
    if not hasattr(container, "__len__"):
        container: Tuple[Any] = tuple(container)
    container_len: int = len(container)

    for elem in iterable:
        elem_len: int = len(elem)
        if elem_len > container_len:
            continue
        elif cmp(elem, container):
            return 0, elem_len, elem
        else:
            for i in range(container_len):
                end: int = i + elem_len
                if cmp(container[i:end], elem):
                    return i, end, elem
                elif i + elem_len + 1 > container_len:
                    break
    return -1, -1, None


BUILTIN_TYPES: FrozenSet[str] = frozenset(
    chain.from_iterable(
        (
            chain.from_iterable(
                map(
                    lambda s: (s, "{}".format(s), "_extensions.{}".format(s)),
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

DUNDERS: FrozenSet[str] = frozenset(
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

INIT_FILENAME: str = "__init__{extsep}py".format(extsep=extsep)


def code_quoted(s):
    """
    Internally user-provided `None` and non `literal_eval`able input is quoted with ```

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
    counter: count = count()
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


def find_module_filepath(module_name, submodule_name=None, none_when_no_spec=False):
    """
    Find module's file location without first importing it

    :param module_name: Module name, e.g., "cdd.tests" or "cdd"
    :type: ```str```

    :param submodule_name: Submodule name, e.g., "test_pure_utils"
    :type: ```Optional[str]```

    :param none_when_no_spec: When `find_spec` returns `None` return that. If `False` raises `AssertionError` then.
    :type none_when_no_spec: ```bool```

    :return: Module location
    :rpath: ```str```
    """
    assert module_name is not None
    module_spec: Optional[ModuleSpec] = find_spec(module_name)
    if module_spec is None:
        if none_when_no_spec:
            return module_spec
        raise AssertionError("spec not found for {}".format(module_name))
    module_origin: Optional[str] = module_spec.origin
    assert module_origin is not None
    module_parent: str = path.dirname(module_origin)
    return (
        module_origin
        if submodule_name is None
        else next(
            filter(
                path.exists,
                (
                    path.join(
                        module_parent,
                        submodule_name,
                        "__init__{}py".format(path.extsep),
                    ),
                    path.join(
                        module_parent, "{}{}py".format(submodule_name, path.extsep)
                    ),
                    path.join(
                        module_parent,
                        submodule_name,
                        "__init__{}py".format(path.extsep),
                    ),
                ),
            ),
            module_origin,
        )
    )


def count_chars_from(
    s, sentinel_char_unseen, char, end, s_len=len, start_idx=0, char_f=None
):
    """
    Count number of chars in string from one or other end, until `ignore` is no longer True (or entire `s` is covered)

    :param s: Input string
    :type s: ```str``

    :param sentinel_char_unseen: Function that takes one char and decided whether to ignore it or not
    :type sentinel_char_unseen: ```Callable[[str], bool]```

    :param char: Single character for counting occurrences of
    :type char: ```str```

    :param end: True to look from the end; False to look from start
    :type end: ```bool```

    :param s_len: String len function to use, override to work at a shorter substr, e.g., `lambda _: 5`
    :type s_len: ```Callable[str, [int]]```

    :param start_idx: Index to start looking at string from, override to work at a shorter substr, e.g., `3`
    :type start_idx: ```int```

    :param char_f: char function, if `True` adds 1 to count. Overrides `char` if provided.
    :type char_f: ```Optional[Callable[[str], bool]]```

    :return: Number of chars counted (until `ignore`)
    :rtype: ```int```
    """
    char_count: int = 0

    if char_f is None:
        char_f: Callable[[str], bool] = rpartial(eq, char)

    for i in range(*((s_len(s) - 1, start_idx, -1) if end else (start_idx, s_len(s)))):
        if char_f(s[i]):
            char_count += 1
        elif not sentinel_char_unseen(s[i]):
            break
    return char_count


num_of_nls: Callable[[str], int] = partial(
    count_chars_from, sentinel_char_unseen=str.isspace, char="\n"
)


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

    :return: Whether IR is empty
    :rtype: ```bool```
    """
    return not intermediate_repr.get("doc") and not any(
        param_d is not None and (param_d.get("typ") or param_d.get("doc"))
        for key in ("params", "returns")
        for param_d in (intermediate_repr.get(key) or {}).values()
    )


if PY_GTE_3_10:
    from itertools import pairwise
else:

    def pairwise(iterable):
        """
        Return successive overlapping pairs taken from the input iterable.

        The number of 2-tuples in the output iterator will be one fewer than the number of inputs.
        It will be empty if the input iterable has fewer than two values.

        https://docs.python.org/3/library/itertools.html#itertools.pairwise
        but it's only avail. from 3.10 onwards

        :param iterable: An iterable
        :type iterable: ```Iterable```

        :return: A pair of 2-tuples or empty
        :rtype: ```zip```
        """
        # pairwise('ABCDEFG') --> AB BC CD DE EF FG
        a, b = tee(iterable)
        next(b, None)
        return zip(a, b)


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


class FilenameProtocol(Protocol):
    """
    Filename protocol
    """

    origin: str


def filename_from_mod_or_filename(mod_or_filename):
    """
    Resolve filename from module name or filename

    :param mod_or_filename: Module name or filename
    :type mod_or_filename: ```str```

    :return: Filename
    :rtype: ```str```
    """
    filename: FilenameProtocol = cast(
        FilenameProtocol, type("", tuple(), {"origin": mod_or_filename})
    )
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
    sep: str = tab * indent_level
    return "\n{sep}{}\n{sep}".format(
        run_per_line(
            "\n".join(map(lambda line: sep if len(line) == 0 else line, s.splitlines()))
        ),
        sep=sep,
    )


def ensure_valid_identifier(s):
    """
    Ensure identifier is valid

    :param s: Potentially valid identifier
    :type s: ```str```

    :return: Valid identifier from `s`
    :rtype: ```str```
    """
    if not s:
        return "_"
    elif iskeyword(s):
        return "{}_".format(s)
    elif s[0].isdigit():
        s: str = "_{}".format(s)
    valid: FrozenSet[str] = frozenset(
        "_{}{}".format(string.ascii_letters, string.digits)
    )
    return "".join(filter(valid.__contains__, s)) or "_"


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


def sliding_window(iterable, n):
    """
    Sliding window

    https://docs.python.org/3/library/itertools.html#itertools-recipes

    :param iterable: An iterable
    :type iterable: ```Iterable```

    :param n: Window size
    :type n: ```int```

    :return: sliding window
    :rtype: ```Generator[tuple]```
    """
    # sliding_window('ABCDEFG', 4) --> ABCD BCDE CDEF DEFG
    it = iter(iterable)
    window = deque(islice(it, n - 1), maxlen=n)
    for x in it:
        window.append(x)
        yield tuple(window)


class SetEncoder(JSONEncoder):
    """
    JSON encoder that supports `set`s
    """

    def default(self, obj):
        """
        Handle `set` by giving a sorted list in its place
        """
        return (sorted if isinstance(obj, set) else partial(JSONEncoder.default, self))(
            obj
        )


def pascal_to_upper_camelcase(s):
    """
    Transform pascal input to upper camelcase

    :param s: Pascal cased string
    :type s: ```str```

    :return: Upper camel case string
    :rtype: ```str```
    """
    return "".join(filterfalse(str.isspace, s.title().replace("_", "")))


def namespaced_pascal_to_upper_camelcase(s, sep="__"):
    """
    Convert potentially namespaced pascal to upper camelcase

    E.g., "foo__bar_can" becomes "Foo__BarCan"

    :param s: Pascal cased string (potentially with namespace, i.e., `sep`)
    :type s: ```str```

    :param sep: Separator (a.k.a., namespace)
    :type sep: ```str```

    :return: Upper camel case string (potentially with namespace)
    :rtype: ```str```
    """
    first, sep, last = s.rpartition(sep)
    return "{}{}{}".format(first.title(), sep, pascal_to_upper_camelcase(last))


def upper_camelcase_to_pascal(s):
    """
    Transform upper camelcase input to pascal case

    :param s: Upper camel case string
    :type s: ```str```

    :return: Pascal cased string
    :rtype: ```str```
    """
    return "_".join(
        map(
            str.lower,
            "".join((" {}".format(c) if c.isupper() else c) for c in s)
            .lstrip(" ")
            .split(" "),
        )
    )


def namespaced_upper_camelcase_to_pascal(s, sep="__"):
    """
    Convert potentially namespaced pascal to upper camelcase

    E.g., "Foo__BarCan" becomes "foo__bar_can"

    :param s: Upper camel case string (potentially with namespace, i.e., `sep`)
    :type s: ```str```

    :param sep: Separator (a.k.a., namespace)
    :type sep: ```str```

    :return: Pascal cased string (potentially with namespace)
    :rtype: ```str```
    """
    first, sep, last = s.rpartition(sep)
    return "{}{}{}".format(first.lower(), sep, upper_camelcase_to_pascal(last))


omit_whitespace: Callable[[str], str] = rpartial(
    str.translate, str.maketrans({" ": "", "\n": "", "\t": ""})
)

sanitise_emit_name: Callable[[str], str] = dict(
    **{
        typ: typ
        for typ in (
            "function",
            "json_schema",
            "pydantic",
            "sqlalchemy",
            "sqlalchemy_hybrid",
            "sqlalchemy_table",
        )
    },
    **{"class": "class_", "argparse": "argparse_function"},
).__getitem__

FakeConstant = type(
    "FakeConstant",
    tuple(),
    {
        "__init__": (
            lambda self, s=None, n=None, constant_value=None, string=None, col_offset=None, lineno=None: setattr(
                self, "s", s or n
            )
            or setattr(self, "n", self.s)
            or setattr(self, "value", self.s)
        )
    },
)

__all__ = [
    "BUILTIN_TYPES",
    "DUNDERS",
    "ENCODING",
    "FakeConstant",
    "INIT_FILENAME",
    "PY3_8",
    "PY_GTE_3_11",
    "PY_GTE_3_12",
    "PY_GTE_3_8",
    "PY_GTE_3_9",
    "SetEncoder",
    "all_dunder_for_module",
    "append_to_dict",
    "assert_equal",
    "balanced_parentheses",
    "blockwise",
    "code_quoted",
    "count_chars_from",
    "count_iter_items",
    "deindent",
    "diff",
    "emit_separating_tabs",
    "ensure_valid_identifier",
    "filename_from_mod_or_filename",
    "fill",
    "find_module_filepath",
    "get_module",
    "identity",
    "indent_all_but_first",
    "is_ir_empty",
    "is_triple_quoted",
    "location_within",
    "lstrip_namespace",
    "multiline",
    "namespaced_pascal_to_upper_camelcase",
    "namespaced_upper_camelcase_to_pascal",
    "none_types",
    "num_of_nls",
    "omit_whitespace",
    "pairwise",
    "paren_wrap_code",
    "parse_comment_from_line",
    "pascal_to_upper_camelcase",
    "pluralise",
    "pp",
    "quote",
    "read_file_to_str",
    "reindent",
    "remove_whitespace_comments",
    "rpartial",
    "sanitise",
    "sanitise_emit_name",
    "set_attr",
    "set_item",
    "simple_types",
    "sliding_window",
    "strip_split",
    "strip_starting",
    "tab",
    "type_to_name",
    "unquote",
    "update_d",
    "upper_camelcase_to_pascal",
    # "previous_line_range",
]  # type: list[str]
