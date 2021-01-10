"""
Pure utils for pure functions. For the same input will always produce the same input_str.
"""
import typing
from ast import Str
from collections import OrderedDict, deque
from functools import partial
from importlib import import_module
from inspect import getmodule
from itertools import chain, count, zip_longest
from keyword import iskeyword
from operator import attrgetter, eq
from pprint import PrettyPrinter
from sys import version_info

pp = PrettyPrinter(indent=4, width=80).pprint
tab = " " * 4
simple_types = {
    "int": 0,
    "float": 0.0,
    "complex": 0j,
    "str": "",
    "bool": False,
    None: None,
}


# From https://github.com/Suor/funcy/blob/0ee7ae8/funcy/funcs.py#L34-L36
def rpartial(func, *args):
    """Partially applies last arguments."""
    return lambda *a: func(*(a + args))


def identity(*args):
    """
    Identity function

    :param args: Any values
    :type args: ```Tuple[Any]```

    :return: the input value
    :rtype: ```Any```
    """
    return args[0] if len(args) == 1 else args


_python_major_minor = version_info[:2]
PY3_8 = _python_major_minor == (3, 8)
PY_GTE_3_8 = _python_major_minor >= (3, 8)
PY_GTE_3_9 = _python_major_minor >= (3, 9)

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


def deindent(s):
    """
    Remove all indentation from the input string

    :param s: Input string
    :type s: ```AnyStr```

    :return: Deindented string
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

    :return: Reindented string
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

    :param s: Input string
    :type s: ```str```

    :param mark: Quote mark to wrap with
    :type mark: ```str```

    :return: Quoted string
    :rtype: ```str```
    """
    s = (
        s
        if isinstance(s, (str, type(None)))
        else s.s
        if isinstance(s, Str)
        else s.value
    )
    # ^ Poor man's `get_value`
    if s is None or len(s) == 0 or s[0] == s[-1] and s[0] in frozenset(("'", '"')):
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
    assert cmp(a, b), "{!r} != {!r}".format(a, b)
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
    return zip_longest(*[iter(t)] * size, fillvalue=fillvalue)


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


BUILTIN_TYPES = (
    frozenset(
        chain.from_iterable(
            map(
                lambda s: (s, "typing.{}".format(s), "_extensions.{}".format(s)),
                filter(lambda s: s[0].isupper() and not s.isupper(), dir(typing)),
            )
        )
    )
    | frozenset(("int", "float", "str", "dict", "list", "tuple"))
)


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


def params_to_ordered_dict(params):
    """
    Convert the old params list with dicts to an OrderedDict

    TODO: Remove this function when codebase is updated

    :param params: list of dict of shape {'name': ..., 'typ': ..., 'doc': ..., 'required': ... }
    :type params: ```List[dict]```

    :return: OrderedDict representation of the params dict, i.e.,
       OrderedDict[str, {'typ': str, 'doc': Optional[str], 'default': Any}]
    :rtype: ```OrderedDict```
    """
    return OrderedDict(
        (param.pop("name"), param)
        for param in ((params,) if isinstance(params, dict) else params)
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


__all__ = [
    "BUILTIN_TYPES",
    "PY3_8",
    "PY_GTE_3_8",
    "PY_GTE_3_9",
    "assert_equal",
    "blockwise",
    "count_iter_items",
    "diff",
    "get_module",
    "identity",
    "location_within",
    "lstrip_namespace",
    "params_to_ordered_dict",
    "paren_wrap_code",
    "pluralise",
    "pp",
    "quote",
    "reindent",
    "rpartial",
    "sanitise",
    "simple_types",
    "strip_split",
    "tab",
    "update_d",
]
