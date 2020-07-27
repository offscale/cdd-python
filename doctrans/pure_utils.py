"""
Pure utils for pure functions. For the same input will always produce the same output.
"""

from pprint import PrettyPrinter

pp = PrettyPrinter(indent=4).pprint
tab = ' ' * 4
simple_types = {'int': 0, float: .0, 'str': '', 'bool': False}


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
