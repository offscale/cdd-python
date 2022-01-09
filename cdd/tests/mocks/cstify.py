# pragma: no cover
# flake8: noqa
# === Copyright 2022 under CC0
"""Module docstring goes here"""

from operator import add


class C(object):
    """My cls"""

    @staticmethod
    def add1(foo):
        """
        :param foo: a foo
        :type foo: ```int```

        :return: foo + 1
        :rtype: ```int```
        """

        """foo"""
        def g(): """foo : bar ; can"""; pass

        def h(): # stuff
            pass

        def adder(a: int,
                  b: int) -> int:
            """
            :param a: First arg

            :param b: Second arg

            :return: first + second arg
            """
            # fmt: off
            res: \
                int \
                = a + b
            return res

        r = (
            add(foo, 1)
            or
            adder(foo, 1)
        )
        if r:
            None
        elif r:
            True
            False
            # ([5,5] @ [5,5]) *\
            -5 / 7 ** 6 + \
            6.0 - 6e1 & 1+2.34j
            r <<= 5
            print(r)
        else:
            pass
        # fmt: on
        # That^ incremented `foo` by 1
        return r


# from contextlib import ContextDecorator

# with ContextDecorator():
#    pass


def f():
    return 1
