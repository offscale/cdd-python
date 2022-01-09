"""
Concrete Syntax Tree for Python 3.6+ source code
"""

from cdd.cst_utils import cst_parser, cst_scanner


def cst_parse(source):
    """
    Parse Python source lines into a Concrete Syntax Tree

    :param source: Python source code
    :type source: ```str```

    :return: List of `namedtuple`s with at least ("line_no_start", "line_no_end", "value") attributes
    :rtype: ```List[Any]```
    """
    scanned = cst_scanner(source)
    parsed = cst_parser(scanned)
    return parsed


__all__ = ["cst_parse"]
