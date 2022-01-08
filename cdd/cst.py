"""
Concrete Syntax Tree for Python 3.6+ source code
"""

from cdd.cst_utils import cst_parser, cst_scanner
from cdd.pure_utils import pp


def cst_parse(source):
    """
    Parse Python source lines into a Concrete Syntax Tree

    :param source: Python source code
    :type source: ```str```

    :return: List of `namedtuple`s with at least ("line_no", "scope", "value") attributes
    :rtype: ```List[Any]```
    """
    scanned = cst_scanner(source)
    parsed = cst_parser(scanned)
    pp(parsed)
    return parsed


__all__ = ["cst_parse"]
