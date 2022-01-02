"""
Concrete Syntax Tree for Python 3.6+ source code
"""

from cdd.cst_utils import MultiLineComment, whitespace_aware_parse


def cst_parse(source_lines):
    """
    Parse Python source lines into a Concrete Syntax Tree

    :param source_lines: Python source lines
    :type source_lines: ```List[str]```

    :return: List of `namedtuple`s with at least ("line_no", "scope", "value") attributes
    :rtype: ```List[Any]```
    """
    concrete_lines, scope = [], []
    multi_line_comment = MultiLineComment(None, None, None, None, None, None)
    for line_no, line in enumerate(source_lines):
        multi_line_comment = whitespace_aware_parse(
            line_no=line_no,
            line=line,
            concrete_lines=concrete_lines,
            multi_line_comment=multi_line_comment,
            scope=scope,
        )
    return concrete_lines


__all__ = ["cst_parse"]
