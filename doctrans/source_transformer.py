"""
Source transformer module. Uses astor on Python < 3.9
"""
from importlib import import_module
from platform import python_version_tuple


def to_code(node):
    """
    Convert the AST input to Python source string

    :param node: AST node
    :type node: ```AST```

    :returns: Python source
    :rtype: ```str```
    """

    return (
        getattr(import_module("astor"), "to_source")
        if python_version_tuple() < ("3", "9")
        else getattr(import_module("ast"), "unparse")
    )(node)
