from platform import python_version_tuple


def to_code(node):
    """
    Convert the AST input to Python source string

    :param node: AST node
    :type node: ```AST```

    :returns: Python source
    :rtype: ```str```
    """

    if python_version_tuple() < ('3', '9'):
        from ast import unparse
        return unparse(node)
    else:
        from astor import to_source
        return to_source(node)
