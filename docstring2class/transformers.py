from ast import parse

from meta.asttools import python_source


def ast2file(ast, filename, mode='a'):
    """
    Convert AST to a file

    :param ast: Constructed object of the `ast` class, usually an `ast.Module`
    :type ast: ```ast.Module```

    :param filename: emit to this file
    :type filename: ```str```

    :param mode: Mode to open the file in, defaults to append
    :type mode: ```str```

    :return: None
    :rtype: ```NoneType```
    """
    with open(filename, mode) as f:
        python_source(ast, file=f)


def docstring2ast(docstring):
    """
    Converts a docstring to an AST

    :param docstring: docstring portion
    :type docstring: ```str```

    :return: Class AST of the docstring
    :rtype: ```ast.ClassDef```
    """
    raise NotImplementedError()


def ast2docstring(ast):
    """
    Converts a docstring to an AST

    :param ast: Class AST or Module AST
    :type ast: ```ast.Module or ast.ClassDef```

    :return: docstring
    :rtype: ```str```
    """
    raise NotImplementedError()


def class2ast(class_string, filename='<unknown>', mode='exec',
              type_comments=False, feature_version=None):
    """
    Converts a class to an AST

    :param class_string: class definition as a str
    :type class_string: ```str```

    :param filename: filename for ast.parse
    :type filename: ```str```

    :param mode: `mode` for `ast.parse`, defaults to 'exec'
    :type mode: ```str```

    :param type_comments: `type_comments` for `ast.parse`, defaults to False
    :type type_comments: ```bool```

    :param feature_version: `feature_version` for `ast.parse`, defaults to None
    :type feature_version: ```None or Tuple[int, int]```

    :return: Class AST
    :rtype: ```ast.ClassDef```
    """
    return parse(class_string, filename=filename, mode=mode,
                 type_comments=type_comments, feature_version=feature_version)


def class2docstring(class_string):
    """
    Converts a class to a docstring

    :param class_string: class definition as a str
    :type class_string: ```str```

    :return: docstring
    :rtype: ```str```
    """
    return ast2docstring(class2ast(class_string))
