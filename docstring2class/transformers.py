from ast import parse, ClassDef, Name, Load, Constant, Expr, Module

from astor import to_source
from black import format_str, FileMode

from docstring2class.info import parse_docstring
from docstring2class.utils import param2ast, tab


def ast2file(ast, filename, mode='a', skip_black=False):
    """
    Convert AST to a file

    :param ast: Constructed object of the `ast` class, usually an `ast.Module`
    :type ast: ```Union[ast.Module, ast.ClassDef]```

    :param filename: emit to this file
    :type filename: ```str```

    :param mode: Mode to open the file in, defaults to append
    :type mode: ```str```

    :param skip_black: Skip formatting with black
    :type skip_black: ```bool```

    :return: None
    :rtype: ```NoneType```
    """
    if isinstance(ast, ClassDef):
        ast = Module(body=[ast], type_ignores=[])
    src = to_source(ast)
    if not skip_black:
        src = format_str(src, mode=FileMode(
            target_versions=set(),
            line_length=119,
            is_pyi=False,
            string_normalization=False,
        ))
    with open(filename, mode) as f:
        f.write(src)


def docstring2ast(docstring, class_name='TargetClass', class_bases=('object',)):
    """
    Converts a docstring to an AST

    :param docstring: docstring portion
    :type docstring: ```str```

    :param class_name: name of class
    :type class_name: ```str```

    :param class_bases: bases of class (the generated class will inherit these)
    :type class_bases: ```Tuple[str]```

    :return: Class AST of the docstring
    :rtype: ```ast.ClassDef```
    """
    parsed = parse_docstring(docstring)

    returns = 'returns' in parsed and 'name' in parsed['returns']
    if returns:
        parsed['returns']['doc'] = parsed['returns']['name']
        parsed['returns']['name'] = 'return_type'

    return ClassDef(bases=[Name(ctx=Load(),
                                id=base_class)
                           for base_class in class_bases],
                    body=[
                             Expr(value=Constant(
                                 kind=None,
                                 value='\n    {description}\n\n{cvars}'.format(
                                     description=parsed['long_description'] or parsed['short_description'],
                                     cvars='\n'.join(
                                         '{tab}:cvar {param[name]}: {param[doc]}'.format(tab=tab, param=param)
                                         for param in parsed['params'] + ([parsed['returns']] if returns else [])
                                     )
                                 )
                             ))
                         ] + list(map(param2ast, parsed['params'])) + [
                             param2ast(parsed['returns'])
                         ] if parsed['returns'] else [],
                    decorator_list=[],
                    keywords=[],
                    name=class_name
                    )


def ast2docstring(ast):
    """
    Converts an AST to a docstring

    :param ast: Class AST or Module AST
    :type ast: ```Union[ast.Module, ast.ClassDef]```

    :return: docstring
    :rtype: ```str```
    """
    if not isinstance(ast, ClassDef):
        raise NotImplementedError(type(ast).__name__)
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
    :type feature_version: ```Optional[Tuple[int, int]]```

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
