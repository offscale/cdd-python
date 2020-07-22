"""
Transform from string or AST representations of input, to AST, file, or str output.
"""

from ast import parse, ClassDef, Name, Load, Constant, Expr, Module, FunctionDef, arguments, arg, Assign, Attribute, \
    Store, Tuple, Return
from functools import partial

from astor import to_source
from black import format_str, FileMode

from doctrans.ast_utils import param2ast, param2argparse_param
from doctrans.defaults_utils import extract_default
from doctrans.docstring_structure_utils import class_def2docstring_structure, argparse_ast2docstring_structure, \
    docstring2docstring_structure
from doctrans.pure_utils import tab


def ast2file(ast, filename, mode='a', skip_black=False):
    """
    Convert AST to a file

    :param ast: Constructed object of the `class_def` class, usually an `ast.Module`
    :type ast: ```Union[Module, ClassDef, FunctionDef]```

    :param filename: emit to this file
    :type filename: ```str```

    :param mode: Mode to open the file in, defaults to append
    :type mode: ```str```

    :param skip_black: Skip formatting with black
    :type skip_black: ```bool```

    :return: None
    :rtype: ```NoneType```
    """
    if isinstance(ast, (ClassDef, FunctionDef)):
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


def docstring2class_def(docstring, class_name='TargetClass',
                        class_bases=('object',), with_default_doc=True):
    """
    Converts a docstring to an AST

    :param docstring: docstring portion
    :type docstring: ```Union[str, Dict]```

    :param class_name: name of class
    :type class_name: ```str```

    :param class_bases: bases of class (the generated class will inherit these)
    :type class_bases: ```Tuple[str]```

    :param with_default_doc: Help/docstring should include 'With default' text
    :type with_default_doc: ```bool``

    :return: Class AST of the docstring
    :rtype: ```ast.ClassDef```
    """
    parsed, returns = docstring2docstring_structure(docstring, with_default_doc=with_default_doc)
    if parsed.get('returns'):
        returns = [parsed['returns']]
        returns[0]['name'] = 'return_type'
    else:
        returns = []
    return ClassDef(
        bases=[Name(ctx=Load(),
                    id=base_class)
               for base_class in class_bases],
        body=[
                 Expr(value=Constant(
                     kind=None,
                     value='\n    {description}\n\n{cvars}'.format(
                         description=parsed['long_description'] or parsed['short_description'],
                         cvars='\n'.join(
                             '{tab}:cvar {param[name]}: {param[doc]}'.format(tab=tab, param=param)
                             for param in parsed['params'] + returns
                         )
                     )
                 ))
             ] + list(map(param2ast, parsed['params'] + returns)),
        decorator_list=[],
        keywords=[],
        name=class_name
    )


def class_def2docstring(class_def, with_default_doc=True):
    """
    Converts an AST to a docstring

    :param class_def: Class AST or Module AST with a ClassDef inside
    :type class_def: ```Union[ast.Module, ast.ClassDef]```

    :param with_default_doc: Help/docstring should include 'With default' text
    :type with_default_doc: ```bool``

    :return: docstring
    :rtype: ```str```
    """
    docstring_struct = class_def2docstring_structure(class_def)
    return '''\n{description}\n\n{params}\n{returns}\n'''.format(
        description=docstring_struct['long_description'] or docstring_struct['short_description'],
        params='\n'.join(':param {param[name]}: {param[doc]}\n'
                         ':type {param[name]}: ```{typ}```\n'.format(param=param,
                                                                     typ=('**{name}'.format(name=param['name'])
                                                                          if 'kwargs' in param['name']
                                                                          else param['typ']))
                         for param in docstring_struct['params']),
        returns=':return: {param[doc]}\n'
                ':rtype: ```{param[typ]}```'.format(param=docstring_struct['returns'])
    )


def str2ast(python_source_str, filename='<unknown>', mode='exec',
            type_comments=False, feature_version=None):
    """
    Converts source_code to an AST

    :param python_source_str: class definition as a str
    :type python_source_str: ```str```

    :param filename: filename for class_def.parse
    :type filename: ```str```

    :param mode: `mode` for `class_def.parse`, defaults to 'exec'
    :type mode: ```str```

    :param type_comments: `type_comments` for `class_def.parse`, defaults to False
    :type type_comments: ```bool```

    :param feature_version: `feature_version` for `class_def.parse`, defaults to None
    :type feature_version: ```Optional[Tuple[int, int]]```

    :return: Class AST
    :rtype: ```ast.ClassDef```
    """
    return parse(python_source_str, filename=filename, mode=mode,
                 type_comments=type_comments, feature_version=feature_version)


def class2docstring(class_string, with_default_doc=True):
    """
    Converts a class to a docstring

    :param class_string: class definition as a str
    :type class_string: ```str```

    :param with_default_doc: Help/docstring should include 'With default' text
    :type with_default_doc: ```bool``

    :return: docstring
    :rtype: ```str```
    """
    return class_def2docstring(str2ast(class_string, with_default_doc=with_default_doc),
                               with_default_doc=with_default_doc)


def ast2argparse(ast, function_name='set_cli_args', with_default_doc=False):
    """

    :param ast: Class AST or Module AST
    :type ast: ```Union[ast.Module, ast.ClassDef]```

    :param function_name: name of function
    :type function_name: ```str```

    :param with_default_doc: Help/docstring should include 'With default' text
    :type with_default_doc: ```bool``
    """
    docstring_struct = class_def2docstring_structure(ast)
    doc, _default = extract_default(docstring_struct['returns']['doc'],
                                    with_default_doc=with_default_doc)
    docstring_struct['returns']['doc'] = doc
    return FunctionDef(args=arguments(args=[arg(annotation=None,
                                                arg='argument_parser',
                                                type_comment=None)],
                                      defaults=[],
                                      kw_defaults=[],
                                      kwarg=None,
                                      kwonlyargs=[],
                                      posonlyargs=[],
                                      vararg=None),
                       body=[
                                Expr(value=Constant(kind=None,
                                                    value='\n    Set CLI arguments\n\n    '
                                                          ':param argument_parser: argument parser\n    '
                                                          ':type argument_parser: ```ArgumentParser```\n\n    '
                                                          ':return: argument_parser, {returns[doc]}\n    '
                                                          ':rtype: ```Tuple[ArgumentParser,'
                                                          ' {returns[typ]}]```\n    '.format(
                                                        returns=docstring_struct['returns']))),
                                Assign(targets=[Attribute(attr='description',
                                                          ctx=Store(),
                                                          value=Name(ctx=Load(),
                                                                     id='argument_parser'))],
                                       type_comment=None,
                                       value=Constant(
                                           kind=None,
                                           value=docstring_struct['long_description'] or docstring_struct[
                                               'short_description']))
                            ] + list(map(partial(param2argparse_param, with_default_doc=with_default_doc),
                                         docstring_struct['params'])) + [
                                Return(
                                    value=Tuple(
                                        ctx=Load(),
                                        elts=[
                                            Name(ctx=Load(),
                                                 id='argument_parser'),
                                            parse(docstring_struct['returns']['default']).body[0].value
                                        ]))],
                       decorator_list=[],
                       name=function_name,
                       returns=None,
                       type_comment=None)


def argparse2class(ast, class_name='TargetClass', with_default_doc=True):
    """
    Converts an argparse function to a class

    :param ast: AST of argparse function
    :type ast: ```FunctionDef```

    :param class_name: class name
    :type class_name: ```str```

    :param with_default_doc: Help/docstring should include 'With default' text
    :type with_default_doc: ```bool``

    :return: docstring
    :rtype: ```str```
    """
    assert isinstance(ast, FunctionDef), 'Expected `FunctionDef` got: `{}`'.format(type(ast).__name__)
    docstring_struct = argparse_ast2docstring_structure(ast, with_default_doc=with_default_doc)
    if 'returns' in docstring_struct:
        docstring_struct['returns']['name'] = 'return_type'
    return docstring2class_def(docstring_struct, class_name=class_name, with_default_doc=with_default_doc)
