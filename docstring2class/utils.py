from _ast import Module, ClassDef
from ast import AnnAssign, Load, Constant, Name, Store, Dict, parse
from collections import OrderedDict
from pprint import PrettyPrinter

from astor import to_source

pp = PrettyPrinter(indent=4).pprint

tab = ' ' * 4

simple_types = {'int': 0, float: .0, 'str': '', 'bool': True}


def param2ast(param):
    """
    Converts a param to an AnnAssign

    :param param: dictionary of shape {'typ': str, 'name': str, 'doc': str}
    :type param: ```dict```

    :return: ast node (AnnAssign)
    :rtype: ```AnnAssign```
    """
    if param['typ'] in simple_types:
        return AnnAssign(annotation=Name(ctx=Load(),
                                         id=param['typ']),
                         simple=1,
                         target=Name(ctx=Store(),
                                     id=param['name']),
                         value=Constant(kind=None,
                                        value=simple_types[param['typ']]))
    elif param['typ'] == 'dict':
        return AnnAssign(annotation=Name(ctx=Load(),
                                         id=param['typ']),
                         simple=1,
                         target=Name(ctx=Store(),
                                     id=param['name']),
                         value=Dict(keys=[],
                                    values=[]))
    elif param['typ'].startswith('*'):
        return AnnAssign(annotation=Name(ctx=Load(),
                                         id='dict'),
                         simple=1,
                         target=Name(ctx=Store(),
                                     id=param['name']),
                         value=Dict(keys=[],
                                    values=[]))
    else:
        return AnnAssign(annotation=parse(param['typ']).body[0].value,
                         simple=1,
                         target=Name(ctx=Store(),
                                     id=param['name']),
                         value=Constant(kind=None,
                                        value=None))


def to_class_def(ast):
    """
    Converts an AST to an `ast.ClassDef`

    :param ast: Class AST or Module AST
    :type ast: ```Union[ast.Module, ast.ClassDef]```

    :return: ClassDef
    :rtype: ```ast.ClassDef```
    """
    if isinstance(ast, Module):
        classes = tuple(e for e in ast.body if isinstance(e, ClassDef))
        if len(classes) > 1:  # We can filter by name I guess? - Or convert every one?
            raise NotImplementedError()
        elif len(classes) > 0:
            return classes[0]
        else:
            raise TypeError('No ClassDef in ast')
    elif isinstance(ast, ClassDef):
        return ast
    else:
        raise NotImplementedError(type(ast).__name__)


def ast2docstring_structure(ast):
    """
    Converts an AST to a docstring structure

    :param ast: Class AST or Module AST
    :type ast: ```Union[ast.Module, ast.ClassDef]```

    :return: docstring structure
    :rtype: ```dict```
    """
    ast = to_class_def(ast)
    docstring_struct = {
        'short_description': '',
        'long_description': '',
        'params': OrderedDict(),
        'returns': {}
    }
    name, key = None, 'params'
    for line in filter(None, map(lambda l: l.lstrip(),
                                 ast.body[0].value.value.replace(':cvar', ':param').split('\n'))):
        if line.startswith(':param'):
            name, _, doc = line.rpartition(':')
            name = name.replace(':param ', '')
            key = 'returns' if name == 'return_type' else 'params'
            docstring_struct[key][name] = {'doc': doc.lstrip(), 'typ': None}
        elif docstring_struct[key]:
            docstring_struct[key][name]['doc'] += line
        else:
            docstring_struct['short_description'] += line

    for e in filter(lambda _e: isinstance(_e, AnnAssign), ast.body[1:]):
        name = e.target.id
        docstring_struct['returns' if name == 'return_type' else 'params'][name]['typ'] = \
            e.annotation.id if isinstance(e.annotation, Name) else to_source(e.annotation).rstrip()

    docstring_struct['params'] = [dict(name=k, **v)
                                  for k, v in docstring_struct['params'].items()]
    docstring_struct['returns'] = docstring_struct['returns']['return_type']

    return docstring_struct


__all__ = ['param2ast', 'pp', 'tab', 'to_class_def', 'ast2docstring_structure']
