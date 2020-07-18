from _ast import Module, ClassDef, Assign, Return
from ast import AnnAssign, Load, Constant, Name, Store, Dict, parse, Expr, Call, keyword, Attribute, Tuple, Subscript, \
    Index
from collections import OrderedDict, namedtuple
from itertools import takewhile
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
    # print('param', param, ';')
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


def class_ast2docstring_structure(ast):
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


def param2argparse_param(param):
    """
    Converts a param to an Expr `argparse.add_argument` call

    :param param: Param dict
    :type param: ```dict```

    :return: argparse.add_argument
    :rtype: ```Expr```
    """
    choices, required = None, False
    if param['typ'] in simple_types:
        typ = param['typ']
    else:
        parsed_type = parse(param['typ']).body[0]
        # print("print_ast(parse(param['typ']))")
        # print_ast(parsed_type)
        # print("</print_ast>")

        # print('parsed_type.value.slice.value:', parsed_type.value.slice.value, ';')
        # print_ast(parsed_type.value.slice.value)

        Param = namedtuple('Param', ('required', 'typ', 'choices'))

        def handle_name(node):
            assert isinstance(node, Name), 'Expected `Name` got `{}`'.format(type(node).__name__)
            if node.id == 'dict':
                _typ = 'loads'
            else:
                _typ = node.id

            return Param(
                required=False,
                typ=_typ,
                choices=None
            )

        def handle_subscript(node):
            assert isinstance(node, Subscript), 'Expected `Subscript` got `{}`'.format(type(node).__name__)
            _choices = None
            _typ = 'str'
            if isinstance(node.slice, Index):
                if isinstance(node.slice.value, Subscript):
                    if isinstance(node.slice.value.value, Name):
                        if node.slice.value.value.id == 'Literal':
                            if isinstance(node.slice, Index):
                                if isinstance(node.slice.value, Subscript):
                                    if isinstance(node.slice.value.slice.value, Tuple):
                                        _choices = tuple(node.id
                                                         for node in node.slice.value.slice.value.elts
                                                         if isinstance(node, Name))
                                        _typ = 'str'  # Convert later?

            return Param(
                required=node.value.id == 'Optional',
                typ=_typ,
                choices=_choices
            )

        if isinstance(parsed_type.value, Name):
            required, typ, choices = handle_name(parsed_type.value)
        elif isinstance(parsed_type.value.slice.value, Name):
            required, typ, choices = handle_name(parsed_type.value.value)
            required = parsed_type.value.value.id != 'Optional'  # TODO: Check for `None` in a `Union`
            typ = parsed_type.value.slice.value.id
            # if parsed_type.value.slice.value.id in simple_types:
            #    typ = parsed_type.value.slice.value.id
            # else:
            #    typ = None
        elif isinstance(parsed_type.value, Subscript):
            required, typ, choices = handle_subscript(parsed_type.value)
        else:
            raise NotImplementedError(type(parsed_type.value).__name__)
    return Expr(
        value=Call(args=[Constant(kind=None,
                                  value='--{param[name]}'.format(param=param))],
                   func=Attribute(attr='add_argument',
                                  ctx=Load(),
                                  value=Name(ctx=Load(),
                                             id='argument_parser')),
                   keywords=list(filter(None, (
                       keyword(arg='type',
                               value=Name(ctx=Load(),
                                          id=typ)),
                       choices if choices is None
                       else keyword(arg='choices',
                                    value=Tuple(ctx=Load(),
                                                elts=[Constant(kind=None,
                                                               value=choice)
                                                      for choice in choices])),
                       keyword(arg='help',
                               value=Constant(kind=None,
                                              value=param['doc'])),
                       keyword(arg='required',
                               value=Constant(kind=None,
                                              value=True)) if required else None
                   ))))
    )


def argparse_ast2docstring_structure(ast):
    """
    Converts an AST to a docstring structure

    :param ast: AST of argparse function
    :type ast: ```FunctionDef``

    :return: docstring structure
    :rtype: ```dict```
    """
    docstring_struct = {'params': []}
    for e in ast.body:
        if isinstance(e, Expr):
            if isinstance(e.value, Constant):
                if e.value.kind is not None:
                    raise NotImplementedError('kind')
                docstring_struct['short_description'] = '\n'.join(
                    takewhile(lambda l: not l.lstrip().startswith(':param'),
                              e.value.value.split('\n'))).strip()
            elif (isinstance(e.value, Call) and len(e.value.args) == 1 and
                  isinstance(e.value.args[0], Constant) and
                  e.value.args[0].kind is None):
                docstring_struct['params'].append({
                    'name': e.value.args[0].value[len('--'):],
                    'doc': next(keyword.value.value
                                for keyword in e.value.keywords
                                if keyword.arg == 'help'
                                ),
                    'typ':
                        next(('Literal[{}]'.format(', '.join('"{}"'.format(elt.value)
                                                             for elt in keyword.value.elts))
                              for keyword in e.value.keywords
                              if keyword.arg == 'choices'
                              ), next(('dict' if keyword.value.id == 'loads' else keyword.value.id
                                       for keyword in e.value.keywords
                                       if keyword.arg == 'type'
                                       ), 'str'))
                })
        elif isinstance(e, Assign):
            if all((len(e.targets) == 1,
                    isinstance(e.targets[0], Attribute),
                    e.targets[0].attr == 'description',
                    isinstance(e.targets[0].value, Name),
                    e.targets[0].value.id == 'argument_parser',
                    isinstance(e.value, Constant))):
                docstring_struct['long_description'] = e.value.value
        elif isinstance(e, Return) and isinstance(e.value, Tuple) and isinstance(e.value.elts[1], Subscript):
            docstring_struct['returns'] = {
                'name': 'return_type',
                'doc': next(line.partition(',')[2].lstrip()
                            for line in ast.body[0].value.value.split('\n')
                            if line.lstrip().startswith(':return')
                            ),
                'typ': to_source(e.value.elts[1]).replace('\n', '')
            }

    return docstring_struct


__all__ = ['param2ast', 'pp', 'tab', 'to_class_def', 'param2argparse_param',
           'class_ast2docstring_structure', 'argparse_ast2docstring_structure']
