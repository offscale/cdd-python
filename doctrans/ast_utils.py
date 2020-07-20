from _ast import AnnAssign, Name, Load, Store, Constant, Dict, Module, ClassDef, Subscript, Tuple, Expr, Call, \
    Attribute, keyword
from ast import parse, Index
from collections import namedtuple

from doctrans.pure_utils import simple_types


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
                                        value=param.get('default', simple_types[param['typ']])))
    elif param['typ'] == 'dict' or param['typ'].startswith('*'):
        return AnnAssign(annotation=Name(ctx=Load(),
                                         id='dict'),
                         simple=1,
                         target=Name(ctx=Store(),
                                     id=param['name']),
                         value=Dict(keys=[],
                                    values=param.get('default', [])))
    else:
        return AnnAssign(annotation=parse(param['typ']).body[0].value,
                         simple=1,
                         target=Name(ctx=Store(),
                                     id=param['name']),
                         value=Constant(kind=None,
                                        value=param.get('default')))


def to_class_def(ast):
    """
    Converts an AST to an `ast.ClassDef`

    :param ast: Class AST or Module AST
    :type ast: ```Union[ast.Module, ast.ClassDef]```

    :return: ClassDef
    :rtype: ```ast.ClassDef```
    """
    if isinstance(ast, Module):
        classes = tuple(e
                        for e in ast.body
                        if isinstance(e, ClassDef))
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
                        if node.slice.value.value.id in frozenset(('Literal', 'Union')):
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
