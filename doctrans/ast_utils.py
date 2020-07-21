"""
ast_utils, bunch of helpers for converting input into ast.* output
"""
from ast import AnnAssign, Name, Load, Store, Constant, Dict, Module, ClassDef, Subscript, Tuple, Expr, Call, \
    Attribute, keyword, parse, walk

from doctrans.pure_utils import simple_types
from doctrans.defaults_utils import extract_default


def param2ast(param):
    """
    Converts a param to an AnnAssign

    :param param: dict of shape {'name': ..., 'typ': ..., 'doc': ..., 'default': ..., 'required': ... }
    :type param: ```dict```

    :return: AST node (AnnAssign)
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
        annotation = parse(param['typ']).body[0].value

        def determine_quoting(node):
            """
            Determine whether the input needs to be quoted

            :param node: AST node
            :type node: ```Union[Subscript, Tuple, Name, Attribute]```

            :returns: True if input needs quoting
            :rtype: ```bool```
            """
            if isinstance(node, Subscript) and isinstance(node.value, Name):
                if node.value.id == 'Optional':
                    return determine_quoting(node.slice.value)
                elif node.value.id == 'Union':
                    if all(True
                           for elt in node.slice.value.elts
                           if isinstance(elt, Subscript)):
                        return any(determine_quoting(elt)
                                   for elt in node.slice.value.elts)
                    return any(True
                               for elt in node.slice.value.elts
                               if elt.id == 'str')
                elif node.value.id == 'Tuple':
                    return any(determine_quoting(elt)
                               for elt in node.slice.value.elts)
                else:
                    raise NotImplementedError(node.value.id)
            elif isinstance(node, Name):
                return node.id == 'str'
            elif isinstance(node, Attribute):
                return determine_quoting(node.value)
            else:
                raise NotImplementedError(type(node).__name__)

        if param.get('default') and not determine_quoting(annotation):
            value = parse(param['default']).body[0].value if 'default' in param \
                else Name(ctx=Load(), id=None)
        else:
            value = Constant(kind=None,
                             value=param.get('default'))

        return AnnAssign(
            annotation=annotation,
            simple=1,
            target=Name(ctx=Store(),
                        id=param['name']),
            value=value
        )


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
            raise TypeError('No ClassDef in AST')
    elif isinstance(ast, ClassDef):
        return ast
    else:
        raise NotImplementedError(type(ast).__name__)


def param2argparse_param(param, with_default_doc=True):
    """
    Converts a param to an Expr `argparse.add_argument` call

    :param param: dict of shape {'name': ..., 'typ': ..., 'doc': ..., 'default': ..., 'required': ... }
    :type param: ```dict```

    :param with_default_doc: Help/docstring should include 'With default' text
    :type with_default_doc: ```bool``

    :return: `argparse.add_argument` call—with arguments—as an AST node
    :rtype: ```Expr```
    """
    typ, choices, required = 'str', None, True
    if param['typ'] in simple_types:
        typ = param['typ']
    elif param['typ'] == 'dict':
        typ = 'loads'
        required = not param['name'].endswith('kwargs')
    else:
        parsed_type = parse(param['typ']).body[0]

        for node in walk(parsed_type):
            if isinstance(node, Tuple):
                maybe_choices = tuple(elt.id
                                      for elt in node.elts
                                      if isinstance(elt, Name))
                if len(maybe_choices) == len(node.elts):
                    choices = maybe_choices
            elif isinstance(node, Name):
                if node.id == 'Optional':
                    required = False
                elif node.id in simple_types:
                    typ = node.id
                elif node.id in frozenset(('Union',)):
                    pass
                else:
                    typ = 'globals().__getitem__'

    doc, _default = extract_default(param['doc'], with_default_doc=with_default_doc)
    default = param.get('default', _default)

    return Expr(
        value=Call(args=[Constant(kind=None,
                                  value='--{param[name]}'.format(param=param))],
                   func=Attribute(attr='add_argument',
                                  ctx=Load(),
                                  value=Name(ctx=Load(),
                                             id='argument_parser')),
                   keywords=list(filter(None, (
                       keyword(
                           arg='type',
                           value=Attribute(
                               attr='__getitem__',
                               ctx=Load(),
                               value=Call(args=[],
                                          func=Name(ctx=Load(),
                                                    id='globals'),
                                          keywords=[])
                           ) if typ == 'globals().__getitem__'
                           else Name(ctx=Load(), id=typ)
                       ),
                       choices if choices is None
                       else keyword(arg='choices',
                                    value=Tuple(ctx=Load(),
                                                elts=[Constant(kind=None,
                                                               value=choice)
                                                      for choice in choices])),
                       keyword(arg='help',
                               value=Constant(kind=None,
                                              value=doc)),
                       keyword(arg='required',
                               value=Constant(kind=None,
                                              value=True)) if required else None,
                       default if default is None
                       else keyword(arg='default',
                                    value=Constant(kind=None,
                                                   value=default))
                   ))))
    )
