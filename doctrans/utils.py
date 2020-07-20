from ast import Assign, Return, AnnAssign, Constant, Name, Expr, Call, Attribute, Tuple, Subscript
from collections import OrderedDict
from typing import Any

from astor import to_source

from doctrans.ast_utils import to_class_def
from doctrans.info import parse_docstring
from doctrans.pure_utils import simple_types
from doctrans.string_utils import extract_default


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
            docstring_struct[key][name] = {
                'doc': doc.lstrip(),
                'typ': None
            }
        elif docstring_struct[key]:
            docstring_struct[key][name]['doc'] += line
        else:
            docstring_struct['short_description'] += line

    for e in filter(lambda _e: isinstance(_e, AnnAssign), ast.body[1:]):
        name = e.target.id
        docstring_struct['returns' if name == 'return_type' else 'params'][name]['typ'] = \
            e.annotation.id if isinstance(e.annotation, Name) else to_source(e.annotation).rstrip()

    def interpolate_defaults(d):
        if 'doc' in d:
            doc, default = extract_default(d['doc'])
            if default:
                d.update({
                    'doc': doc,
                    'default': default
                })
        return d

    docstring_struct['params'] = [
        dict(name=k, **interpolate_defaults(v))
        for k, v in docstring_struct['params'].items()
    ]
    docstring_struct['returns'] = docstring_struct['returns']['return_type']

    return docstring_struct


def argparse_ast2docstring_structure(ast):
    """
    Converts an AST to a docstring structure

    :param ast: AST of argparse function
    :type ast: ```FunctionDef``

    :return: docstring structure
    :rtype: ```dict```
    """
    docstring_struct = {'short_description': '', 'long_description': '', 'params': []}
    for e in ast.body:
        if isinstance(e, Expr):
            if isinstance(e.value, Constant):
                if e.value.kind is not None:
                    raise NotImplementedError('kind')
                # docstring_struct['short_description'] = '\n'.join(
                #     takewhile(lambda l: not l.lstrip().startswith(':param'),
                #               e.value.value.split('\n'))).strip()
            elif (isinstance(e.value, Call) and len(e.value.args) == 1 and
                  isinstance(e.value.args[0], Constant) and
                  e.value.args[0].kind is None):
                required = next(
                    (keyword
                     for keyword in e.value.keywords
                     if keyword.arg == 'required'), Constant(value=False)
                ).value

                def handle_value(value):
                    if isinstance(value, Attribute):
                        return Any
                    elif isinstance(value, Name):
                        return 'dict' if value.id == 'loads' else value.id
                    raise NotImplementedError(type(value).__name__)

                typ = next((
                    handle_value(keyword.value)
                    for keyword in e.value.keywords
                    if keyword.arg == 'type'
                ), 'str')
                name = e.value.args[0].value[len('--'):]
                default = next(
                    (key_word.value.value
                     for key_word in e.value.keywords
                     if key_word.arg == 'default'),
                    None
                )
                doc = (
                    lambda help: (
                        help if default is None or (hasattr(default, '__len__') and len(
                            default) == 0) or 'defaults to' in help or 'Defaults to' in help
                        else '{help}. Defaults to {default}'.format(help=help, default=default)
                    )
                )(next(
                    key_word.value.value
                    for key_word in e.value.keywords
                    if key_word.arg == 'help'
                ))
                if default is None:
                    _, default = extract_default(doc)
                if default is None:
                    # if name.endswith('kwargs'):
                    #    default = {}
                    # required = True
                    # el
                    if typ in simple_types:
                        if required:
                            default = simple_types[typ]
                docstring_struct['params'].append(dict(
                    name=name,
                    doc=doc,
                    typ=(lambda typ: (typ if required or name.endswith('kwargs')
                                      else 'Optional[{typ}]'.format(typ=typ)))(
                        typ=next(
                            ('Union[{}]'.format(', '.join(elt.value if typ == Any
                                                          else '"{}"'.format(elt.value)
                                                          for elt in keyword.value.elts))
                             for keyword in e.value.keywords
                             if keyword.arg == 'choices'
                             ), typ
                        )
                    ),
                    **({} if default is None else {'default': default})
                ))
        elif isinstance(e, Assign):
            if all((len(e.targets) == 1,
                    isinstance(e.targets[0], Attribute),
                    e.targets[0].attr == 'description',
                    isinstance(e.targets[0].value, Name),
                    e.targets[0].value.id == 'argument_parser',
                    isinstance(e.value, Constant))):
                docstring_struct['short_description'] = e.value.value
        elif isinstance(e, Return) and isinstance(e.value, Tuple) and isinstance(e.value.elts[1], Subscript):
            docstring_struct['returns'] = {
                # 'name': 'return_type',
                'doc': next(
                    line.partition(',')[2].lstrip()
                    for line in ast.body[0].value.value.split('\n')
                    if line.lstrip().startswith(':return')
                ),
                'typ': to_source(e.value.elts[1]).replace('\n', '')
            }
    return docstring_struct


def docstring2docstring_structure(docstring):
    """
    Converts a docstring to an AST

    :param docstring: docstring portion
    :type docstring: ```Union[str, Dict]```

    :return: Class AST of the docstring
    :rtype: ```Tuple[dict, bool]```
    """
    parsed = docstring if isinstance(docstring, dict) else parse_docstring(docstring)
    returns = 'returns' in parsed and 'name' in parsed['returns']
    if returns:
        parsed['returns']['doc'] = parsed['returns'].get('doc', parsed['returns']['name'])
        # parsed['returns']['name'] = 'return_type'
    return parsed, returns


__all__ = ['class_ast2docstring_structure', 'argparse_ast2docstring_structure',
           'docstring2docstring_structure']
