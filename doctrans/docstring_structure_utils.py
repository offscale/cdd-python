"""
Functions which produce docstring_structure from various different inputs
"""

from ast import Assign, Return, AnnAssign, Constant, Name, Expr, Call, Attribute, Tuple, Subscript, parse, \
    get_docstring
from collections import OrderedDict
from typing import Any

from astor import to_source

from doctrans.ast_utils import to_class_def
from doctrans.rest_docstring_parser import parse_docstring
from doctrans.pure_utils import simple_types
from doctrans.defaults_utils import extract_default


def class_def2docstring_structure(class_def, with_default_doc=True):
    """
    Converts an AST to a docstring structure

    :param class_def: Class AST or Module AST with a ClassDef inside
    :type class_def: ```Union[Module, ClassDef]```

    :param with_default_doc: Help/docstring should include 'With default' text
    :type with_default_doc: ```bool``

    :return: docstring structure
    :rtype: ```dict```
    """
    class_def = to_class_def(class_def)
    docstring_struct = {
        'short_description': '',
        'long_description': '',
        'params': OrderedDict(),
        'returns': {}
    }
    name, key = None, 'params'
    for line in filter(None, map(lambda l: l.lstrip(),
                                 class_def.body[0].value.value.replace(':cvar', ':param').split('\n'))):
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

    for e in filter(lambda _e: isinstance(_e, AnnAssign), class_def.body[1:]):
        name = e.target.id
        docstring_struct['returns' if name == 'return_type' else 'params'][name]['typ'] = \
            e.annotation.id if isinstance(e.annotation, Name) else to_source(e.annotation).rstrip()

    def interpolate_defaults(param):
        """
        Correctly set the 'default' and 'doc' parameters

        :param param: dict of shape {'name': ..., 'typ': ..., 'doc': ..., 'required': ... }
        :type param: ```dict```

        :returns: dict of shape {'name': ..., 'typ': ..., 'doc': ..., 'default': ..., 'required': ... }
        :rtype: ```dict```
        """
        if 'doc' in param:
            doc, default = extract_default(param['doc'], with_default_doc=with_default_doc)
            param['doc'] = doc
            if default:
                param['default'] = default
        return param

    docstring_struct['params'] = [
        dict(name=k, **interpolate_defaults(v))
        for k, v in docstring_struct['params'].items()
    ]
    docstring_struct['returns'] = interpolate_defaults(
        docstring_struct['returns']['return_type']
    )

    return docstring_struct


def argparse_ast2docstring_structure(function_def, with_default_doc=False):
    """
    Converts an AST to a docstring structure

    :param function_def: AST of argparse function
    :type function_def: ```FunctionDef``

    :param with_default_doc: Help/docstring should include 'With default' text
    :type with_default_doc: ```bool``

    :return: docstring structure
    :rtype: ```dict```
    """
    docstring_struct = {'short_description': '', 'long_description': '', 'params': []}
    _docstring_struct = parse_docstring(get_docstring(function_def), with_default_doc=with_default_doc)
    for e in function_def.body:
        if isinstance(e, Expr):
            if isinstance(e.value, Constant):
                if e.value.kind is not None:
                    raise NotImplementedError('kind')
                # docstring_struct['short_description'] = '\n'.join(
                #     takewhile(lambda l: not l.lstrip().startswith(':param'),
                #               expr.value.value.split('\n'))).strip()
            elif (isinstance(e.value, Call) and len(e.value.args) == 1 and
                  isinstance(e.value.args[0], Constant) and
                  e.value.args[0].kind is None):
                docstring_struct['params'].append(parse_out_param(e, with_default_doc=with_default_doc))
        elif isinstance(e, Assign):
            if all((len(e.targets) == 1,
                    isinstance(e.targets[0], Attribute),
                    e.targets[0].attr == 'description',
                    isinstance(e.targets[0].value, Name),
                    e.targets[0].value.id == 'argument_parser',
                    isinstance(e.value, Constant))):
                docstring_struct['short_description'] = e.value.value
        elif isinstance(e, Return) and isinstance(e.value, Tuple):
            if isinstance(e.value.elts[1], Subscript):
                docstring_struct['returns'] = {
                    # 'name': 'return_type',
                    'doc': next(
                        line.partition(',')[2].lstrip()
                        for line in function_def.body[0].value.value.split('\n')
                        if line.lstrip().startswith(':return')
                    ),
                    'typ': to_source(e.value.elts[1]).replace('\n', '')
                }
            else:
                default = to_source(e.value.elts[1]).replace('\n', '')
                doc = next(
                    line.partition(',')[2].lstrip()
                    for line in function_def.body[0].value.value.split('\n')
                    if line.lstrip().startswith(':return')
                )
                if not with_default_doc:
                    doc, _ = extract_default(doc, with_default_doc=with_default_doc)

                docstring_struct['returns'] = {
                    # 'name': 'return_type',
                    'doc': '{doc}{maybe_dot} Defaults to {default}'.format(
                        maybe_dot='' if doc.endswith('.') else '.',
                        doc=doc,
                        default=default
                    ) if all((default is not None,
                              with_default_doc,
                              'Defaults to' not in doc,
                              'defaults to' not in doc))
                    else doc,
                    'default': default,
                    'typ': to_source(
                        parse(_docstring_struct['returns']['typ']).body[0].value.slice.value.elts[1]
                    ).rstrip()
                    # 'Tuple[ArgumentParser, {typ}]'.format(typ=_docstring_struct['returns']['typ'])
                }
    return docstring_struct


def parse_out_param(expr, with_default_doc=True):
    """
    Turns the class_def repr of '--dataset_name', type=str, help='name of dataset.', required=True, default='mnist'
      into
          {'name': 'dataset_name', 'typ': 'str', doc='name of dataset.',
           'required': True, 'default': 'mnist'}

    :param expr: Expr
    :type expr: ```Expr```

    :param with_default_doc: Help/docstring should include 'With default' text
    :type with_default_doc: ```bool``

    :returns: dict of shape {'name': ..., 'typ': ..., 'doc': ..., 'default': ..., 'required': ... }
    :rtype ```dict```
    """
    required = next(
        (keyword
         for keyword in expr.value.keywords
         if keyword.arg == 'required'),
        Constant(value=False)
    ).value

    def handle_value(node):
        """
        Handle keyword.value types, returning the correct one as a `str` or `Any`

        :param node: AST node from keyword.value
        :type node: ```Union[Attribute, Name]```

        :returns: `str` or `Any`, representing the type for argparse
        :rtype: ```Union[str, Any]```
        """
        if isinstance(node, Attribute):
            return Any
        elif isinstance(node, Name):
            return 'dict' if node.id == 'loads' else node.id
        raise NotImplementedError(type(node).__name__)

    typ = next((
        handle_value(keyword.value)
        for keyword in expr.value.keywords
        if keyword.arg == 'type'
    ), 'str')
    name = expr.value.args[0].value[len('--'):]
    default = next(
        (key_word.value.value
         for key_word in expr.value.keywords
         if key_word.arg == 'default'),
        None
    )
    doc = (
        lambda help: (
            help if default is None or with_default_doc is False or (hasattr(default, '__len__') and len(
                default) == 0) or 'defaults to' in help or 'Defaults to' in help
            else '{help} Defaults to {default}'.format(
                help=help if help.endswith('.') else '{}.'.format(help),
                default=default
            )
        )
    )(next(
        key_word.value.value
        for key_word in expr.value.keywords
        if key_word.arg == 'help'
    ))
    if default is None:
        doc, default = extract_default(doc, with_default_doc=with_default_doc)
    if default is None:
        # if name.endswith('kwargs'):
        #    default = {}
        # required = True
        # el
        if typ in simple_types:
            if required:
                default = simple_types[typ]
    return dict(
        name=name,
        doc=doc,
        typ=(lambda typ: (typ if required or name.endswith('kwargs')
                          else 'Optional[{typ}]'.format(typ=typ)))(
            typ=next(
                ('Union[{}]'.format(', '.join(elt.value if typ == Any
                                              else '"{}"'.format(elt.value)
                                              for elt in keyword.value.elts))
                 for keyword in expr.value.keywords
                 if keyword.arg == 'choices'
                 ), typ
            )
        ),
        **({} if default is None else {'default': default})
    )


def docstring2docstring_structure(docstring, with_default_doc=True):
    """
    Converts a docstring to an AST

    :param docstring: docstring portion
    :type docstring: ```Union[str, Dict]```

    :param with_default_doc: Help/docstring should include 'With default' text
    :type with_default_doc: ```bool``

    :return: docstring_structure, whether it returns or not
    :rtype: ```Tuple[dict, bool]```
    """
    parsed = docstring if isinstance(docstring, dict) \
        else parse_docstring(docstring, with_default_doc=with_default_doc)
    returns = 'returns' in parsed and 'name' in parsed['returns']
    if returns:
        parsed['returns']['doc'] = parsed['returns'].get('doc', parsed['returns']['name'])
        # parsed['returns']['name'] = 'return_type'
    return parsed, returns


__all__ = ['class_def2docstring_structure',
           'argparse_ast2docstring_structure',
           'docstring2docstring_structure']
