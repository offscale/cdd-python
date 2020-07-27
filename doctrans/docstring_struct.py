"""
Transform from string or AST representations of input, to docstring_structure dict of shape {
            'name': ..., 'platform': ...,
            'module': ..., 'title': ..., 'description': ...,
            'parameters': ..., 'schema': ...,'returns': ...}.
"""

from ast import AnnAssign, Name, FunctionDef, Return, Expr, Constant, Call, \
    Assign, Attribute, Tuple, get_docstring, parse, Subscript
from collections import OrderedDict
from functools import partial
from operator import itemgetter

from astor import to_source
from docstring_parser import DocstringParam, DocstringMeta

from doctrans.ast_utils import to_class_def, get_function_type
from doctrans.defaults_utils import remove_defaults_from_docstring_structure, extract_default
from doctrans.docstring_structure_utils import parse_out_param
from doctrans.pure_utils import tab, rpartial
from doctrans.rest_docstring_parser import parse_docstring


def from_class(class_def, emit_default_doc=True):
    """
    Converts an AST to a docstring structure

    :param class_def: Class AST or Module AST with a ClassDef inside
    :type class_def: ```Union[Module, ClassDef]```

    :param emit_default_doc: Whether help/docstring should include 'With default' text
    :type emit_default_doc: ```bool``

    :return: docstring structure
    :rtype: ```dict```
    """
    class_def = to_class_def(class_def)
    docstring_structure = from_docstring(get_docstring(class_def).replace(':cvar', ':param'),
                                         emit_default_doc=emit_default_doc)
    docstring_structure['params'] = OrderedDict((param.pop('name'), param)
                                                for param in docstring_structure['params'])
    if 'return_type' in docstring_structure['params']:
        docstring_structure['returns'] = {'return_type': docstring_structure['params'].pop('return_type')}

    for e in filter(rpartial(isinstance, AnnAssign), class_def.body):
        docstring_structure['returns' if e.target.id == 'return_type' else 'params'][e.target.id]['typ'] = \
            to_source(e.annotation)[:-1]

    docstring_structure['params'] = [
        dict(name=k, **v)
        for k, v in docstring_structure['params'].items()
    ]
    docstring_structure['returns'] = (lambda k: dict(name=k, **docstring_structure['returns'][k]))('return_type')

    return docstring_structure


def from_class_with_method(class_def, method_name, emit_default_doc=True):
    """
    Converts an AST of a class with a method to a docstring structure

    :param class_def: Class AST or Module AST with a ClassDef inside
    :type class_def: ```Union[Module, ClassDef]```

    :param method_name: Method name
    :type method_name: ```str```

    :param emit_default_doc: Whether help/docstring should include 'With default' text
    :type emit_default_doc: ```bool``

    :return: docstring structure
    :rtype: ```dict```
    """
    return from_function(
        function_def=next(
            node
            for node in to_class_def(class_def).body
            if isinstance(node, FunctionDef) and node.name == method_name
        ),
        emit_default_doc=emit_default_doc
    )


def from_function(function_def, emit_default_doc=True):
    """
    Converts an AST of a class with a method to a docstring structure

    :param function_def: FunctionDef
    :type function_def: ```FunctionDef```

    :param emit_default_doc: Whether help/docstring should include 'With default' text
    :type emit_default_doc: ```bool``

    :return: docstring structure
    :rtype: ```dict```
    """

    docstring_structure = from_docstring(get_docstring(function_def).replace(':cvar', ':param'),
                                         emit_default_doc=emit_default_doc)
    function_type = get_function_type(function_def)
    offset = 0 if function_type is None else 1

    for idx, arg in enumerate(function_def.args.args):
        if arg.annotation is not None:
            docstring_structure['params'][idx - offset]['typ'] = to_source(arg.annotation)[:-1]

    if emit_default_doc:
        for idx, const in enumerate(function_def.args.defaults):
            assert isinstance(const, Constant) and const.kind is None
            if const.value is not None:
                docstring_structure['params'][idx]['default'] = const.value

        # Convention - the final top-level `return` is the default
        return_ast = next(filter(rpartial(isinstance, Return), function_def.body[::-1]), None)
        if return_ast is not None and return_ast.value is not None:
            docstring_structure['returns']['default'] = to_source(return_ast.value)[:-1]

    if isinstance(function_def.returns, Subscript):
        docstring_structure['returns']['typ'] = to_source(function_def.returns)[:-1]

    for e in filter(lambda el: el.target.id not in frozenset(('self', 'cls')),
                    filter(rpartial(isinstance, AnnAssign),
                           function_def.body)):
        docstring_structure['returns' if e.target.id == 'return_type' else 'params'][e.target.id]['typ'] = \
            e.annotation.id if isinstance(e.annotation, Name) else to_source(e.annotation).rstrip()

    return docstring_structure


def from_argparse_ast(function_def, emit_default_doc=False):
    """
    Converts an AST to a docstring structure

    :param function_def: AST of argparse function
    :type function_def: ```FunctionDef``

    :param emit_default_doc: Whether help/docstring should include 'With default' text
    :type emit_default_doc: ```bool``

    :return: docstring structure
    :rtype: ```dict```
    """
    docstring_structure = {'short_description': '', 'long_description': '', 'params': []}
    _docstring_struct = parse_docstring(get_docstring(function_def), emit_default_doc=emit_default_doc)
    for e in function_def.body:
        if isinstance(e, Expr):
            if isinstance(e.value, Constant):
                if e.value.kind is not None:
                    raise NotImplementedError('kind')
                # docstring_structure['short_description'] = '\n'.join(
                #     takewhile(lambda l: not l.lstrip().startswith(':param'),
                #               expr.value.value.split('\n'))).strip()
            elif (isinstance(e.value, Call) and len(e.value.args) == 1 and
                  isinstance(e.value.args[0], Constant) and
                  e.value.args[0].kind is None):
                docstring_structure['params'].append(parse_out_param(e, emit_default_doc=emit_default_doc))
        elif isinstance(e, Assign):
            if all((len(e.targets) == 1,
                    isinstance(e.targets[0], Attribute),
                    e.targets[0].attr == 'description',
                    isinstance(e.targets[0].value, Name),
                    e.targets[0].value.id == 'argument_parser',
                    isinstance(e.value, Constant))):
                docstring_structure['short_description'] = e.value.value
        elif isinstance(e, Return) and isinstance(e.value, Tuple):
            default = to_source(e.value.elts[1]).replace('\n', '')
            doc = next(
                line.partition(',')[2].lstrip()
                for line in function_def.body[0].value.value.split('\n')
                if line.lstrip().startswith(':return')
            )
            if not emit_default_doc:
                doc, _ = extract_default(doc, emit_default_doc=emit_default_doc)

            docstring_structure['returns'] = {
                'name': 'return_type',
                'doc': '{doc}{maybe_dot} Defaults to {default}'.format(
                    maybe_dot='' if doc.endswith('.') else '.',
                    doc=doc,
                    default=default
                ) if all((default is not None,
                          emit_default_doc,
                          'Defaults to' not in doc,
                          'defaults to' not in doc))
                else doc,
                'default': default,
                'typ': to_source(
                    parse(_docstring_struct['returns']['typ']).body[0].value.slice.value.elts[1]
                ).rstrip()
                # 'Tuple[ArgumentParser, {typ}]'.format(typ=_docstring_structure['returns']['typ'])
            }
    if not emit_default_doc:
        remove_defaults_from_docstring_structure(docstring_structure, emit_defaults=False)
    return docstring_structure


def from_docstring(docstring, emit_default_doc=True, return_tuple=False):
    """
    Converts a docstring to an AST

    :param docstring: docstring portion
    :type docstring: ```Union[str, Dict]```

    :param emit_default_doc: Whether help/docstring should include 'With default' text
    :type emit_default_doc: ```bool``

    :param return_tuple: Whether to return a tuple, or just the docstring_struct
    :type return_tuple: ```bool```

    :return: docstring_structure, whether it returns or not
    :rtype: ```Optional[Union[dict, Tuple[dict, bool]]]```
    """
    parsed = docstring if isinstance(docstring, dict) \
        else parse_docstring(docstring, emit_default_doc=emit_default_doc)
    returns = 'returns' in parsed and parsed['returns'] is not None and 'name' in parsed['returns']
    if returns:
        parsed['returns']['doc'] = parsed['returns'].get('doc', parsed['returns']['name'])
    if return_tuple:
        return parsed, returns
    return parsed


def to_docstring(docstring_structure, emit_default_doc=True,
                 docstring_format='rest', indent_level=2,
                 emit_types=False, emit_separating_tab=True):
    """
    Converts a docstring to an AST

    :param docstring_structure: docstring struct
    :type docstring_structure: ```dict```

    :param emit_default_doc: Whether help/docstring should include 'With default' text
    :type emit_default_doc: ```bool``

    :param docstring_format: Format of docstring
    :type docstring_format: ```Literal['rest', 'numpy', 'google']```

    :param indent_level: indentation level whence: 0=no_tabs, 1=one tab; 2=two tabs
    :type indent_level: ```int```

    :param emit_types: whether to show `:type` lines
    :type emit_types: ```bool```

    :param emit_separating_tab: whether to put a tab between :param and return and desc
    :type emit_separating_tab: ```bool```

    :return: docstring
    :rtype: ```str```
    """
    if docstring_format != 'rest':
        raise NotImplementedError(docstring_format)

    def param2docstring_param(param, docstring_format='rest',
                              emit_default_doc=True, indent_level=1,
                              emit_types=False):
        """
        Converts param dict from docstring_structure to the right string representation

        :param param: dict of shape {'name': ..., 'typ': ..., 'doc': ..., 'default': ..., 'required': ... }
        :type param: ```dict```

        :param docstring_format: Format of docstring
        :type docstring_format: ```Literal['rest', 'numpy', 'google']```

        :param emit_default_doc: Whether help/docstring should include 'With default' text
        :type emit_default_doc: ```bool``

        :param indent_level: indentation level whence: 0=no_tabs, 1=one tab; 2=two tabs
        :type indent_level: ```int```

        :param emit_types: whether to show `:type` lines
        :type emit_types: ```bool```
        """
        doc, default = extract_default(param['doc'], emit_default_doc=False)
        if default is None:
            default = param.get('default')

        typ = '**{param[name]}'.format(param=param) \
            if param.get('typ') == 'dict' and param['name'].endswith('kwargs') \
            else param.get('typ')

        return ''.join(filter(
            None, (
                '{tab}:param {param[name]}: {doc}'.format(tab=tab * indent_level, param=param, doc=doc),
                ' Defaults to {default}'.format(default=default) if emit_default_doc and 'default' in param
                else None,
                None if typ is None or not emit_types else '\n{tab}:type {param[name]}: ```{typ}```'.format(
                    tab=tab * indent_level,
                    param=param,
                    typ=typ
                )
            )
        ))

    param2docstring_param = partial(
        param2docstring_param,
        emit_default_doc=emit_default_doc,
        docstring_format=docstring_format,
        indent_level=indent_level,
        emit_types=emit_types
    )
    sep = tab if emit_separating_tab else ''
    return '\n{tab}{description}\n{sep}\n{params}\n{sep}\n{returns}\n{tab}'.format(
        sep=sep,
        tab=tab * indent_level,
        description=docstring_structure.get('long_description') or docstring_structure['short_description'],
        params='\n{sep}\n'.format(sep=sep).join(map(param2docstring_param,
                                                    docstring_structure['params'])),
        returns=(param2docstring_param(docstring_structure['returns'])
                 .replace(':param return_type:', ':return:')
                 .replace(':type return_type:', ':rtype:'))
        if docstring_structure.get('returns') else ''
    )


def from_docstring_parser(docstring):
    """
    Converts Docstring from the docstring_parser library to our internal representation / docstring structure

    :param docstring: Docstring from the docstring_parser library
    :type docstring: ```docstring_parser.common.Docstring```

    :return: docstring structure
    :rtype: ```dict```
    """

    def parse_dict(d):
        """
        Restructure dictionary to match expectations

        :param d: input dictionary
        :type d: ```dict```

        :returns: restructured dict
        :rtype: ```dict```
        """
        if 'args' in d and len(d['args']) in frozenset((1, 2)):
            d['name'] = d.pop('args')[0]
            if d['name'] == 'return':
                d['name'] = 'return_type'
        if 'type_name' in d:
            d['typ'] = d.pop('type_name')
        if 'description' in d:
            d['doc'] = d.pop('description')

        return {k: v
                for k, v in d.items()
                if v is not None}

    def evaluate_to_docstring_value(name_value):
        """
        Turn the second element of the tuple into the final representation (e.g., a bool, str, int)

        :param name_value: name value tuple
        :type name_value: ```Tuple[str, Any]```

        :return: Same shape as input
        :rtype: ```Tuple[str, Tuple[Union[str, int, bool, float]]]```
        """
        name, value = name_value
        if isinstance(value, (list, tuple)):
            value = list(map(itemgetter(1),
                             map(lambda v: evaluate_to_docstring_value((name, v)),
                                 value)))
        elif isinstance(value, DocstringParam):
            assert len(value.args) == 2 and value.args[1] == value.arg_name
            value = {attr: getattr(value, attr)
                     for attr in ('type_name', 'arg_name', 'is_optional',
                                  'default', 'description')
                     if getattr(value, attr) is not None}
            if 'arg_name' in value:
                value['name'] = value.pop('arg_name')
            if 'description' in value:
                value['doc'] = extract_default(value.pop('description'), emit_default_doc=False)[0]
        elif isinstance(value, DocstringMeta):
            value = parse_dict({
                attr: getattr(value, attr)
                for attr in dir(value)
                if not attr.startswith('_') and getattr(value, attr)
            })
        elif not isinstance(value, (str, int, float, bool, type(None))):
            raise NotImplementedError(type(value).__name__)
        return name, value

    docstring_structure = dict(map(evaluate_to_docstring_value,
                                   filter(lambda k_v: not isinstance(k_v[1], (type(None), bool)),
                                          map(lambda attr: (attr, getattr(docstring, attr)),
                                              filter(lambda attr: not attr.startswith('_'),
                                                     dir(docstring))))))
    if 'meta' in docstring_structure and 'params' in docstring_structure:
        meta = {e['name']: e
                for e in docstring_structure.pop('meta')}
        docstring_structure['params'] = [
            dict(**param, **{k: v
                             for k, v in meta[param['name']].items()
                             if k not in param})
            for param in docstring_structure['params']
        ]

    return docstring_structure
