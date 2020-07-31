"""
Functions which produce docstring_structure from various different inputs
"""
from ast import Constant, Name, Attribute, Return, parse
from typing import Any

from doctrans.ast_utils import get_value
from doctrans.defaults_utils import extract_default, set_default_doc
from doctrans.pure_utils import simple_types
from doctrans.source_transformer import to_code


def _handle_value(node):
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


def _handle_keyword(keyword, typ):
    """
    Decide which type to wrap the keyword tuples in

    :param keyword: AST keyword
    :type keyword: ```keyword```

    :param typ: string representation of type
    :type typ: ```str```

    :returns: string representation of type
    :rtype: ```str``
    """
    global quote_f

    type_ = 'Union'
    if typ == Any or typ in simple_types:
        if typ == 'str' or typ == Any:
            def quote_f(s):
                """
                Wrap the input in quotes

                :param s: Any value
                :type s: ```Any```

                :returns: the input value
                :rtype: ```Any```
                """
                return '\'{}\''.format(s)

        type_ = 'Literal'

    return '{type}[{typs}]'.format(
        type=type_,
        typs=', '.join(quote_f(get_value(elt))
                       for elt in keyword.value.elts)
    )


def parse_out_param(expr, emit_default_doc=True):
    """
    Turns the class_def repr of '--dataset_name', type=str, help='name of dataset.', required=True, default='mnist'
      into
          {'name': 'dataset_name', 'typ': 'str', doc='name of dataset.',
           'required': True, 'default': 'mnist'}

    :param expr: Expr
    :type expr: ```Expr```

    :param emit_default_doc: Whether help/docstring should include 'With default' text
    :type emit_default_doc: ```bool``

    :returns: dict of shape {'name': ..., 'typ': ..., 'doc': ..., 'default': ..., 'required': ... }
    :rtype ```dict```
    """
    required = next(
        (keyword
         for keyword in expr.value.keywords
         if keyword.arg == 'required'),
        Constant(value=False)
    ).value

    typ = next((
        _handle_value(get_value(keyword))
        for keyword in expr.value.keywords
        if keyword.arg == 'type'
    ), 'str')
    name = get_value(expr.value.args[0])[len('--'):]
    default = next(
        (get_value(key_word.value)
         for key_word in expr.value.keywords
         if key_word.arg == 'default'),
        None
    )
    doc = (
        lambda help: help if help is None else (
            help if default is None or emit_default_doc is False or (hasattr(default, '__len__') and len(
                default) == 0) or 'defaults to' in help or 'Defaults to' in help
            else '{help} Defaults to {default}'.format(
                help=help if help.endswith('.') else '{}.'.format(help),
                default=default
            )
        )
    )(next(
        (get_value(key_word.value)
         for key_word in expr.value.keywords
         if key_word.arg == 'help'),
        None
    ))
    if default is None:
        doc, default = extract_default(doc, emit_default_doc=emit_default_doc)
    if default is None and typ in simple_types and required:
        default = simple_types[typ]

    return dict(
        name=name,
        doc=doc,
        typ=(lambda typ: (typ if required or name.endswith('kwargs')
                          else 'Optional[{typ}]'.format(typ=typ)))(
            typ=next(
                (_handle_keyword(keyword, typ)
                 for keyword in expr.value.keywords
                 if keyword.arg == 'choices'
                 ), typ
            )
        ),
        **({} if default is None else {'default': default})
    )


def interpolate_defaults(param, emit_default_doc=True):
    """
    Correctly set the 'default' and 'doc' parameters

    :param param: dict of shape {'name': ..., 'typ': ..., 'doc': ..., 'required': ... }
    :type param: ```dict```

    :param emit_default_doc: Whether help/docstring should include 'With default' text
    :type emit_default_doc: ```bool``

    :returns: dict of shape {'name': ..., 'typ': ..., 'doc': ..., 'default': ..., 'required': ... }
    :rtype: ```dict```
    """
    if 'doc' in param:
        doc, default = extract_default(param['doc'], emit_default_doc=emit_default_doc)
        param['doc'] = doc
        if default:
            param['default'] = default
    return param


def _parse_return(e, docstring_structure, function_def, emit_default_doc):
    """
    Parse return into a param dict

    :param e: Return AST node
    :type e: Return

    :param docstring_structure: a dictionary of form
          {
              'short_description': ...,
              'long_description': ...,
              'params': [{'name': ..., 'typ': ..., 'doc': ..., 'default': ..., 'required': ... }, ...],
              "returns': {'name': ..., 'typ': ..., 'doc': ..., 'default': ..., 'required': ... }
          }
    :type docstring_structure: ```dict```

    :param function_def: FunctionDef
    :type function_def: ```FunctionDef```

    :param emit_default_doc: Whether help/docstring should include 'With default' text
    :type emit_default_doc: ```bool``

    :returns: dict of shape {'name': ..., 'typ': ..., 'doc': ..., 'default': ..., 'required': ... }
    :rtype: ```dict```
    """
    assert isinstance(e, Return)

    return set_default_doc({
        'name': 'return_type',
        'doc': extract_default(next(
            line.partition(',')[2].lstrip()
            for line in get_value(function_def.body[0].value).split('\n')
            if line.lstrip().startswith(':return')
        ), emit_default_doc=emit_default_doc)[0],
        'default': to_code(e.value.elts[1])[:-1],
        'typ': to_code(
            get_value(parse(docstring_structure['returns']['typ']).body[0].value.slice).elts[1]
        ).rstrip()
        # 'Tuple[ArgumentParser, {typ}]'.format(typ=_docstring_structure['returns']['typ'])
    })


__all__ = ['parse_out_param', 'interpolate_defaults']
