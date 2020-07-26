"""
Functions which produce docstring_structure from various different inputs
"""
from ast import Constant, Name, Expr, Attribute, keyword
from typing import Any

from doctrans.defaults_utils import extract_default
from doctrans.pure_utils import simple_types


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
            help if default is None or emit_default_doc is False or (hasattr(default, '__len__') and len(
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
        doc, default = extract_default(doc, emit_default_doc=emit_default_doc)
    if default is None:
        # if name.endswith('kwargs'):
        #    default = {}
        # required = True
        # el
        if typ in simple_types:
            if required:
                default = simple_types[typ]

    def handle_keyword(keyword):
        """
        Decide which type to wrap the keyword tuples in

        :param keyword: AST keyword
        :type keyword: ```keyword```

        :returns: string representation of type
        :rtype: ```str``
        """
        quote_f = lambda s: s
        type_ = 'Union'
        if typ == Any or typ in simple_types:
            if typ == 'str' or typ == Any:
                quote_f = lambda s: '\'{}\''.format(s)
            type_ = 'Literal'

        return '{type}[{typs}]'.format(
            type=type_,
            typs=', '.join(quote_f(elt.value)
                           for elt in keyword.value.elts)
        )

    return dict(
        name=name,
        doc=doc,
        typ=(lambda typ: (typ if required or name.endswith('kwargs')
                          else 'Optional[{typ}]'.format(typ=typ)))(
            typ=next(
                (handle_keyword(keyword)
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


__all__ = ['parse_out_param', 'interpolate_defaults']
