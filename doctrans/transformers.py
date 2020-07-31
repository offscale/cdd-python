"""
Transform from string or AST representations of input, to AST, file, or str output.
"""

from ast import parse, ClassDef, Name, Load, Expr, Module, \
    FunctionDef, arguments, Assign, Attribute, Store, Tuple, Return, arg
from functools import partial

from black import format_str, FileMode

from doctrans import docstring_struct
from doctrans.ast_utils import param2argparse_param, param2ast, set_value
from doctrans.defaults_utils import set_default_doc
from doctrans.pure_utils import tab, simple_types, PY_GTE_3_9
from doctrans.source_transformer import to_code


def to_argparse(docstring_structure, emit_default_doc, function_name='set_cli_args'):
    """
    Convert to an argparse function definition

    :param docstring_structure: a dictionary of form
          {
              'short_description': ...,
              'long_description': ...,
              'params': [{'name': ..., 'typ': ..., 'doc': ..., 'default': ..., 'required': ... }, ...],
              "returns': {'name': ..., 'typ': ..., 'doc': ..., 'default': ..., 'required': ... }
          }
    :type docstring_structure: ```dict```

    :param emit_default_doc: Whether help/docstring should include 'With default' text
    :type emit_default_doc: ```bool``

    :param function_name: name of function
    :type function_name: ```str```

    :returns: function which constructs argparse
    :rtype: ```FunctionDef``
    """
    return FunctionDef(
        args=arguments(args=[arg(annotation=None,
                                 arg='argument_parser',
                                 type_comment=None)],
                       defaults=[],
                       kw_defaults=[],
                       kwarg=None,
                       kwonlyargs=[],
                       posonlyargs=[],
                       vararg=None),
        body=[Expr(value=set_value(kind=None,
                                   value='\n    Set CLI arguments\n\n    '
                                         ':param argument_parser: argument parser\n    '
                                         ':type argument_parser: ```ArgumentParser```\n\n    '
                                         ':return: argument_parser, {returns[doc]}\n    '
                                         ':rtype: ```Tuple[ArgumentParser,'
                                         ' {returns[typ]}]```\n    '.format(returns=docstring_structure['returns']))
                   ),
              Assign(targets=[Attribute(attr='description',
                                        ctx=Store(),
                                        value=Name(ctx=Load(),
                                                   id='argument_parser'))],
                     type_comment=None,
                     value=set_value(
                         kind=None,
                         value=docstring_structure['long_description'] or docstring_structure['short_description']),
                     lineno=None),
              ] + list(map(partial(param2argparse_param, emit_default_doc=emit_default_doc),
                           docstring_structure['params'])
                       ) + [Return(value=Tuple(ctx=Load(),
                                               elts=[
                                                   Name(ctx=Load(),
                                                        id='argument_parser'),
                                                   parse(docstring_structure['returns']['default']).body[0].value]))],
        decorator_list=[],
        name=function_name,
        returns=None,
        type_comment=None,
        lineno=None
    )


def to_class(docstring_structure, class_name='TargetClass', class_bases=('object',)):
    """
    Construct a class

    :param docstring_structure: a dictionary of form
          {
              'short_description': ...,
              'long_description': ...,
              'params': [{'name': ..., 'typ': ..., 'doc': ..., 'default': ..., 'required': ... }, ...],
              "returns': {'name': ..., 'typ': ..., 'doc': ..., 'default': ..., 'required': ... }
          }
    :type docstring_structure: ```dict```

    :param class_name: name of class
    :type class_name: ```str```

    :param class_bases: bases of class (the generated class will inherit these)
    :type class_bases: ```Iterable[str]```

    :return: Class AST of the docstring
    :rtype: ```ClassDef```
    """
    returns = [docstring_structure['returns']] if docstring_structure.get('returns') else []
    return ClassDef(
        bases=[Name(ctx=Load(),
                    id=base_class)
               for base_class in class_bases],
        body=[Expr(
            value=set_value(
                kind=None,
                value='\n    {description}\n\n{cvars}'.format(
                    description=docstring_structure['long_description'] or docstring_structure['short_description'],
                    cvars='\n'.join(
                        '{tab}:cvar {param[name]}: {param[doc]}'.format(tab=tab, param=set_default_doc(param))
                        for param in docstring_structure['params'] + returns
                    )
                )
            ))] + list(map(param2ast, docstring_structure['params'] + returns)),
        decorator_list=[],
        keywords=[],
        name=class_name
    )


def to_docstring(docstring_structure, docstring_format='rest', emit_default_doc=True):
    """
    Converts an AST to a docstring

    :param docstring_structure: a dictionary of form
          {
              'short_description': ...,
              'long_description': ...,
              'params': [{'name': ..., 'typ': ..., 'doc': ..., 'default': ..., 'required': ... }, ...],
              "returns': {'name': ..., 'typ': ..., 'doc': ..., 'default': ..., 'required': ... }
          }
    :type docstring_structure: ```dict```

    :param docstring_format: Format of docstring
    :type docstring_format: ```Literal['rest', 'numpy', 'google']```

    :param emit_default_doc: Whether help/docstring should include 'With default' text
    :type emit_default_doc: ```bool``

    :returns: docstring
    :rtype: ```str``
    """
    if docstring_format != 'rest':
        raise NotImplementedError()

    return '''\n{description}\n\n{params}\n{returns}\n'''.format(
        description=docstring_structure['long_description'] or docstring_structure['short_description'],
        params='\n'.join(
            ':param {param[name]}: {param[doc]}\n'
            ':type {param[name]}: ```{typ}```\n'.format(
                param=set_default_doc(param),
                typ=('**{name}'.format(name=param['name'])
                     if 'kwargs' in param['name']
                     else param['typ']))
            for param in docstring_structure['params']
        ),
        returns=':return: {param[doc]}\n'
                ':rtype: ```{param[typ]}```'.format(
            param=set_default_doc(docstring_structure['returns'])
        )
    )


def to_file(ast, filename, mode='a', skip_black=PY_GTE_3_9):
    """
    Convert AST to a file

    :param ast: AST node
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
    src = to_code(ast)
    if not skip_black:
        src = format_str(src, mode=FileMode(
            target_versions=set(),
            line_length=119,
            is_pyi=False,
            string_normalization=False,
        ))
    with open(filename, mode) as f:
        f.write(src)


def to_function(docstring_structure, emit_default_doc, function_name,
                function_type, docstring_format='rest', indent_level=2,
                emit_separating_tab=False, inline_types=True):
    """
    Construct a function

    :param docstring_structure: a dictionary of form
          {
              'short_description': ...,
              'long_description': ...,
              'params': [{'name': ..., 'typ': ..., 'doc': ..., 'default': ..., 'required': ... }, ...],
              "returns': {'name': ..., 'typ': ..., 'doc': ..., 'default': ..., 'required': ... }
          }
    :type docstring_structure: ```dict```

    :param emit_default_doc: Whether help/docstring should include 'With default' text
    :type emit_default_doc: ```bool``

    :param function_name: name of function
    :type function_name: ```str```

    :param function_type: None is a loose function (def f()`), others self-explanatory
    :type function_type: ```Optional[Literal['self', 'cls']]```

    :param docstring_format: Format of docstring
    :type docstring_format: ```Literal['rest', 'numpy', 'google']```

    :param indent_level: docstring indentation level whence: 0=no_tabs, 1=one tab; 2=two tabs
    :type indent_level: ```int```

    :param emit_separating_tab: docstring decider for whether to put a tab between :param and return and desc
    :type emit_separating_tab: ```bool```

    :param inline_types: Whether the type should be inline or in docstring
    :type inline_types: ```bool```

    :returns: function (could be a method on a class)
    :rtype: ```FunctionDef``
    """
    params_no_kwargs = tuple(filter(
        lambda param: not param['name'].endswith('kwargs'),
        docstring_structure['params']
    ))

    return FunctionDef(
        args=arguments(
            args=list(filter(
                None,
                (function_type if function_type is None else arg(annotation=None,
                                                                 arg=function_type,
                                                                 type_comment=None),
                 *map(lambda param:
                      arg(
                          annotation=(
                              Name(
                                  ctx=Load(),
                                  id=param['typ']
                              )
                              if param['typ'] in simple_types
                              else parse(param['typ']).body[0].value)
                          if inline_types and 'typ' in param else None,
                          arg=param['name'],
                          type_comment=None
                      ),
                      params_no_kwargs)
                 )
            )),
            defaults=list(
                map(lambda param: set_value(kind=None,
                                            value=param['default']),
                    filter(lambda param: 'default' in param,
                           params_no_kwargs))
            ) + [set_value(kind=None,
                           value=None)],
            kw_defaults=[],
            kwarg=next(map(
                lambda param: arg(annotation=None,
                                  arg=param['name'],
                                  type_comment=None),
                filter(
                    lambda param: param['name'].endswith('kwargs'),
                    docstring_structure['params']
                ))
            ),
            kwonlyargs=[],
            posonlyargs=[],
            vararg=None
        ),
        body=list(filter(
            None,
            (
                Expr(value=set_value(
                    kind=None,
                    value=docstring_struct.to_docstring(
                        docstring_structure,
                        emit_default_doc=emit_default_doc,
                        docstring_format=docstring_format,
                        emit_types=not inline_types,
                        indent_level=indent_level,
                        emit_separating_tab=emit_separating_tab
                    )
                )),
                Return(value=parse(docstring_structure['returns']['default']).body[0].value)
                if 'returns' in docstring_structure and docstring_structure['returns'].get('default')
                else None
            )
        )),
        decorator_list=[],
        name=function_name,
        returns=(parse(docstring_structure['returns']['typ']).body[0].value
                 if 'returns' in docstring_structure and 'typ' in docstring_structure[
            'returns'] else None) if inline_types else None,
        type_comment=None,
        lineno=None
    )
