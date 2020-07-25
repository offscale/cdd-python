"""
Transform from string or AST representations of input, to AST, file, or str output.
"""
from ast import parse, ClassDef, Name, Load, Constant, Expr, Module, FunctionDef, arguments, Assign, Attribute, \
    Store, Tuple, Return, arg
from functools import partial

from astor import to_source
from black import format_str, FileMode

from doctrans.ast_utils import param2argparse_param, param2ast
from doctrans.defaults_utils import extract_default
from doctrans.docstring_structure_utils import class_def2docstring_structure, argparse_ast2docstring_structure, \
    docstring_structure2docstring
from doctrans.pure_utils import tab, simple_types
from doctrans.rest_docstring_parser import parse_docstring


class BaseTransform(object):
    """
    BaseTransformer

    :cvar generated_ast: Generated AST.
    :type generated_ast: ```Optional[Module, ClassDef, FunctionDef]```

    :cvar docstring_struct: dict of shape {'name': ..., 'platform': ...,
            'module': ..., 'title': ..., 'description': ...,
            'parameters': ..., 'schema': ...,'returns': ...}
    :type docstring_struct: ```dict```
    """

    generated_ast = None
    docstring_struct = None

    def __init__(self, inline_types, emit_default_doc):
        """
        Construct a transformer object

        :param inline_types: Whether the type should be inline or in docstring
        :type inline_types: ```bool```

        :param emit_default_doc: Help/docstring should include 'With default' text
        :type emit_default_doc: ```bool``
        """

        self.inline_types = inline_types
        self.emit_default_doc = emit_default_doc

    def to_argparse(self, function_name='set_cli_args'):
        """
        Convert to an argparse function definition

        :param function_name: name of function
        :type function_name: ```str```

        :returns: function which constructs argparse
        :rtype: ```FunctionDef``
        """
        doc, _default = extract_default(self.docstring_struct['returns']['doc'],
                                        emit_default_doc=self.emit_default_doc)
        self.docstring_struct['returns']['doc'] = doc
        self.generated_ast = FunctionDef(
            args=arguments(args=[arg(annotation=None,
                                     arg='argument_parser',
                                     type_comment=None)],
                           defaults=[],
                           kw_defaults=[],
                           kwarg=None,
                           kwonlyargs=[],
                           posonlyargs=[],
                           vararg=None),
            body=[
                     Expr(value=Constant(kind=None,
                                         value='\n    Set CLI arguments\n\n    '
                                               ':param argument_parser: argument parser\n    '
                                               ':type argument_parser: ```ArgumentParser```\n\n    '
                                               ':return: argument_parser, {returns[doc]}\n    '
                                               ':rtype: ```Tuple[ArgumentParser,'
                                               ' {returns[typ]}]```\n    '.format(
                                             returns=self.docstring_struct['returns']))),
                     Assign(targets=[Attribute(attr='description',
                                               ctx=Store(),
                                               value=Name(ctx=Load(),
                                                          id='argument_parser'))],
                            type_comment=None,
                            value=Constant(
                                kind=None,
                                value=self.docstring_struct['long_description'] or
                                      self.docstring_struct[
                                          'short_description']))
                 ] + list(map(partial(param2argparse_param, emit_default_doc=self.emit_default_doc),
                              self.docstring_struct['params'])) + [
                     Return(
                         value=Tuple(
                             ctx=Load(),
                             elts=[
                                 Name(ctx=Load(),
                                      id='argument_parser'),
                                 parse(self.docstring_struct['returns']['default']).body[0].value
                             ]))],
            decorator_list=[],
            name=function_name,
            returns=None,
            type_comment=None
        )
        return self.generated_ast

    def to_class(self, class_name='TargetClass', class_bases=('object',)):
        """
        Construct a class

        :param class_name: name of class
        :type class_name: ```str```

        :param class_bases: bases of class (the generated class will inherit these)
        :type class_bases: ```Tuple[str]```

        :return: Class AST of the docstring
        :rtype: ```ClassDef```
        """
        returns = [self.docstring_struct['returns']] if self.docstring_struct.get('returns') else []
        self.generated_ast = ClassDef(
            bases=[Name(ctx=Load(),
                        id=base_class)
                   for base_class in class_bases],
            body=[
                     Expr(value=Constant(
                         kind=None,
                         value='\n    {description}\n\n{cvars}'.format(
                             description=self.docstring_struct['long_description'] or self.docstring_struct[
                                 'short_description'],
                             cvars='\n'.join(
                                 '{tab}:cvar {param[name]}: {param[doc]}'.format(tab=tab, param=param)
                                 for param in self.docstring_struct['params'] + returns
                             )
                         )
                     ))
                 ] + list(map(param2ast, self.docstring_struct['params'] + returns)),
            decorator_list=[],
            keywords=[],
            name=class_name
        )
        return self.generated_ast

    def to_docstring(self):
        """
        Converts an AST to a docstring

        :returns: docstring
        :rtype: ```str``
        """
        return '''\n{description}\n\n{params}\n{returns}\n'''.format(
            description=self.docstring_struct['long_description'] or self.docstring_struct['short_description'],
            params='\n'.join(
                ':param {param[name]}: {param[doc]}\n'
                ':type {param[name]}: ```{typ}```\n'.format(param=param,
                                                            typ=('**{name}'.format(name=param['name'])
                                                                 if 'kwargs' in param['name']
                                                                 else param['typ']))
                for param in self.docstring_struct['params']
            ),
            returns=':return: {param[doc]}\n'
                    ':rtype: ```{param[typ]}```'.format(param=self.docstring_struct['returns'])
        )

    def to_file(self, filename, ast=None, mode='a', skip_black=False):
        """
        Convert AST to a file

        :param filename: emit to this file
        :type filename: ```str```

        :param ast: Constructed object of the `class_def` class, usually an `ast.Module`
        :type ast: ```Optional[Union[Module, ClassDef, FunctionDef]]```

        :param mode: Mode to open the file in, defaults to append
        :type mode: ```str```

        :param skip_black: Skip formatting with black
        :type skip_black: ```bool```

        :return: None
        :rtype: ```NoneType```
        """
        if ast is None:
            ast = self.generated_ast
        if isinstance(ast, (ClassDef, FunctionDef)):
            ast = Module(body=[ast], type_ignores=[])
        src = to_source(ast)
        if not skip_black:
            src = format_str(src, mode=FileMode(
                target_versions=set(),
                line_length=119,
                is_pyi=False,
                string_normalization=False,
            ))
        with open(filename, mode) as f:
            f.write(src)

    def to_function(self, function_name, function_type):
        """
        Construct a function

        :param function_name: name of function
        :type function_name: ```str```

        :param function_type: None is a loose function (def f()`), others self-explanatory
        :type function_type: ```Optional[Literal['self', 'cls']]```

        :returns: function (could be a method on a class)
        :rtype: ```FunctionDef``
        """
        params_no_kwargs = tuple(filter(
            lambda param: not param['name'].endswith('kwargs'),
            self.docstring_struct['params']
        ))

        self.generated_ast = FunctionDef(
            args=arguments(
                args=list(filter(
                    None,
                    (function_type if function_type is None else arg(annotation=None,
                                                                     arg=function_type,
                                                                     type_comment=None),
                     *map(lambda param:
                          arg(
                              annotation=Name(
                                  ctx=Load(),
                                  id=param['typ']
                              )
                              if param['typ'] in simple_types
                              else parse(param['typ']).body[0].value,
                              arg=param['name'],
                              type_comment=None
                          ),
                          params_no_kwargs)
                     )
                )),
                defaults=list(
                    map(lambda param: Constant(kind=None,
                                               value=param['default']),
                        filter(lambda param: 'default' in param,
                               params_no_kwargs))
                ) + [Constant(kind=None,
                              value=None)],
                kw_defaults=[],
                kwarg=next(map(
                    lambda param: arg(annotation=None,
                                      arg=param['name'],
                                      type_comment=None),
                    filter(
                        lambda param: param['name'].endswith('kwargs'),
                        self.docstring_struct['params']
                    ))
                ),
                kwonlyargs=[],
                posonlyargs=[],
                vararg=None
            ),
            body=list(filter(
                None,
                (
                    Expr(value=Constant(
                        kind=None,
                        value=docstring_structure2docstring(
                            self.docstring_struct,
                            emit_default_doc=self.emit_default_doc
                        )
                    )),
                    Return(value=parse(self.docstring_struct['returns']['default']).body[0].value)
                    if 'returns' in self.docstring_struct and self.docstring_struct['returns'].get('default')
                    else None
                )
            )),
            decorator_list=[],
            name=function_name,
            returns=(parse(self.docstring_struct['returns']['typ']).body[0].value
                     if 'returns' in self.docstring_struct else None),
            type_comment=None
        )
        return self.generated_ast


class ArgparseTransform(BaseTransform):
    """
    Transformer
    """

    def __init__(self, ast, inline_types, emit_default_doc):
        """
        Construct the object

        :param ast: FunctionDef AST or Module AST
        :type ast: ```Union[ast.Module, ast.FunctionDef]```

        :param inline_types: Whether the type should be inline or in docstring
        :type inline_types: ```bool```

        :param emit_default_doc: Help/docstring should include 'With default' text
        :type emit_default_doc: ```bool``
        """
        super(ArgparseTransform, self).__init__(inline_types=inline_types,
                                                emit_default_doc=emit_default_doc)
        self.docstring_struct = argparse_ast2docstring_structure(ast, emit_default_doc=emit_default_doc)


class ClassTransform(BaseTransform):
    """
    Transformer
    """

    def __init__(self, ast, inline_types, emit_default_doc):
        """
        Construct the object

        :param ast: Class AST or Module AST
        :type ast: ```Union[ast.Module, ast.ClassDef]```

        :param inline_types: Whether the type should be inline or in docstring
        :type inline_types: ```bool```

        :param emit_default_doc: Help/docstring should include 'With default' text
        :type emit_default_doc: ```bool``
        """
        super(ClassTransform, self).__init__(inline_types=inline_types,
                                             emit_default_doc=emit_default_doc)
        self.docstring_struct = class_def2docstring_structure(ast)


class DocstringTransform(BaseTransform):
    """
    Transformer
    """

    def __init__(self, docstring, inline_types, emit_default_doc, docstring_format='rest'):
        """
        Construct the object

        :param docstring: the docstring
        :type docstring: ```str```

        :param inline_types: Whether the type should be inline or in docstring
        :type inline_types: ```bool```

        :param emit_default_doc: Help/docstring should include 'With default' text
        :type emit_default_doc: ```bool``

        :param docstring_format: Format of docstring
        :type docstring_format: ```Literal['rest', 'numpy', 'google']```
        """
        super(DocstringTransform, self).__init__(inline_types=inline_types,
                                                 emit_default_doc=emit_default_doc)
        self.docstring_struct = parse_docstring(docstring, emit_default_doc=emit_default_doc)


class MethodTransform(BaseTransform):
    """
    Transformer
    """

    def __init__(self, ast, inline_types, emit_default_doc):
        """
        Construct the object

        :param ast: FunctionDef AST or Module AST
        :type ast: ```Union[ast.Module, ast.FunctionDef]```

        :param inline_types: Whether the type should be inline or in docstring
        :type inline_types: ```bool```

        :param emit_default_doc: Help/docstring should include 'With default' text
        :type emit_default_doc: ```bool``
        """
        super(MethodTransform, self).__init__(inline_types=inline_types,
                                              emit_default_doc=emit_default_doc)
        self.docstring_struct = class_def2docstring_structure(ast)
