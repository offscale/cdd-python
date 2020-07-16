try:
    import tensorflow as tf
    import numpy as np
except ImportError:
    from collections import namedtuple

    tf = namedtuple('TensorFlow', ('data',))(namedtuple('data', ('Dataset',)))
    np = namedtuple('numpy', ('ndarray',))

from ast import Module, ClassDef, Name, Load, Expr, Constant, AnnAssign, \
    Subscript, Store, Dict, Tuple, Attribute, Index, arguments, FunctionDef, arg, Assign, Call, keyword, Return

docstring0 = """
Acquire from the official tensorflow_datasets model zoo, or the ophthalmology focussed ml-prepare library

:param dataset_name: name of dataset
:type dataset_name: ```str```

:param tfds_dir: directory to look for models in. Default is ~/tensorflow_datasets.
:type tfds_dir: ```Optional[str]```

:param K: backend engine, e.g., `np` or `tf`
:type K: ```Optional[Literal[np, tf]]```

:param as_numpy: Convert to numpy ndarrays
:type as_numpy: ```bool```

:param data_loader_kwargs: pass this as arguments to data_loader function
:type data_loader_kwargs: ```**data_loader_kwargs```

:return: Train and tests dataset splits
:rtype: ```Union[Tuple[tf.data.Dataset, tf.data.Dataset], Tuple[np.ndarray, np.ndarray]]```
"""

cls = '''
class TargetClass(object):
    """
    Acquire from the official tensorflow_datasets model zoo, or the ophthalmology focussed ml-prepare library

    :cvar dataset_name: name of dataset
    :cvar tfds_dir: directory to look for models in. Default is ~/tensorflow_datasets.
    :cvar K: backend engine, e.g., `np` or `tf`
    :cvar as_numpy: Convert to numpy ndarrays
    :cvar data_loader_kwargs: pass this as arguments to data_loader function
    :cvar return_type: Train and tests dataset splits"""

    dataset_name: str = ''
    tfds_dir: Optional[str] = None
    K: Optional[Literal[np, tf]] = None
    as_numpy: bool = True
    data_loader_kwargs: dict = {}
    return_type: Union[Tuple[tf.data.Dataset, tf.data.Dataset], Tuple[np.ndarray, np.ndarray]] = None
'''

# Generated from `print_ast(parse(cls, mode='exec'))`
ast_def = Module(body=[
    ClassDef(bases=[Name(ctx=Load(),
                         id='object')],
             body=[Expr(value=Constant(kind=None,
                                       value='\n    Acquire from the official tensorflow_datasets model zoo, '
                                             'or the ophthalmology focussed ml-prepare library\n\n    '
                                             ':cvar dataset_name: name of dataset\n    '
                                             ':cvar tfds_dir: directory to look for models in.'
                                             ' Default is ~/tensorflow_datasets.\n    '
                                             ':cvar K: backend engine, e.g., `np` or `tf`\n    '
                                             ':cvar as_numpy: Convert to numpy ndarrays\n    '
                                             ':cvar data_loader_kwargs: pass this as arguments to'
                                             ' data_loader function\n    '
                                             ':cvar return_type: Train and tests dataset splits')),
                   AnnAssign(annotation=Name(ctx=Load(),
                                             id='str'),
                             simple=1,
                             target=Name(ctx=Store(),
                                         id='dataset_name'),
                             value=Constant(kind=None,
                                            value='')),
                   AnnAssign(annotation=Subscript(ctx=Load(),
                                                  slice=Index(value=Name(ctx=Load(),
                                                                         id='str')),
                                                  value=Name(ctx=Load(),
                                                             id='Optional')),
                             simple=1,
                             target=Name(ctx=Store(),
                                         id='tfds_dir'),
                             value=Constant(kind=None,
                                            value=None)),
                   AnnAssign(
                       annotation=Subscript(ctx=Load(),
                                            slice=Index(
                                                value=Subscript(ctx=Load(),
                                                                slice=Index(value=Tuple(
                                                                    ctx=Load(),
                                                                    elts=[Name(ctx=Load(),
                                                                               id='np'),
                                                                          Name(ctx=Load(),
                                                                               id='tf')])),
                                                                value=Name(ctx=Load(),
                                                                           id='Literal'))),
                                            value=Name(ctx=Load(),
                                                       id='Optional')),
                       simple=1,
                       target=Name(ctx=Store(),
                                   id='K'),
                       value=Constant(kind=None,
                                      value=None)),
                   AnnAssign(annotation=Name(ctx=Load(),
                                             id='bool'),
                             simple=1,
                             target=Name(ctx=Store(),
                                         id='as_numpy'),
                             value=Constant(kind=None,
                                            value=True)),
                   AnnAssign(annotation=Name(ctx=Load(),
                                             id='dict'),
                             simple=1,
                             target=Name(ctx=Store(),
                                         id='data_loader_kwargs'),
                             value=Dict(keys=[],
                                        values=[])),
                   AnnAssign(annotation=Subscript(
                       ctx=Load(),
                       slice=Index(value=Tuple(
                           ctx=Load(),
                           elts=[
                               Subscript(ctx=Load(),
                                         slice=Index(value=Tuple(ctx=Load(),
                                                                 elts=[Attribute(attr='Dataset',
                                                                                 ctx=Load(),
                                                                                 value=Attribute(attr='data',
                                                                                                 ctx=Load(),
                                                                                                 value=Name(ctx=Load(),
                                                                                                            id='tf'))),
                                                                       Attribute(attr='Dataset',
                                                                                 ctx=Load(),
                                                                                 value=Attribute(attr='data',
                                                                                                 ctx=Load(),
                                                                                                 value=Name(ctx=Load(),
                                                                                                            id='tf')))
                                                                       ])),
                                         value=Name(ctx=Load(),
                                                    id='Tuple')),
                               Subscript(ctx=Load(),
                                         slice=Index(value=Tuple(ctx=Load(),
                                                                 elts=[Attribute(attr='ndarray',
                                                                                 ctx=Load(),
                                                                                 value=Name(ctx=Load(),
                                                                                            id='np')),
                                                                       Attribute(attr='ndarray',
                                                                                 ctx=Load(),
                                                                                 value=Name(ctx=Load(),
                                                                                            id='np'))])),
                                         value=Name(ctx=Load(),
                                                    id='Tuple'))])),
                       value=Name(ctx=Load(),
                                  id='Union')),
                       simple=1,
                       target=Name(ctx=Store(),
                                   id='return_type'),
                       value=Constant(kind=None,
                                      value=None))],
             decorator_list=[],
             keywords=[],
             name='TargetClass')],
    type_ignores=[])

set_cli_func = '''
def set_cli_args(argument_parser):
    """
    Set CLI arguments

    :param argument_parser: argument parser
    :type argument_parser: ```ArgumentParser```

    :return: argument parser and return type
    :rtype: ```Tuple[ArgumentParser, Union[Tuple[tf.data.Dataset, tf.data.Dataset], Tuple[np.ndarray, np.ndarray]]]```
    """

    argument_parser.description = 'Acquire from the official tensorflow_datasets model zoo, ' \
                                  'or the ophthalmology focussed ml-prepare library'

    argument_parser.add_argument('--dataset_name', type=str, help='name of dataset', required=True)
    argument_parser.add_argument('--tfds_dir', type=str, default='~/tensorflow_datasets',
                                 help='directory to look for models in.')
    argument_parser.add_argument('--K', type=str,
                                 choices=('np', 'tf'),
                                 help='backend engine, e.g., `np` or `tf`',
                                 required=True)
    argument_parser.add_argument('--as_numpy', type=bool, default=True, help='Convert to numpy ndarrays')
    argument_parser.add_argument('--data_loader_kwargs', type=loads,
                                 help='pass this as arguments to data_loader function')

    return argument_parser, (Union[Tuple[tf.data.Dataset, tf.data.Dataset], Tuple[np.ndarray, np.ndarray]])
'''

set_cli_func_ast = Module(body=[
    FunctionDef(args=arguments(args=[arg(annotation=None,
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
                                              ':return: argument parser and return type\n    '
                                              ':rtype: ```Tuple[ArgumentParser,'
                                              ' Union[Tuple[tf.data.Dataset, tf.data.Dataset],'
                                              ' Tuple[np.ndarray, np.ndarray]]]```\n    ')),
                    Assign(targets=[Attribute(attr='description',
                                              ctx=Store(),
                                              value=Name(ctx=Load(),
                                                         id='argument_parser'))],
                           type_comment=None,
                           value=Constant(kind=None,
                                          value='Acquire from the official tensorflow_datasets model zoo,'
                                                ' or the ophthalmology focussed ml-prepare library')),
                    Expr(value=Call(args=[Constant(kind=None,
                                                   value='--dataset_name')],
                                    func=Attribute(attr='add_argument',
                                                   ctx=Load(),
                                                   value=Name(ctx=Load(),
                                                              id='argument_parser')),
                                    keywords=[keyword(arg='type',
                                                      value=Name(ctx=Load(),
                                                                 id='str')),
                                              keyword(arg='help',
                                                      value=Constant(kind=None,
                                                                     value='name of dataset')),
                                              keyword(arg='required',
                                                      value=Constant(kind=None,
                                                                     value=True))])),
                    Expr(value=Call(args=[Constant(kind=None,
                                                   value='--tfds_dir')],
                                    func=Attribute(attr='add_argument',
                                                   ctx=Load(),
                                                   value=Name(ctx=Load(),
                                                              id='argument_parser')),
                                    keywords=[keyword(arg='type',
                                                      value=Name(ctx=Load(),
                                                                 id='str')),
                                              keyword(arg='default',
                                                      value=Constant(kind=None,
                                                                     value='~/tensorflow_datasets')),
                                              keyword(arg='help',
                                                      value=Constant(kind=None,
                                                                     value='directory to look for models in.'))])),
                    Expr(value=Call(args=[Constant(kind=None,
                                                   value='--K')],
                                    func=Attribute(attr='add_argument',
                                                   ctx=Load(),
                                                   value=Name(ctx=Load(),
                                                              id='argument_parser')),
                                    keywords=[keyword(arg='type',
                                                      value=Name(ctx=Load(),
                                                                 id='str')),
                                              keyword(arg='choices',
                                                      value=Tuple(ctx=Load(),
                                                                  elts=[Constant(kind=None,
                                                                                 value='np'),
                                                                        Constant(kind=None,
                                                                                 value='tf')])),
                                              keyword(arg='help',
                                                      value=Constant(kind=None,
                                                                     value='backend engine, e.g., `np` or `tf`')),
                                              keyword(arg='required',
                                                      value=Constant(kind=None,
                                                                     value=True))
                                              ])),
                    Expr(value=Call(args=[Constant(kind=None,
                                                   value='--as_numpy')],
                                    func=Attribute(attr='add_argument',
                                                   ctx=Load(),
                                                   value=Name(ctx=Load(),
                                                              id='argument_parser')),
                                    keywords=[keyword(arg='type',
                                                      value=Name(ctx=Load(),
                                                                 id='bool')),
                                              keyword(arg='default',
                                                      value=Constant(kind=None,
                                                                     value=True)),
                                              keyword(arg='help',
                                                      value=Constant(kind=None,
                                                                     value='Convert to numpy ndarrays'))])),
                    Expr(value=Call(args=[Constant(kind=None,
                                                   value='--data_loader_kwargs')],
                                    func=Attribute(attr='add_argument',
                                                   ctx=Load(),
                                                   value=Name(ctx=Load(),
                                                              id='argument_parser')),
                                    keywords=[keyword(arg='type',
                                                      value=Name(ctx=Load(),
                                                                 id='loads')),
                                              keyword(arg='help',
                                                      value=Constant(kind=None,
                                                                     value='pass this as arguments'
                                                                           ' to data_loader function'))])),
                    Return(
                        value=Tuple(
                            ctx=Load(),
                            elts=[
                                Name(ctx=Load(),
                                     id='argument_parser'),
                                Subscript(
                                    ctx=Load(),
                                    slice=Index(value=Tuple(ctx=Load(),
                                                            elts=[Subscript(
                                                                ctx=Load(),
                                                                slice=Index(
                                                                    value=Tuple(ctx=Load(),
                                                                                elts=[Attribute(
                                                                                    attr='Dataset',
                                                                                    ctx=Load(),
                                                                                    value=Attribute(
                                                                                        attr='data',
                                                                                        ctx=Load(),
                                                                                        value=Name(
                                                                                            ctx=Load(),
                                                                                            id='tf'))),
                                                                                    Attribute(
                                                                                        attr='Dataset',
                                                                                        ctx=Load(),
                                                                                        value=Attribute(
                                                                                            attr='data',
                                                                                            ctx=Load(),
                                                                                            value=Name(
                                                                                                ctx=Load(),
                                                                                                id='tf')))])),
                                                                value=Name(ctx=Load(),
                                                                           id='Tuple')),
                                                                Subscript(ctx=Load(),
                                                                          slice=Index(
                                                                              value=Tuple(ctx=Load(),
                                                                                          elts=[Attribute(
                                                                                              attr='ndarray',
                                                                                              ctx=Load(),
                                                                                              value=Name(
                                                                                                  ctx=Load(),
                                                                                                  id='np')),
                                                                                              Attribute(
                                                                                                  attr='ndarray',
                                                                                                  ctx=Load(),
                                                                                                  value=Name(
                                                                                                      ctx=Load(),
                                                                                                      id='np'))])),
                                                                          value=Name(ctx=Load(),
                                                                                     id='Tuple'))])),
                                    value=Name(ctx=Load(),
                                               id='Union'))]))],
                decorator_list=[],
                name='set_cli_args',
                returns=None,
                type_comment=None)],
    type_ignores=[])
