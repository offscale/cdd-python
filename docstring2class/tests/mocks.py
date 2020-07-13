try:
    import tensorflow as tf
    import numpy as np
except ImportError:
    from collections import namedtuple

    tf = namedtuple('TensorFlow', ('data',))(namedtuple('data', ('Dataset',)))
    np = namedtuple('numpy', ('ndarray',))

from ast import Module, ClassDef, Name, Load, Expr, Constant, AnnAssign, \
    Subscript, Store, Dict, Tuple, Attribute, Index

docstring0 = """
Acquire from the official tensorflow_datasets model zoo, or the ophthalmology focussed ml-prepare library

:param dataset_name: name of dataset
:type dataset_name: ```str```

:param tfds_dir: directory to look for models in. Default is ~/tensorflow_datasets.
:type tfds_dir: ```Optional[str]```

:param K: backend engine, e.g., `np` or `tf`
:type K: ```Optional[Union[np, tf, Any]]```

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
    :cvar return_type: Train and tests dataset splits
    """

    dataset_name: Optional[str] = None
    tfds_dir: Optional[str] = None
    K: Optional[str]
    as_numpy: bool = True
    data_loader_kwargs: dict = {}
    return_type: Union[Tuple[tf.data.Dataset, tf.data.Dataset], Tuple[np.ndarray, np.ndarray]]
'''

# Generated from `print_ast(parse(cls, mode='exec'))`
ast_def = Module(body=[
    ClassDef(bases=[Name(ctx=Load(),
                         id='object')],
             body=[
                 Expr(value=Constant(kind=None,
                                     value='\n    Acquire from the official tensorflow_datasets model zoo, or the'
                                           ' ophthalmology focussed ml-prepare library\n\n'
                                           '    :cvar dataset_name: name of dataset\n'
                                           '    :cvar tfds_dir: directory to look for models in. '
                                           'Default is ~/tensorflow_datasets.\n'
                                           '    :cvar K: backend engine, e.g., `np` or `tf`\n'
                                           '    :cvar as_numpy: Convert to numpy ndarrays\n'
                                           '    :cvar data_loader_kwargs: pass this as arguments to '
                                           'data_loader function\n'
                                           '    :cvar return_type: Train and tests dataset splits\n    ')),
                 AnnAssign(annotation=Subscript(ctx=Load(),
                                                slice=Index(value=Name(ctx=Load(),
                                                                       id='str')),
                                                value=Name(ctx=Load(),
                                                           id='Optional')),
                           simple=1,
                           target=Name(ctx=Store(),
                                       id='dataset_name'),
                           value=Constant(kind=None,
                                          value=None)),
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
                 AnnAssign(annotation=Subscript(ctx=Load(),
                                                slice=Index(value=Name(ctx=Load(),
                                                                       id='str')),
                                                value=Name(ctx=Load(),
                                                           id='Optional')),
                           simple=1,
                           target=Name(ctx=Store(),
                                       id='K'),
                           value=None),
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
                 AnnAssign(
                     annotation=Subscript(
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
                                                                                                   value=Name(
                                                                                                       ctx=Load(),
                                                                                                       id='tf'))),
                                                                         Attribute(attr='Dataset',
                                                                                   ctx=Load(),
                                                                                   value=Attribute(attr='data',
                                                                                                   ctx=Load(),
                                                                                                   value=Name(
                                                                                                       ctx=Load(),
                                                                                                       id='tf')))])),
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
                     value=None)],
             decorator_list=[],
             keywords=[],
             name='TargetClass')
], type_ignores=[])
