"""
Mocks for the `class`
"""

from ast import ClassDef, Name, Load, Expr, Constant, AnnAssign, \
    Store, Subscript, Tuple, Dict, Attribute, Index, Call, parse

from doctrans.pure_utils import PY3_8

class_str = '''
class TargetClass(object):
    """
    Acquire from the official tensorflow_datasets model zoo, or the ophthalmology focussed ml-prepare library

    :cvar dataset_name: name of dataset. Defaults to mnist
    :cvar tfds_dir: directory to look for models in. Defaults to ~/tensorflow_datasets
    :cvar K: backend engine, e.g., `np` or `tf`. Defaults to np
    :cvar as_numpy: Convert to numpy ndarrays
    :cvar data_loader_kwargs: pass this as arguments to data_loader function
    :cvar return_type: Train and tests dataset splits. Defaults to (np.empty(0), np.empty(0))"""

    dataset_name: str = 'mnist'
    tfds_dir: Optional[str] = '~/tensorflow_datasets'
    K: Literal['np', 'tf'] = 'np'
    as_numpy: Optional[bool] = None
    data_loader_kwargs: dict = {}
    return_type: Union[Tuple[tf.data.Dataset, tf.data.Dataset], Tuple[np.ndarray, np.ndarray]] = (
        np.empty(0),
        np.empty(0),
    )
'''

class_ast = ClassDef(
    bases=[Name(ctx=Load(),
                id='object')],
    body=[
        Expr(
            value=Constant(
                kind=None,
                value='\n    Acquire from the official tensorflow_datasets model zoo,'
                      ' or the ophthalmology focussed ml-prepare library\n\n    '
                      ':cvar dataset_name: name of dataset. Defaults to mnist\n    '
                      ':cvar tfds_dir: directory to look for models in. Defaults to ~/tensorflow_datasets\n    '
                      ':cvar K: backend engine, e.g., `np` or `tf`. Defaults to np\n    '
                      ':cvar as_numpy: Convert to numpy ndarrays\n    '
                      ':cvar data_loader_kwargs: pass this as arguments to data_loader function\n    '
                      ':cvar return_type: Train and tests dataset splits. Defaults to (np.empty(0), np.empty(0))'
            )),
        AnnAssign(annotation=Name(ctx=Load(),
                                  id='str'),
                  simple=1,
                  target=Name(ctx=Store(),
                              id='dataset_name'),
                  value=Constant(kind=None,
                                 value='mnist')),
        AnnAssign(annotation=Subscript(ctx=Load(),
                                       slice=Index(value=Name(ctx=Load(),
                                                              id='str')),
                                       value=Name(ctx=Load(),
                                                  id='Optional')),
                  simple=1,
                  target=Name(ctx=Store(),
                              id='tfds_dir'),
                  value=Constant(kind=None,
                                 value='~/tensorflow_datasets')),
        AnnAssign(annotation=Subscript(ctx=Load(),
                                       slice=Index(value=Tuple(ctx=Load(),
                                                               elts=[Constant(kind=None,
                                                                              value='np'),
                                                                     Constant(kind=None,
                                                                              value='tf')])),
                                       value=Name(ctx=Load(),
                                                  id='Literal')),
                  simple=1,
                  target=Name(ctx=Store(),
                              id='K'),
                  value=Constant(kind=None,
                                 value='np')),
        AnnAssign(annotation=Subscript(ctx=Load(),
                                       slice=Index(value=Name(ctx=Load(),
                                                              id='bool')),
                                       value=Name(ctx=Load(),
                                                  id='Optional')),
                  simple=1,
                  target=Name(ctx=Store(),
                              id='as_numpy'),
                  value=Constant(kind=None,
                                 value=None)),
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
                slice=Index(
                    value=Tuple(
                        ctx=Load(),
                        elts=[
                            Subscript(
                                ctx=Load(),
                                slice=Index(
                                    value=Tuple(
                                        ctx=Load(),
                                        elts=[
                                            Attribute(
                                                attr='Dataset',
                                                ctx=Load(),
                                                value=Attribute(attr='data',
                                                                ctx=Load(),
                                                                value=Name(ctx=Load(),
                                                                           id='tf'))),
                                            Attribute(
                                                attr='Dataset',
                                                ctx=Load(),
                                                value=Attribute(attr='data',
                                                                ctx=Load(),
                                                                value=Name(ctx=Load(),
                                                                           id='tf')))
                                        ]
                                    )
                                ),
                                value=Name(ctx=Load(),
                                           id='Tuple')),
                            Subscript(ctx=Load(),
                                      slice=Index(
                                          value=Tuple(
                                              ctx=Load(),
                                              elts=[
                                                  Attribute(attr='ndarray',
                                                            ctx=Load(),
                                                            value=Name(ctx=Load(),
                                                                       id='np')),
                                                  Attribute(attr='ndarray',
                                                            ctx=Load(),
                                                            value=Name(ctx=Load(),
                                                                       id='np'))
                                              ])),
                                      value=Name(ctx=Load(),
                                                 id='Tuple'))])),
                value=Name(ctx=Load(),
                           id='Union')),
            simple=1,
            target=Name(ctx=Store(),
                        id='return_type'),
            value=Tuple(ctx=Load(),
                        elts=[Call(args=[Constant(kind=None,
                                                  value=0)],
                                   func=Attribute(attr='empty',
                                                  ctx=Load(),
                                                  value=Name(ctx=Load(),
                                                             id='np')),
                                   keywords=[]),
                              Call(args=[Constant(kind=None,
                                                  value=0)],
                                   func=Attribute(attr='empty',
                                                  ctx=Load(),
                                                  value=Name(ctx=Load(),
                                                             id='np')),
                                   keywords=[])]))],
    decorator_list=[],
    keywords=[],
    name='TargetClass'
) if PY3_8 else parse(class_str).body[0]
