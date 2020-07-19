from ast import FunctionDef, arguments, arg, Expr, Constant, Assign, Store, \
    Attribute, Name, Load, Call, keyword, Return, Tuple, Subscript, Index

argparse_func_str = '''
def set_cli_args(argument_parser):
    """
    Set CLI arguments

    :param argument_parser: argument parser
    :type argument_parser: ```ArgumentParser```

    :return: argument_parser, Train and tests dataset splits
    :rtype: ```Tuple[ArgumentParser, Union[Tuple[tf.data.Dataset, tf.data.Dataset], Tuple[np.ndarray, np.ndarray]]]```
    """

    argument_parser.description = 'Acquire from the official tensorflow_datasets model zoo, ' \
                                  'or the ophthalmology focussed ml-prepare library'

    argument_parser.add_argument('--dataset_name', type=str, help='name of dataset', required=True, default='mnist')
    argument_parser.add_argument('--tfds_dir', type=str,
                                 help='directory to look for models in. Default is ~/tensorflow_datasets.')
    argument_parser.add_argument('--K', type=globals().__getitem__,
                                 choices=('np', 'tf'),
                                 help='backend engine, e.g., `np` or `tf`',
                                 required=True)
    argument_parser.add_argument('--as_numpy', type=bool, help='Convert to numpy ndarrays')
    argument_parser.add_argument('--data_loader_kwargs', type=loads,
                                 help='pass this as arguments to data_loader function')

    return argument_parser, Union[Tuple[tf.data.Dataset, tf.data.Dataset], Tuple[np.ndarray, np.ndarray]]
'''

argparse_func_ast = FunctionDef(
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
                                  ':return: argument_parser, Train and tests dataset splits\n    '
                                  ':rtype: ```Tuple[ArgumentParser,'
                                  ' Union[Tuple[tf.data.Dataset, tf.data.Dataset], Tuple[np.ndarray, np.ndarray]]]```'
                                  '\n    ')),
        Assign(targets=[Attribute(attr='description',
                                  ctx=Store(),
                                  value=Name(ctx=Load(),
                                             id='argument_parser'))],
               type_comment=None,
               value=Constant(
                   kind=None,
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
                                                         value=True)),
                                  keyword(arg='default',
                                          value=Constant(kind=None,
                                                         value='mnist'))
                                  ])),
        Expr(value=Call(args=[Constant(kind=None,
                                       value='--tfds_dir')],
                        func=Attribute(attr='add_argument',
                                       ctx=Load(),
                                       value=Name(ctx=Load(),
                                                  id='argument_parser')),
                        keywords=[keyword(arg='type',
                                          value=Name(ctx=Load(),
                                                     id='str')),
                                  keyword(arg='help',
                                          value=Constant(kind=None,
                                                         value='directory to look for models in.'
                                                               ' Default is ~/tensorflow_datasets.'))])),
        Expr(value=Call(args=[Constant(kind=None,
                                       value='--K')],
                        func=Attribute(attr='add_argument',
                                       ctx=Load(),
                                       value=Name(ctx=Load(),
                                                  id='argument_parser')),
                        keywords=[keyword(arg='type',
                                          value=Attribute(attr='__getitem__',
                                                          ctx=Load(),
                                                          value=Call(args=[],
                                                                     func=Name(ctx=Load(),
                                                                               id='globals'),
                                                                     keywords=[]))),
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
                                                         value=True))])),
        Expr(value=Call(args=[Constant(kind=None,
                                       value='--as_numpy')],
                        func=Attribute(attr='add_argument',
                                       ctx=Load(),
                                       value=Name(ctx=Load(),
                                                  id='argument_parser')),
                        keywords=[keyword(arg='type',
                                          value=Name(ctx=Load(),
                                                     id='bool')),
                                  keyword(arg='help',
                                          value=Constant(kind=None,
                                                         value='Convert to numpy ndarrays'))])),
        Expr(value=Call(args=[Constant(kind=None,
                                       value='--data_loader_kwargs')],
                        func=Attribute(attr='add_argument',
                                       ctx=Load(),
                                       value=Name(ctx=Load(),
                                                  id='argument_parser')),
                        keywords=[
                            keyword(arg='type',
                                    value=Name(ctx=Load(),
                                               id='loads')),
                            keyword(arg='help',
                                    value=Constant(kind=None,
                                                   value='pass this as arguments to data_loader function'))
                        ])),
        Return(
            value=Tuple(
                ctx=Load(),
                elts=[
                    Name(ctx=Load(),
                         id='argument_parser'),
                    Subscript(
                        ctx=Load(),
                        slice=Index(value=Tuple(ctx=Load(),
                                                elts=[
                                                    Subscript(ctx=Load(),
                                                              slice=Index(value=Tuple(
                                                                  ctx=Load(),
                                                                  elts=[
                                                                      Attribute(
                                                                          attr='Dataset',
                                                                          ctx=Load(),
                                                                          value=Attribute(
                                                                              attr='data',
                                                                              ctx=Load(),
                                                                              value=Name(ctx=Load(),
                                                                                         id='tf'))),
                                                                      Attribute(
                                                                          attr='Dataset',
                                                                          ctx=Load(),
                                                                          value=Attribute(
                                                                              attr='data',
                                                                              ctx=Load(),
                                                                              value=Name(ctx=Load(),
                                                                                         id='tf')))])),
                                                              value=Name(ctx=Load(),
                                                                         id='Tuple')),
                                                    Subscript(
                                                        ctx=Load(),
                                                        slice=Index(value=Tuple(
                                                            ctx=Load(),
                                                            elts=[
                                                                Attribute(
                                                                    attr='ndarray',
                                                                    ctx=Load(),
                                                                    value=Name(ctx=Load(),
                                                                               id='np')),
                                                                Attribute(
                                                                    attr='ndarray',
                                                                    ctx=Load(),
                                                                    value=Name(ctx=Load(),
                                                                               id='np'))])),
                                                        value=Name(ctx=Load(),
                                                                   id='Tuple'))])),
                        value=Name(ctx=Load(),
                                   id='Union'))]))],
    decorator_list=[],
    name='set_cli_args',
    returns=None,
    type_comment=None
)
