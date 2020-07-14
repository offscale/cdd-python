from ast import AnnAssign, Load, Constant, Name, Store, Dict, parse
from pprint import PrettyPrinter

pp = PrettyPrinter(indent=4).pprint

tab = ' ' * 4

simple_types = {'int': 0, float: .0, 'str': '', 'bool': True}


def param2ast(param):
    """
    Converts a param to an AnnAssign

    :param param: dictionary of shape {'typ': str, 'name': str, 'doc': str}
    :type param: ```dict```

    :return: ast node (AnnAssign)
    :rtype: ```AnnAssign```
    """
    if param['typ'] in simple_types:
        return AnnAssign(annotation=Name(ctx=Load(),
                                         id=param['typ']),
                         simple=1,
                         target=Name(ctx=Store(),
                                     id=param['name']),
                         value=Constant(kind=None,
                                        value=simple_types[param['typ']]))
    elif param['typ'] == 'dict':
        return AnnAssign(annotation=Name(ctx=Load(),
                                         id=param['typ']),
                         simple=1,
                         target=Name(ctx=Store(),
                                     id=param['name']),
                         value=Dict(keys=[],
                                    values=[]))
    elif param['typ'].startswith('*'):
        return AnnAssign(annotation=Name(ctx=Load(),
                                         id='dict'),
                         simple=1,
                         target=Name(ctx=Store(),
                                     id=param['name']),
                         value=Dict(keys=[],
                                    values=[]))
    else:
        return AnnAssign(annotation=parse(param['typ']).body[0].value,
                         simple=1,
                         target=Name(ctx=Store(),
                                     id=param['name']),
                         value=Constant(kind=None,
                                        value=None))


__all__ = ['param2ast', 'pp', 'tab']
