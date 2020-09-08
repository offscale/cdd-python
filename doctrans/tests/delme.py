import ast
import operator


def add_6_5(*, a=6, b=5):
    """
    :param a: first param
    :type a: ```int```

    :param b: second param
    :type b: ```int```

    :returns: Aggregated summation of `a` and `b`
    :rtype: ```int```
    """
    return operator.add(a, b)


s = '''
def add_6_5(*, a=6, b=5):
    """
    :param a: first param
    :type a: ```int```

    :param b: second param
    :type b: ```int```

    :returns: Aggregated summation of `a` and `b`
    :rtype: ```int```
    """
    return operator.add(a, b)
'''

print(ast.dump(ast.parse(s).body[0], indent=4))
