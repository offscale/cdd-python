"""
Pure utils for pure functions. For the same input will always produce the same output.
"""

from pprint import PrettyPrinter

pp = PrettyPrinter(indent=4).pprint
tab = ' ' * 4
simple_types = {'int': 0, float: .0, 'str': '', 'bool': False}
