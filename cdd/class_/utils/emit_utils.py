"""
Utility functions for `cdd.emit.class_`
"""

import ast
from ast import Attribute, Load, Name


class RewriteName(ast.NodeTransformer):
    """
    A :class:`NodeTransformer` subclass that walks the abstract syntax tree and
    allows modification of nodes. Here it modifies parameter names to be `self.param_name`
    """

    def __init__(self, node_ids):
        """
        Set parameter

        :param node_ids: Container of AST `id`s to match for rename
        :type node_ids: ```Optional[Iterator[str]]```
        """
        self.node_ids = node_ids

    def visit_Name(self, node):
        """
        Rename parameter name with a `self.` attribute prefix

        :param node: The AST node
        :type node: ```Name```

        :return: `Name` iff `Name` is not a parameter else `Attribute`
        :rtype: ```Union[Name, Attribute]```
        """
        return (
            Attribute(Name("self", Load()), node.id, Load())
            if not self.node_ids or node.id in self.node_ids
            else ast.NodeTransformer.generic_visit(self, node)
        )


__all__ = ["RewriteName"]
