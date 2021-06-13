"""
Helpers to traverse the AST, extract the docstring out, parse and format to intended style
"""

from ast import (
    AnnAssign,
    Assign,
    AsyncFunctionDef,
    ClassDef,
    FunctionDef,
    NodeTransformer,
    get_docstring,
    walk,
)
from collections import OrderedDict
from operator import attrgetter

from cdd import emit, parse
from cdd.ast_utils import find_in_ast, set_arg, set_docstring, to_annotation
from cdd.docstring_parsers import parse_docstring
from cdd.parser_utils import ir_merge


def has_type_annotations(node):
    """
    Whether the node—incl. any nodes within this node—have type annotations

    :param node: AST node
    :type node: ```AST```
    """
    return any(
        filter(
            lambda _node: hasattr(_node, "annotation")
            and _node.annotation is not None
            or hasattr(_node, "returns")
            and _node.returns is not None,
            walk(node),
        )
    )


class DocTrans(NodeTransformer):
    """
    Walk the nodes modifying the docstring and inlining||commenting types as it goes
    """

    def __init__(
        self, docstring_format, type_annotations, existing_type_annotations, whole_ast
    ):
        """
        Transform the docstrings found to intended docstring_format, potentially manipulating type annotations also

        :param docstring_format: Format of docstring
        :type docstring_format: ```Literal['rest', 'numpydoc', 'google']```

        :param type_annotations: True to have type annotations (3.6+), False to place in docstring
        :type type_annotations: ```bool```

        :param existing_type_annotations: Whether there are any type annotations (3.6+)
        :type existing_type_annotations: ```bool```

        :param whole_ast: The entire input AST, useful for lookups by location
        :type whole_ast: ```AST``
        """
        self.docstring_format = docstring_format
        self.type_annotations = type_annotations
        self.existing_type_annotations = existing_type_annotations
        self.whole_ast = whole_ast
        self.memoized = {}

    # def generic_visit(self, node):
    #     """
    #     visits the `AST` node, if it could have a docstring pass it off to that handler
    #
    #     :param node: The AST node
    #     :type node: ```AST```
    #
    #     :returns: Potentially changed AST node
    #     :rtype: ```AST```
    #     """
    #     is_func, doc_str = isinstance(node, (AsyncFunctionDef, FunctionDef)), None
    #     if is_func or isinstance(node, ClassDef):
    #         node, doc_str = self._handle_node_with_docstring(node)
    #         doc_str = ast.get_docstring(node)
    #     if is_func:
    #         node = self._handle_function(node, doc_str)
    #     return super(DocTrans, self).generic_visit(node)

    def visit_AnnAssign(self, node):
        """
        Handle `AnnAssign`

        :param node: AnnAssign
        :type node: ```AnnAssign```

        :returns: `AnnAssign` if `type_annotations` and type found else `Assign`
        :rtype: ```Union[AnnAssign, Assign]```
        """
        if self.type_annotations:
            node.annotation = self._get_ass_typ(node)
            node.type_comment = None
            return node

        return Assign(
            targets=[node.target],
            value=node.value,
            type_comment=node.annotation,
            lineno=None,
        )

    def visit_Assign(self, node):
        """
        Handle `Assign`

        :param node: Assign
        :type node: ```Assign```

        :returns: `AnnAssign` if `type_annotations` and type found else `Assign`
        :rtype: ```Union[Assign, AnnAssign]```
        """
        typ = self._get_ass_typ(node)
        if self.type_annotations:
            assert len(node.targets) == 1
            return AnnAssign(
                annotation=typ,
                value=node.value,
                type_comment=None,
                lineno=None,
                simple=1,
                target=node.targets[0],
                expr=None,
                expr_target=None,
                expr_annotation=None,
            )
        else:
            node.type_comment = typ
        return node

    def visit_Module(self, node):
        """
        visits the `Module`, potentially augmenting its docstring indentation

        :param node: Module
        :type node: ```Module```

        :returns: Potentially changed Module
        :rtype: ```Module```
        """
        # Clean might be wrong if the header is a license or other long-spiel documentation
        doc_str = get_docstring(node, clean=True)
        empty = doc_str is None
        if not empty:
            set_docstring("\n{}\n".format(doc_str), empty, node)
        node.body = list(map(self.visit, node.body))
        return node

    def visit_FunctionDef(self, node):
        """
        visits the `FunctionDef`, potentially augmenting its docstring and argument types

        :param node: FunctionDef
        :type node: ```FunctionDef```

        :returns: Potentially changed FunctionDef
        :rtype: ```FunctionDef```
        """
        return self._handle_function(node, get_docstring(node))

    def _get_ass_typ(self, node):
        """
        Get the type of the assignment

        :param node: Assignment
        :type node: ```Union[Assign, AnnAssign]```

        :returns: The type of the assignment, e.g., `int`
        :rtype: ```Optional[str]```
        """
        name, typ_dict = (
            (node.targets[0], {"typ": node.type_comment})
            if isinstance(node, Assign)
            else (node.target, {"typ": node.annotation or node.type_comment})
        )
        if not hasattr(node, "_location"):
            return typ_dict["typ"]
        search = node._location[:-1]
        search_str = ".".join(search)

        self.memoized[search_str] = ir = (
            self.memoized.get(
                search_str,
                (
                    lambda parent: (
                        (
                            lambda doc_str: None
                            if doc_str is None
                            else parse.docstring(doc_str)
                        )(get_docstring(parent))
                        if isinstance(parent, (ClassDef, AsyncFunctionDef, FunctionDef))
                        else {"params": OrderedDict()}
                    )
                )(find_in_ast(search, self.whole_ast)),
            )
            or {"params": OrderedDict()}
        )

        return ir["params"].get(name, typ_dict)[
            "typ"
        ]  # if ir is not None and ir.get("params") else None

    def _handle_function(self, node, doc_str):
        """
        Handle functions

        :param node: AsyncFunctionDef | FunctionDef
        :type node: ```Union[AsyncFunctionDef, FunctionDef]```

        :param doc_str: The docstring
        :type doc_str: ```Optional[str]```

        :returns: Same type as input with args, returns, and docstring potentially modified
        :rtype: ```Union[AsyncFunctionDef, FunctionDef]```
        """
        ir = parse_docstring(doc_str)
        ir_merge(ir, parse.function(node))
        ir["name"] = node.name
        indent_level = max(
            len(node._location) - 1, 1
        )  # function docstrings always have at least 1 indent level
        _doc_str = emit.docstring(
            ir,
            emit_types=not self.type_annotations,
            emit_default_doc=False,
            docstring_format=self.docstring_format,
            indent_level=indent_level,
        )
        if _doc_str.isspace():
            if doc_str is not None:
                del node.body[0]
        else:
            set_docstring(_doc_str, False, node)
        if self.type_annotations:
            # Add annotations
            if ir["params"]:
                node.args.args = list(
                    map(
                        lambda _arg: set_arg(
                            _arg.arg,
                            annotation=to_annotation(
                                _arg.annotation
                                if _arg.annotation is not None
                                else ir["params"][_arg.arg].get("typ")
                            ),
                        ),
                        node.args.args,
                    )
                )
            if (
                "return_type" in (ir.get("returns") or iter(()))
                and ir["returns"]["return_type"].get("typ") is not None
            ):
                node.returns = to_annotation(ir["returns"]["return_type"]["typ"])
        else:
            # Remove annotations
            node.args.args = list(map(set_arg, map(attrgetter("arg"), node.args.args)))
            node.returns = None

        node.body = list(map(self.visit, node.body))
        return node


def clear_annotation(node):
    """
    Remove annotations and type_comments from node

    :param node: AST node
    :type node: ```AST```

    :returns: AST node with annotations and type_comments set to `None`
    :rtype: ```AST```
    """
    if getattr(node, "annotation", None) is not None:
        node.annotation = None
    if getattr(node, "type_comment", None) is not None:
        node.type_comment = None
    return node


__all__ = ["DocTrans", "has_type_annotations"]
