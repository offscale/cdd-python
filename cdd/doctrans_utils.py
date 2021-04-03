"""
Helpers to traverse the AST, extract the docstring out, parse and format to intended style
"""
import ast
from ast import (
    AST,
    AnnAssign,
    Assign,
    AsyncFunctionDef,
    ClassDef,
    Expr,
    FunctionDef,
    Load,
    Name,
    NodeTransformer,
    get_docstring,
    walk,
)
from collections import OrderedDict

from cdd import emit, parse
from cdd.ast_utils import find_in_ast, get_value, set_value
from cdd.docstring_parsers import parse_docstring
from cdd.parser_utils import ir_merge
from cdd.pure_utils import set_attr, set_item, simple_types


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

    def generic_visit(self, node):
        """
        visits the `AST` node, if it could have a docstring pass it off to that handler

        :param node: The AST node
        :type node: ```AST```

        :returns: Potentially changed AST node
        :rtype: ```AST```
        """
        is_func, doc_str = isinstance(node, (AsyncFunctionDef, FunctionDef)), None
        if is_func or isinstance(node, ClassDef):
            # node, doc_str = self._handle_node_with_docstring(node)
            doc_str = ast.get_docstring(node)
        if is_func:
            node = self._handle_function(node, doc_str)
        return super(DocTrans, self).generic_visit(node)

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

    # TODO: Implement class and test docstring type conversion
    #
    # def _handle_node_with_docstring(
    #     self, node, word_wrap=False, emit_default_doc=False
    # ):
    #     """
    #  Potentially change a node with docstring, by inlining or doc-stringing its type & changing its docstring_format
    #
    #     :param node: AST node that `ast.get_docstring` can work with
    #     :type node: ```Union[AsyncFunctionDef, FunctionDef, ClassDef]```
    #
    #     :param word_wrap: Whether to word-wrap. Set `DOCTRANS_LINE_LENGTH` to configure length.
    #     :type word_wrap: ```bool```
    #
    #     :param emit_default_doc: Whether help/docstring should include 'With default' text
    #     :type emit_default_doc: ```bool```
    #
    #     :returns: Potentially changed `node`, i.e., inlined||docstringed types and changed docstring_format, doc_str
    #     :rtype: ```Union[AST, str]```
    #     """
    #     doc_str = get_docstring(node)
    #     if doc_str is None:
    #         return node, doc_str
    #
    #     style = derive_docstring_format(doc_str)
    #     if (
    #         style == self.docstring_format
    #         and self.type_annotations
    #         and self.existing_type_annotations
    #     ):
    #         return node, doc_str
    #
    #     parsed_emit_common_kwargs = dict(
    #        word_wrap=word_wrap, emit_default_doc=emit_default_doc
    #     )
    #     ir = parse_docstring(
    #         docstring=doc_str,
    #         parse_original_whitespace=True,
    #         **parsed_emit_common_kwargs
    #     )
    #     node.body[0] = Expr(
    #         set_value(
    #             emit.docstring(
    #                 ir,
    #                 docstring_format=self.docstring_format,
    #                 indent_level=1,
    #                 **parsed_emit_common_kwargs
    #             )
    #         )
    #     )
    #     return super(DocTrans, self).generic_visit(node), doc_str

    def _get_ass_typ(self, node):
        """
        Get the type of the assignment

        :param node: Assignment
        :type node: ```Union[Assign, AnnAssign]```

        :returns: The type of the assignment, e.g., `int`
        :rtype: ```Optional[str]```
        """
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

        return ir["params"].get(
            *(node.targets[0], {"typ": node.type_comment})
            if isinstance(node, Assign)
            else (node.target, {"typ": node.annotation or node.type_comment})
        )[
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
        ir, changed = parse_docstring(doc_str), False
        ir["name"] = node.name
        if node.returns:
            return_value = get_value(node.returns)
            if return_value is not None:
                ir.__setitem__(
                    "returns", OrderedDict((("return_type", {"typ": return_value}),))
                ) if ir.get("returns") is None else ir["returns"][
                    "return_type"
                ].__setitem__(
                    "typ", return_value
                )
                if not self.type_annotations:
                    ir["returns"] = node.returns = None
            changed = True
        if node.args:
            ir_merge(
                target=ir,
                other={
                    "params": OrderedDict(
                        map(
                            lambda _arg: (
                                _arg.arg,
                                {
                                    "typ": get_value(
                                        _arg.annotation
                                        or getattr(_arg, "type_comment", None)
                                    )
                                },
                            ),
                            node.args.args,
                        )
                    ),
                    "returns": None,
                },
            )
            changed = True
            if not self.type_annotations:
                node.args.args = list(map(clear_annotation, node.args.args))
                ir["params"] = OrderedDict()

        if changed:
            if self.type_annotations:

                def _to_ast_typ(typ):
                    """
                    :param typ: The type as stored in the IR
                    :type typ: ```Union[str, AST]```

                    :returns: Type to annotate with
                    :rtype: ```AST```
                    """
                    return (
                        Name(typ, Load())
                        if typ in simple_types
                        else typ
                        if isinstance(typ, AST)
                        else ast.parse(typ)
                    )

                if ir["returns"].get("return_type", {"typ": None}).get("typ"):
                    node.returns = _to_ast_typ(ir["returns"]["return_type"]["typ"])
                node.args.args = list(
                    map(
                        lambda _arg: set_attr(
                            _arg,
                            "annotation",
                            _to_ast_typ(ir["params"][_arg.arg]["typ"]),
                        )
                        if ir["params"][_arg.arg].get("typ")
                        else _arg,
                        node.args.args,
                    )
                )

            else:
                # Remove types from IR
                ir.update(
                    {
                        k: OrderedDict(
                            map(
                                lambda _param: set_item(_param, "typ", None),
                                ir[k].items(),
                            )
                        )
                        if ir[k]
                        else ir[k]
                        for k in ("params", "returns")
                    }
                )
            (
                node.body.__setitem__
                if isinstance(node.body[0], Expr)
                and isinstance(get_value(node.body[0].value), str)
                else node.body.insert
            )(
                0,
                Expr(
                    set_value(
                        emit.docstring(
                            ir,
                            emit_types=not self.type_annotations,
                            docstring_format=self.docstring_format,
                            indent_level=1,
                        )
                    )
                ),
            )
            if not ir["doc"] or get_value(node.body[0].value).isspace():
                del node.body[0]
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
