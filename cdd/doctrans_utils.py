"""
Helpers to traverse the AST, extract the docstring out, parse and format to intended style
"""
import ast
from ast import (
    AnnAssign,
    Assign,
    AsyncFunctionDef,
    ClassDef,
    Expr,
    FunctionDef,
    NodeTransformer,
    get_docstring,
    walk,
)
from collections import OrderedDict

from cdd import emit, parse
from cdd.ast_utils import find_in_ast, get_value, set_value
from cdd.docstring_parsers import derive_docstring_format, parse_docstring
from cdd.parser_utils import ir_merge


def has_inline_types(node):
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
        self, docstring_format, inline_types, existing_inline_types, whole_ast
    ):
        """
        Transform the docstrings found to intended docstring_format, potentially manipulating type annotations also

        :param docstring_format: Format of docstring
        :type docstring_format: ```Literal['rest', 'numpydoc', 'google']```

        :param inline_types: Whether the type should be inline or in docstring
        :type inline_types: ```bool```

        :param existing_inline_types: Whether any inline types exist
        :type existing_inline_types: ```bool```

        :param whole_ast: The entire input AST, useful for lookups by location
        :type whole_ast: ```AST``
        """
        self.docstring_format = docstring_format
        self.inline_types = inline_types
        self.existing_inline_types = existing_inline_types
        self.whole_ast = whole_ast

    def _handle_node_with_docstring(
        self, node, word_wrap=False, emit_default_doc=False
    ):
        """
        Potentially change a node with docstring, by inlining or doc-stringing its type & changing its docstring_format

        :param node: AST node that `ast.get_docstring` can work with
        :type node: ```Union[AsyncFunctionDef, FunctionDef, ClassDef]```

        :param word_wrap: Whether to word-wrap. Set `DOCTRANS_LINE_LENGTH` to configure length.
        :type word_wrap: ```bool```

        :param emit_default_doc: Whether help/docstring should include 'With default' text
        :type emit_default_doc: ```bool```

        :returns: Potentially changed `node`, i.e., inlined||docstringed types and changed docstring_format, doc_str
        :rtype: ```Union[AST, str]```
        """
        doc_str = get_docstring(node)
        if doc_str is None:
            return node, doc_str

        style = derive_docstring_format(doc_str)
        if (
            style == self.docstring_format
            and self.inline_types
            and self.existing_inline_types
        ):
            return node, doc_str

        # parsed_emit_common_kwargs = dict(
        #    word_wrap=word_wrap, emit_default_doc=emit_default_doc
        # )
        # ir = parse_docstring(
        #     docstring=doc_str,
        #     parse_original_whitespace=True,
        #     **parsed_emit_common_kwargs
        # )
        # node.body[0] = Expr(
        #     set_value(
        #         emit.docstring(
        #             ir,
        #             docstring_format=self.docstring_format,
        #             indent_level=1,
        #             **parsed_emit_common_kwargs
        #         )
        #     )
        # )
        return super(DocTrans, self).generic_visit(node), doc_str

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

        :returns: `AnnAssign` if `inline_types` and type found else `Assign`
        :rtype: ```Union[AnnAssign, Assign]```
        """
        parent = find_in_ast(node._location[:-1], self.whole_ast)
        ir = (
            parse.docstring(get_docstring(parent))
            if isinstance(parent, (ClassDef, AsyncFunctionDef, FunctionDef))
            else {"params": OrderedDict()}
        )

        typ = ir["params"].get(
            node.target, {"typ": node.annotation or node.type_comment}
        )["typ"]
        if self.inline_types:
            node.annotation = typ
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

        :returns: `AnnAssign` if `inline_types` and type found else `Assign`
        :rtype: ```Union[Assign, AnnAssign]```
        """
        print("visit_Assign::node._location:", node._location, ";")
        if self.inline_types:
            if node.type_comment is None:
                # TODO: Look at parent structure to see if IR contains the type
                pass
            else:
                assert len(node.targets) == 1
                return AnnAssign(
                    target=node.targets[0],
                    value=node.value,
                    annotation=node.type_comment,
                )
        return node

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
                if not self.inline_types:
                    node.returns = None
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
                                        _arg.annotation or _arg.type_comment
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
            if not self.inline_types:
                node.args.args = list(map(clear_annotation, node.args.args))

        if changed:
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
                            emit_types=not self.inline_types,
                            docstring_format=self.docstring_format,
                            indent_level=1,
                        )
                    )
                ),
            )
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


__all__ = ["DocTrans", "has_inline_types"]
