"""
Helpers to traverse the AST of the input file, extract the docstring out, parse and format to intended style, and emit
"""
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
from copy import deepcopy

from cdd import emit
from cdd.ast_utils import cmp_ast, set_value
from cdd.docstring_parsers import derive_docstring_format, parse_docstring
from cdd.source_transformer import ast_parse


def doctrans(filename, docstring_format, inline_types):
    """
    Transform the docstrings found within provided filename to intended docstring_format

    :param filename: Python file to convert docstrings within. Edited in place.
    :type filename: ```str```

    :param docstring_format: Format of docstring
    :type docstring_format: ```Literal['rest', 'numpydoc', 'google']```

    :param inline_types: Whether the type should be inline or in docstring
    :type inline_types: ```bool```
    """
    with open(filename, "rt") as f:
        node = ast_parse(f.read(), skip_docstring_remit=True)
    orig_node = deepcopy(node)

    DocTrans(
        docstring_format=docstring_format,
        inline_types=inline_types,
        existing_inline_types=has_inline_types(node),
    ).visit(node)

    if not cmp_ast(node, orig_node):
        emit.file(node, filename, mode="wt", skip_black=True)


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

    def __init__(self, docstring_format, inline_types, existing_inline_types):
        """
        Transform the docstrings found to intended docstring_format, potentially manipulating type annotations also

        :param docstring_format: Format of docstring
        :type docstring_format: ```Literal['rest', 'numpydoc', 'google']```

        :param inline_types: Whether the type should be inline or in docstring
        :type inline_types: ```bool```

        :param existing_inline_types: Whether any inline types exist
        :type existing_inline_types: ```bool```
        """
        self.docstring_format = docstring_format
        self.inline_types = inline_types
        self.existing_inline_types = existing_inline_types

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

        :returns: Potentially changed `node`, i.e., inlined||docstringed types and changed docstring_format
        :rtype: ```AST```
        """
        doc_str = get_docstring(node)
        if doc_str is None:
            return node

        style = derive_docstring_format(doc_str)
        if (
            style == self.docstring_format
            and self.inline_types
            and self.existing_inline_types
        ):
            return node

        parsed_emit_common_kwargs = dict(
            word_wrap=word_wrap, emit_default_doc=emit_default_doc
        )
        ir = parse_docstring(
            docstring=doc_str,
            parse_original_whitespace=True,
            **parsed_emit_common_kwargs
        )
        node.body[0] = Expr(
            set_value(
                emit.docstring(
                    ir,
                    docstring_format=self.docstring_format,
                    **parsed_emit_common_kwargs
                )
            )
        )
        return node

    def generic_visit(self, node):
        """
        visits the `AST` node, if it could have a docstring pass it off to that handler

        :param node: The AST node
        :type node: ```AST```

        :returns: Potentially changed AST node
        :rtype: ```AST```
        """
        return (
            self._handle_node_with_docstring
            if isinstance(node, (AsyncFunctionDef, FunctionDef, ClassDef))
            else super(DocTrans, self).generic_visit
        )(node)

    def visit_AnnAssign(self, node):
        """
        Handle `AnnAssign`

        :param node: AnnAssign
        :type node: ```AnnAssign```

        :returns: `AnnAssign` if `inline_types` and type found else `Assign`
        :rtype: ```Union[AnnAssign, Assign]```
        """
        if self.inline_types:
            if node.annotation is None:
                # TODO: Look at parent structure to see if IR contains the type
                pass
            return node
        return Assign(
            targets=[node.target], value=node.value, type_comment=node.annotation
        )

    def visit_Assign(self, node):
        """
        Handle `Assign`

        :param node: Assign
        :type node: ```Assign```

        :returns: `AnnAssign` if `inline_types` and type found else `Assign`
        :rtype: ```Union[Assign, AnnAssign]```
        """
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


__all__ = ["doctrans"]
