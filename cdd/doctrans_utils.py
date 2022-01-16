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
    arg,
    get_docstring,
    walk,
)
from collections import OrderedDict
from copy import deepcopy
from operator import attrgetter, eq

from cdd import emit, parse
from cdd.ast_cst_utils import (
    find_cst_at_ast,
    maybe_replace_doc_str_in_function_or_class,
    maybe_replace_function_args,
    maybe_replace_function_return_type,
)
from cdd.ast_utils import (
    annotate_ancestry,
    find_in_ast,
    maybe_type_comment,
    set_arg,
    set_docstring,
    set_value,
    to_annotation,
    to_type_comment,
)
from cdd.cst_utils import reindent_block_with_pass_body
from cdd.docstring_parsers import parse_docstring
from cdd.parser_utils import ir_merge
from cdd.pure_utils import PY_GTE_3_8, is_ir_empty, none_types, omit_whitespace
from cdd.source_transformer import ast_parse


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
        self,
        docstring_format,
        word_wrap,
        type_annotations,
        existing_type_annotations,
        whole_ast,
    ):
        """
        Transform the docstrings found to intended docstring_format, potentially manipulating type annotations also

        :param docstring_format: Format of docstring
        :type docstring_format: ```Literal['rest', 'numpydoc', 'google']```

        :param word_wrap: Whether to word-wrap. Set `DOCTRANS_LINE_LENGTH` to configure length.
        :type word_wrap: ```bool```

        :param type_annotations: True to have type annotations (3.6+), False to place in docstring
        :type type_annotations: ```bool```

        :param existing_type_annotations: Whether there are any type annotations (3.6+)
        :type existing_type_annotations: ```bool```

        :param whole_ast: The entire input AST, useful for lookups by location
        :type whole_ast: ```AST``
        """

        self.docstring_format = docstring_format
        self.word_wrap = word_wrap
        self.type_annotations = type_annotations
        self.existing_type_annotations = existing_type_annotations
        if not hasattr(whole_ast, "_location"):
            self.whole_ast = deepcopy(whole_ast)
            annotate_ancestry(self.whole_ast)
        else:
            self.whole_ast = whole_ast
        self.memoized = {}

    # def generic_visit(self, node):
    #     """
    #     visits the `AST` node, if it could have a docstring pass it off to that handler
    #
    #     :param node: The AST node
    #     :type node: ```AST```
    #
    #     :return: Potentially changed AST node
    #     :rtype: ```AST```
    #     """
    #     is_func, doc_str = isinstance(node, (AsyncFunctionDef, FunctionDef)), None
    #     if is_func or isinstance(node, ClassDef):
    #         node, doc_str = self._handle_node_with_docstring(node)
    #         doc_str = ast.get_docstring(node, clean=True)
    #     if is_func:
    #         node = self._handle_function(node, doc_str)
    #     return super(DocTrans, self).generic_visit(node)

    def visit_AnnAssign(self, node):
        """
        Handle `AnnAssign`

        :param node: AnnAssign
        :type node: ```AnnAssign```

        :return: `AnnAssign` if `type_annotations` and type found else `Assign`
        :rtype: ```Union[AnnAssign, Assign]```
        """
        if self.type_annotations:
            node.annotation = self._get_ass_typ(node)
            setattr(node, "type_comment", None)
            return node

        return Assign(
            targets=[node.target],
            lineno=node.lineno,
            col_offset=getattr(node, "col_offset", None),
            end_lineno=getattr(node, "end_lineno", None),
            end_col_offset=getattr(node, "end_col_offset", None),
            type_comment=to_type_comment(node.annotation),
            # `var: int` is valid and turning it to `var = None  # type_comment int` would
            # be wrong, as the upcoming smarter type tracer will reverse this to `var: Optional[int] = None`
            value=set_value(none_types[-1]) if node.value is None else node.value,
        )

    def visit_Assign(self, node):
        """
        Handle `Assign`

        :param node: Assign
        :type node: ```Assign```

        :return: `AnnAssign` if `type_annotations` and type found else `Assign`
        :rtype: ```Union[Assign, AnnAssign]```
        """
        typ = self._get_ass_typ(node)
        annotation = (
            None if not self.type_annotations or typ is None else to_annotation(typ)
        )
        if annotation:
            assert len(node.targets) == 1
            return AnnAssign(
                annotation=to_annotation(typ),
                lineno=node.lineno,
                col_offset=getattr(node, "col_offset", None),
                end_lineno=getattr(node, "end_lineno", None),
                end_col_offset=getattr(node, "end_col_offset", None),
                simple=1,
                target=node.targets[0],
                expr=None,
                expr_target=None,
                expr_annotation=None,
                **{} if node.value is None else {"value": node.value},
                **maybe_type_comment,
            )
        else:
            setattr(node, "type_comment", typ)
        return node

    def visit_FunctionDef(self, node):
        """
        visits the `FunctionDef`, potentially augmenting its docstring and argument types

        :param node: FunctionDef
        :type node: ```FunctionDef```

        :return: Potentially changed FunctionDef
        :rtype: ```FunctionDef```
        """
        return self._handle_function(node, get_docstring(node, clean=False))

    def _get_ass_typ(self, node):
        """
        Get the type of the assignment

        :param node: Assignment
        :type node: ```Union[Assign, AnnAssign]```

        :return: The type of the assignment, e.g., `int`
        :rtype: ```Optional[str]```
        """
        name, typ_dict = (
            (node.targets[0], {"typ": getattr(node, "type_comment", None)})
            if isinstance(node, Assign)
            else (
                node.target,
                {"typ": node.annotation or getattr(node, "type_comment", None)},
            )
        )
        if not hasattr(node, "_location") or node._location[:-1] == [None]:
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
                        )(get_docstring(parent, clean=False))
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

    def _handle_function(self, node, original_doc_str):
        """
        Handle functions

        :param node: AsyncFunctionDef | FunctionDef
        :type node: ```Union[AsyncFunctionDef, FunctionDef]```

        :param original_doc_str: The docstring
        :type original_doc_str: ```Optional[str]```

        :return: Same type as input with args, returns, and docstring potentially modified
        :rtype: ```Union[AsyncFunctionDef, FunctionDef]```
        """
        ir = parse_docstring(original_doc_str)
        ir_merge(ir, parse.function(node))
        ir["name"] = node.name
        indent_level = max(
            len(node._location), 1
        )  # function docstrings always have at least 1 indent level
        doc_str = (
            None
            if is_ir_empty(ir)
            else emit.docstring(
                ir,
                emit_types=not self.type_annotations,
                emit_default_doc=False,
                docstring_format=self.docstring_format,
                indent_level=indent_level,
                word_wrap=self.word_wrap,
            )
        )
        if not doc_str or doc_str.isspace():
            if original_doc_str is not None:
                del node.body[0]
        else:
            set_docstring(
                original_doc_str
                if original_doc_str
                and eq(*map(omit_whitespace, (original_doc_str, doc_str)))
                else doc_str,
                False,
                node,
            )
        if self.type_annotations:
            # Add annotations
            if ir["params"]:
                node.args.args = list(
                    map(
                        lambda _arg: arg(
                            arg=_arg.arg,
                            annotation=to_annotation(
                                _arg.annotation
                                if _arg.annotation is not None
                                or _arg.arg in frozenset(("self", "cls"))
                                else ir["params"][_arg.arg].get("typ")
                            ),
                            identifier_arg=None,
                            end_col_offset=getattr(_arg, "end_col_offset", None),
                            **dict(expr=None, **maybe_type_comment)
                            if PY_GTE_3_8
                            else {},
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

    :return: AST node with annotations and type_comments set to `None`
    :rtype: ```AST```
    """
    if getattr(node, "annotation", None) is not None:
        node.annotation = None
    if getattr(node, "type_comment", None) is not None:
        setattr(node, "type_comment", None)
    return node


def doctransify_cst(cst_list, node):
    """
    Carefully replace only docstrings, function return annotations, assignment and annotation assignments.
    (maintaining all other existing whitespace, comments, &etc.); and only when cdd has changed them

    :param cst_list: List of `namedtuple`s with at least ("line_no_start", "line_no_end", "value") attributes
    :type cst_list: ```List[NamedTuple]```

    :param node: AST node with a `.body`, probably the `ast.Module`
    :type node: ```AST```
    """
    for _node in walk(node):
        if hasattr(_node, "_location"):
            is_func = isinstance(_node, (AsyncFunctionDef, FunctionDef))
            if isinstance(_node, ClassDef) or is_func:
                cst_idx, cst_node = find_cst_at_ast(cst_list, _node)

                if cst_node is not None:
                    maybe_replace_doc_str_in_function_or_class(_node, cst_idx, cst_list)

                    if is_func:
                        cur_ast_node = ast_parse(
                            reindent_block_with_pass_body(cst_list[cst_idx].value),
                            skip_annotate=True,
                            skip_docstring_remit=True,
                        ).body[0]

                        maybe_replace_function_return_type(
                            _node, cur_ast_node, cst_idx, cst_list
                        )
                        maybe_replace_function_args(
                            _node, cur_ast_node, cst_idx, cst_list
                        )
            # TODO: AnnAssign|Assign
            # AnnAssign|Assign is separate task than the `maybe_replace_function_args` as inferring types is done
            #   better with knowledge of function return types and function arguments (`default` being the only issue)
            # elif isinstance(_node, (AnnAssign, Assign)):
            #     print("(AnnAssign | Assign)._location:", _node._location, ";")
            #     print_ast(_node)


__all__ = ["DocTrans", "clear_annotation", "doctransify_cst", "has_type_annotations"]
