"""
OpenAPI emit_utils
"""

from ast import Assign, Call, Load, Name, Store, keyword

from cdd.shared.ast_utils import set_value

column_fk: Assign = Assign(
    targets=[Name(id="column_name", ctx=Store(), lineno=None, col_offset=None)],
    value=Call(
        func=Name(id="Column", ctx=Load(), lineno=None, col_offset=None),
        args=[
            Name(id="TableName0", ctx=Load(), lineno=None, col_offset=None),
            Call(
                func=Name(id="ForeignKey", ctx=Load(), lineno=None, col_offset=None),
                args=[set_value("TableName0")],
                keywords=[],
                lineno=None,
                col_offset=None,
            ),
        ],
        keywords=[keyword(arg="nullable", value=set_value(True))],
        lineno=None,
        col_offset=None,
    ),
    type_comment=None,
    expr=None,
    lineno=None,
)

column_fk_gold: Assign = Assign(
    targets=[Name(id="column_name", ctx=Store(), lineno=None, col_offset=None)],
    value=Call(
        func=Name(id="Column", ctx=Load(), lineno=None, col_offset=None),
        args=[
            Name(id="Integer", ctx=Load(), lineno=None, col_offset=None),
            Call(
                func=Name(id="ForeignKey", ctx=Load(), lineno=None, col_offset=None),
                args=[set_value("table_name0.id")],
                keywords=[],
                lineno=None,
                col_offset=None,
            ),
        ],
        keywords=[keyword(arg="nullable", value=set_value(True), identifier=None)],
        lineno=None,
        col_offset=None,
    ),
    type_comment=None,
    expr=None,
    lineno=None,
)

id_column: Assign = Assign(
    targets=[Name(id="id", ctx=Store(), lineno=None, col_offset=None)],
    value=Call(
        func=Name(id="Column", ctx=Load(), lineno=None, col_offset=None),
        args=[Name(id="Integer", ctx=Load(), lineno=None, col_offset=None)],
        keywords=[
            keyword(arg="primary_key", value=set_value(True)),
            keyword(
                arg="server_default",
                value=Call(
                    func=Name(id="Identity", ctx=Load(), lineno=None, col_offset=None),
                    args=[],
                    keywords=[],
                    lineno=None,
                    col_offset=None,
                ),
            ),
        ],
        lineno=None,
        col_offset=None,
    ),
    type_comment=None,
    expr=None,
    lineno=None,
)

__all__ = ["column_fk", "column_fk_gold", "id_column"]  # type: list[str]
