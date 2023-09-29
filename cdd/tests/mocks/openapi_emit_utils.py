"""
OpenAPI emit_utils
"""

from ast import Assign, Call, Load, Name, Store, keyword

from cdd.shared.ast_utils import set_value

column_fk = Assign(
    targets=[Name(id="column_name", ctx=Store())],
    value=Call(
        func=Name(id="Column", ctx=Load()),
        args=[
            Name(id="TableName0", ctx=Load()),
            Call(
                func=Name(id="ForeignKey", ctx=Load()),
                args=[set_value("TableName0")],
                keywords=[],
            ),
        ],
        keywords=[keyword(arg="nullable", value=set_value(True))],
    ),
    type_comment=None,
    expr=None,
    lineno=None,
)

column_fk_gold = Assign(
    targets=[Name(id="column_name", ctx=Store())],
    value=Call(
        func=Name(id="Column", ctx=Load()),
        args=[
            Name(id="Integer", ctx=Load()),
            Call(
                func=Name(id="ForeignKey", ctx=Load()),
                args=[set_value("table_name0.id")],
                keywords=[],
            ),
        ],
        keywords=[keyword(arg="nullable", value=set_value(True), identifier=None)],
    ),
    type_comment=None,
    expr=None,
    lineno=None,
)

id_column = Assign(
    targets=[Name(id="id", ctx=Store())],
    value=Call(
        func=Name(id="Column", ctx=Load()),
        args=[Name(id="Integer", ctx=Load())],
        keywords=[
            keyword(arg="primary_key", value=set_value(True)),
            keyword(
                arg="server_default",
                value=Call(
                    func=Name(id="Identity", ctx=Load()),
                    args=[],
                    keywords=[],
                ),
            ),
        ],
    ),
    type_comment=None,
    expr=None,
    lineno=None,
)

__all__ = ["column_fk", "column_fk_gold", "id_column"]
