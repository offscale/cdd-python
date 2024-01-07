"""
Pydantic mocks
"""

from ast import AnnAssign, ClassDef, Index, Load, Name, Store, Subscript

from cdd.shared.ast_utils import maybe_type_comment, set_value

pydantic_class_str: str = """
class Cat(BaseModel):
    pet_type: Literal['cat']
    cat_name: str
"""

pydantic_class_cls_def: ClassDef = ClassDef(
    bases=[Name(ctx=Load(), id="BaseModel", lineno=None, col_offset=None)],
    body=[
        AnnAssign(
            annotation=Subscript(
                ctx=Load(),
                slice=Index(value=set_value("cat")),
                value=Name(ctx=Load(), id="Literal", lineno=None, col_offset=None),
                lineno=None,
                col_offset=None,
            ),
            simple=1,
            target=Name(ctx=Store(), id="pet_type", lineno=None, col_offset=None),
            value=None,
            expr=None,
            expr_target=None,
            expr_annotation=None,
            lineno=None,
            col_offset=None,
            **maybe_type_comment
        ),
        AnnAssign(
            annotation=Name("str", Load(), lineno=None, col_offset=None),
            simple=1,
            target=Name(ctx=Store(), id="cat_name", lineno=None, col_offset=None),
            value=None,
            expr=None,
            expr_target=None,
            expr_annotation=None,
            lineno=None,
            col_offset=None,
            **maybe_type_comment
        ),
    ],
    decorator_list=[],
    type_params=[],
    keywords=[],
    name="Cat",
    lineno=None,
    col_offset=None,
)

__all__ = ["pydantic_class_str", "pydantic_class_cls_def"]  # type: list[str]
