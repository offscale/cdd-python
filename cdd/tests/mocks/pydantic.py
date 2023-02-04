"""
Pydantic mocks
"""

from ast import AnnAssign, ClassDef, Index, Load, Name, Store, Subscript

from cdd.shared.ast_utils import maybe_type_comment, set_value

pydantic_class_str = """
class Cat(BaseModel):
    pet_type: Literal['cat']
    cat_name: str
"""

pydantic_class_cls_def = ClassDef(
    bases=[Name(ctx=Load(), id="BaseModel")],
    body=[
        AnnAssign(
            annotation=Subscript(
                ctx=Load(),
                slice=Index(value=set_value("cat")),
                value=Name(ctx=Load(), id="Literal"),
            ),
            simple=1,
            target=Name(ctx=Store(), id="pet_type"),
            value=None,
            expr=None,
            expr_target=None,
            expr_annotation=None,
            lineno=None,
            **maybe_type_comment
        ),
        AnnAssign(
            annotation=Name(ctx=Load(), id="str"),
            simple=1,
            target=Name(ctx=Store(), id="cat_name"),
            value=None,
            expr=None,
            expr_target=None,
            expr_annotation=None,
            lineno=None,
            **maybe_type_comment
        ),
    ],
    decorator_list=[],
    keywords=[],
    name="Cat",
)

__all__ = ["pydantic_class_str", "pydantic_class_cls_def"]
