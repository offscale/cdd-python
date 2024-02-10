"""
Mocks for SQLalchemy
"""

from ast import (
    Assign,
    Attribute,
    Call,
    ClassDef,
    Compare,
    DictComp,
    Expr,
    FunctionDef,
    IsNot,
    Load,
    Module,
    Name,
    Return,
    Store,
    Tuple,
    arguments,
    comprehension,
    keyword,
)
from copy import deepcopy
from functools import partial
from itertools import chain, islice
from textwrap import indent

from cdd.shared.ast_utils import maybe_type_comment, set_arg, set_value
from cdd.shared.pure_utils import reindent, tab
from cdd.tests.mocks.docstrings import (
    docstring_header_and_return_str,
    docstring_header_and_return_two_nl_str,
    docstring_repr_str,
)

_docstring_header_and_return_str: str = "\n{docstring}\n{tab}".format(
    docstring="\n".join(indent(docstring_header_and_return_str, tab).split("\n")),
    tab=tab,
)

sqlalchemy_imports_str: str = "\n".join(
    map(
        "def {}(*args, **kwargs): pass\n".format,
        (
            "Table",
            "Boolean",
            "JSON",
            "String",
            "Column",
            "Enum",
        ),
    )
)

config_tbl_with_comments_str: str = (
    """
config_tbl = Table(
    "config_tbl",
    metadata,
    Column(
        "dataset_name",
        String,
        comment="name of dataset",
        default="mnist",
        primary_key=True,
    ),
    Column(
        "tfds_dir",
        String,
        comment="directory to look for models in",
        default="~/tensorflow_datasets",
        nullable=False,
    ),
    Column(
        "K",
        Enum("np", "tf", name="K"),
        comment="backend engine, e.g., `np` or `tf`",
        default="np",
        nullable=False,
    ),
    Column(
        "as_numpy",
        Boolean,
        comment="Convert to numpy ndarrays",
        nullable=True,
    ),
    Column(
        "data_loader_kwargs",
        JSON,
        comment="pass this as arguments to data_loader function",
        nullable=True,
    ),
    comment={comment!r},
    keep_existing=True
)
""".format(
        comment=docstring_header_and_return_two_nl_str
    )
)

config_tbl_with_comments_ast: Assign = Assign(
    targets=[Name(ctx=Store(), id="config_tbl", lineno=None, col_offset=None)],
    value=Call(
        args=[
            set_value("config_tbl"),
            Name(ctx=Load(), id="metadata", lineno=None, col_offset=None),
            Call(
                args=[
                    set_value("dataset_name"),
                    Name(ctx=Load(), id="String", lineno=None, col_offset=None),
                ],
                func=Name(ctx=Load(), id="Column", lineno=None, col_offset=None),
                keywords=[
                    keyword(
                        arg="comment",
                        value=set_value("name of dataset"),
                        identifier=None,
                    ),
                    keyword(arg="default", value=set_value("mnist"), identifier=None),
                    keyword(arg="primary_key", value=set_value(True), identifier=None),
                ],
                lineno=None,
                col_offset=None,
            ),
            Call(
                args=[
                    set_value("tfds_dir"),
                    Name(ctx=Load(), id="String", lineno=None, col_offset=None),
                ],
                func=Name(ctx=Load(), id="Column", lineno=None, col_offset=None),
                keywords=[
                    keyword(
                        arg="comment",
                        value=set_value("directory to look for models in"),
                        identifier=None,
                    ),
                    keyword(
                        arg="default",
                        value=set_value("~/tensorflow_datasets"),
                        identifier=None,
                    ),
                    keyword(arg="nullable", value=set_value(False), identifier=None),
                ],
                lineno=None,
                col_offset=None,
            ),
            Call(
                args=[
                    set_value("K"),
                    Call(
                        args=[set_value("np"), set_value("tf")],
                        func=Name(ctx=Load(), id="Enum", lineno=None, col_offset=None),
                        keywords=[
                            keyword(arg="name", value=set_value("K"), identifier=None)
                        ],
                        lineno=None,
                        col_offset=None,
                    ),
                ],
                func=Name(ctx=Load(), id="Column", lineno=None, col_offset=None),
                keywords=[
                    keyword(
                        arg="comment",
                        value=set_value("backend engine, e.g., `np` or `tf`"),
                        identifier=None,
                    ),
                    keyword(arg="default", value=set_value("np"), identifier=None),
                    keyword(arg="nullable", value=set_value(False), identifier=None),
                ],
                lineno=None,
                col_offset=None,
            ),
            Call(
                args=[
                    set_value("as_numpy"),
                    Name(ctx=Load(), id="Boolean", lineno=None, col_offset=None),
                ],
                func=Name(ctx=Load(), id="Column", lineno=None, col_offset=None),
                keywords=[
                    keyword(
                        arg="comment",
                        value=set_value("Convert to numpy ndarrays"),
                        identifier=None,
                    ),
                    keyword(arg="nullable", value=set_value(True), identifier=None),
                ],
                lineno=None,
                col_offset=None,
            ),
            Call(
                args=[
                    set_value("data_loader_kwargs"),
                    Name(ctx=Load(), id="JSON", lineno=None, col_offset=None),
                ],
                func=Name(ctx=Load(), id="Column", lineno=None, col_offset=None),
                keywords=[
                    keyword(
                        arg="comment",
                        value=set_value(
                            "pass this as arguments to data_loader function"
                        ),
                        identifier=None,
                    ),
                    keyword(arg="nullable", value=set_value(True)),
                ],
                lineno=None,
                col_offset=None,
            ),
        ],
        func=Name(ctx=Load(), id="Table", lineno=None, col_offset=None),
        keywords=[
            keyword(
                arg="comment",
                value=set_value(docstring_header_and_return_two_nl_str),
                identifier=None,
            ),
            keyword(
                arg="keep_existing",
                value=set_value(True),
                identifier=None,
                expr=None,
                lineno=None,
                **maybe_type_comment,
            ),
        ],
        lineno=None,
        col_offset=None,
    ),
    expr=None,
    lineno=None,
    **maybe_type_comment,
)

config_decl_base_str: str = (
    '''
class Config(Base):
    """{_docstring_header_and_return_str}"""
    __tablename__ = "config_tbl"

    dataset_name = Column(
        String,
        comment="name of dataset",
        default="mnist",
        primary_key=True,
    )

    tfds_dir = Column(
        String,
        comment="directory to look for models in",
        default="~/tensorflow_datasets",
        nullable=False,
    )

    K = Column(
        Enum("np", "tf", name="K"),
        comment="backend engine, e.g., `np` or `tf`",
        default="np",
        nullable=False,
    )

    as_numpy = Column(
        Boolean,
        comment="Convert to numpy ndarrays",
        nullable=True,
    )

    data_loader_kwargs = Column(
        JSON,
        comment="pass this as arguments to data_loader function",
        nullable=True,
    )

    def __repr__(self):
    {tab}"""\n{tab}{tab}{_repr_docstring}"""
    {__repr___body}
    '''.format(
        _docstring_header_and_return_str=reindent(_docstring_header_and_return_str),
        tab=tab,
        _repr_docstring=docstring_repr_str.lstrip(),
        __repr___body="""
        return "Config(dataset_name={dataset_name!r}, tfds_dir={tfds_dir!r}, " \
               "K={K!r}, as_numpy={as_numpy!r}, data_loader_kwargs={data_loader_kwargs!r})".format(
            dataset_name=self.dataset_name, tfds_dir=self.tfds_dir, K=self.K,
            as_numpy=self.as_numpy, data_loader_kwargs=self.data_loader_kwargs
        )
""",
    )
)

dataset_primary_key_column_assign: Assign = Assign(
    targets=[Name("dataset_name", Store(), lineno=None, col_offset=None)],
    value=Call(
        func=Name("Column", Load(), lineno=None, col_offset=None),
        args=[Name("String", Load(), lineno=None, col_offset=None)],
        keywords=[
            keyword(
                arg="comment",
                value=set_value("name of dataset"),
                identifier=None,
            ),
            keyword(arg="default", value=set_value("mnist"), identifier=None),
            keyword(arg="primary_key", value=set_value(True), identifier=None),
        ],
        expr=None,
        expr_func=None,
        lineno=None,
        col_offset=None,
    ),
    expr=None,
    lineno=None,
    **maybe_type_comment,
)

config_decl_base_ast: ClassDef = ClassDef(
    name="Config",
    bases=[Name("Base", Load(), lineno=None, col_offset=None)],
    keywords=[],
    body=[
        Expr(
            set_value(reindent(_docstring_header_and_return_str)),
            lineno=None,
            col_offset=None,
        ),
        Assign(
            targets=[Name("__tablename__", Store(), lineno=None, col_offset=None)],
            value=set_value("config_tbl"),
            expr=None,
            lineno=None,
            **maybe_type_comment,
        ),
        dataset_primary_key_column_assign,
        Assign(
            targets=[Name("tfds_dir", Store(), lineno=None, col_offset=None)],
            value=Call(
                func=Name("Column", Load(), lineno=None, col_offset=None),
                args=[Name("String", Load(), lineno=None, col_offset=None)],
                keywords=[
                    keyword(
                        arg="comment",
                        value=set_value("directory to look for models in"),
                        identifier=None,
                    ),
                    keyword(
                        arg="default",
                        value=set_value("~/tensorflow_datasets"),
                        identifier=None,
                    ),
                    keyword(arg="nullable", value=set_value(False), identifier=None),
                ],
                expr=None,
                expr_func=None,
                lineno=None,
                col_offset=None,
            ),
            expr=None,
            lineno=None,
            **maybe_type_comment,
        ),
        Assign(
            targets=[Name("K", Store(), lineno=None, col_offset=None)],
            value=Call(
                func=Name("Column", Load(), lineno=None, col_offset=None),
                args=[
                    Call(
                        func=Name("Enum", Load(), lineno=None, col_offset=None),
                        args=[set_value("np"), set_value("tf")],
                        keywords=[
                            keyword(arg="name", value=set_value("K"), identifier=None)
                        ],
                        expr=None,
                        expr_func=None,
                        lineno=None,
                        col_offset=None,
                    )
                ],
                keywords=[
                    keyword(
                        arg="comment",
                        value=set_value("backend engine, e.g., `np` or `tf`"),
                        identifier=None,
                    ),
                    keyword(arg="default", value=set_value("np"), identifier=None),
                    keyword(arg="nullable", value=set_value(False), identifier=None),
                ],
                expr=None,
                expr_func=None,
                lineno=None,
                col_offset=None,
            ),
            expr=None,
            lineno=None,
            col_offset=None,
            **maybe_type_comment,
        ),
        Assign(
            targets=[Name("as_numpy", Store(), lineno=None, col_offset=None)],
            value=Call(
                func=Name("Column", Load(), lineno=None, col_offset=None),
                args=[Name("Boolean", Load(), lineno=None, col_offset=None)],
                keywords=[
                    keyword(
                        arg="comment",
                        value=set_value("Convert to numpy ndarrays"),
                        identifier=None,
                    ),
                    keyword(arg="nullable", value=set_value(True), identifier=None),
                ],
                expr=None,
                expr_func=None,
                lineno=None,
                col_offset=None,
            ),
            expr=None,
            lineno=None,
            **maybe_type_comment,
        ),
        Assign(
            targets=[Name("data_loader_kwargs", Store(), lineno=None, col_offset=None)],
            value=Call(
                func=Name("Column", Load(), lineno=None, col_offset=None),
                args=[Name("JSON", Load(), lineno=None, col_offset=None)],
                keywords=[
                    keyword(
                        arg="comment",
                        value=set_value(
                            "pass this as arguments to data_loader function"
                        ),
                        identifier=None,
                    ),
                    keyword(arg="nullable", value=set_value(True), identifier=None),
                ],
                expr=None,
                expr_func=None,
                lineno=None,
                col_offset=None,
            ),
            expr=None,
            lineno=None,
            **maybe_type_comment,
        ),
        FunctionDef(
            name="__repr__",
            args=arguments(
                posonlyargs=[],
                args=[set_arg("self")],
                kwonlyargs=[],
                kw_defaults=[],
                defaults=[],
                vararg=None,
                kwarg=None,
            ),
            body=[
                Expr(
                    set_value(docstring_repr_str.lstrip(" ")),
                    lineno=None,
                    col_offset=None,
                ),
                Return(
                    value=Call(
                        func=Attribute(
                            set_value(
                                "Config(dataset_name={dataset_name!r}, tfds_dir={tfds_dir!r}, K={K!r}, "
                                "as_numpy={as_numpy!r}, data_loader_kwargs={data_loader_kwargs!r})"
                            ),
                            "format",
                            Load(),
                            lineno=None,
                            col_offset=None,
                        ),
                        args=[],
                        keywords=[
                            keyword(
                                arg="dataset_name",
                                value=Attribute(
                                    Name("self", Load(), lineno=None, col_offset=None),
                                    "dataset_name",
                                    Load(),
                                    lineno=None,
                                    col_offset=None,
                                ),
                                identifier=None,
                            ),
                            keyword(
                                arg="tfds_dir",
                                value=Attribute(
                                    Name("self", Load(), lineno=None, col_offset=None),
                                    "tfds_dir",
                                    Load(),
                                    lineno=None,
                                    col_offset=None,
                                ),
                                identifier=None,
                            ),
                            keyword(
                                arg="K",
                                value=Attribute(
                                    Name("self", Load(), lineno=None, col_offset=None),
                                    "K",
                                    Load(),
                                    lineno=None,
                                    col_offset=None,
                                ),
                                identifier=None,
                            ),
                            keyword(
                                arg="as_numpy",
                                value=Attribute(
                                    Name("self", Load(), lineno=None, col_offset=None),
                                    "as_numpy",
                                    Load(),
                                    lineno=None,
                                    col_offset=None,
                                ),
                                identifier=None,
                            ),
                            keyword(
                                arg="data_loader_kwargs",
                                value=Attribute(
                                    Name("self", Load(), lineno=None, col_offset=None),
                                    "data_loader_kwargs",
                                    Load(),
                                    lineno=None,
                                    col_offset=None,
                                ),
                                identifier=None,
                            ),
                        ],
                        expr=None,
                        expr_func=None,
                        lineno=None,
                        col_offset=None,
                    ),
                    expr=None,
                ),
            ],
            decorator_list=[],
            type_params=[],
            arguments_args=None,
            identifier_name=None,
            stmt=None,
            lineno=None,
            returns=None,
            **maybe_type_comment,
        ),
    ],
    decorator_list=[],
    type_params=[],
    expr=None,
    lineno=None,
    col_offset=None,
    end_lineno=None,
    end_col_offset=None,
    identifier_name=None,
)

config_hybrid_ast = deepcopy(config_decl_base_ast)
config_hybrid_ast.body = list(
    chain.from_iterable(
        (islice(config_hybrid_ast.body, 2), (deepcopy(config_tbl_with_comments_ast),))
    )
)
config_hybrid_ast.body[-1].targets[0].id = "__table__"

empty_with_inferred_pk_column_assign: Assign = Assign(
    targets=[
        Name(id="empty_with_inferred_pk_tbl", ctx=Store(), lineno=None, col_offset=None)
    ],
    value=Call(
        func=Name(id="Table", ctx=Load(), lineno=None, col_offset=None),
        args=[
            set_value("empty_with_inferred_pk_tbl"),
            Name(id="metadata", ctx=Load(), lineno=None, col_offset=None),
            Call(
                func=Name(id="Column", ctx=Load(), lineno=None, col_offset=None),
                args=[
                    set_value("id"),
                    Name(id="Integer", ctx=Load(), lineno=None, col_offset=None),
                ],
                keywords=[
                    keyword(arg="primary_key", value=set_value(True)),
                    keyword(
                        arg="server_default",
                        value=Call(
                            func=Name(
                                id="Identity", ctx=Load(), lineno=None, col_offset=None
                            ),
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
        ],
        keywords=[
            keyword(
                arg="keep_existing",
                value=set_value(True),
                identifier=None,
                expr=None,
                lineno=None,
                **maybe_type_comment,
            )
        ],
        lineno=None,
        col_offset=None,
    ),
    expr=None,
    lineno=None,
    **maybe_type_comment,
)

foreign_sqlalchemy_tbls_str: str = """node = Table(
    "node",
    metadata_obj,
    Column("node_id", Integer, primary_key=True),
    Column("primary_element", Integer, ForeignKey("element.element_id")),
    keep_existing=True
)

element = Table(
    "element",
    metadata_obj,
    Column("element_id", Integer, primary_key=True),
    Column("parent_node_id", Integer),
    keep_existing=True
)"""

node_fk_call: Call = Call(
    func=Name(id="Column", ctx=Load(), lineno=None, col_offset=None),
    args=[
        set_value("primary_element"),
        Name(id="Integer", ctx=Load(), lineno=None, col_offset=None),
        Call(
            func=Name(id="ForeignKey", ctx=Load(), lineno=None, col_offset=None),
            args=[set_value("element.element_id")],
            keywords=[],
            lineno=None,
            col_offset=None,
        ),
    ],
    keywords=[],
    lineno=None,
    col_offset=None,
)

node_pk_tbl_call: Call = Call(
    func=Name(id="Table", ctx=Load(), lineno=None, col_offset=None),
    args=[
        set_value("node"),
        Name(id="metadata_obj", ctx=Load(), lineno=None, col_offset=None),
        Call(
            func=Name(id="Column", ctx=Load(), lineno=None, col_offset=None),
            args=[
                set_value("node_id"),
                Name(id="Integer", ctx=Load(), lineno=None, col_offset=None),
            ],
            keywords=[keyword(arg="primary_key", value=set_value(True))],
        ),
        node_fk_call,
    ],
    keywords=[
        keyword(
            arg="keep_existing",
            value=set_value(True),
            identifier=None,
            expr=None,
            lineno=None,
            **maybe_type_comment,
        )
    ],
)

node_pk_tbl_ass: Assign = Assign(
    targets=[Name(id="node", ctx=Store(), lineno=None, col_offset=None)],
    value=node_pk_tbl_call,
    expr=None,
    lineno=None,
    **maybe_type_comment,
)

node_pk_tbl_class: ClassDef = ClassDef(
    name="node",
    bases=[Name(id="Base", ctx=Load(), lineno=None, col_offset=None)],
    keywords=[],
    body=[
        Assign(
            targets=[
                Name(id="__tablename__", ctx=Store(), lineno=None, col_offset=None)
            ],
            value=set_value("node"),
            lineno=None,
            **maybe_type_comment,
        ),
        Assign(
            targets=[Name(id="node_id", ctx=Store(), lineno=None, col_offset=None)],
            value=Call(
                func=Name(id="Column", ctx=Load(), lineno=None, col_offset=None),
                args=[Name(id="Integer", ctx=Load(), lineno=None, col_offset=None)],
                keywords=[keyword(arg="primary_key", value=set_value(True))],
            ),
            lineno=None,
            **maybe_type_comment,
        ),
        Assign(
            targets=[
                Name(id="primary_element", ctx=Store(), lineno=None, col_offset=None)
            ],
            value=Call(
                func=Name(id="Column", ctx=Load(), lineno=None, col_offset=None),
                args=[
                    Name(id="Integer", ctx=Load(), lineno=None, col_offset=None),
                    Call(
                        func=Name(
                            id="ForeignKey", ctx=Load(), lineno=None, col_offset=None
                        ),
                        args=[set_value("element.element_id")],
                        keywords=[],
                    ),
                ],
                keywords=[],
            ),
            lineno=None,
            **maybe_type_comment,
        ),
    ],
    decorator_list=[],
    type_params=[],
    lineno=None,
    col_offset=None,
)

element_pk_fk_tbl: Call = Call(
    func=Name(id="Table", ctx=Load(), lineno=None, col_offset=None),
    args=[
        set_value("element"),
        Name(id="metadata_obj", ctx=Load(), lineno=None, col_offset=None),
        Call(
            func=Name(id="Column", ctx=Load(), lineno=None, col_offset=None),
            args=[
                set_value("element_id"),
                Name(id="Integer", ctx=Load(), lineno=None, col_offset=None),
            ],
            keywords=[keyword(arg="primary_key", value=set_value(True))],
        ),
        Call(
            func=Name(id="Column", ctx=Load(), lineno=None, col_offset=None),
            args=[
                set_value("parent_node_id"),
                Name(id="Integer", ctx=Load(), lineno=None, col_offset=None),
            ],
            keywords=[],
        ),
    ],
    keywords=[
        keyword(
            arg="keep_existing",
            value=set_value(True),
            identifier=None,
            expr=None,
            lineno=None,
            **maybe_type_comment,
        )
    ],
)

element_pk_fk_ass: Assign = Assign(
    targets=[Name(id="element", ctx=Store(), lineno=None, col_offset=None)],
    value=element_pk_fk_tbl,
    expr=None,
    lineno=None,
    **maybe_type_comment,
)

foreign_sqlalchemy_tbls_mod: Module = Module(
    body=[node_pk_tbl_ass, element_pk_fk_ass],
    type_ignores=[],
)

create_from_attr_mock: FunctionDef = FunctionDef(
    name="create_from_attr",
    args=arguments(
        posonlyargs=[],
        args=[set_arg("record")],
        kwonlyargs=[],
        kw_defaults=[],
        defaults=[],
        vararg=None,
        kwarg=None,
    ),
    body=[
        Expr(
            value=set_value(
                "\n".join(
                    map(
                        partial(indent, prefix=tab * 2),
                        (
                            "",
                            "Construct an instance from an object with identical columns (as attributes)"
                            " as this `class`/`Table`",
                            tab * 2,
                            ":return: A new instance made from the input object's attributes",
                            ":rtype: ```foo```",
                            tab * 2,
                        ),
                    )
                )
            ),
            lineno=None,
            col_offset=None,
        ),
        Return(
            value=Call(
                func=Name(id="foo", ctx=Load(), lineno=None, col_offset=None),
                args=[],
                keywords=[
                    keyword(
                        arg=None,
                        value=DictComp(
                            key=Name(
                                id="attr", ctx=Load(), lineno=None, col_offset=None
                            ),
                            value=Call(
                                func=Name(
                                    id="getattr",
                                    ctx=Load(),
                                    lineno=None,
                                    col_offset=None,
                                ),
                                args=[
                                    Name(
                                        id="record",
                                        ctx=Load(),
                                        lineno=None,
                                        col_offset=None,
                                    ),
                                    Name(
                                        id="attr",
                                        ctx=Load(),
                                        lineno=None,
                                        col_offset=None,
                                    ),
                                ],
                                keywords=[],
                            ),
                            generators=[
                                comprehension(
                                    target=Name(
                                        id="attr",
                                        ctx=Store(),
                                        lineno=None,
                                        col_offset=None,
                                    ),
                                    iter=Tuple(
                                        elts=[
                                            set_value("id"),
                                            set_value("not_pk_id"),
                                        ],
                                        ctx=Load(),
                                        lineno=None,
                                        col_offset=None,
                                    ),
                                    ifs=[
                                        Compare(
                                            left=Call(
                                                func=Name(
                                                    id="getattr",
                                                    ctx=Load(),
                                                    lineno=None,
                                                    col_offset=None,
                                                ),
                                                args=[
                                                    Name(
                                                        id="record",
                                                        ctx=Load(),
                                                    ),
                                                    Name(
                                                        id="attr",
                                                        ctx=Load(),
                                                    ),
                                                    set_value(None),
                                                ],
                                                keywords=[],
                                            ),
                                            ops=[IsNot()],
                                            comparators=[set_value(None)],
                                            lineno=None,
                                            col_offset=None,
                                        )
                                    ],
                                    is_async=0,
                                )
                            ],
                            lineno=None,
                            col_offset=None,
                        ),
                    )
                ],
            )
        ),
    ],
    lineno=None,
    returns=None,
    type_comment=None,
    decorator_list=[Name(id="staticmethod", ctx=Load(), lineno=None, col_offset=None)],
)

__all__ = [
    "config_decl_base_ast",
    "config_decl_base_str",
    "config_hybrid_ast",
    "config_tbl_with_comments_ast",
    "config_tbl_with_comments_ast",
    "config_tbl_with_comments_str",
    "create_from_attr_mock",
    "dataset_primary_key_column_assign",
    "element_pk_fk_ass",
    "empty_with_inferred_pk_column_assign",
    "foreign_sqlalchemy_tbls_mod",
    "foreign_sqlalchemy_tbls_str",
    "node_fk_call",
    "node_pk_tbl_ass",
    "node_pk_tbl_call",
    "node_pk_tbl_class",
    "sqlalchemy_imports_str",
]  # type: list[str]
