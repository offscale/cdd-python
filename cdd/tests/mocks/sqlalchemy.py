"""
Mocks for SQLalchemy
"""
from ast import (
    Assign,
    Attribute,
    Call,
    ClassDef,
    Expr,
    FunctionDef,
    Load,
    Name,
    Return,
    Store,
    arg,
    arguments,
    keyword,
)
from textwrap import indent

from cdd.ast_utils import maybe_type_comment, set_value
from cdd.pure_utils import reindent, tab
from cdd.tests.mocks.docstrings import (
    docstring_header_and_return_str,
    docstring_repr_str,
)

_docstring_header_and_return_str = "\n{docstring}\n{tab}".format(
    docstring="\n".join(indent(docstring_header_and_return_str, tab).split("\n")),
    tab=tab,
)

sqlalchemy_imports_str = "\n".join(
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

config_tbl_str = """
config_tbl = Table(
    "config_tbl",
    metadata,
    Column(
        "dataset_name",
        String,
        doc="name of dataset",
        default="mnist",
        primary_key=True,
    ),
    Column(
        "tfds_dir",
        String,
        doc="directory to look for models in",
        default="~/tensorflow_datasets",
        nullable=False,
    ),
    Column(
        "K",
        Enum("np", "tf", name="K"),
        doc="backend engine, e.g., `np` or `tf`",
        default="np",
        nullable=False,
    ),
    Column(
        "as_numpy",
        Boolean,
        doc="Convert to numpy ndarrays",
        default=None,
        nullable=True,
    ),
    Column(
        "data_loader_kwargs",
        JSON,
        doc="pass this as arguments to data_loader function",
        default=None,
        nullable=True,
    ),
    comment={comment!r},
)
""".format(
    comment=docstring_header_and_return_str
)

config_tbl_ast = Assign(
    targets=[Name("config_tbl", Store())],
    value=Call(
        func=Name("Table", Load()),
        args=[
            set_value("config_tbl"),
            Name("metadata", Load()),
            Call(
                func=Name("Column", Load()),
                args=[set_value("dataset_name"), Name("String", Load())],
                keywords=[
                    keyword(
                        arg="doc", value=set_value("name of dataset"), identifier=None
                    ),
                    keyword(arg="default", value=set_value("mnist"), identifier=None),
                    keyword(arg="primary_key", value=set_value(True), identifier=None),
                ],
                expr=None,
                expr_func=None,
            ),
            Call(
                func=Name("Column", Load()),
                args=[set_value("tfds_dir"), Name("String", Load())],
                keywords=[
                    keyword(
                        arg="doc",
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
            ),
            Call(
                func=Name("Column", Load()),
                args=[
                    set_value("K"),
                    Call(
                        func=Name("Enum", Load()),
                        args=list(map(set_value, ("np", "tf"))),
                        keywords=[
                            keyword(arg="name", value=set_value("K"), identifier=None)
                        ],
                        expr=None,
                        expr_func=None,
                    ),
                ],
                keywords=[
                    keyword(
                        arg="doc",
                        value=set_value("backend engine, e.g., `np` or `tf`"),
                        identifier=None,
                    ),
                    keyword(arg="default", value=set_value("np"), identifier=None),
                    keyword(arg="nullable", value=set_value(False), identifier=None),
                ],
                expr=None,
                expr_func=None,
            ),
            Call(
                func=Name("Column", Load()),
                args=[set_value("as_numpy"), Name("Boolean", Load())],
                keywords=[
                    keyword(
                        arg="doc",
                        value=set_value("Convert to numpy ndarrays"),
                        identifier=None,
                    ),
                    keyword(arg="default", value=set_value(None), identifier=None),
                    keyword(arg="nullable", value=set_value(True), identifier=None),
                ],
                expr=None,
                expr_func=None,
            ),
            Call(
                func=Name("Column", Load()),
                args=[set_value("data_loader_kwargs"), Name("JSON", Load())],
                keywords=[
                    keyword(
                        arg="doc",
                        value=set_value(
                            "pass this as arguments to data_loader function"
                        ),
                        identifier=None,
                    ),
                    keyword(arg="default", value=set_value(None), identifier=None),
                    keyword(arg="nullable", value=set_value(True), identifier=None),
                ],
                expr=None,
                expr_func=None,
            ),
        ],
        keywords=[
            keyword(
                arg="comment",
                value=set_value(docstring_header_and_return_str),
                identifier=None,
            )
        ],
        expr=None,
        expr_func=None,
    ),
    lineno=None,
    expr=None,
    **maybe_type_comment
)

config_decl_base_str = '''
class Config(Base):
    """{_docstring_header_and_return_str}"""
    __tablename__ = "config_tbl"

    dataset_name = Column(
        String,
        doc="name of dataset",
        default="mnist",
        primary_key=True,
    )

    tfds_dir = Column(
        String,
        doc="directory to look for models in",
        default="~/tensorflow_datasets",
        nullable=False,
    )

    K = Column(
        Enum("np", "tf", name="K"),
        doc="backend engine, e.g., `np` or `tf`",
        default="np",
        nullable=False,
    )

    as_numpy = Column(
        Boolean,
        doc="Convert to numpy ndarrays",
        default=None,
        nullable=True,
    )

    data_loader_kwargs = Column(
        JSON,
        doc="pass this as arguments to data_loader function",
        default=None,
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

config_decl_base_ast = ClassDef(
    name="Config",
    bases=[Name("Base", Load())],
    keywords=[],
    body=[
        Expr(set_value(reindent(_docstring_header_and_return_str))),
        Assign(
            targets=[Name("__tablename__", Store())],
            value=set_value("config_tbl"),
            expr=None,
            lineno=None,
            **maybe_type_comment
        ),
        Assign(
            targets=[Name("dataset_name", Store())],
            value=Call(
                func=Name("Column", Load()),
                args=[Name("String", Load())],
                keywords=[
                    keyword(
                        arg="doc", value=set_value("name of dataset"), identifier=None
                    ),
                    keyword(arg="default", value=set_value("mnist"), identifier=None),
                    keyword(arg="primary_key", value=set_value(True), identifier=None),
                ],
                expr=None,
                expr_func=None,
            ),
            expr=None,
            lineno=None,
            **maybe_type_comment
        ),
        Assign(
            targets=[Name("tfds_dir", Store())],
            value=Call(
                func=Name("Column", Load()),
                args=[Name("String", Load())],
                keywords=[
                    keyword(
                        arg="doc",
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
            ),
            expr=None,
            lineno=None,
            **maybe_type_comment
        ),
        Assign(
            targets=[Name("K", Store())],
            value=Call(
                func=Name("Column", Load()),
                args=[
                    Call(
                        func=Name("Enum", Load()),
                        args=[set_value("np"), set_value("tf")],
                        keywords=[
                            keyword(arg="name", value=set_value("K"), identifier=None)
                        ],
                        expr=None,
                        expr_func=None,
                    )
                ],
                keywords=[
                    keyword(
                        arg="doc",
                        value=set_value("backend engine, e.g., `np` or `tf`"),
                        identifier=None,
                    ),
                    keyword(arg="default", value=set_value("np"), identifier=None),
                    keyword(arg="nullable", value=set_value(False), identifier=None),
                ],
                expr=None,
                expr_func=None,
            ),
            expr=None,
            lineno=None,
            **maybe_type_comment
        ),
        Assign(
            targets=[Name("as_numpy", Store())],
            value=Call(
                func=Name("Column", Load()),
                args=[Name("Boolean", Load())],
                keywords=[
                    keyword(
                        arg="doc",
                        value=set_value("Convert to numpy ndarrays"),
                        identifier=None,
                    ),
                    keyword(arg="default", value=set_value(None), identifier=None),
                    keyword(arg="nullable", value=set_value(True), identifier=None),
                ],
                expr=None,
                expr_func=None,
            ),
            expr=None,
            lineno=None,
            **maybe_type_comment
        ),
        Assign(
            targets=[Name("data_loader_kwargs", Store())],
            value=Call(
                func=Name("Column", Load()),
                args=[Name("JSON", Load())],
                keywords=[
                    keyword(
                        arg="doc",
                        value=set_value(
                            "pass this as arguments to data_loader function"
                        ),
                        identifier=None,
                    ),
                    keyword(arg="default", value=set_value(None), identifier=None),
                    keyword(arg="nullable", value=set_value(True), identifier=None),
                ],
                expr=None,
                expr_func=None,
            ),
            expr=None,
            lineno=None,
            **maybe_type_comment
        ),
        FunctionDef(
            name="__repr__",
            args=arguments(
                posonlyargs=[],
                arg=None,
                args=[
                    arg(
                        arg="self",
                        annotation=None,
                        expr=None,
                        identifier_arg=None,
                        **maybe_type_comment
                    )
                ],
                kwonlyargs=[],
                kw_defaults=[],
                defaults=[],
                vararg=None,
                kwarg=None,
            ),
            body=[
                Expr(set_value(docstring_repr_str.lstrip(" "))),
                Return(
                    value=Call(
                        func=Attribute(
                            set_value(
                                "Config(dataset_name={dataset_name!r}, tfds_dir={tfds_dir!r}, K={K!r}, "
                                "as_numpy={as_numpy!r}, data_loader_kwargs={data_loader_kwargs!r})"
                            ),
                            "format",
                            Load(),
                        ),
                        args=[],
                        keywords=[
                            keyword(
                                arg="dataset_name",
                                value=Attribute(
                                    Name("self", Load()), "dataset_name", Load()
                                ),
                                identifier=None,
                            ),
                            keyword(
                                arg="tfds_dir",
                                value=Attribute(
                                    Name("self", Load()), "tfds_dir", Load()
                                ),
                                identifier=None,
                            ),
                            keyword(
                                arg="K",
                                value=Attribute(Name("self", Load()), "K", Load()),
                                identifier=None,
                            ),
                            keyword(
                                arg="as_numpy",
                                value=Attribute(
                                    Name("self", Load()), "as_numpy", Load()
                                ),
                                identifier=None,
                            ),
                            keyword(
                                arg="data_loader_kwargs",
                                value=Attribute(
                                    Name("self", Load()), "data_loader_kwargs", Load()
                                ),
                                identifier=None,
                            ),
                        ],
                        expr=None,
                        expr_func=None,
                    ),
                    expr=None,
                ),
            ],
            decorator_list=[],
            arguments_args=None,
            identifier_name=None,
            stmt=None,
            lineno=None,
            returns=None,
            **maybe_type_comment
        ),
    ],
    decorator_list=[],
    expr=None,
    lineno=None,
    col_offset=None,
    end_lineno=None,
    end_col_offset=None,
    identifier_name=None,
)

__all__ = [
    "config_tbl_ast",
    "config_tbl_str",
    "config_decl_base_ast",
    "config_decl_base_str",
]
