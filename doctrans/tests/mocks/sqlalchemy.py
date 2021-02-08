"""
Mocks for SQLalchemy
"""
from ast import Assign, Call, Load, Name, Store, keyword

from doctrans.ast_utils import maybe_type_comment, set_value
from doctrans.tests.mocks.docstrings import docstring_header_and_return_str

config_tbl_str = """
config_tbl = Table(
    "config_tbl",
    metadata,
    Column(
        "dataset_name", String, doc="name of dataset", default="mnist", primary_key=True
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
    Column("as_numpy", Boolean, doc="Convert to numpy ndarrays"),
    Column(
        "data_loader_kwargs", JSON, doc="pass this as arguments to data_loader function"
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
                        keywords=[keyword(arg="name", value=set_value("K"))],
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
                    )
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
                    )
                ],
                expr=None,
                expr_func=None,
            ),
        ],
        keywords=[
            keyword(arg="comment", value=set_value(docstring_header_and_return_str))
        ],
        expr=None,
        expr_func=None,
    ),
    lineno=None,
    expr=None,
    **maybe_type_comment
)

__all__ = ["config_tbl_ast", "config_tbl_str"]
