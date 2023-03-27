"""
Gen mocks
"""

from ast import Import, ImportFrom, alias

from cdd.shared.source_transformer import to_code

import_star_from_input_ast = ImportFrom(
    module="input",
    names=[
        alias(
            name="input_map",
            asname=None,
            identifier=None,
            identifier_name=None,
        ),
        alias(
            name="Foo",
            asname=None,
            identifier=None,
            identifier_name=None,
        ),
    ],
    level=1,
    identifier=None,
)
import_star_from_input_str = to_code(import_star_from_input_ast)
import_gen_test_module_ast = Import(
    names=[
        alias(
            name="gen_test_module",
            asname=None,
            identifier=None,
            identifier_name=None,
        )
    ],
    alias=None,
)
import_gen_test_module_str = "{}\n".format(
    to_code(import_gen_test_module_ast).rstrip("\n")
)

__all__ = [
    "import_gen_test_module_ast",
    "import_gen_test_module_str",
    "import_star_from_input_ast",
    "import_star_from_input_str",
]
