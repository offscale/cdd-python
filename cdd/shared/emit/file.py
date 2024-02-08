"""
File emitter
"""

from ast import Module
from importlib import import_module
from importlib.util import find_spec

import cdd.shared.source_transformer

black = (
    import_module("black")
    if find_spec("black") is not None
    else type(
        "black",
        tuple(),
        {
            "format_str": lambda src_contents, mode: src_contents,
            "Mode": (
                lambda target_versions, line_length, is_pyi, string_normalization: None
            ),
        },
    )
)


def file(node, filename, mode="a", skip_black=False):
    """
    Convert AST to a file

    :param node: AST node
    :type node: ```Union[Module, ClassDef, FunctionDef]```

    :param filename: emit to this file
    :type filename: ```str```

    :param mode: Mode to open the file in, defaults to append
    :type mode: ```str```

    :param skip_black: Whether to skip formatting with black
    :type skip_black: ```bool```

    :return: None
    :rtype: ```NoneType```
    """
    if not isinstance(node, Module):
        node: Module = Module(body=[node], type_ignores=[], stmt=None)
    src: str = cdd.shared.source_transformer.to_code(node)
    if not skip_black:
        src = black.format_str(
            src,
            mode=black.Mode(
                target_versions=set(),
                line_length=119,
                is_pyi=False,
                string_normalization=False,
            ),
        )
    with open(filename, mode) as f:
        f.write(src)


__all__ = ["file"]  # type: list[str]
