"""
Shared utility functions used by many tests
"""

import ast
from ast import Expr
from copy import deepcopy
from functools import partial
from importlib import import_module
from importlib.abc import Loader
from importlib.util import find_spec, module_from_spec, spec_from_loader
from itertools import takewhile
from operator import add
from os import path
from os.path import extsep
from sys import modules
from tempfile import NamedTemporaryFile
from typing import Optional
from unittest import main
from unittest.mock import MagicMock, patch

from cdd.shared import source_transformer
from cdd.shared.ast_utils import cmp_ast, get_value, set_value
from cdd.shared.docstring_utils import TOKENS
from cdd.shared.pure_utils import PY3_8, count_iter_items, identity, reindent, tab

black = (
    import_module("black")
    if find_spec("black") is not None
    else type(
        "black",
        tuple(),
        {
            "format_str": lambda src_contents, mode: None,
            "Mode": (
                lambda target_versions, line_length, is_pyi, string_normalization: None
            ),
        },
    )
)


def run_ast_test(test_case_instance, gen_ast, gold, skip_black=False):
    """
    Compares `gen_ast` with `gold` standard. Uses only the first element if `Module`.

    :param test_case_instance: instance of `TestCase`
    :type test_case_instance: ```unittest.TestCase```

    :param gen_ast: generated AST
    :type gen_ast: ```Union[ast.Module, ast.ClassDef, ast.FunctionDef]```

    :param gold: mocked AST
    :type gold: ```Union[ast.Module, ast.ClassDef, ast.FunctionDef]```

    :param skip_black: Whether to skip formatting with black. Turned off for performance, turn on for pretty debug.
    :type skip_black: ```bool```
    """
    if isinstance(gen_ast, str):
        gen_ast = ast.parse(gen_ast).body[0]

    assert gen_ast is not None, "gen_ast is None"
    assert gold is not None, "gold is None"

    gen_ast = deepcopy(gen_ast)
    gold = deepcopy(gold)

    # if reindent_docstring:
    #     gen_docstring = ast.get_docstring(gen_ast, clean=True)
    #     if gen_docstring is not None:
    #         gen_ast.body[0] = set_value(
    #             "\n{}".format(indent(cleandoc(gen_docstring), tab))
    #         )
    #     gold.body[0] = set_value(
    #       "\n{}".format(indent(ast.get_docstring(gold, clean=True), tab))
    #     )

    _gen_ast, _gold_ast = (
        (gold.body[0], gen_ast.body[0])
        if isinstance(gen_ast, ast.Module) and gen_ast.body
        else (gold, gen_ast)
    )

    # if not cmp_ast(_gen_ast, _gold_ast):
    #     import sys
    #
    #     if sys.version_info > (3, 7):
    #         from ast import dump
    #
    #         def print_ast(*args, file=sys.stdout, **kwargs):
    #             """
    #             Proxy for `ast.dump` that matches interface of `meta.asttools.print_ast`
    #             """
    #             if sys.version_info > (3, 9):
    #                 kwargs["indent"] = 4
    #             return print(dump(*args, **kwargs), file=file)
    #
    #     else:
    #         from meta.asttools import print_ast
    #
    #     print("#gen", file=sys.stderr)
    #     print_ast(_gen_ast, file=sys.stderr)
    #     print("#gold", file=sys.stderr)
    #     print_ast(_gold_ast, file=sys.stderr)

    if isinstance(
        _gen_ast, (ast.Module, ast.ClassDef, ast.AsyncFunctionDef, ast.FunctionDef)
    ):
        test_case_instance.assertEqual(
            *map(partial(ast.get_docstring, clean=True), (_gold_ast, _gen_ast))
        )

    coded_gold_gen = tuple(map(source_transformer.to_code, (_gold_ast, _gen_ast)))
    # test_case_instance.assertEqual(*map("\n".join, zip(("#gen", "#gold"), coded_gold_gen)))

    test_case_instance.assertEqual(*coded_gold_gen)
    test_case_instance.assertEqual(
        *map(
            (
                identity
                if skip_black
                else partial(
                    black.format_str,
                    mode=black.Mode(
                        target_versions=set(),
                        line_length=60,
                        is_pyi=False,
                        string_normalization=False,
                    ),
                )
            ),
            coded_gold_gen,
        )
    )

    test_case_instance.assertTrue(cmp_ast(_gen_ast, _gold_ast))


def run_cli_test(
    test_case_instance,
    cli_argv,
    exit_code,
    output,
    output_checker=lambda output: (lambda q: output[output.find(q) + len(q) :])(
        "error: "
    ),
    exception=SystemExit,
    return_args=False,
):
    """
    CLI test helper, wraps exit code and stdout/stderr input_str

    :param test_case_instance: instance of `TestCase`
    :type test_case_instance: ```unittest.TestCase```

    :param cli_argv: cli_argv, can be sys.argv or proxy
    :type cli_argv: ```list[str]```

    :param exit_code: exit code
    :type exit_code: ```Optional[int]```

    :param output: string representation (from stdout/stderr)
    :type output: ```Optional[str]```

    :param output_checker: Function to check the input_str with
    :type output_checker: ```Callable[[str], bool]```

    :param exception: The exception that is expected to be raised
    :type exception: ```Union[BaseException, Exception]```

    :param return_args: Primarily use is for tests. Returns the args rather than executing anything.
    :type return_args: ```bool```

    :return: input_str
    :rtype: ```tuple[str, Optional[Namespace]]```
    """
    argparse_mock, args = MagicMock(), None
    with patch("argparse.ArgumentParser._print_message", argparse_mock), patch(
        "sys.argv", cli_argv
    ):
        from cdd.__main__ import main

        main_f = partial(main, cli_argv=cli_argv, return_args=return_args)
        if exit_code is None:
            args = main_f()
        else:
            with test_case_instance.assertRaises(exception) as e:
                args = main_f()
    if exit_code is not None:
        test_case_instance.assertEqual(
            *(
                (e.exception.code, exception(exit_code).code)
                if exception is SystemExit
                else (str(e.exception), output)
            )
        )
    if exception is not SystemExit:
        pass
    elif argparse_mock.call_args is None:
        test_case_instance.assertIsNone(output)
    else:
        test_case_instance.assertEqual(
            output_checker(
                (argparse_mock.call_args.args if PY3_8 else argparse_mock.call_args[0])[
                    0
                ]
            ),
            output,
        )
    return output, args


def unittest_main():
    """Runs `unittest.main` if __main__"""
    if __name__ == "__main__":
        main()


# These next two are mostly https://stackoverflow.com/a/64960448


class ShowSourceLoader(Loader):
    """
    Loader that will enable `inspect.getsource` and friends to work from an in-memory construct
    """

    def __init__(self, modname, source):
        """
        :param modname: Name of module
        :type modname: ```str```

        :param source: Source string
        :type source: ```str```
        """
        self.modname = modname
        self.source = source

    def get_source(self, modname):
        """
        Return the source for the module

        :param modname: Name of module
        :type modname: ```str```

        :return: Source string
        :rtype: ```str```
        """
        assert modname == self.modname, ImportError(modname)
        return self.source


def inspectable_compile(s, modname=None):
    """
    Compile and executable the input string, returning what was constructed

    :param s: Input source
    :type s: ```str```

    :param modname: Module name, generates a random one if None
    :type modname: ```Optional[str]```

    :return: The compiled and executed input source module, such that `inspect.getsource` works
    :rtype: ```Any```
    """
    fh = NamedTemporaryFile(suffix="{extsep}py".format(extsep=extsep))
    filename: str = fh.name
    try:
        modname: str = modname or path.splitext(path.basename(filename))[0]
        assert modname not in modules
        # our loader is a dummy one which just spits out our source
        loader = ShowSourceLoader(modname, s)
        spec = spec_from_loader(modname, loader, origin=filename)
        module = module_from_spec(spec)
        # the code must be compiled so the function's code object has a filename
        code = compile(s, mode="exec", filename=filename)
        exec(code, module.__dict__)
        # inspect.getmodule(...) requires it to be in sys.modules
        setattr(module, "__file__", s)
        modules[modname] = module
        return module
    finally:
        fh.close()  # Is auto-deleted on close


# def module_from_file(file_path, module_name):
#     """
#     Creates a module out of the file_path
#
#     :param file_path: Input source
#     :type file_path: ```str```
#
#     :param module_name: Module name
#     :type module_name: ```Optional[str]```
#
#     :return: The module itself. Alternative `import` should now work from it.
#     :rtype: ```Any```
#     """
# spec = spec_from_file_location(module_name, file_path)
# assert spec is not None
# TODO
# print("spec:", spec, ";")
# module = module_from_spec(spec)
# sys.modules[module_name] = module
# spec.loader.exec_module(module)
# return module

# spec = spec_from_file_location(module_name, file_path)
# assert spec is not None
# module = module_from_spec(spec)
# spec.loader.exec_module(module)
# return module


def mock_function(*args, **kwargs):
    """
    Mock function to check if it is called

    :return: True
    :rtype: ```Literal[True]```
    """
    return True


def reindent_docstring(node, indent_level=1, smart=True):
    """
    Reindent the docstring

    :param node: AST node
    :type node: ```ast.AST```

    :param indent_level: docstring indentation level whence: 0=no_tabs, 1=one tab; 2=two tabs
    :type indent_level: ```int```

    :param smart: Smart indent mode, if False `lstrip`s each line of the input and adds the indent
    :type smart: ```bool```

    :return: Node with reindent docstring
    :rtype: ```ast.AST```
    """
    doc_str: Optional[str] = ast.get_docstring(node, clean=True)
    if doc_str is not None:
        _sep: str = tab * abs(indent_level)
        node.body[0] = ast.Expr(
            set_value(
                "\n{_sep}{s}\n{_sep}".format(
                    _sep=_sep,
                    s=(
                        "\n".join(
                            map(
                                lambda line: (
                                    "{sep}{line}".format(
                                        sep=tab * 2, line=line.lstrip()
                                    )
                                    if line.startswith(tab)
                                    and len(line) > len(tab)
                                    and line[
                                        len(tab) : line.lstrip().find(" ") + len(tab)
                                    ].rstrip(":s")
                                    not in frozenset((False,) + TOKENS.rest)
                                    else line
                                ),
                                reindent(doc_str).splitlines(),
                            )
                        )
                        if smart
                        else "\n".join(
                            map(
                                partial(add, _sep),
                                map(str.lstrip, doc_str.splitlines()),
                            )
                        )
                    ),
                )
            ),
            lineno=None,
            col_offset=None,
        )
    return node


def replace_docstring(node, new_docstring):
    """
    Replace the docstring. If no docstring assertion error.

    :param node: AST node
    :type node: ```ast.AST```

    :param new_docstring: Replace docstring
    :type new_docstring: ```str```

    :return: Node with reindent docstring
    :rtype: ```ast.AST```
    """
    assert (
        node
        and hasattr(node, "body")
        and node.body
        and isinstance(node.body[0], Expr)
        and isinstance(get_value(get_value(node.body[0])), str)
    )
    node.body[0] = Expr(value=set_value(new_docstring), lineno=None, col_offset=None)
    return node


def remove_args_from_docstring(doc_str):
    """
    Remove args, kwargs, raises, and any other "args" from the docstring

    :param doc_str: The doc str (any style)
    :type doc_str: ```str```

    :return: Docstring excluding args
    :rtype: ```str```
    """
    stack, in_args = [], False
    assert isinstance(doc_str, str), "Expected `str` got `{type_name}`".format(
        type_name=type(doc_str).__name__
    )
    for line in doc_str.splitlines():
        stripped_line = line.lstrip()
        if any(filter(stripped_line.startswith, TOKENS)):
            in_args = True
        elif line.isspace():
            stack.append("")
        elif (
            not in_args
            or stripped_line.endswith(":")
            and count_iter_items(takewhile(str.isalpha, stripped_line[:-1]))
            == len(stripped_line) - 1
        ):
            stack.append(line)
            in_args = False
    return "\n".join(stack)  # + ("\n" if stack[-1] == "" else "")


__all__ = [
    "inspectable_compile",
    "mock_function",
    # "module_from_file",
    "reindent_docstring",
    "remove_args_from_docstring",
    "replace_docstring",
    "run_ast_test",
    "run_cli_test",
    "unittest_main",
]  # type: list[str]
