"""
Shared utility functions used by many tests
"""
import ast
from copy import deepcopy
from functools import partial
from importlib.abc import Loader
from importlib.util import module_from_spec, spec_from_loader
from os import path
from sys import modules
from tempfile import NamedTemporaryFile
from unittest import main
from unittest.mock import MagicMock, patch

from black import Mode, format_str
from meta.asttools import cmp_ast

from cdd import source_transformer
from cdd.ast_utils import set_value
from cdd.docstring_utils import TOKENS
from cdd.pure_utils import PY3_8, identity, reindent, tab


def run_ast_test(test_case_instance, gen_ast, gold, skip_black=False):
    """
    Compares `gen_ast` with `gold` standard

    :param test_case_instance: instance of `TestCase`
    :type test_case_instance: ```unittest.TestCase```

    :param gen_ast: generated AST
    :type gen_ast: ```Union[ast.Module, ast.ClassDef, ast.FunctionDef]```

    :param skip_black: Whether to skip formatting with black. Turned off for performance, turn on for pretty debug.
    :type skip_black: ```bool```

    :param gold: mocked AST
    :type gold: ```Union[ast.Module, ast.ClassDef, ast.FunctionDef]```
    """
    if isinstance(gen_ast, str):
        gen_ast = ast.parse(gen_ast).body[0]

    assert gen_ast is not None, "gen_ast is None"
    assert gold is not None, "gold is None"

    gen_ast = deepcopy(gen_ast)
    gold = deepcopy(gold)

    # if reindent_docstring:
    #           gen_docstring = ast.get_docstring(gen_ast)
    #           if gen_docstring is not None:
    #               gen_ast.body[0] = set_value(
    #                   "\n{}".format(indent(cleandoc(gen_docstring), tab))
    #               )
    #           gold.body[0] = set_value(
    #               "\n{}".format(indent(ast.get_docstring(gold, clean=True), tab))
    #           )

    # from meta.asttools import print_ast
    #
    # print("#gen")
    # print_ast(gen_ast)
    # print("#gold")
    # print_ast(gold)

    test_case_instance.assertEqual(
        *map(
            identity
            if skip_black
            else partial(
                format_str,
                mode=Mode(
                    target_versions=set(),
                    line_length=60,
                    is_pyi=False,
                    string_normalization=False,
                ),
            ),
            map(source_transformer.to_code, (gold, gen_ast)),
        )
    )

    test_case_instance.assertTrue(
        cmp_ast(gen_ast, gold), "Generated AST doesn't match reference AST"
    )


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
    :type cli_argv: ```List[str]```

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

    :returns: input_str
    :rtype: ```Tuple[str, Optional[Namespace]]```
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
            *(e.exception.code, exception(exit_code).code)
            if exception is SystemExit
            else (str(e.exception), output)
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
    """ Runs unittest.main if __main__ """
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

        :returns: Source string
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

    :returns: The compiled and executed input source module, such that `inspect.getsource` works
    :rtype: ```Any```
    """
    fh = NamedTemporaryFile(suffix=".py")
    filename = fh.name
    try:
        modname = modname or path.splitext(path.basename(filename))[0]
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


def mock_function(*args, **kwargs):
    """
    Mock function to check if it is called

    :returns: True
    :rtype: ```Literal[True]```
    """
    return True


def reindent_docstring(node, indent_level=1):
    """
    Reindent the docstring

    :param node: AST node
    :type node: ```ast.AST```

    :param indent_level: docstring indentation level whence: 0=no_tabs, 1=one tab; 2=two tabs
    :type indent_level: ```int```

    :returns: Node with reindent docstring
    :rtype: ```ast.AST```
    """
    doc_str = ast.get_docstring(node)
    if doc_str is not None:
        node.body[0] = ast.Expr(
            set_value(
                "\n{tab}{s}\n{tab}".format(
                    tab=tab * abs(indent_level),
                    s="\n".join(
                        map(
                            lambda line: "{sep}{line}".format(
                                sep=tab * 2, line=line.lstrip()
                            )
                            if line.startswith(tab)
                            and len(line) > len(tab)
                            and line[
                                len(tab) : line.lstrip().find(" ") + len(tab)
                            ].rstrip(":s")
                            not in frozenset((False,) + TOKENS.rest)
                            else line,
                            reindent(doc_str).splitlines(),
                        )
                    ),
                )
            )
        )
    return node


def emit_separating_tab(s, indent_level=1):
    """
    Emit a separating tab between paragraphs

    :param s: Input string (probably a docstring)
    :type s: ```str```

    :param indent_level: docstring indentation level whence: 0=no_tabs, 1=one tab; 2=two tabs
    :type indent_level: ```int```

    """
    sep = tab * indent_level
    return "\n{sep}{}\n{sep}".format(
        "\n".join(
            map(lambda line: sep if len(line) == 0 else line, s.splitlines())
        ).lstrip(),
        sep=sep,
    )


__all__ = [
    "inspectable_compile",
    "mock_function",
    "run_ast_test",
    "run_cli_test",
    "unittest_main",
]
