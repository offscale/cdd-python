"""
Shared utility functions used by many tests
"""
import ast
from copy import deepcopy
from functools import partial
from unittest import main
from unittest.mock import MagicMock, patch

from meta.asttools import cmp_ast

import doctrans.source_transformer
from doctrans.pure_utils import PY3_8


def run_ast_test(test_case_instance, gen_ast, gold, run_cmp_ast=True):
    """
    Compares `gen_ast` with `gold` standard

    :param test_case_instance: instance of `TestCase`
    :type test_case_instance: ```unittest.TestCase```

    :param gen_ast: generated AST
    :type gen_ast: ```Union[ast.Module, ast.ClassDef, ast.FunctionDef]```

    :param run_cmp_ast: whether to `cmp_ast` on output; otherwise only compare string representation
    :type run_cmp_ast: ```bool```

    :param gold: mocked AST
    :type gold: ```Union[ast.Module, ast.ClassDef, ast.FunctionDef]```
    """
    if isinstance(gen_ast, str):
        gen_ast = ast.parse(gen_ast).body[0]

    gen_ast = deepcopy(gen_ast)
    gold = deepcopy(gold)

    if hasattr(gen_ast, "body") and len(gen_ast.body) > 0:
        gen_docstring = ast.get_docstring(gen_ast)
        gold_docstring = ast.get_docstring(gold)
        if gen_docstring is not None and gold_docstring is not None:
            test_case_instance.assertEqual(
                gold_docstring.strip(), gen_docstring.strip()
            )
            # Following test issue with docstring indentation, remove them from the AST, as symmetry has been confirmed
            gen_ast.body.pop(0)
            gold.body.pop(0)

    test_case_instance.assertEqual(
        *map(doctrans.source_transformer.to_code, (gen_ast, gold))
    )

    if run_cmp_ast:
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

    :param return_args: Primarily use is for tests. Returns the args rather than executing anything.
    :type return_args: ```bool```

    :return: input_str
    :rtype: ```Tuple[str, Optional[Namespace]]```
    """
    argparse_mock, args = MagicMock(), None
    with patch("argparse.ArgumentParser._print_message", argparse_mock), patch(
        "sys.argv", cli_argv
    ):
        from doctrans.__main__ import main

        main_f = partial(main, cli_argv=cli_argv, return_args=return_args)
        if exit_code is None:
            args = main_f()
        else:
            with test_case_instance.assertRaises(SystemExit) as e:
                args = main_f()
    if exit_code is not None:
        test_case_instance.assertEqual(e.exception.code, SystemExit(exit_code).code)
    if argparse_mock.call_args is None:
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


__all__ = ["run_ast_test", "run_cli_test", "unittest_main"]
