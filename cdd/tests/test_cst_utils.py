""" Tests for cst_utils """
from unittest import TestCase

from cdd.tests.utils_for_tests import unittest_main


class TestCstUtils(TestCase):
    """Test class for cst_utils"""

    # def test_handle_multiline_comment(self) -> None:
    #     """Tests that `handle_multiline_comment` can do its namesake"""
    #
    #     multi_line_comment = MultiComment(None, None, None, None, None, None)
    #     cst_list = []
    #     lines = "\n".join(("def f():", '    """', "foo", 'bar"""'))
    #     deque(
    #         map(
    #             lambda line: handle_multiline_comment(
    #                 line_no=0,
    #                 line=line,
    #                 line_lstripped=line.lstrip(),
    #                 is_double_q=True,
    #                 multi_line_comment=multi_line_comment,
    #                 cst_list=cst_list,
    #                 scope=[],
    #             ),
    #             lines.splitlines(True),
    #         ),
    #         maxlen=0,
    #     )
    #     pp(cst_list)
    #     self.assertListEqual(
    #         cst_list,
    #         [
    #             MultiComment(
    #                 is_double_q=True,
    #                 is_docstr=False,
    #                 scope=[],
    #                 line_no_start=0,
    #                 line_no_end=0,
    #                 value='def f():\n    """\nfoo\nbar"""',
    #             ),
    #             MultiComment(
    #                 is_double_q=True,
    #                 is_docstr=False,
    #                 scope=[],
    #                 line_no_start=0,
    #                 line_no_end=0,
    #                 value='def f():\n    """\nfoo\nbar"""',
    #             ),
    #             MultiComment(
    #                 is_double_q=True,
    #                 is_docstr=False,
    #                 scope=[],
    #                 line_no_start=0,
    #                 line_no_end=0,
    #                 value='def f():\n    """\nfoo\nbar"""',
    #             ),
    #         ],
    #     )
    #     self.assertEqual(multi_line_comment.value, "")


unittest_main()
