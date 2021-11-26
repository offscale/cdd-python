""" Tests for docstring_utils """

from unittest import TestCase

from cdd.docstring_utils import ensure_doc_args_whence_original
from cdd.tests.utils_for_tests import unittest_main


class TestDocstringUtils(TestCase):
    """Test class for emitter_utils"""

    def test_ensure_doc_args_whence_original(self) -> None:
        """Test that ensure_doc_args_whence_original moves the doc to the right place"""
        original_doc_str = "\n".join(
            (
                "\nfoo\n\n",
                ":param a:",
                ":type a: ```int```\n",
                "can haz",
            )
        )

        current_doc_str = "\n".join(
            (
                "\nfoo\n\n",
                "can haz\n",
                ":param a:",
                ":type a: ```int```\n",
            )
        )

        self.assertEqual(
            ensure_doc_args_whence_original(
                current_doc_str=current_doc_str, original_doc_str=original_doc_str
            ),
            original_doc_str,
        )

    def test_ensure_doc_args_whence_original_no_header(self) -> None:
        """Test that ensure_doc_args_whence_original moves the doc to the right place when no header exists"""
        original_doc_str = "\n".join(
            (
                ":param a:",
                ":type a: ```int```\n",
                "can haz",
            )
        )

        current_doc_str = "\n".join(
            (
                "can haz\n",
                ":param a:",
                ":type a: ```int```\n",
            )
        )

        self.assertEqual(
            ensure_doc_args_whence_original(
                current_doc_str=current_doc_str, original_doc_str=original_doc_str
            ),
            original_doc_str,
        )

    def test_ensure_doc_args_whence_original_no_footer(self) -> None:
        """Test that ensure_doc_args_whence_original moves the doc to the right place when no footer exists"""
        original_doc_str = "\n".join(("\nfoo\n\n", ":param a:", ":type a: ```int```\n"))

        current_doc_str = "\n".join(
            (
                "\nfoo\n\n",
                ":param a:",
                ":type a: ```int```\n",
            )
        )

        self.assertEqual(
            ensure_doc_args_whence_original(
                current_doc_str=current_doc_str, original_doc_str=original_doc_str
            ),
            original_doc_str,
        )

    def test_ensure_doc_args_whence_original_no_header_or_footer(self) -> None:
        """Test that ensure_doc_args_whence_original moves the doc to the right place when no header|footer exists"""
        original_doc_str = "\n".join((":param a:", ":type a: ```int```\n"))

        current_doc_str = "\n".join(
            (
                ":param a:",
                ":type a: ```int```\n",
            )
        )

        self.assertEqual(
            ensure_doc_args_whence_original(
                current_doc_str=current_doc_str, original_doc_str=original_doc_str
            ),
            original_doc_str,
        )

    def test_ensure_doc_args_whence_different_multiline_original_docstring_format(
        self,
    ) -> None:
        """Test that ensure_doc_args_whence_original moves the doc to the right place when no header|footer exists"""
        original_doc_str = "\n".join(
            (
                "Header\n",
                "Parameters",
                "----------",
                "as_numpy : Optional[bool]",
                "  Convert to numpy ndarrays. Defaults to None\n",
                "Footer",
            )
        )

        current_doc_str = "\n".join(
            (
                "Header\n",
                "Footer",
                ":param as_numpy: Convert to numpy ndarrays. Defaults to None",
                ":type as_numpy: ```Optional[bool]```",
            )
        )

        self.assertEqual(
            ensure_doc_args_whence_original(
                current_doc_str=current_doc_str, original_doc_str=original_doc_str
            ),
            "\n".join(
                (
                    "Header\n",
                    ":param as_numpy: Convert to numpy ndarrays. Defaults to None",
                    ":type as_numpy: ```Optional[bool]```",
                    "Footer",
                )
            ),
        )

    def test_ensure_doc_args_whence_different_multiline_generated_docstring_format(
        self,
    ) -> None:
        """Test that ensure_doc_args_whence_original moves the doc to the right place when no header|footer exists"""
        original_doc_str = "\n".join(
            (
                "Header\n",
                ":param as_numpy: Convert to numpy ndarrays. Defaults to None",
                ":type as_numpy: ```Optional[bool]```",
                "Footer",
            )
        )

        current_doc_str = "\n".join(
            (
                "Header\n",
                "Footer",
                "Parameters",
                "----------",
                "as_numpy : Optional[bool]",
                "  Convert to numpy ndarrays. Defaults to None\n",
            )
        )

        self.assertEqual(
            ensure_doc_args_whence_original(
                current_doc_str=current_doc_str, original_doc_str=original_doc_str
            ),
            "\n".join(
                (
                    "Header\n",
                    "Parameters",
                    "----------",
                    "as_numpy : Optional[bool]",
                    "  Convert to numpy ndarrays. Defaults to None\n",
                    "Footer",
                )
            ),
        )


unittest_main()
