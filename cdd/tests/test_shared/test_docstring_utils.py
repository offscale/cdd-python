""" Tests for docstring_utils """

from textwrap import indent
from unittest import TestCase

from cdd.shared.docstring_utils import (
    ensure_doc_args_whence_original,
    parse_docstring_into_header_args_footer,
)
from cdd.shared.pure_utils import emit_separating_tabs, tab
from cdd.tests.mocks.classes import class_doc_str
from cdd.tests.mocks.docstrings import (
    docstring_google_tf_mean_squared_error_args_tuple,
    docstring_google_tf_mean_squared_error_footer_tuple,
    docstring_google_tf_mean_squared_error_header_tuple,
    docstring_google_tf_mean_squared_error_str,
    docstring_no_nl_str,
)
from cdd.tests.utils_for_tests import unittest_main


class TestDocstringUtils(TestCase):
    """Test class for emitter_utils"""

    header = "Header\n\n"
    footer = "Footer"

    def test_ensure_doc_args_whence_original(self) -> None:
        """Test that ensure_doc_args_whence_original moves the doc to the right place"""
        original_doc_str = "\n".join(
            (
                "\n{header}".format(header=self.header),
                ":param a:",
                ":type a: ```int```",
                self.footer,
            )
        )

        current_doc_str = "\n".join(
            (
                "\n{header}\n\n".format(header=self.header),
                self.footer,
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
                "",
                "",
                "",
                ":param a:",
                ":type a: ```int```",
                self.footer,
            )
        )

        current_doc_str = "\n".join(
            (
                self.footer,
                "",
                ":param a:",
                ":type a: ```int```",
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
        original_doc_str = "\n".join(
            (
                "\n{header}".format(header=self.header),
                ":param a:",
                ":type a: ```int```\n",
            )
        )

        current_doc_str = "\n".join(
            (
                "\n{header}".format(header=self.header),
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
                self.header.rstrip("\n"),
                "",
                "Parameters",
                "----------",
                "as_numpy : Optional[bool]",
                "  Convert to numpy ndarrays",
                "",
                "",
                self.footer,
            )
        )

        current_doc_str = "\n".join(
            (
                "{header}\n".format(header=self.header.rstrip("\n")),
                self.footer,
                ":param as_numpy: Convert to numpy ndarrays",
                ":type as_numpy: ```Optional[bool]```",
            )
        )

        self.assertEqual(
            ensure_doc_args_whence_original(
                current_doc_str=current_doc_str, original_doc_str=original_doc_str
            ),
            "\n".join(
                (
                    "{header}\n".format(header=self.header.rstrip("\n")),
                    ":param as_numpy: Convert to numpy ndarrays",
                    ":type as_numpy: ```Optional[bool]```\n",
                    self.footer,
                )
            ),
        )

    def test_ensure_doc_args_whence_different_multiline_generated_docstring_format(
        self,
    ) -> None:
        """Test that ensure_doc_args_whence_original moves the doc to the right place when no header|footer exists"""
        original_doc_str = "\n".join(
            (
                "{header}\n".format(header=self.header.rstrip("\n")),
                ":param as_numpy: Convert to numpy ndarrays",
                ":type as_numpy: ```Optional[bool]```",
                self.footer,
            )
        )

        current_doc_str = "\n".join(
            (
                "Header\n",
                self.footer,
                "Parameters",
                "----------",
                "as_numpy : Optional[bool]",
                "  Convert to numpy ndarrays\n",
            )
        )

        header, args_returns, footer = parse_docstring_into_header_args_footer(
            current_doc_str=current_doc_str, original_doc_str=original_doc_str
        )
        self.assertEqual(header, self.header)
        self.assertEqual(
            args_returns,
            "\n".join(
                (
                    "Parameters",
                    "----------",
                    "as_numpy : Optional[bool]",
                    "  Convert to numpy ndarrays\n",
                )
            ),
        )
        self.assertEqual(footer, self.footer)

        self.assertEqual(
            ensure_doc_args_whence_original(
                current_doc_str=current_doc_str, original_doc_str=original_doc_str
            ),
            "\n".join(
                (
                    "{header}\n".format(header=self.header.rstrip("\n")),
                    "Parameters",
                    "----------",
                    "as_numpy : Optional[bool]",
                    "  Convert to numpy ndarrays",
                    self.footer,
                )
            ),
        )

    def test_ensure_doc_args_whence_original_to_docstring_str(self) -> None:
        """Test that ensure_doc_args_whence_original reworks the header and args_returns whence indent"""
        original_doc_str = class_doc_str
        current_doc_str = docstring_no_nl_str

        self.assertEqual(
            ensure_doc_args_whence_original(
                current_doc_str=current_doc_str, original_doc_str=original_doc_str
            ),
            emit_separating_tabs(indent(current_doc_str, tab))[: -len(tab)],
        )

    def test_from_docstring_google_tf_mean_squared_error_str_to_three_parts(
        self,
    ) -> None:
        """
        Tests whether `docstring_google_tf_mean_squared_error_str` is correctly parsed into 3 parts:
        0. header
        1. args|return
        2. footer
        """
        header, args_returns, footer = parse_docstring_into_header_args_footer(
            current_doc_str=docstring_google_tf_mean_squared_error_str,
            # current_doc_str=docstring_google_tf_mean_squared_error_str.replace(
            #     "Args:", "Parameters\n----------", 1
            # ).replace("Returns:", "Returns\n-------", 1),
            original_doc_str=docstring_google_tf_mean_squared_error_str,
        )
        self.assertEqual(
            header,
            "\n".join(docstring_google_tf_mean_squared_error_header_tuple) + "\n",
        )
        self.assertEqual(
            args_returns, "\n".join(docstring_google_tf_mean_squared_error_args_tuple)
        )
        self.assertEqual(
            footer,
            "\n" + "\n".join(docstring_google_tf_mean_squared_error_footer_tuple),
        )


unittest_main()
