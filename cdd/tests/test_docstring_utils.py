""" Tests for docstring_utils """

from textwrap import indent
from unittest import TestCase

from cdd.docstring_utils import (
    ensure_doc_args_whence_original,
    parse_docstring_into_header_args_footer,
)
from cdd.pure_utils import emit_separating_tabs, tab
from cdd.tests.mocks.docstrings import (
    docstring_google_tf_mean_squared_error_args_tuple,
    docstring_google_tf_mean_squared_error_footer_tuple,
    docstring_google_tf_mean_squared_error_header_tuple,
    docstring_google_tf_mean_squared_error_str,
    docstring_str,
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
                ":type a: ```int```\n",
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
                ":param a:",
                ":type a: ```int```\n",
                self.footer,
            )
        )

        current_doc_str = "\n".join(
            (
                "{footer}\n".format(footer=self.footer),
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
                "{header}\n".format(header=self.header.rstrip("\n")),
                "Parameters",
                "----------",
                "as_numpy : Optional[bool]",
                "  Convert to numpy ndarrays. Defaults to None\n",
                self.footer,
            )
        )

        current_doc_str = "\n".join(
            (
                "{header}\n".format(header=self.header.rstrip("\n")),
                self.footer,
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
                    "{header}\n".format(header=self.header.rstrip("\n")),
                    ":param as_numpy: Convert to numpy ndarrays. Defaults to None",
                    ":type as_numpy: ```Optional[bool]```",
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
                ":param as_numpy: Convert to numpy ndarrays. Defaults to None",
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
                "  Convert to numpy ndarrays. Defaults to None\n",
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
                    "  Convert to numpy ndarrays. Defaults to None\n",
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
                    "  Convert to numpy ndarrays. Defaults to None\n",
                    self.footer,
                )
            ),
        )

    def test_ensure_doc_args_whence_original_to_docstring_str(self) -> None:
        """Test that ensure_doc_args_whence_original reworks the header and args_returns whence indent"""
        original_doc_str = (
            "\n"
            "    Acquire from the official tensorflow_datasets model zoo, or the ophthalmology "
            "focussed ml-prepare\n"
            "    \n"
            '    :cvar dataset_name: name of dataset. Defaults to "mnist"\n'
            "    :cvar tfds_dir: directory to look for models in. Defaults to "
            '"~/tensorflow_datasets"\n'
            '    :cvar K: backend engine, e.g., `np` or `tf`. Defaults to "np"\n'
            "    :cvar as_numpy: Convert to numpy ndarrays. Defaults to None\n"
            "    :cvar data_loader_kwargs: pass this as arguments to data_loader function\n"
            "    :cvar return_type: Train and tests dataset splits. Defaults to (np.empty(0), "
            "np.empty(0))\n"
        )
        current_doc_str = (
            "\n"
            "Acquire from the official tensorflow_datasets model zoo, or the ophthalmology focussed "
            "ml-prepare\n"
            "\n"
            ':param dataset_name: name of dataset. Defaults to "mnist"\n'
            ":type dataset_name: ```str```\n"
            "\n"
            ':param tfds_dir: directory to look for models in. Defaults to "~/tensorflow_datasets"\n'
            ":type tfds_dir: ```str```\n"
            "\n"
            ':param K: backend engine, e.g., `np` or `tf`. Defaults to "np"\n'
            ":type K: ```Literal['np', 'tf']```\n"
            "\n"
            ":param as_numpy: Convert to numpy ndarrays. Defaults to None\n"
            ":type as_numpy: ```Optional[bool]```\n"
            "\n"
            ":param data_loader_kwargs: pass this as arguments to data_loader function\n"
            ":type data_loader_kwargs: ```Optional[dict]```\n"
            "\n"
            ":return: Train and tests dataset splits. Defaults to (np.empty(0), np.empty(0))\n"
            ":rtype: ```Union[Tuple[tf.data.Dataset, tf.data.Dataset], Tuple[np.ndarray, np.ndarray]]```"
        )

        self.assertEqual(
            ensure_doc_args_whence_original(
                current_doc_str=current_doc_str, original_doc_str=original_doc_str
            ),
            emit_separating_tabs(indent(docstring_str, tab))[: -len(tab) - 1],
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
