""" Tests for docstring_utils """
from textwrap import indent
from unittest import TestCase

from cdd.docstring_utils import ensure_doc_args_whence_original
from cdd.pure_utils import tab, emit_separating_tabs
from cdd.tests.mocks.docstrings import docstring_str
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

    def test_ensure_doc_args_whence_original_to_docstring_str(self) -> None:
        """Test that ensure_doc_args_whence_original reworks the header and footer whence indent"""
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
            "np.empty(0))"
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
            ":rtype: ```Union[Tuple[tf.data.Dataset, tf.data.Dataset], Tuple[np.ndarray, "
            "np.ndarray]]```\n"
        )

        self.assertEqual(
            ensure_doc_args_whence_original(
                current_doc_str=current_doc_str, original_doc_str=original_doc_str
            ),
            emit_separating_tabs(indent(docstring_str, tab))[: -len(tab)],
        )


unittest_main()
