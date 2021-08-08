""" Tests for exmod_utils """
from collections import deque
from io import StringIO
from os import path
from tempfile import TemporaryDirectory
from unittest import TestCase
from unittest.mock import MagicMock, patch

from cdd.exmod_utils import _emit_symbol, emit_file_on_hierarchy
from cdd.pure_utils import INIT_FILENAME, quote, rpartial
from cdd.tests.utils_for_tests import unittest_main


class TestExmodUtils(TestCase):
    """Test class for emitter_utils"""

    def test_emit_file_on_hierarchy_dry_run(self) -> None:
        """Test that `emit_file_on_hierarchy` works with dry_run"""

        ir = {"name": "YEP", "doc": None}
        with patch("sys.stdout", new_callable=StringIO) as f:
            emit_file_on_hierarchy(
                ("", "foo_dir", ir), "argparse", "", "", None, "", dry_run=True
            )
        self.assertListEqual(
            deque(map(rpartial(str.split, "\t"), f.getvalue().splitlines()), maxlen=1)[
                0
            ],
            ["write", quote("{name}.py".format(name=ir["name"]), "'")],
        )

    def test_emit_file_on_hierarchy(self) -> None:
        """Test `emit_file_on_hierarchy`"""

        ir = {"name": "YEP", "doc": None}
        with patch(
            "sys.stdout", new_callable=StringIO
        ), TemporaryDirectory() as tempdir:
            open(path.join(tempdir, INIT_FILENAME), "a").close()
            emit_file_on_hierarchy(
                ("foo.bar", "foo_dir", ir),
                "argparse",
                "",
                "",
                filesystem_layout="as_input",
                output_directory=tempdir,
                dry_run=False,
            )
            self.assertTrue(path.isdir(tempdir))

    def test__emit_symbols_isfile_emit_filename_true(self):
        """Test `_emit_symbol` when `isfile_emit_filename is True`"""
        with patch("sys.stdout", new_callable=StringIO), patch(
            "cdd.exmod_utils.merge_modules", MagicMock()
        ) as f, patch("cdd.exmod_utils.merge_assignment_lists", MagicMock()) as g:
            _emit_symbol(
                name_orig_ir=("", "", dict()),
                emit_name="argparse",
                module_name="module_name",
                emit_filename="emit_filename",
                existent_mod=None,
                init_filepath="",
                intermediate_repr={"name": None, "doc": None},
                isfile_emit_filename=True,
                name="",
                dry_run=True,
            )
            f.assert_called_once()
            g.assert_called_once()


unittest_main()
