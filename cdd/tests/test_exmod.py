""" Tests for exmod subcommand """
from os import path
from tempfile import TemporaryDirectory
from unittest import TestCase

from cdd.exmod import exmod
from cdd.tests.utils_for_tests import unittest_main


class TestExMod(TestCase):
    """Test class for exmod.py"""

    def test_exmod_blacklist(self) -> None:
        """Tests `exmod` blacklist"""

        with TemporaryDirectory() as tempdir, self.assertRaises(NotImplementedError):
            exmod(
                module="unittest",
                emit=None,
                blacklist=("unittest.TestCase",),
                whitelist=tuple(),
                output_directory=tempdir,
            )

    def test_exmod_whitelist(self) -> None:
        """Tests `exmod` whitelist"""

        with TemporaryDirectory() as tempdir, self.assertRaises(NotImplementedError):
            exmod(
                module="unittest",
                emit=None,
                blacklist=tuple(),
                whitelist=("unittest.TestCase",),
                output_directory=tempdir,
            )

    def test_exmod_module_directory(self) -> None:
        """Tests `exmod` module whence directory"""

        with TemporaryDirectory() as tempdir, self.assertRaises(NotImplementedError):
            exmod(
                module=tempdir,
                emit=None,
                blacklist=tuple(),
                whitelist=tuple(),
                output_directory=tempdir,
            )

    def test_exmod_output_directory_nonexistent(self) -> None:
        """Tests `exmod` module whence directory does not exist"""

        with TemporaryDirectory() as tempdir:
            output_directory = path.join(tempdir, "stuff")
            self.assertFalse(path.isdir(output_directory))
            exmod(
                module="unittest",
                emit=None,
                blacklist=tuple(),
                whitelist=tuple(),
                output_directory=output_directory,
            )
            self.assertTrue(path.isdir(output_directory))

    def test_exmod(self) -> None:
        """Tests `exmod`"""

        with TemporaryDirectory() as tempdir:
            exmod(
                module="unittest",
                emit=None,
                blacklist=tuple(),
                whitelist=tuple(),
                output_directory=tempdir,
            )


unittest_main()
