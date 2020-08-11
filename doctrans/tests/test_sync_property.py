""" Tests for sync_property """
import os
from tempfile import TemporaryDirectory
from unittest import TestCase

from doctrans.pure_utils import tab
from doctrans.sync_property import sync_property
from doctrans.tests.utils_for_tests import unittest_main


class TestSyncProperty(TestCase):
    """ Test class for sync_property.py """

    def test_sync_property(self) -> None:
        """ Tests `sync_property` call failure cases """
        with TemporaryDirectory() as tempdir:
            class_py = os.path.join(tempdir, "class_.py")
            method_py = os.path.join(tempdir, "method.py")
            with open(class_py, "wt") as f:
                f.write(
                    "from typing import Literal\n\n"
                    "class Foo(object):\n"
                    "{tab}def g(f: Literal['a']):\n"
                    "{tab}{tab}pass".format(tab=tab)
                )
            with open(method_py, "wt") as f:
                f.write(
                    "from typing import Literal\n\n"
                    "def f(h: Literal['b']):"
                    "{tab}{tab}pass".format(tab=tab)
                )

            self.assertRaises(
                NotImplementedError,
                lambda: sync_property(
                    input_file=class_py,
                    input_param="Foo.g.f",
                    input_eval=False,
                    output_file=method_py,
                    output_param="f.h",
                ),
            )


unittest_main()
