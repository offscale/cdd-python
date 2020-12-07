""" Tests for default utils """
from unittest import TestCase

from doctrans.defaults_utils import extract_default, set_default_doc
from doctrans.tests.utils_for_tests import unittest_main


class TestDefaultUtils(TestCase):
    """ Test class for default utils """

    def test_extract_default(self) -> None:
        """ Tests that `extract_default` produces the expected output """
        sample = "This defaults to foo."
        self.assertTupleEqual(extract_default(sample), (sample, "foo"))
        self.assertTupleEqual(
            extract_default(sample, emit_default_doc=False), ("This", "foo")
        )

    def test_extract_default_middle(self) -> None:
        """ Tests that `extract_default` produces the expected output """
        sample = "Why would you. Have this defaults to something. In the middle?"
        self.assertTupleEqual(extract_default(sample), (sample, "something"))

        self.assertTupleEqual(
            extract_default(sample, rstrip_default=False, emit_default_doc=False),
            ("Why would you. Have this. In the middle?", "something"),
        )
        self.assertTupleEqual(
            extract_default(sample, rstrip_default=True, emit_default_doc=False),
            ("Why would you. Have thisIn the middle?", "something"),
        )

    def test_extract_default_with_dot(self) -> None:
        """ Tests that `extract_default` works when there is a `.` in the default """
        sample = "This. defaults to (np.empty(0), np.empty(0))"
        self.assertTupleEqual(
            extract_default(sample, emit_default_doc=False),
            ("This.", "(np.empty(0), np.empty(0))"),
        )

    def test_extract_default_with_int(self) -> None:
        """ Tests that `extract_default` works for an integer default """
        sample = "learning rate. Defaults to 0001."
        self.assertTupleEqual(
            extract_default(sample, emit_default_doc=False),
            ("learning rate.", 1),
        )

    def test_extract_default_with_float(self) -> None:
        """ Tests that `extract_default` works when there is a `.` in the default referring to a decimal place """
        sample = "learning rate. Defaults to 0.001."
        self.assertTupleEqual(
            extract_default(sample, emit_default_doc=False),
            ("learning rate.", 0.001),
        )

    def test_extract_default_with_bool(self) -> None:
        """ Tests that `extract_default` works for an integer default """
        sample = (
            "Boolean. Whether to apply AMSGrad variant of this algorithm from"
            'the paper "On the Convergence of Adam and beyond". Defaults to `True`.'
        )
        self.assertTupleEqual(
            extract_default(sample, emit_default_doc=True),
            (sample, True),
        )

    def test_set_default_doc_none(self) -> None:
        """ Tests that `set_default_doc` does nop whence no doc in param """
        param = {"name": "foo"}
        self.assertDictEqual(set_default_doc(param), param)


unittest_main()
