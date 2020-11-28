""" Tests for default utils """
from unittest import TestCase

from doctrans.defaults_utils import extract_default
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


unittest_main()
