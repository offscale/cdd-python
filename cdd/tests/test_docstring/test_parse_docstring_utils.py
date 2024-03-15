"""
Tests for cdd.docstring.utils.parse_utils
"""

from collections import deque
from unittest import TestCase

import cdd.shared.docstring_parsers
from cdd.docstring.utils.parse_utils import parse_adhoc_doc_for_typ
from cdd.shared.pure_utils import pp
from cdd.tests.mocks.docstrings import docstring_google_keras_tensorboard_return_str
from cdd.tests.mocks.ir import class_google_keras_tensorboard_ir
from cdd.tests.utils_for_tests import unittest_main


class TestParseDocstringUtils(TestCase):
    """
    Tests whether the intermediate representation is consistent when parsed from different inputs.

    IR is a dictionary consistent with `IntermediateRepr`, defined as:
        ParamVal = TypedDict("ParamVal", {"typ": str, "doc": Optional[str], "default": Any})
        IntermediateRepr = TypedDict("IntermediateRepr", {
            "name": Optional[str],
            "type": Optional[str],
            "doc": Optional[str],
            "params": OrderedDict[str, ParamVal],
            "returns": Optional[OrderedDict[Literal["return_type"], ParamVal]],
        })
    """

    def test_parse_adhoc_doc_for_typ(self) -> None:
        """
        Test that `parse_adhoc_doc_for_typ` works for various found-in-wild Keras variants
        """
        pp(
            parse_adhoc_doc_for_typ(
                "Dictionary of `{str: object}` pairs, where the `str` key is the object name.",
                name="",
                default_is_none=False,
            )
        )
        deque(
            map(
                lambda output_input: self.assertEqual(
                    output_input[0],
                    parse_adhoc_doc_for_typ(
                        output_input[1], name="", default_is_none=False
                    ),
                ),
                (
                    (
                        "str",
                        class_google_keras_tensorboard_ir["params"]["log_dir"]["doc"],
                    ),
                    (
                        "int",
                        class_google_keras_tensorboard_ir["params"]["histogram_freq"][
                            "doc"
                        ],
                    ),
                    ("Union[list,tuple]", "A list/tuple"),
                    (
                        'Union[Literal["batch", "epoch"], int]',
                        class_google_keras_tensorboard_ir["params"]["update_freq"][
                            "doc"
                        ],
                    ),
                    (
                        "int",
                        "Explicit `int64`-castable monotonic step value for this summary.",
                    ),
                    (
                        "bool",
                        cdd.shared.docstring_parsers.parse_docstring(
                            docstring_google_keras_tensorboard_return_str
                        )["returns"]["return_type"]["typ"],
                    ),
                    (
                        "Literal['auto', 'min', 'max']",
                        "String. One of `{'auto', 'min', 'max'}`. In `'min'` mode,",
                    ),
                    (
                        'Union[Literal["epoch"], int, bool]',
                        '`"epoch"`, integer, or `False`.'
                        'When set to `"epoch" the callback saves the checkpoint at the end of each epoch.',
                    ),
                    ("Optional[int]", "Int or None, defaults to None."),
                    (
                        "Literal['bfloat16', 'float16', 'float32', 'float64']",
                        "String; `'bfloat16'`, `'float16'`, `'float32'`, or `'float64'`.",
                    ),
                    ("List[str]", "List of string."),
                    ("Mapping[str, object]", "Dictionary of `{str: object}` pairs."),
                    (None, ""),
                ),
            ),
            maxlen=0,
        )


unittest_main()
