from unittest import TestCase
from cdd.tests.mocks.ndb import ndb_model_example
from cdd.ndb.parse import ndb_model
import ast
class TestParseNDB(TestCase):
    """
    Tests whether the intermediate representation is consistent when parsed from different inputs.

    IR is a dictionary of form:
        {  "name": Optional[str],
           "type": Optional[str],
           "doc": Optional[str],
           "params": OrderedDict[str, {'typ': str, 'doc': Optional[str], 'default': Any}]
           "returns": Optional[OrderedDict[Literal['return_type'],
                                           {'typ': str, 'doc': Optional[str], 'default': Any}),)]] }
    """
    def test_from_ndb_model(self) -> None:
        """
        Tests that `ndb.parse` produces `intermediate_repr_ndb` properly
        """

        asts = ast.parse(ndb_model_example);
        ndb_model(ast.parse(ndb_model_example).body[0])
        print("hey")


