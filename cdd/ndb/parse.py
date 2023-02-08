from cdd.shared.source_transformer import to_code
from cdd.ndb.utils.parser_utils import property_to_param
from collections import OrderedDict
def ndb_model(model, parse_original_whitespace=False):
    """
    Parse out a `ndb.Model`, into the IR

    :param model: The ClassDef for the ndb model
    :type model: ```ClassDef```

    :param parse_original_whitespace: Whether to parse original whitespace or strip it out
    :type parse_original_whitespace: ```bool```

    :return: a dictionary of form
        {  "name": Optional[str],
           "type": Optional[str],
           "doc": Optional[str],
           "params": OrderedDict[str, {'typ': str, 'doc': Optional[str], 'default': Any}]
           "returns": Optional[OrderedDict[Literal['return_type'],
                                           {'typ': str, 'doc': Optional[str], 'default': Any}),)]] }
    :rtype: ```dict```
    """
    assert ("ndb.Model" in list(map(lambda b: to_code(b).rstrip(), model.bases)))
    intermediate_repr = {"type": None, "doc": "", "params": OrderedDict()}
    intermediate_repr["name"] = model.name
    merge_ir = {
        "params": OrderedDict(map(property_to_param, filter(rpartial model.body))),
        "returns": None,
    }



