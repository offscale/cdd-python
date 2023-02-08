import cdd
from cdd.shared.source_transformer import to_code
import cdd.shared.parse.utils.parser_utils
from cdd.ndb.utils.parser_utils import property_to_param
from cdd.docstring.parse import docstring
from collections import OrderedDict
from cdd.shared.pure_utils import rpartial
from ast import Assign, Expr, Constant


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
    assert (any(map(lambda b: to_code(b).rstrip() == "ndb.Model", model.bases)))
    doc = model.body[0].value.value if isinstance(model.body[0], Expr) and isinstance(model.body[0].value,
                                                                                            Constant) else None
    intermediate_repr = (
        {"type": None, "doc": "", "params": OrderedDict()}
        if docstring is None
        else docstring(
            doc, parse_original_whitespace=parse_original_whitespace
        )
    )
    # intermediate_repr = {"type": None, "doc": "", "params": OrderedDict()}
    intermediate_repr["name"] = model.name
    merge_ir = {
        "params": OrderedDict(map(property_to_param, filter(rpartial(isinstance, Assign), model.body))),
        "returns": None,
    }

    cdd.shared.parse.utils.parser_utils.ir_merge(
        target=intermediate_repr, other=merge_ir
    )
    return intermediate_repr

