"""
JSON schema emitter
"""

from collections import OrderedDict
from functools import partial
from json import dump
from operator import add

from cdd.docstring.emit import docstring
from cdd.json_schema.utils.emit_utils import param2json_schema_property
from cdd.shared.pure_utils import SetEncoder, deindent


def json_schema(
    intermediate_repr,
    identifier=None,
    emit_original_whitespace=False,
    emit_default_doc=False,
    word_wrap=False,
):
    """
    Construct a JSON schema dict

    :param intermediate_repr: a dictionary consistent with `IntermediateRepr`, defined as:
        ParamVal = TypedDict("ParamVal", {"typ": str, "doc": Optional[str], "default": Any})
        IntermediateRepr = TypedDict("IntermediateRepr", {
            "name": Optional[str],
            "type": Optional[str],
            "doc": Optional[str],
            "params": OrderedDict[str, ParamVal],
            "returns": Optional[OrderedDict[Literal["return_type"], ParamVal]],
        })
    :type intermediate_repr: ```dict```

    :param identifier: The `$id` of the schema
    :type identifier: ```str```

    :param emit_original_whitespace: Whether to emit original whitespace (in top-level `description`) or strip it out
    :type emit_original_whitespace: ```bool```

    :param emit_default_doc: Whether help/docstring should include 'With default' text
    :type emit_default_doc: ```bool```

    :param word_wrap: Whether to word-wrap. Set `DOCTRANS_LINE_LENGTH` to configure length.
    :type word_wrap: ```bool```

    :return: JSON Schema dict
    :rtype: ```dict```
    """
    del emit_default_doc, word_wrap
    assert isinstance(
        intermediate_repr, dict
    ), "Expected `dict` got `{type_name}`".format(
        type_name=type(intermediate_repr).__name__
    )
    if "$id" in intermediate_repr and "params" not in intermediate_repr:
        return intermediate_repr  # Somehow this function got JSON schema as input
    if identifier is None:
        identifier: str = intermediate_repr.get(
            "$id",
            "https://offscale.io/{}.schema.json".format(
                intermediate_repr.get("name", "INFERRED")
            ),
        )
    required = []
    _param2json_schema_property = partial(param2json_schema_property, required=required)
    properties = dict(
        map(_param2json_schema_property, intermediate_repr["params"].items())
    )

    return {
        "$id": identifier,
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "description": (
            deindent(
                add(
                    *map(
                        partial(
                            docstring,
                            emit_default_doc=True,
                            emit_original_whitespace=emit_original_whitespace,
                            emit_types=True,
                        ),
                        (
                            {
                                "doc": intermediate_repr["doc"],
                                "params": OrderedDict(),
                                "returns": None,
                            },
                            {
                                "doc": "",
                                "params": OrderedDict(),
                                "returns": intermediate_repr["returns"],
                            },
                        ),
                    )
                )
            ).lstrip("\n")
            or None
        ),
        "type": "object",
        "properties": properties,
        "required": required,
    }


def json_schema_file(input_mapping, output_filename):
    """
    Emit `input_mapping`—as JSON schema—into `output_filename`

    :param input_mapping: Import location of mapping/2-tuple collection.
    :type input_mapping: ```Dict[str, AST]```

    :param output_filename: Output file to write to
    :type output_filename: ```str```
    """
    schemas_it = (json_schema(v) for k, v in input_mapping.items())
    schemas = (
        {"schemas": list(schemas_it)} if len(input_mapping) > 1 else next(schemas_it)
    )
    with open(output_filename, "a") as f:
        dump(schemas, f, cls=SetEncoder)


__all__ = ["json_schema", "json_schema_file"]
