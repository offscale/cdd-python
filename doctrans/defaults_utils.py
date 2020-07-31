"""
Functions to handle default parameterisation
"""


def extract_default(line, emit_default_doc=True):
    """
    Extract the a tuple of (doc, default) from a doc line

    :param line: Example - "dataset. Defaults to mnist"
    :type line: ```str``

    :param emit_default_doc: Whether help/docstring should include 'With default' text
    :type emit_default_doc: ```bool``

    :returns: Example - ("dataset. Defaults to mnist", "mnist") if emit_default_doc else ("dataset", "mnist")
    :rtype: Tuple[str, Optional[str]]
    """
    search_str = 'defaults to '
    if line is None:
        return line, line
    doc, _, default = (lambda parts: parts if parts[1] else line.partition(search_str.capitalize()))(
        line.partition(search_str)
    )
    return line if emit_default_doc else doc.rstrip(';\n, '), default if len(default) else None


def remove_defaults_from_docstring_structure(docstring_structure, emit_defaults=True):
    """
    Remove "Default of" text from docstring structure

    :param docstring_structure: a dictionary of form
              {
                  'short_description': ...,
                  'long_description': ...,
                  'params': [{'name': ..., 'typ': ..., 'doc': ..., 'default': ..., 'required': ... }, ...],
                  "returns': {'name': ..., 'typ': ..., 'doc': ..., 'default': ..., 'required': ... }
              }
    :type docstring_structure: ```dict```

    :param emit_defaults: Whether to emit default property
    :type emit_defaults: ```bool```

    :returns: a dictionary of form
              {
                  'short_description': ...,
                  'long_description': ...,
                  'params': [{'name': ..., 'typ': ..., 'doc': ..., 'default': ..., 'required': ... }, ...],
                  "returns': {'name': ..., 'typ': ..., 'doc': ..., 'default': ..., 'required': ... }
              }
    :rtype: ```dict```
    """

    def handle_param(param):
        """
        :param param: dict of shape {'name': ..., 'typ': ..., 'doc': ..., 'default': ..., 'required': ... }
        :type param: ```dict```

        :returns: dict of shape {'name': ..., 'typ': ..., 'doc': ..., 'default': ..., 'required': ... }
        :rtype: ```dict```
        """
        doc, default = extract_default(param['doc'], emit_default_doc=False)
        param.update({
            'doc': doc,
            'default': default
        })
        if default is None or not emit_defaults:
            del param['default']
        return param

    docstring_structure['params'] = list(map(handle_param, docstring_structure['params']))
    docstring_structure['returns'] = handle_param(docstring_structure['returns'])
    return docstring_structure


def set_default_doc(param, emit_default_doc=True):
    """
    Emit param with 'doc' set to include 'Defaults'

    :param param: dict of shape {'name': ..., 'typ': ..., 'doc': ..., 'default': ..., 'required': ... }
    :type param: ```dict```

    :param emit_default_doc: Whether help/docstring should include 'With default' text
    :type emit_default_doc: ```bool``

    :returns: Same shape as input but with Default append to doc.
    :rtype: ```dict``
    """

    if emit_default_doc and 'default' in param and 'Defaults' not in param['doc'] and 'defaults' not in param['doc']:
        param['doc'] = '{doc} Defaults to {default}'.format(
            doc=(param['doc'] if param['doc'][-1] in frozenset(('.', ','))
                 else '{doc}.'.format(doc=param['doc'])),
            default=param['default']
        )

    return param
