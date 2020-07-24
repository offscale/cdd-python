"""
Functions to handle default parameterisation
"""


def extract_default(line, with_default_doc=True):
    """
    Extract the a tuple of (doc, default) from a doc line

    :param line: Example - "dataset. Defaults to mnist"
    :type line: ```str``

    :param with_default_doc: Help/docstring should include 'With default' text
    :type with_default_doc: ```bool``

    :returns: Example - ("dataset. Defaults to mnist", "mnist") if with_default_doc else ("dataset", "mnist")
    :rtype: Tuple[str, str]
    """
    search_str = 'defaults to '
    doc, _, default = (lambda parts: parts if parts[1] else line.partition(search_str.capitalize()))(
        line.partition(search_str))
    return line if with_default_doc else doc.rstrip(), default if len(default) else None


def remove_defaults_from_docstring_structure(docstring_struct, remove_defaults=False):
    """
    Remove "Default of" text from docstring structure

    :param docstring_struct: a dictionary of form
              {
                  'short_description': ...,
                  'long_description': ...,
                  'params': [{'name': ..., 'typ': ..., 'doc': ..., 'default': ..., 'required': ... }, ...],
                  "returns': {'name': ..., 'typ': ..., 'doc': ..., 'default': ..., 'required': ... }
              }
    :type docstring_struct: ```dict```

    :param remove_defaults: Whether to remove default property
    :type remove_defaults: ```bool```

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
        doc, default = extract_default(param['doc'], with_default_doc=False)
        param.update({
            'doc': doc,
            'default': default
        })
        if default is None or remove_defaults:
            del param['default']
        return param

    docstring_struct['params'] = list(map(handle_param, docstring_struct['params']))
    docstring_struct['returns'] = handle_param(docstring_struct['returns'])
    return docstring_struct
