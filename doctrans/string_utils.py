def extract_default(line, with_default_doc=True):
    search_str = 'defaults to '
    doc, _, default = (lambda parts: parts if parts[1] else line.partition(search_str.capitalize()))(
        line.partition(search_str))
    return line if with_default_doc else doc.rstrip(), default if len(default) else None


def remove_defaults_from_docstring_structure(docstring_struct):
    def one(param):
        doc, default = extract_default(param['doc'], with_default_doc=False)
        param.update({
            'doc': doc,
            'default': default
        })
        if default is None:
            del param['default']
        return param

    docstring_struct['params'] = list(map(one, docstring_struct['params']))
    docstring_struct['returns'] = one(docstring_struct['returns'])
    return docstring_struct
