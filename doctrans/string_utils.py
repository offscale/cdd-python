def extract_default(line):
    search_str = 'defaults to '
    doc, _, default = (lambda parts: parts if parts[1] else line.partition(search_str.capitalize()))(
        line.partition(search_str))
    # return doc.rstrip(), default
    return line, default if len(default) else None
