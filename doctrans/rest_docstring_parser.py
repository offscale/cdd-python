"""
ReST docstring parser.
Translates from the [ReST docstring format (Sphinx)](
  https://sphinx-rtd-tutorial.readthedocs.io/en/latest/docstrings.html#the-sphinx-docstring-format)
"""

# Stolen from https://raw.githubusercontent.com/openstack/rally/ab365e9/rally/common/plugin/info.py
#
# New things:
# - Added docstrings to all functions and the module itself
# - Handled default parameters
# - Handle type parameters
# - Remove MixIn
#
# Copyright 2020: Copyright 2020 Samuel Marks (for Offscale.io)
# Copyright 2015: Mirantis Inc.
# All Rights Reserved.
#
#    Licensed under the Apache License, Version 2.0 (the "License"); you may
#    not use this file except in compliance with the License. You may obtain
#    a copy of the License at
#
#         http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
#    WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
#    License for the specific language governing permissions and limitations
#    under the License.

import re
import sys

from doctrans.defaults_utils import extract_default

PARAM_OR_RETURNS_REGEX = re.compile(":(?:param|returns?)")
RETURNS_REGEX = re.compile(":returns?: (?P<doc>.*)", re.S)
PARAM_REGEX = re.compile(r":param (?P<name>[\w]+): (?P<doc>.*?)"
                         r"(?:(?=:param)|(?=:return)|(?=:raises)|\Z)", re.S)


def trim(docstring):
    """
    trim function from PEP-257

    :param docstring: the docstring
    :type docstring: ```str```

    :return: Trimmed input
    :rtype ```str```
    """
    # Convert tabs to spaces (following the normal Python rules)
    # and split into a list of lines:
    lines = docstring.expandtabs().splitlines()
    # Determine minimum indentation (first line doesn't count):
    indent = sys.maxsize
    for line in lines[1:]:
        stripped = line.lstrip()
        if stripped:
            indent = min(indent, len(line) - len(stripped))
    # Remove indentation (first line is special):
    trimmed = [lines[0].strip()]
    if indent < sys.maxsize:
        for line in lines[1:]:
            trimmed.append(line[indent:].rstrip())
    # Strip off trailing and leading blank lines:
    while trimmed and not trimmed[-1]:
        trimmed.pop()
    while trimmed and not trimmed[0]:
        trimmed.pop(0)

    # Current code/unittests expects a line return at
    # end of multiline docstrings
    # workaround expected behavior from unittests
    if '\n' in docstring:
        trimmed.append("")

    # Return a single string:
    return '\n'.join(trimmed)


def reindent(s):
    """
    Reindent the input string

    :param s: input string
    :type s: ```str```

    :return: reindented—and stripped—string
    :rtype: ```str```
    """
    return '\n'.join(line.strip() for line in s.strip().split('\n'))


def doc_to_type_doc(name, doc, emit_default_doc=True):
    """
    Convert input string to default and type (if those are present)

    :param name: name
    :type name: ```str```

    :param doc: doc
    :type doc: ```str```

    :param emit_default_doc: Whether help/docstring should include 'With default' text
    :type emit_default_doc: ```bool``

    :return: dict of shape {'name': ..., 'typ': ..., 'doc': ..., 'default': ..., 'required': ... }
    :rtype: ```dict```
    """
    doc = trim(doc).splitlines()
    docs, typ, default = [], [], None
    for line in doc:
        if line.startswith(':type'):
            line = line[len(':type '):]
            colon_at = line.find(':')
            found_name = line[:colon_at]
            assert name == found_name, '{!r} != {!r}'.format(name, found_name)
            line = line[colon_at + 2:]
            typ.append(line[3:-3] if line.startswith('```') and line.endswith('```') else line)
        elif len(typ):
            typ.append(line)
        else:
            doc, default = extract_default(line, emit_default_doc=emit_default_doc)
            docs.append(doc)
    return dict(doc='\n'.join(docs), **{'default': default} if default else {},
                **{'typ': (lambda typ: 'dict' if typ.endswith('kwargs') else typ)('\n'.join(typ))}
                if len(typ) else {})


def parse_docstring(docstring, emit_default_doc=True):
    """Parse the docstring into its components.

    :param docstring: the docstring
    :type docstring: ```Union[str, dict]```

    :param emit_default_doc: Whether help/docstring should include 'With default' text
    :type emit_default_doc: ```bool``

    :returns: a dictionary of form
              {
                  'short_description': ...,
                  'long_description': ...,
                  'params': [{'name': ..., 'typ': ..., 'doc': ..., 'default': ..., 'required': ... }, ...],
                  "returns': {'name': ..., 'typ': ..., 'doc': ..., 'default': ..., 'required': ... }
              }
    :rtype ```dict```
    """

    short_description = long_description = returns = ""
    params = []

    if docstring:
        docstring = trim(docstring.lstrip('\n'))

        lines = docstring.split('\n', 1)
        short_description = lines[0]

        if len(lines) > 1:
            long_description = lines[1].strip()

            params_returns_desc = None

            match = PARAM_OR_RETURNS_REGEX.search(long_description)
            if match:
                long_desc_end = match.start()
                params_returns_desc = long_description[long_desc_end:].strip()
                long_description = long_description[:long_desc_end].rstrip()

            if params_returns_desc:
                params = [
                    dict(name=name, **doc_to_type_doc(name, doc, emit_default_doc=emit_default_doc))
                    for name, doc in PARAM_REGEX.findall(params_returns_desc)
                ]

                returns = extract_return_params(params_returns_desc, emit_default_doc=emit_default_doc)

    return {
        'short_description': short_description,
        'long_description': long_description,
        'params': params,
        'returns': returns
    }


def extract_return_params(params_returns_desc, emit_default_doc):
    """
    Parse out the doc, type, and defaults from the ReST doctype

    :param params_returns_desc: return text from docstring
    :type params_returns_desc: ```str```

    :param emit_default_doc: Whether help/docstring should include 'With default' text
    :type emit_default_doc: ```bool``

    :return: dict of shape {'typ': ..., 'doc': ..., 'default': ...}
    :rdict: ```dict```
    """
    match = RETURNS_REGEX.search(params_returns_desc)
    returns = None
    if match:
        returns = reindent(match.group('doc'))
    if returns:
        r_dict = {'doc': ''}
        for idx, char in enumerate(returns):
            if char == ':':
                r_dict['typ'] = returns[idx + len(':rtype:'):].strip()
                if r_dict['typ'].startswith('```') and r_dict['typ'].endswith('```'):
                    r_dict['typ'] = r_dict['typ'][3:-3]
                break
            else:
                r_dict['doc'] += char
        doc, default = extract_default(r_dict['doc'].rstrip('\n'), emit_default_doc=emit_default_doc)
        r_dict.update({
            'name': 'return_type',
            'doc': doc,
            'default': default
        })
        if not r_dict.get('default', True):
            del r_dict['default']
        returns = r_dict
    return returns
