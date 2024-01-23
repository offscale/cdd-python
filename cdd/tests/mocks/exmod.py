""" Exmod mocks """

from ast import Assign, ImportFrom, List, Load, Module, Name, Store, alias
from itertools import chain
from operator import itemgetter
from os import environ

from cdd.shared.ast_utils import maybe_type_comment, set_value
from cdd.shared.pure_utils import ENCODING
from cdd.shared.source_transformer import to_code

setup_py_mock: str = '''{encoding}

"""
setup.py implementation, interesting because it parsed the first __init__.py and
    extracts the `__author__` and `__version__`
"""

from sys import version_info

if version_info[:2] < (3, 8):
    from ast import Assign, Str, parse

    Constant = type("_Never", tuple(), {{}})
else:
    from ast import Assign, Constant, parse

    Str = type("_Never", tuple(), {{}})

from ast import Name
from operator import attrgetter
from os import path
from os.path import extsep

from setuptools import find_packages, setup

package_name = {package_name!r}
module_name = {module_name!r}


def main():
    """Main function for setup.py; this actually does the installation"""
    with open(
        path.join(
            path.abspath(path.dirname(__file__)),
            package_name,
            "__init__{{extsep}}py".format(extsep=extsep),
        )
    ) as f:
        parsed_init = parse(f.read())

    __author__, __version__ = map(
        lambda node: node.value if isinstance(node, Constant) else node.s,
        filter(
            lambda node: isinstance(node, (Constant, Str)),
            map(
                attrgetter("value"),
                filter(
                    lambda node: isinstance(node, Assign)
                    and any(
                        filter(
                            lambda name: isinstance(name, Name)
                            and name.id
                            in frozenset(("__author__", "__version__")),
                            node.targets,
                        )
                    ),
                    parsed_init.body,
                ),
            ),
        ),
    )

    setup(
        name=package_name,
        author=__author__,
        version=__version__,
        packages=find_packages(),
        package_dir={{package_name: package_name}},
        python_requires=">=3.6",
    )


def setup_py_main():
    """Calls main if `__name__ == '__main__'`"""
    if __name__ == "__main__":
        main()


setup_py_main()
'''


def create_init_mock(package_root_name, module_hierarchy):
    """
    Create mock for __init__.py file with `__all__` set

    :param package_root_name: package root name
    :type package_root_name: ```str```

    :param module_hierarchy: (
        (parent_name, parent_dir),
        (child_name, child_dir),
        (grandchild_name, grandchild_dir),
    )
    :type module_hierarchy: ```Tuple[Tuple[str, str], Tuple[str, str], Tuple[str, str]]```

    :return: str for __init__.py file with `__all__` set
    :rtype: ```str```
    """
    return "{encoding}\n\n" "{mod}\n".format(
        encoding=ENCODING,
        mod=to_code(
            Module(
                body=[
                    ImportFrom(
                        module=".".join((package_root_name, "gen")),
                        names=[
                            alias(
                                "*",
                                None,
                                identifier=None,
                                identifier_name=None,
                            )
                        ],
                        level=0,
                        identifier=None,
                    ),
                    Assign(
                        targets=[
                            Name(
                                "__author__",
                                Store(),
                                lineno=None,
                                col_offset=None,
                            )
                        ],
                        value=set_value(environ.get("CDD_AUTHOR", "Samuel Marks")),
                        expr=None,
                        lineno=None,
                        **maybe_type_comment,
                    ),
                    Assign(
                        targets=[
                            Name(
                                "__version__",
                                Store(),
                                lineno=None,
                                col_offset=None,
                            )
                        ],
                        value=set_value(environ.get("CDD_VERSION", "0.0.0")),
                        expr=None,
                        lineno=None,
                        **maybe_type_comment,
                    ),
                    Assign(
                        targets=[
                            Name(
                                "__all__",
                                Store(),
                                lineno=None,
                                col_offset=None,
                            )
                        ],
                        value=List(
                            ctx=Load(),
                            elts=list(
                                map(
                                    set_value,
                                    chain.from_iterable(
                                        (
                                            (
                                                "__author__",
                                                "__version__",
                                            ),
                                            map(
                                                itemgetter(0),
                                                module_hierarchy,
                                            ),
                                        )
                                    ),
                                )
                            ),
                            expr=None,
                        ),
                        expr=None,
                        lineno=None,
                        **maybe_type_comment,
                    ),
                ],
                type_ignores=[],
                stmt=None,
            )
        ),
    )


__all__ = ["create_init_mock", "setup_py_mock"]  # type: list[str]
