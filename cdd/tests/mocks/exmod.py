""" Exmod mocks """

setup_py_mock = '''{encoding}

"""
setup.py implementation, interesting because it parsed the first __init__.py and
    extracts the `__author__` and `__version__`
"""

from ast import Assign, Constant, Str, parse
from operator import attrgetter
from os import path
from os.path import extsep

from setuptools import find_packages, setup

package_name = {package_name!r}


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
                filter(lambda node: isinstance(node, Assign), parsed_init.body),
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

__all__ = ["setup_py_mock"]
