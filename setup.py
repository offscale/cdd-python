# -*- coding: utf-8 -*-

"""
setup.py implementation, interesting because it parsed the first __init__.py and
    extracts the `__author__` and `__version__`
"""

from ast import parse
from distutils.sysconfig import get_python_lib
from functools import partial
from os import listdir, path

from setuptools import find_packages, setup

package_name = "doctrans"


def to_funcs(*paths):
    """
    Produce function tuples that produce the local and install dir, respectively.

    :param paths: one or more str, referring to relative folder names
    :type paths: ```*paths```

    :return: 2 functions
    :rtype: ```Tuple[Callable[Optional[List[str]], str], Callable[Optional[List[str]], str]]```
    """
    return (
        partial(path.join, path.dirname(__file__), package_name, *paths),
        partial(path.join, get_python_lib(prefix=""), package_name, *paths),
    )


def main():
    """ Main function for setup.py; this actually does the installation """
    with open(
        path.join(path.abspath(path.dirname(__file__)), package_name, "__init__.py")
    ) as f:
        __author__, __version__ = map(
            lambda buf: next(map(lambda e: e.value.s, parse(buf).body)),
            filter(
                lambda line: line.startswith("__version__")
                or line.startswith("__author__"),
                f,
            ),
        )

    _data_join, _data_install_dir = to_funcs("_data")

    setup(
        name=package_name,
        author=__author__,
        version=__version__,
        install_requires=["pyyaml"],
        test_suite=package_name + ".tests",
        packages=find_packages(),
        package_dir={package_name: package_name},
        classifiers=[
            "Development Status :: 3 - Alpha",
            "Environment :: Console",
            "Intended Audience :: Developers",
            "License :: OSI Approved",
            "License :: OSI Approved :: Apache Software License",
            "License :: OSI Approved :: MIT License",
            "Natural Language :: English",
            "Operating System :: OS Independent",
            "Programming Language :: Python :: 3 :: Only",
            "Programming Language :: Python :: 3.8",
            "Programming Language :: Python :: Implementation",
            "Topic :: Scientific/Engineering :: Interface Engine/Protocol Translator",
            "Topic :: Software Development",
            "Topic :: Software Development :: Build Tools",
            "Topic :: Software Development :: Code Generators",
            "Topic :: Software Development :: Compilers",
            "Topic :: Software Development :: Pre-processors",
            "Topic :: Internet :: WWW/HTTP :: HTTP Servers",
        ],
        data_files=[
            (_data_install_dir(), list(map(_data_join, listdir(_data_join()))))
        ],
    )


def setup_py_main():
    """ Calls main if `__name__ == '__main__'` """
    if __name__ == "__main__":
        main()


setup_py_main()
