# -*- coding: utf-8 -*-

"""
setup.py implementation, interesting because it parsed the first __init__.py and
    extracts the `__author__` and `__version__`
"""

import sys
from ast import Assign, Constant, Name, parse
from functools import partial
from operator import attrgetter
from os import path
from os.path import extsep

from setuptools import find_packages, setup

if sys.version_info[:2] >= (3, 12):
    import os
    from sysconfig import _BASE_EXEC_PREFIX as BASE_EXEC_PREFIX
    from sysconfig import _BASE_PREFIX as BASE_PREFIX
    from sysconfig import _EXEC_PREFIX as EXEC_PREFIX
    from sysconfig import _PREFIX as PREFIX
    from sysconfig import get_python_version

    Str = type(
        "_Never",
        tuple(),
        {
            "__init__": lambda s=None, n=None, constant_value=None, string=None, col_offset=None, lineno=None: s
            or n
        },
    )

    def is_virtual_environment():
        """
        Whether one is in a virtual environment
        """
        return sys.base_prefix != sys.prefix or hasattr(sys, "real_prefix")

    def get_python_lib(plat_specific=0, standard_lib=0, prefix=None):
        """Return the directory containing the Python library (standard or
        site additions).

        If 'plat_specific' is true, return the directory containing
        platform-specific modules, i.e. any module from a non-pure-Python
        module distribution; otherwise, return the platform-shared library
        directory.  If 'standard_lib' is true, return the directory
        containing standard Python library modules; otherwise, return the
        directory for site-specific modules.

        If 'prefix' is supplied, use it instead of sys.base_prefix or
        sys.base_exec_prefix -- i.e., ignore 'plat_specific'.
        """
        is_default_prefix = not prefix or os.path.normpath(prefix) in (
            "/usr",
            "/usr/local",
        )
        prefix = (
            prefix or plat_specific and (BASE_EXEC_PREFIX or BASE_PREFIX)
            if standard_lib
            else (EXEC_PREFIX or PREFIX)
        )

        class DistutilsPlatformError(Exception):
            """DistutilsPlatformError"""

        assert os.name in frozenset(("posix", "nt")), DistutilsPlatformError(
            "I don't know where Python installs its library "
            "on platform '{}'".format(os.name)
        )
        return (
            (
                # plat_specific or standard_lib:
                # Platform-specific modules (any module from a non-pure-Python
                # module distribution) or standard Python library modules.
                # else:
                # Pure Python
                lambda libpython: (
                    libpython
                    if standard_lib
                    else (
                        os.path.join(prefix, "lib", "python3", "dist-packages")
                        if is_default_prefix and not is_virtual_environment()
                        else os.path.join(libpython, "site-packages")
                    )
                )
            )(
                os.path.join(
                    prefix,
                    sys.platlibdir if plat_specific or standard_lib else "lib",
                    "python" + get_python_version(),
                )
            )
            if os.name == "posix"
            else (
                os.path.join(prefix, "Lib")
                if standard_lib
                else os.path.join(prefix, "Lib", "site-packages")
            )
        )

else:
    from ast import Str
    from distutils.sysconfig import get_python_lib

package_name = "cdd"

with open(
    path.join(path.dirname(__file__), "README{extsep}md".format(extsep=extsep)),
    "rt",
    encoding="utf-8",
) as fh:
    long_description = fh.read()


def to_funcs(*paths):
    """
    Produce function tuples that produce the local and install dir, respectively.

    :param paths: one or more str, referring to relative folder names
    :type paths: ```*paths```

    :return: 2 functions
    :rtype: ```tuple[Callable[Optional[List[str]], str], Callable[Optional[List[str]], str]]```
    """
    return (
        partial(path.join, path.dirname(__file__), package_name, *paths),
        partial(path.join, get_python_lib(prefix=""), package_name, *paths),
    )


def main():
    """Main function for setup.py; this actually does the installation"""
    with open(
        path.join(
            path.abspath(path.dirname(__file__)),
            package_name,
            "__init__{extsep}py".format(extsep=extsep),
        )
    ) as f:
        parsed_init = parse(f.read())

    __author__, __version__, __description__ = map(
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
                            in frozenset(
                                ("__author__", "__version__", "__description__")
                            ),
                            node.targets,
                        )
                    ),
                    parsed_init.body,
                ),
            ),
        ),
    )

    setup(
        name="python-{}".format(package_name),
        author=__author__,
        author_email="807580+SamuelMarks@users.noreply.github.com",
        version=__version__,
        description=__description__,
        long_description=long_description,
        long_description_content_type="text/markdown",
        url="https://github.com/offscale/{}-python".format(package_name),
        install_requires=["pyyaml"],
        test_suite="{}{}tests".format(package_name, path.extsep),
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
            "Programming Language :: Python :: 3.6",
            "Programming Language :: Python :: 3.7",
            "Programming Language :: Python :: 3.8",
            "Programming Language :: Python :: 3.9",
            "Programming Language :: Python :: 3.10",
            "Programming Language :: Python :: 3.11",
            "Programming Language :: Python :: 3.12",
            "Programming Language :: Python :: Implementation",
            "Programming Language :: Python :: Implementation :: CPython",
            "Topic :: Scientific/Engineering :: Interface Engine/Protocol Translator",
            "Topic :: Software Development",
            "Topic :: Software Development :: Build Tools",
            "Topic :: Software Development :: Code Generators",
            "Topic :: Software Development :: Compilers",
            "Topic :: Software Development :: Pre-processors",
        ],
        python_requires=">=3.6",
    )


def setup_py_main():
    """Calls main if `__name__ == '__main__'`"""
    if __name__ == "__main__":
        main()


setup_py_main()
