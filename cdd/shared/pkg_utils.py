"""
pkg_utils
"""

from cdd.shared.pure_utils import PY_GTE_3_12

if PY_GTE_3_12:
    import os
    import sys
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
    from distutils.sysconfig import get_python_lib


def relative_filename(filename, remove_hints=tuple()):
    """
    Remove all the paths which are not relevant

    :param filename: Filename
    :type filename: ```str```

    :param remove_hints: Hints as to what can be removed
    :type remove_hints: ```tuple[str, ...]```

    :return: Relative `os.path` (if derived) else original
    :rtype: ```str```
    """
    _filename: str = filename.casefold()
    lib = get_python_lib(), get_python_lib(prefix="")  # type: tuple[str, str]
    return next(
        map(
            lambda elem: filename[len(elem) + 1 :],
            filter(
                lambda elem: _filename.startswith(elem.casefold()), remove_hints + lib
            ),
        ),
        filename,
    )


__all__ = ["get_python_lib", "relative_filename"]  # type: list[str]
