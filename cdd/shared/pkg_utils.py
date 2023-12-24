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
        if prefix is None:
            if standard_lib:
                prefix = plat_specific and BASE_EXEC_PREFIX or BASE_PREFIX
            else:
                prefix = plat_specific and EXEC_PREFIX or PREFIX

        if os.name == "posix":
            if plat_specific or standard_lib:
                # Platform-specific modules (any module from a non-pure-Python
                # module distribution) or standard Python library modules.
                libdir = sys.platlibdir
            else:
                # Pure Python
                libdir = "lib"
            libpython = os.path.join(prefix, libdir, "python" + get_python_version())
            if standard_lib:
                return libpython
            elif is_default_prefix and not is_virtual_environment():
                return os.path.join(prefix, "lib", "python3", "dist-packages")
            else:
                return os.path.join(libpython, "site-packages")
        elif os.name == "nt":
            if standard_lib:
                return os.path.join(prefix, "Lib")
            else:
                return os.path.join(prefix, "Lib", "site-packages")
        else:

            class DistutilsPlatformError(Exception):
                """DistutilsPlatformError"""

            raise DistutilsPlatformError(
                "I don't know where Python installs its library "
                "on platform '%s'" % os.name
            )

else:
    from distutils.sysconfig import get_python_lib


def relative_filename(filename, remove_hints=tuple()):
    """
    Remove all the paths which are not relevant

    :param filename: Filename
    :type filename: ```str```

    :param remove_hints: Hints as to what can be removed
    :type remove_hints: ```tuple[str]```

    :return: Relative `os.path` (if derived) else original
    :rtype: ```str```
    """
    _filename: str = filename.casefold()
    lib = get_python_lib(), get_python_lib(prefix="")
    for elem in remove_hints + lib:
        if _filename.startswith(elem.casefold()):
            return filename[len(elem) + 1 :]
    return filename


__all__ = ["relative_filename"]  # type: list[str]
