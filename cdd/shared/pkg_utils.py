"""
pkg_utils
"""

from cdd.shared.pure_utils import PY_GTE_3_12

if PY_GTE_3_12:
    from sysconfig import get_paths

    get_python_lib = lambda prefix="", *args, **kwargs: get_paths(*args, **kwargs)[
        prefix or "purelib"
    ]
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
