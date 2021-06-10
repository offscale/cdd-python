"""
pkg_utils
"""
from distutils.sysconfig import get_python_lib


def relative_filename(filename, remove_hints=tuple()):
    """
    Remove all the paths which are not relevant

    :param filename: Filename
    :type filename: ```str```

    :param remove_hints: Hints as to what can be removed
    :type remove_hints: ```Tuple[str]```

    :returns: Relative path (if derived) else original
    :rtype: ```str```
    """
    for elem in remove_hints + (get_python_lib(), get_python_lib(prefix="")):
        if filename.startswith(elem):
            return filename[len(elem) + 1 :]
    return filename


__all__ = ["relative_filename"]
