"""
Not a dead module
"""
from functools import partial
from importlib import import_module
from inspect import ismodule
from itertools import chain, filterfalse
from operator import methodcaller
from os import makedirs, path

from cdd import parse
from cdd.pure_utils import no_magic_dir2attr, pp, rpartial
from cdd.tests.utils_for_tests import module_from_file


def exmod(module, emit, blacklist, whitelist, output_directory):
    """
    Expose module as `emit` types into `output_directory`

    :param module: Module name or path
    :type module: ```str```

    :param emit: What type(s) to generate.
    :type emit: ```List[Literal["argparse", "class", "function", "sqlalchemy", "sqlalchemy_table"]]```

    :param blacklist: Modules/FQN to omit. If unspecified will emit all (unless whitelist).
    :type blacklist: ```List[str]```

    :param whitelist: Modules/FQN to emit. If unspecified will emit all (minus blacklist).
    :type whitelist: ```List[str]```

    :param output_directory: Where to place the generated exposed interfaces to the given `--module`.
    :type output_directory: ```str```
    """
    if not path.isdir(output_directory):
        makedirs(output_directory)
    if blacklist:
        raise NotImplementedError("blacklist")
    elif whitelist:
        raise NotImplementedError("whitelist")
    module = (
        partial(module_from_file, module_name=path.basename(module))
        if path.isdir(module)
        else import_module
    )(module)

    def get_module_contents(obj):
        """
        Helper function to get the recursive inner module contents

        :param obj: Something to `dir` on
        :type obj: ```Any```

        :returns: Values (could be modules, classes, and whatever other symbols are exposed)
        :rtype: ```Generator[Any]```
        """
        return (
            v
            for val in map(
                methodcaller("values"),
                map(no_magic_dir2attr, no_magic_dir2attr(obj).values()),
            )
            for v in val
        )

    pp(
        tuple(
            map(
                parse.class_,
                filter(
                    rpartial(isinstance, type),
                    chain.from_iterable(
                        filterfalse(ismodule, get_module_contents(v))
                        if ismodule(v)
                        else (v,)
                        for v in get_module_contents(module)
                    ),
                ),
            )
        )[:-1]  # last element is already included
    )


__all__ = ["exmod"]
