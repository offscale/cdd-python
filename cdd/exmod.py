"""
Not a dead module
"""

from os import makedirs, path


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
    if path.isdir(module):
        raise NotImplementedError("--module as directory")


__all__ = ["exmod"]
