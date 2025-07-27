#!/usr/bin/env python

"""Generate the code reference pages and navigation."""

from pathlib import Path

import mkdocs_gen_files

nav = mkdocs_gen_files.Nav()

# Project root is two levels up from this script's directory
root = Path(__file__).parent.parent

# Source code is in the 'cdd' directory
src = root / "cdd"

# Create a single API reference page
with mkdocs_gen_files.open("api.md", "w") as fd:
    fd.write("# API Reference\n\n")

    for path in sorted(src.rglob("*.py")):
        module_path = path.relative_to(root).with_suffix("")

        parts = list(module_path.parts)

        if "tests" in parts:
            continue

        if parts[-1] == "__init__":
            parts = parts[:-1]

        if not parts or parts[-1] == "__main__":
            continue

        ident = ".".join(parts)

        fd.write(f"## `{ident}`\n\n")
        fd.write(f"::: {ident}\n\n")

nav["Home"] = "index.md"
nav["API Reference"] = "api.md"

with mkdocs_gen_files.open("SUMMARY.md", "w") as nav_file:
    nav_file.writelines(nav.build_literate_nav())
