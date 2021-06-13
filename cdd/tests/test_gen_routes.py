""" Tests for gen_routes subcommand """

import ast
from ast import FunctionDef
from binascii import crc32
from itertools import tee
from os import path, remove
from os.path import extsep
from tempfile import TemporaryDirectory
from unittest import TestCase

from cdd.openapi.gen_routes import gen_routes, upsert_routes
from cdd.routes.parser_utils import get_route_meta
from cdd.tests.mocks.routes import (
    create_route,
    destroy_route,
    read_route,
    route_config,
    route_mock_prelude,
)
from cdd.tests.mocks.sqlalchemy import config_decl_base_str
from cdd.tests.utils_for_tests import unittest_main


def populate_files(tempdir, init_with_crud):
    """
    Populate files in the tempdir

    :param tempdir: Temporary directory
    :type tempdir: ```str```

    :param init_with_crud: Initialise the routes to test against with these CRUD (routes)
    :type init_with_crud: ```Union[Literal['C', 'R'], Literal['C', 'U'], Literal['C', 'D'], Literal['R', 'C'],
                     Literal['R', 'U'], Literal['R', 'D'], Literal['U', 'C'], Literal['U', 'R'],
                     Literal['U', 'D'], Literal['D', 'C'], Literal['D', 'R'], Literal['D', 'U'],
                     Literal['C', 'R', 'U'], Literal['C', 'R', 'D'], Literal['C', 'U', 'R'],
                     Literal['C', 'U', 'D'], Literal['C', 'D', 'R'], Literal['C', 'D', 'U'],
                     Literal['R', 'C', 'U'], Literal['R', 'C', 'D'], Literal['R', 'U', 'C'],
                     Literal['R', 'U', 'D'], Literal['R', 'D', 'C'], Literal['R', 'D', 'U'],
                     Literal['U', 'C', 'R'], Literal['U', 'C', 'D'], Literal['U', 'R', 'C'],
                     Literal['U', 'R', 'D'], Literal['U', 'D', 'C'], Literal['U', 'D', 'R'],
                     Literal['D', 'C', 'R'], Literal['D', 'C', 'U'], Literal['D', 'R', 'C'],
                     Literal['D', 'R', 'U'], Literal['D', 'U', 'C'], Literal['D', 'U', 'R']]```

    :returns: model_path, routes_path
    :rtype: ```Tuple[str, str]```
    """
    model_path = path.join(tempdir, "model{extsep}py".format(extsep=extsep))
    routes_path = path.join(tempdir, "routes{extsep}py".format(extsep=extsep))
    open(path.join(tempdir, "__init__{extsep}py".format(extsep=extsep)), "a").close()

    model = "\n".join(
        (
            "Base = JSON = object",
            "def Column(a, doc, default, nullable, primary_key): pass",
            "def Enum(a, b, name): pass",
            config_decl_base_str,
        )
    )
    with open(model_path, "wt") as f:
        f.write(model)

    routes = "\n".join(
        filter(
            None,
            (
                route_mock_prelude,
                create_route if "C" in init_with_crud else None,
                read_route if "R" in init_with_crud else None,
                destroy_route if "D" in init_with_crud else None,
            ),
        )
    )

    with open(routes_path, "wt") as f:
        f.write(routes)

    return model_path, routes_path


class TestGenRoutes(TestCase):
    """Test class for gen_routes.py"""

    def further_tests(self, mod):
        """
        Callback to run after the initial generic tests are done

        :param mod: Parsed AST of the generated + augmented file
        :type mod: ```Module```
        """
        self.assertEqual(len(mod.body), 4)
        self.assertTupleEqual(
            tuple(get_route_meta(mod)),
            (
                ("create", "rest_api", "/api/config", "post"),
                ("read", "rest_api", "/api/config/:dataset_name", "get"),
                ("destroy", "rest_api", "/api/config/:dataset_name", "delete"),
            ),
        )

    def test_gen_routes_no_change(self) -> None:
        """Tests `gen_routes` when routes are identical in app name, route path and method"""

        self.gen_routes_tester(
            approach="update",
            init_with_crud="CRD",
            upsert_crud="CRD",
            file_change=False,
            further_tests=self.further_tests,
        )

    def test_gen_routes_update(self) -> None:
        """Tests `gen_routes` when routes do exists"""

        self.gen_routes_tester(
            approach="update",
            init_with_crud="CRD",
            upsert_crud="CRD",
            file_change=False,
            further_tests=self.further_tests,
        )

    def test_gen_routes_update_missing(self) -> None:
        """Tests `gen_routes` when routes do exists, but `destroy` route (DELETE method) is missing"""

        self.gen_routes_tester(
            approach="update",
            init_with_crud="CR",
            upsert_crud="CRD",
            file_change=True,
            further_tests=self.further_tests,
        )

    def test_gen_routes_insert(self) -> None:
        """Tests `gen_routes` when routes do not exist"""

        def _further_tests(mod):
            """
            Callback to run after the initial generic tests are done

            :param mod: Parsed AST of the generated + augmented file
            :type mod: ```Module```
            """
            self.assertEqual(len(mod.body), 5)
            self.assertTupleEqual(
                tuple(get_route_meta(mod)),
                (
                    ("create", "rest_api", "/api/config", "post"),
                    ("read", "rest_api", "/api/config/:dataset_name", "get"),
                    ("destroy", "rest_api", "/api/config/:dataset_name", "delete"),
                ),
            )

        self.gen_routes_tester(
            approach="insert",
            init_with_crud=iter(()),
            upsert_crud="CRD",
            file_change=True,
            further_tests=_further_tests,
        )

    def gen_routes_tester(
        self, approach, init_with_crud, upsert_crud, file_change, further_tests
    ):
        """
        :param approach: How to upsert
        :type approach: ```Literal["insert", "update"]```

        :param init_with_crud: Initialise the routes to test against with these CRUD (routes)
        :type init_with_crud: ```Union[Literal['C', 'R'], Literal['C', 'U'], Literal['C', 'D'], Literal['R', 'C'],
                         Literal['R', 'U'], Literal['R', 'D'], Literal['U', 'C'], Literal['U', 'R'],
                         Literal['U', 'D'], Literal['D', 'C'], Literal['D', 'R'], Literal['D', 'U'],
                         Literal['C', 'R', 'U'], Literal['C', 'R', 'D'], Literal['C', 'U', 'R'],
                         Literal['C', 'U', 'D'], Literal['C', 'D', 'R'], Literal['C', 'D', 'U'],
                         Literal['R', 'C', 'U'], Literal['R', 'C', 'D'], Literal['R', 'U', 'C'],
                         Literal['R', 'U', 'D'], Literal['R', 'D', 'C'], Literal['R', 'D', 'U'],
                         Literal['U', 'C', 'R'], Literal['U', 'C', 'D'], Literal['U', 'R', 'C'],
                         Literal['U', 'R', 'D'], Literal['U', 'D', 'C'], Literal['U', 'D', 'R'],
                         Literal['D', 'C', 'R'], Literal['D', 'C', 'U'], Literal['D', 'R', 'C'],
                         Literal['D', 'R', 'U'], Literal['D', 'U', 'C'], Literal['D', 'U', 'R']]```

        :param upsert_crud: Upsert these CRUD (routes)
        :type upsert_crud: ```Union[Literal['C', 'R'], Literal['C', 'U'], Literal['C', 'D'], Literal['R', 'C'],
                         Literal['R', 'U'], Literal['R', 'D'], Literal['U', 'C'], Literal['U', 'R'],
                         Literal['U', 'D'], Literal['D', 'C'], Literal['D', 'R'], Literal['D', 'U'],
                         Literal['C', 'R', 'U'], Literal['C', 'R', 'D'], Literal['C', 'U', 'R'],
                         Literal['C', 'U', 'D'], Literal['C', 'D', 'R'], Literal['C', 'D', 'U'],
                         Literal['R', 'C', 'U'], Literal['R', 'C', 'D'], Literal['R', 'U', 'C'],
                         Literal['R', 'U', 'D'], Literal['R', 'D', 'C'], Literal['R', 'D', 'U'],
                         Literal['U', 'C', 'R'], Literal['U', 'C', 'D'], Literal['U', 'R', 'C'],
                         Literal['U', 'R', 'D'], Literal['U', 'D', 'C'], Literal['U', 'D', 'R'],
                         Literal['D', 'C', 'R'], Literal['D', 'C', 'U'], Literal['D', 'R', 'C'],
                         Literal['D', 'R', 'U'], Literal['D', 'U', 'C'], Literal['D', 'U', 'R']]```

        :param file_change: Whether to expect a file change
        :type file_change: ```bool```

        :param further_tests: Run this function, which takes the parsed AST of the generated + augmented file
        :type further_tests: ```Callback[[Module], None]```
        """
        with TemporaryDirectory() as tempdir:
            model_path, routes_path = populate_files(tempdir, init_with_crud)
            with open(routes_path, "rb") as f:
                init_hash = crc32(f.read())

            if approach == "insert":
                remove(routes_path)
                init_hash = None

            routes, primary_key = gen_routes(
                app=route_config["app"],
                route=route_config["route"],
                model_path=model_path,
                model_name=route_config["name"],
                crud=upsert_crud,
            )

            routes, testable_routes = tee(routes)
            testable_routes = tuple(testable_routes)

            self.assertIsInstance(testable_routes, tuple)
            self.assertGreaterEqual(len(testable_routes), 1)
            self.assertIsInstance(testable_routes[0], FunctionDef)

            self.assertIsNone(
                upsert_routes(
                    app=route_config["app"],
                    route=route_config["route"],
                    routes=routes,
                    routes_path=routes_path,
                    primary_key=primary_key,
                )
            )
            with open(routes_path, "rb") as f:
                routes_bin = f.read()

            getattr(self, "assertNotEqual" if file_change else "assertEqual")(
                init_hash,
                crc32(routes_bin),
                "File after processing compared to initial file",
            )
            parsed_routes = ast.parse(routes_bin)

            further_tests(parsed_routes)


unittest_main()
