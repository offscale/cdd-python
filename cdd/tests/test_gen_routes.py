""" Tests for gen_routes subcommand """
from itertools import tee
from os import path, remove
from tempfile import TemporaryDirectory
from unittest import TestCase

from _ast import FunctionDef

from cdd.openapi.gen_routes import gen_routes, upsert_routes
from cdd.tests.mocks.routes import (
    create_route,
    destroy_route,
    read_route,
    route_config,
    route_mock_prelude,
)
from cdd.tests.mocks.sqlalchemy import config_decl_base_str
from cdd.tests.utils_for_tests import unittest_main


def populate_files(tempdir):
    """
    Populate files in the tempdir

    :param tempdir: Temporary directory
    :type tempdir: ```str```

    :returns: model_path, routes_path
    :rtype: ```Tuple[str, str]```
    """
    model_path = path.join(tempdir, "model.py")
    routes_path = path.join(tempdir, "routes.py")
    open("__init__.py", "a").close()

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

    routes = "\n".join((route_mock_prelude, create_route, read_route, destroy_route))

    with open(routes_path, "wt") as f:
        f.write(routes)

    return model_path, routes_path


class TestGenRoutes(TestCase):
    """ Test class for gen_routes.py """

    def test_gen_routes_update(self) -> None:
        """ Tests `gen_routes` when routes do exists """
        self.gen_routes_tester("update")

    def test_gen_routes_insert(self) -> None:
        """ Tests `gen_routes` when routes do not exist """
        self.gen_routes_tester("insert")

    def gen_routes_tester(self, approach):
        """
        :param approach: How to upsert
        :type approach: ```Literal["insert", "update"]```
        """
        with TemporaryDirectory() as tempdir:
            model_path, routes_path = populate_files(tempdir)

            if approach == "insert":
                remove(routes_path)

            routes = gen_routes(
                app=route_config["app"],
                route=route_config["route"],
                model_path=model_path,
                model_name=route_config["name"],
                crud="CRD",
            )

            routes, testable_routes = tee(routes)
            testable_routes = list(testable_routes)

            self.assertIsInstance(testable_routes, list)
            self.assertGreaterEqual(len(testable_routes), 1)
            self.assertIsInstance(testable_routes[0], FunctionDef)

            self.assertIsNone(
                upsert_routes(
                    app=route_config["app"],
                    route=route_config["route"],
                    routes=routes,
                    routes_path=routes_path,
                )
            )


unittest_main()
