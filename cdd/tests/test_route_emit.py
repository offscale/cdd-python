"""
Tests route emission
"""
from copy import deepcopy
from unittest import TestCase

from cdd.routes.emit_constants import (
    create_route_variants,
    delete_route_variants,
    read_route_variants,
)
from cdd.tests.mocks.routes import create_route, destroy_route, read_route, route_config
from cdd.tests.utils_for_tests import unittest_main


class TestRouteEmit(TestCase):
    """ Tests `routes.emit` """

    @classmethod
    def setUpClass(cls) -> None:
        """
        Setup a couple of class-wide config variables
        """
        cls.config = deepcopy(route_config)
        del cls.config["primary_key"]
        cls.config_with_id = deepcopy(route_config)
        cls.config_with_id["id"] = cls.config_with_id.pop("primary_key")

    def test_create(self) -> None:
        """
        Tests whether `create_route` produces the right `create_route_variants`
        """
        self.assertEqual(create_route_variants[-1].format(**self.config), create_route)

    def test_read(self) -> None:
        """
        Tests whether `read_route` produces the right `read_route_variants`
        """
        self.assertEqual(
            read_route_variants[-1].format(**self.config_with_id), read_route
        )

    def test_delete(self) -> None:
        """
        Tests whether `destroy_route` produces the right `delete_route_variants`
        """
        self.assertEqual(
            delete_route_variants[-1].format(**self.config_with_id), destroy_route
        )


unittest_main()
