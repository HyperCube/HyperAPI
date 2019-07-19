import unittest
from hdp_lib_api.hdp_api.base.route import Route
from tests._utils import DummySession


class CompatibilityTestCase(unittest.TestCase):

    def setUp(self):
        from tests.data.resource_simple import TestResource

        self.route_compatible = TestResource.RouteCompatible

        self.resourceCls = TestResource

    def test_route_none(self):
        self.session = DummySession(version='2.9')
        resource = self.resourceCls(session=self.session, watcher=None)
        self.assertFalse(hasattr(resource, self.route_compatible.get_route_name()))

    def test_route_base(self):
        self.session = DummySession(version='3.0')
        resource = self.resourceCls(session=self.session, watcher=None)
        self.assertTrue(hasattr(resource, self.route_compatible.get_route_name()))
        route_instance = resource.routecompatible
        self.assertEqual(route_instance.name, "Route Compatible 3.0")
        self.assertEqual(route_instance.httpMethod, Route.GET)
        self.assertEqual(route_instance.path, "/route/compatible/v0")

    def test_route_args(self):
        self.session = DummySession(version='3.1')
        resource = self.resourceCls(session=self.session, watcher=None)
        self.assertTrue(hasattr(resource, self.route_compatible.get_route_name()))
        route_instance = resource.routecompatible
        self.assertEqual(route_instance.name, "Route Compatible 3.1")
        self.assertEqual(route_instance.httpMethod, Route.POST)
        self.assertEqual(route_instance.path, "/route/compatible/v1")

    def test_route_convert(self):
        self.session = DummySession(version='3.2')
        resource = self.resourceCls(session=self.session, watcher=None)
        self.assertTrue(hasattr(resource, self.route_compatible.get_route_name()))
        route_instance = resource.routecompatible
        self.assertEqual(route_instance.name, "Route Compatible 3.0")
        self.assertEqual(route_instance.httpMethod, Route.GET)
        self.assertEqual(route_instance.path, "/route/compatible/{route_ID}")
        res = route_instance(json=1)
        self.assertDictEqual(res, {'httpMethod': 'GET', 'path': 'route/compatible/0', 'json': 1})

    def test_route_removed(self):
        self.session = DummySession(version='3.3')
        resource = self.resourceCls(session=self.session, watcher=None)
        self.assertFalse(hasattr(resource, self.route_compatible.get_route_name()))
