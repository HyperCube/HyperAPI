import unittest
from hdp_lib_api.hdp_api.base.validators import RouteConsistencyException


class ConsistencyTestCase(unittest.TestCase):

    def setUp(self):
        from tests.data.routes_integrity import TestResource
        self.resourceCls = TestResource

    def test_route_overlap(self):
        self.assertRaises(RouteConsistencyException, self.resourceCls.RouteOverlap.check_routes_integrity)

    def test_route_wrong(self):
        self.assertRaises(RouteConsistencyException, self.resourceCls.RouteWrong.check_routes_integrity)
        self.assertRaises(RouteConsistencyException, self.resourceCls.RouteOk.check_routes_integrity)
