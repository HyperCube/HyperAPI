import unittest
from hdp_lib_api import Router
from tests._utils import DummySession


class ResourceTestCase(unittest.TestCase):

    def setUp(self):
        from tests.data.resource_simple import TestResource

        self.resourceCls = TestResource
        self.create_resources = Router._create_resources

        if hasattr(self, self.resourceCls.__name__):
            delattr(self, self.resourceCls.__name__)

    def tearDown(self):
        if hasattr(self, self.resourceCls.__name__):
            delattr(self, self.resourceCls.__name__)

    def test_resource_availability(self):
        self.assertFalse(self.resourceCls.is_available('0.0'))
        self.assertTrue(self.resourceCls.is_available('1.0'))
        self.assertFalse(self.resourceCls.is_available('4.0'))

    def test_resource_not_available(self):
        self.session = DummySession(version='0.0')
        self.create_resources(self, [self.resourceCls], None)
        self.assertFalse(hasattr(self, self.resourceCls.__name__), 'Future Resources should not be created')

    def test_resource_created(self):
        self.session = DummySession(version='1.0')
        self.create_resources(self, [self.resourceCls], None)
        self.assertTrue(hasattr(self, self.resourceCls.__name__), 'Available Resources should be created')

    def test_resource_removed(self):
        self.session = DummySession(version='4.0')
        self.create_resources(self, [self.resourceCls], None)
        self.assertFalse(hasattr(self, self.resourceCls.__name__), 'Obsolete Resources should not be created')
