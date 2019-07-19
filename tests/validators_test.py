import unittest
from hdp_lib_api.hdp_api.base.route import Route


class ValidatorTestCase(unittest.TestCase):
    """Test validators with various inputs"""

    def test_any(self):
        self.assertTrue(Route.VALIDATOR_ANY('Hello'))
        self.assertTrue(Route.VALIDATOR_ANY(42))
        self.assertFalse(Route.VALIDATOR_ANY(' '))
        self.assertFalse(Route.VALIDATOR_ANY(None))

    def test_objectId(self):
        self.assertTrue(Route.VALIDATOR_OBJECTID('0123456789ab0123456789ab'))
        self.assertTrue(Route.VALIDATOR_OBJECTID(123456789012123456789012))
        self.assertFalse(Route.VALIDATOR_OBJECTID('Plop '))
        self.assertFalse(Route.VALIDATOR_OBJECTID(None))
