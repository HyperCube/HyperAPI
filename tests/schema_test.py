import unittest
from hdp_lib_api import Router
from HyperAPI.utils.version import Version
import json
import pathlib


class SchemaTestCase(unittest.TestCase):

    def test_schemas(self):
        _path = pathlib.Path(__file__)
        for _file in _path.parent.joinpath('data', 'schema_json').iterdir():
            _file_name = _file.parts[-1]
            if not _file_name.endswith('.json'):
                continue

            with open(_file, 'r', encoding="utf-8") as F:
                schema = json.load(F)

            version = Version(schema.get('version'))
            results = Router.validate_schema(schema, version)

            message = '\n' + json.dumps(results, sort_keys=True, indent=4)

            # Empty sets evaluates to False so assertFalse checks empty
            self.assertFalse(results.get('unexpected_resources'), f'{_file_name} - Resources not present in the schema{message}')
            self.assertFalse(results.get('missing_resources'), f'{_file_name}Resources missing in the API{message}')

            for _resource_name, _route_diffs in results.get('different_resources').items():
                self.assertFalse(_route_diffs.get('unexpected_routes'), f'{_file_name} - Routes not present in the schema for {_resource_name}{message}')
                self.assertFalse(_route_diffs.get('missing_routes'), f'{_file_name} - Routes missing in the API for {_resource_name}{message}')
