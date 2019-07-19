from hdp_lib_api.hdp_api.base.resource import Resource
from hdp_lib_api.hdp_api.base.route import Route, SubRoute


class TestResource(Resource):
    name = "TestResource"
    available_since = "1.0"
    removed_since = "4.0"

    class RouteAvailable(Route):
        name = "Route Available"
        httpMethod = Route.GET
        path = "/route/available"
        available_since = "3.1"

    class RouteFuture(Route):
        name = "Route Future"
        httpMethod = Route.GET
        path = "/route/future"
        available_since = "3.2"

    class RouteRemoved(Route):
        name = "Route Removed"
        httpMethod = Route.GET
        path = "/route/removed"
        removed_since = "3.2"

    class RouteCompatible(Route):
        name = "Route Compatible 3.0"
        httpMethod = Route.GET
        path = "/route/compatible/v0"
        available_since = "3.0"
        removed_since = "3.1"

        class RouteCompatible_31(SubRoute):
            name = "Route Compatible 3.1"
            httpMethod = Route.POST
            path = "/route/compatible/v1"
            available_since = "3.1"
            removed_since = "3.2"

        class RouteCompatible_32(SubRoute):
            httpMethod = Route.GET
            available_since = "3.2"
            removed_since = "3.3"
            path = "/route/compatible/{route_ID}"
            _path_keys = {
                'route_ID': Route.VALIDATOR_ANY,
            }

            @staticmethod
            def _convert_args(self, **kwargs):
                kwargs["route_ID"] = 0
                return kwargs
