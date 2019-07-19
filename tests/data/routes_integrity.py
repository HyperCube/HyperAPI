from hdp_lib_api.hdp_api.base.resource import Resource
from hdp_lib_api.hdp_api.base.route import Route, SubRoute


class TestResource(Resource):
    name = "TestResource"
    available_since = "1.0"
    removed_since = "4.0"

    class RouteOverlap(Route):
        name = "Route Overlap"
        httpMethod = Route.GET
        path = "/route"
        available_since = "3.0"
        removed_since = "3.2"

        class SubRouteOverlap(SubRoute):
            available_since = "3.1"
            removed_since = "3.3"

    class RouteWrong(Route):
        name = "Route Wrong"
        httpMethod = Route.GET
        path = "/route"
        available_since = "3.2"
        removed_since = "3.0"

    class RouteOk(Route):
        name = "Route Ok"
        httpMethod = Route.GET
        path = "/route"
        available_since = "3.0"
        removed_since = "3.2"

        class SubRouteWrong(SubRoute):
            name = "SubRoute Wrong"
            available_since = "3.2"
            removed_since = "3.1"
