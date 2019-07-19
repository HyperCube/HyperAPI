from hdp_lib_api import Router
from HyperAPI.util import Helper
from HyperAPI.hyper_api.project import ProjectFactory


class Api:
    @Helper.try_catch
    def __init__(self, token=None, url='', username=None, password=None, watcher=None):
        self.__api = Router(token=token, url=url, watcher=watcher, username=username, password=password)
        print('{}\n{} - {}\nVersion: {}\nBuild date: {}'.format(self.__api.session_details.url,
                                                                self.__api.session_details.name,
                                                                self.__api.session_details.product,
                                                                self.__api.session_details.version,
                                                                self.__api.session_details.build_date
                                                                ))
        self.Project = ProjectFactory(self.__api)

        self.timeout_settings = self.__api._default_timeout_settings
