from HyperAPI.hyper_api.dataset import DatasetFactory
from HyperAPI.hyper_api.target import TargetFactory
from HyperAPI.hyper_api.base import Base
from HyperAPI.util import Helper
from HyperAPI.hyper_api.model import ModelFactory
from HyperAPI.hyper_api.automatedmodel import AutomatedModelFactory
from HyperAPI.hyper_api.ruleset import RulesetFactory
from HyperAPI.hyper_api.xray import XrayFactory


class ProjectFactory:
    """
    """
    def __init__(self, api):
        self.__api = api

    @Helper.try_catch
    def create(self, name, description='', type_id=None, wait=False):
        """
        create(name, description='', type_id=None, wait=False)
        Create a HyperCube project.

        Args:
            name (str): The name of the project
            description (str): the description of the project, default is ''
            type_id (str): id for a demo project (eg: 'TitanicDemoProject'), default is None, which is a blank project 
                available only for HDP versions equal to or later than 6.0.1
            wait (bool) : waits for the end of all works in the demo project specified by type_id, default is False
                available only for HDP versions equal to or later than 6.0.1

        Returns:
            Project: Created project
        """
        if type_id is None or self.__api.session.version < self.__api.session.version.__class__('6.0.1'):
            json = {'name': name, 'description': description}
            return Project(self.__api, json, self.__api.Projects.addproject(json=json))
        else:
            json = {'name': name, 'description': description, 'projectTypeId': type_id}
            project = Project(self.__api, json, self.__api.Projects.addproject(json=json))
            if wait is False or self.__api.session.version < self.__api.session.version.__class__('6.0.1'):
                return project
            self.__api.handle_work_states(project_id=project.project_id, work_id=project.__json_returned.get('workflowId'))
            return project
        json = {'name': name, 'description': description}
        return Project(self.__api, json, self.__api.Projects.addproject(json=json))

    @Helper.try_catch
    def filter(self):
        """
        filter()
        Get all projects.

        Returns:
            list(Project): All projects
        """
        if self.__api.session.version >= self.__api.session.version.__class__('3.6'):
            return list(map(lambda x: Project(self.__api, {}, x), self.__api.Projects.projects()))        
        return list(map(lambda x: Project(self.__api, {}, x), self.__api.Projects.projects().get('projects')))

    @Helper.try_catch
    def get(self, name):
        """
        get(name)
        Returns a project found by name or None if no match.

        Args:
            name (str): The name of the project

        Returns:
            Project or None: Retrieved project
        """
        projects = list(filter(lambda x: x.name == name, self.filter()))
        if projects:
            return projects[0]
        return None

    @Helper.try_catch
    def get_by_id(self, project_id):
        """
        get_by_id(project_id)
        Returns a project found by ID or None if no match.

        Args:
            id (str): ID of the project

        Returns:
            Project or None: Retrieved project
        """
        json = {'project_ID': project_id}
        json_returned = self.__api.Projects.getaproject(**json)
        if json_returned is None:
            return None
        return Project(self.__api, json, json_returned)

    @Helper.try_catch
    def get_default(self):
        """
        get_default()
        Returns the default project.

        Returns:
            Project: Default project
        """
        if self.__api.session.version >= self.__api.session.version.__class__('3.6'):
            defaultProjectId = self.__api.Settings.getusersettings().get('defaultProjectId')
        else:
            defaultProjectId = self.__api.Projects.projects().get('defaultProject')

        if defaultProjectId is None:
            return None

        defaultProject = self.get_by_id(defaultProjectId)
        if defaultProject is None:
            projects = self.filter()
            if len(projects) > 0:
                return projects[0].set_as_default()
            return None
        return defaultProject

    @Helper.try_catch
    def get_or_create(self, name, description='', type_id=None, wait=False):
        """
        get_or_create(name, description='', type_id=None, wait=False)
        Returns an existing project matching the given name. If no match, create a new project.

        Args:
            name (str): The name of the project
            description (str): the description of the project, default is ''
            type_id (str): id for a demo project (eg: 'TitanicDemoProject'), default is None, which is a blank project
            wait (bool) : waits for the end of all works in the demo project specified by type_id, default is False

        Returns:
            Project: retrieved or new project
        """
        _res = self.get(name)
        if _res is None:
            self.create(name, description, type_id, wait)
            _res = self.get(name)
        return _res


class Project(Base):
    """
    Project()
    """
    def __init__(self, api, json_sent, json_return):
        self.__api = api
        self.__json_sent = json_sent
        self.__json_returned = json_return
        self._is_deleted = False

    def __repr__(self):
        return """\n{} : {} <{}>\n""".format(
            self.__class__.__name__,
            self.name,
            self.project_id
        ) + ("\t<This is the default Project>\n" if self.is_default else "") + \
            ("\t<! This project has been deleted>\n" if self._is_deleted else "") + \
            """\t- Description : {}\n\t- Created by : {}\n\t- Created on : {}\n""".format(
            self.description,
            self.user_name,
            self.created.strftime('%Y-%m-%d %H:%M:%S UTC'))

    # Factory part
    @property
    def Dataset(self):
        """
        DatasetFactory: Tool class for creating and retrieving existing datasets in this project
        """
        return DatasetFactory(self.__api, self.project_id)

    @property
    def Target(self):
        """
        TargetFactory: Tool class for creating and retrieving existing targets in this project
        """
        return TargetFactory(self.__api, self.project_id)

    @property
    def Ruleset(self):
        """
        RulesetFactory: Tool class for creating and retrieving existing rulesets in this project
        """
        return RulesetFactory(self.__api, self.project_id)

    @property
    def Model(self):
        """
        ModelFactory: Tool class for creating and retrieving existing models in this project
        """
        return ModelFactory(self.__api, self.project_id)

    @property
    def AutomatedModel(self):
        """
        AutomatedModelFactory: Tool class for creating and retrieving existing automated models in this project
        """
        if hasattr(self.__api, 'AutomatedPrediction'):
            return AutomatedModelFactory(self.__api, self.project_id)
        else:
            raise NotImplementedError('The feature is not available on this platform')

    @property
    def Xray(self):
        """
         XrayFactory: Tool class for creating and retrieving existing automated XRays in this project
        """
        return XrayFactory(self.__api, self.project_id)

    # Property  part
    @property
    def _json(self):
        return self.__json_returned

    @property
    def project_id(self):
        """
        str: project ID
        """
        return self.__json_returned.get('_id')

    @property
    def is_default(self):
        """
        Boolean: indicating if this project is the default project
        """
        if self._is_deleted:
            return False
        if self.__api.session.version >= self.__api.session.version.__class__('3.6'):
            defaultProjectId = self.__api.Settings.getusersettings().get('defaultProjectId')
        else:
            defaultProjectId = self.__api.Projects.projects().get('defaultProject')

        return self.project_id == defaultProjectId

    @property
    def user_name(self):
        """
        str: User name
        """
        owners = list(filter(lambda x: x["role"] == "O", self.__json_returned.get("users", [])))
        if len(owners) > 0:
            return owners[0].get("username", "")
        else:
            return ""

    @property
    def datasets(self):
        """
        list(Dataset): all datasets in this project
        """
        return self.Dataset.filter()

    @property
    def rulesets(self):
        """
        list(Ruleset): all rulesets in this project
        """
        return self.Ruleset.filter()

    @property
    def models(self):
        """
        list(Models): all models in this project
        """
        return self.Model.filter()

    @property
    def targets(self):
        """
        list(Target): all targets in this project
        """
        target_family = self.Target.KPI_FAMILY_TARGET
        return [t for t in self.Target.filter() if t.indicator_family == target_family]

    @property
    def descriptions(self):
        """
        list: all descriptions in this project
        """
        description_family = self.Target.KPI_FAMILY_DESCRIPTION
        return [t for t in self.Target.filter() if t.indicator_family == description_family]

    @property
    def _tags(self):
        return NotImplemented

    @property
    def name(self):
        """
        str: Project name
        """
        return self.__json_returned.get('name')

    @property
    def description(self):
        """
        str: Project description
        """
        return self.__json_returned.get('description')

    @property
    def created(self):
        """
        datetime: Creation date of this project
        """
        created_date = None
        if 'createdOn' in self.__json_returned.keys():
            created_date = self.__json_returned.get('createdOn')
        elif 'created' in self.__json_returned.keys():
            created_date = self.__json_returned.get('created')
        else:
            return None
        if isinstance(created_date, int):
            return self.timestamp2date(created_date)
        return self.str2date(created_date, '%Y-%m-%dT%H:%M:%S.%fZ')

    # Methods part
    @Helper.try_catch
    def delete(self):
        """
        delete()
        Delete this project.
        """
        if not self._is_deleted:
            json = {'project_ID': self.project_id}
            self.__api.Projects.deleteproject(**json)
            self._is_deleted = True
        return self

    @Helper.try_catch
    def set_as_default(self):
        """
        set_as_default()
        Set this project as default.

        Returns:
            Project: The project itself
        """
        if not self._is_deleted:
            self.__json_sent = {
                'json': { 
                    'settings': { 
                        'defaultProjectId': self.project_id 
                    }
                }
            }
            self.__api.Settings.updateusersettings(**self.__json_sent)
            self.__json_returned = ProjectFactory(self.__api).get_by_id(self.project_id).__json_returned
        return self

    @Helper.try_catch
    def update(self, **kwargs):
        """
        update()
        Update this project.

        Returns:
            Project: The project itself
        """
        if not self._is_deleted:
            json = dict([x for x in kwargs.items() if x[1] is not None])
            if json:
                self.__json_sent = {'project_ID': self.project_id, 'json': json}
                self.__api.Projects.updateproject(**self.__json_sent)
                self.__json_returned = ProjectFactory(self.__api).get_by_id(self.project_id).__json_returned
        return self
