from HyperAPI.util import Helper
from HyperAPI.hyper_api.base import Base
from HyperAPI.utils.exceptions import ApiException


class VariableFactory:
    """
    """

    def __init__(self, api, project_id, dataset_id):
        self.__api = api
        self.__project_id = project_id
        self.__dataset_id = dataset_id

    @Helper.try_catch
    def filter(self):
        """
        Get all variables on the dataset.

        Returns:
            Variable[]: list of variables on the dataset
        """
        json = {'project_ID': self.__project_id, 'dataset_ID': self.__dataset_id}
        variable_res = self.__api.Variable.getvariable(**json)

        return [DiscreteVariable(self.__api, json, variable_info) if variable_info.get('type') == 'D'
                else ContinuousVariable(self.__api, json, variable_info) for variable_info in variable_res['variables']]

    @Helper.try_catch
    def get(self, name):
        """
        Get a variable matching the given name

        Args:
            name (str): The name of the variable

        Returns:
            Variable: variable found by name
        """
        variables = list(filter(lambda x: x.name == name, self.filter()))
        if variables:
            return variables[0]
        return None

    @Helper.try_catch
    def ignore(self, *args):
        """
        Variables you don't want to keep.

        Args:
            args (str): list of variable names to ignore

        Returns:
            Variable[]: print the final status of listed variables
        """
        result = []
        for variable_name in args:
            variable = self.get(variable_name)
            if variable is not None:
                variable.ignore()
                result.append(variable)
            else:
                print('variable "{}" was not found'.format(variable_name))
        return result

    @Helper.try_catch
    def keep(self, *args):
        """
        Variables you want to keep.

        Args:
            args (str): list of variable names to keep

        Returns:
            Variable[]: print the final status of listed variables
        """
        result = []
        for variable_name in args:
            variable = self.get(variable_name)
            if variable is not None:
                variable.keep()
                result.append(variable)
            else:
                print('variable "{}" was not found'.format(variable_name))
        return result


class Variable(Base):
    TYPE_MANUAL = "manual"
    TYPE_EQUALFREQ = "equal-freq"
    TYPE_EQUALWIDTH = "equal-width"
    TYPE_LEGACY = "legacy"

    def __init__(self, api, json_sent, json_returned):
        self.__api = api
        self.__json_sent = json_sent
        self.__json_returned = json_returned

    def __repr__(self):
        return """\n{} : {}\n""".format(
            self.__class__.__name__,
            self.name
        ) + ("\t<! This variable is ignored>\n" if self.is_ignored else "") + \
            """\t- Count : {}\n\t- Missing count : {}\n""".format(
            self.count,
            self.missing_value_count)

    # Property part
    @property
    def _json(self):
        return self.__json_returned

    @property
    def name(self):
        return self.__json_returned.get('name')

    @property
    def is_ignored(self):
        return self.__json_returned.get('ignored')

    @property
    def is_discrete(self):
        return self.__json_returned.get('type') == 'D'

    @property
    def dataset_id(self):
        return self.__json_sent.get('dataset_ID')

    @property
    def project_id(self):
        return self.__json_sent.get('project_ID')

    @property
    def count(self):
        return self.__json_returned['stats']['count']

    @property
    def missing_value_count(self):
        return self.__json_returned['stats']['missing']

    # Method part
    @Helper.try_catch
    def ignore(self):
        """
        Returns:
            Variable: variable that has been ignored
        """
        if self.__api.session.version >= self.__api.session.version.__class__('3.6'):
            varname = self.name
            data = {'updateFields': {varname: {'ignored': True}}}
            returned_json = self.__api.Datasets.metadata(project_ID=self.project_id, dataset_ID=self.dataset_id, json=data)
            self.__json_returned['ignored'] = list(filter(lambda x: x.get('varName') == self.name, returned_json['metadata']['variables']))[0].get('ignored')
        elif not self.is_ignored:
            data = {'changedMetadata': [self.name]}
            self.__api.Datasets.metadata(project_ID=self.project_id, dataset_ID=self.dataset_id, json=data)
            self.__json_returned = self._update()
        return self

    @Helper.try_catch
    def keep(self):
        """
        Returns:
            Variable: variable that has been kept
        """
        if self.__api.session.version >= self.__api.session.version.__class__('3.6'):
            varname = self.name
            data = {'updateFields': {varname: {'ignored': False}}}
            returned_json = self.__api.Datasets.metadata(project_ID=self.project_id, dataset_ID=self.dataset_id, json=data)
            self.__json_returned['ignored'] = list(filter(lambda x: x.get('varName') == self.name, returned_json['metadata']['variables']))[0].get('ignored')
        elif self.is_ignored:
            data = {'changedMetadata': [self.name]}
            self.__api.Datasets.metadata(project_ID=self.project_id, dataset_ID=self.dataset_id, json=data)
            self.__json_returned = self._update()
        return self

    @Helper.try_catch
    def _update(self):
        json = {'project_ID': self.project_id, 'dataset_ID': self.dataset_id}
        variable_res = self.__api.Variable.getvariable(**json)
        return list(filter(lambda x: x.get('name') == self.name, variable_res['variables']))[0]


class DiscreteVariable(Variable):
    def __init__(self, api, json_sent, json_returned):
        self.__api = api
        self.__json_sent = json_sent
        self.__json_returned = json_returned
        super().__init__(api, json_sent, json_returned)

    def __repr__(self):
        return super().__repr__() + """\t- Top Modality : {}\n\t- Top Modality Count : {}\n""".format(self.top_modality, self.top_modality_frequency)

    # Property part
    @property
    def modalities(self):
        """ Returns the top 50 modalities by frequency for the variable """
        return self.__json_returned['stats']['distribution']['X']

    @property
    def modality_count(self):
        return self.__json_returned['stats']['modalitiesCount']

    @property
    def frequencies(self):
        """ Returns the frequencies for the top 50 modalities by frequency for the variable """
        return self.__json_returned['stats']['distribution']['Y']

    @property
    def top_modality(self):
        return self.__json_returned['stats']['topModality']

    @property
    def top_modality_frequency(self):
        return self.__json_returned['stats']['freqTopModality']

    @property
    def purities(self):
        """ Returns the purities for the top 50 modalities by frequency for the variable """
        return self.__json_returned['stats']['distribution']['Purities']


class ContinuousVariable(Variable):
    def __init__(self, api, json_sent, json_returned):
        self.__api = api
        self.__json_sent = json_sent
        self.__json_returned = json_returned
        super().__init__(api, json_sent, json_returned)

    def __repr__(self):
        return super().__repr__() + \
            ("\t<! This variable is discretized>\n" if self.discretization is not None else "") + \
            """\t- Min : {}\n\t- Max : {}\n\t- Mean : {}\n\t- Std deviation : {}\n""".format(self.min, self.max, self.mean, self.std)

    # Property part
    @property
    def discretization(self):
        return self.__json_returned.get('discretization', None)

    @property
    def min(self):
        return self.__json_returned['stats']['min']

    @property
    def max(self):
        return self.__json_returned['stats']['max']

    @property
    def mean(self):
        return self.__json_returned['stats']['mean']

    @property
    def std(self):
        return self.__json_returned['stats']['std']

    @property
    def first_quartile(self):
        return self.__json_returned['stats']['25%']

    @property
    def median(self):
        return self.__json_returned['stats']['50%']

    @property
    def third_quartile(self):
        return self.__json_returned['stats']['75%']

    @property
    def bin_boundaries(self):
        return self.__json_returned['stats']['distribution']['X']

    @property
    def bin_sizes(self):
        return self.__json_returned['stats']['distribution']['Y'][1:]

    # Method part
    @Helper.try_catch
    def discretize(self, discretization_type=Variable.TYPE_EQUALFREQ, nb_bins=10):
        """
        Args:
            discretization_type (str): "equal-freq" or "equal-width", default is "equal-freq"
            nb_bins (int): number of bins to target after discretization, default is 10

        Returns:
            ContinuousVariable : variable that has been discretized
        """
        # Eventually existing discretization will be overriden by the new one
        if not isinstance(nb_bins, int):
            raise ApiException('Number of bins must be an integer')
        if discretization_type == Variable.TYPE_EQUALFREQ:
            data = [{'name': self.name, 'equalFreqBins': nb_bins}]
        elif discretization_type == Variable.TYPE_EQUALWIDTH:
            data = [{'name': self.name, 'equalWidthBins': nb_bins}]
        else:
            raise ApiException('Discretization Type does not exist: {}'.format(discretization_type))
        creation_json = self.__api.Datasets.discretize(project_ID=self.project_id, dataset_ID=self.dataset_id, json=data)

        try:
            self.__api.handle_work_states(self.project_id, work_type='discretization', work_id=creation_json.get('_id'))
        except Exception as E:
            raise ApiException('Unable to get the discretization status', str(E))

        self.__json_returned = self._update()
        return self

    @Helper.try_catch
    def delete_discretization(self):
        """
        Returns:
            ContinuousVariable : variable whose discretization has been removed
        """
        if self.discretization is not None:
            data = {'name': self.name}
            self.__api.Datasets.deletediscretization(project_ID=self.project_id, dataset_ID=self.dataset_id, json=data)
            self.__json_returned = self._update()
        return self

