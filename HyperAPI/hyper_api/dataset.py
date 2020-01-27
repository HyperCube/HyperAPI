from os.path import getsize, split
import sys
import uuid
import io
from HyperAPI.util import Helper
from HyperAPI.utils.exceptions import ApiException
from HyperAPI.utils.imports import get_required_module
from HyperAPI.hyper_api.base import Base
from HyperAPI.hyper_api.variable import VariableFactory
from HyperAPI.hyper_api.xray import XrayFactory
from HyperAPI.hyper_api.ruleset import RulesetFactory


class DatasetFactory:
    """
    """
    string_delimiters = ["semicolon", "comma", "tab", "pipe"]
    char_delimiters = [";", ",", "\t", "|"]

    def __init__(self, api, project_id):
        self.__api = api
        self.__project_id = project_id

    @Helper.try_catch
    def create(self, name, file_path, decimal='.',
               delimiter=';', encoding='UTF-8', selectedSheet=1,
               description='', modalities=2, continuous_threshold=0.95, missing_threshold=0.95,
               metadata_file_path=None, discreteDict_file_path=None, keepVariableName=None):
        """
        Create a Dataset from a file (csv, Excel)
        Args:
            name (str): The name of the dataset
            file_path (str): The origin path of the file
            decimal (str): Decimal separator - csv files only, default is '.'
            delimiter (str): The csv field delimiter - csv files only, default is ';'
            encoding (str): The file encoding - csv files only, default is 'UTF-8'
            selectedSheet (int): The worksheet to use (starts at 1 like in Hypercube User Interface) - Excel files only, default is 1
            description (str): The dataset description, default is ''
            modalities (int): Modality threshold for discrete variables, default is 2
            continuous_threshold (float): % of continuous values threshold for continuous variables, default is 0.95
            missing_threshold (float): % of missing values threshold for ignored variables, default is 0.95
        Returns:
            Dataset
        """

        project_id = self.__project_id
        _, file_name = split(file_path)
        if metadata_file_path:
            _, metadata_file_name = split(metadata_file_path)
        else:
            metadata_file_name = None
        if discreteDict_file_path:
            _, discreteDict_file_name = split(discreteDict_file_path)
        else:
            discreteDict_file_name = None
        selectedSheet = max(1, selectedSheet)

        # historically, delimiter/separator were stored as explicit strings in our database (ex: "semicolon")
        # we want to keep it that way
        if delimiter in self.char_delimiters:
            delimiter = self.string_delimiters[self.char_delimiters.index(delimiter)]
        elif delimiter not in self.string_delimiters:
            raise ApiException(f'Unsupported value for delimiter: {delimiter}', f'Supported values: {self.string_delimiters}')

        data = {
            'name': name,
            'fileName': file_name,
            'decimalDelimiter': decimal,
            'delimiter': delimiter,
            'separator': delimiter,
            'encoding': encoding,
            'usePython': description,
            'useSpark': 'False',
            'sourceFileName': file_name,
            'selectedSheet': str(selectedSheet),
            'description': description,
            'size': '{}'.format(getsize(file_path)),
            'nbModalitiesThreshold': str(modalities),
            'percentageContinuousThreshold': str(continuous_threshold),
            'percentageMissingThreshold': str(missing_threshold)
        }

        if keepVariableName:
            data['keepVariableName'] = keepVariableName

        def apihandle():
                json = {'project_ID': project_id, 'data': data, 'streaming': True}

                creation_json = self.__api.Datasets.uploaddatasets(**json)
                print('\n')

                try:
                    self.__api.handle_work_states(project_id, work_type='datasetValidation', query={"datasetId": creation_json.get('_id')})
                except Exception as E:
                    raise ApiException('Unable to get the dataset validation status', str(E))
                try:
                    self.__api.handle_work_states(project_id, work_type='datasetDescription', query={"datasetId": creation_json.get('_id')})
                except Exception as E:
                    raise ApiException('Unable to get the dataset description status', str(E))

                returned_json = self.__api.Datasets.getadataset(project_ID=project_id, dataset_ID=creation_json.get('_id'))
                return json, returned_json

        if metadata_file_name and discreteDict_file_name:
            data['metadataFileName'] = metadata_file_name,
            data['discreteDictFileName'] = discreteDict_file_name,
            with open(file_path, 'rb') as FILE:
                with open(metadata_file_path, 'rb') as METADATA:
                    with open(discreteDict_file_path, 'rb') as DISCRETEDICT:
                        data['file[0]'] = (
                            file_name,
                            FILE,
                            'application/vnd.ms-excel',
                        )
                        data['file[1]'] = (
                            metadata_file_name,
                            METADATA,
                            'application/json',
                        )
                        data['file[2]'] = (
                            discreteDict_file_name,
                            DISCRETEDICT,
                            'application/json',
                        )
                        json, returned_json = apihandle()
        elif metadata_file_name:
            data['metadataFileName'] = metadata_file_name,
            with open(file_path, 'rb') as FILE:
                with open(metadata_file_path, 'rb') as METADATA:
                    data['file[0]'] = (
                        file_name,
                        FILE,
                        'application/vnd.ms-excel',
                    )
                    data['file[1]'] = (
                        metadata_file_name,
                        METADATA,
                        'application/json',
                    )
                    json, returned_json = apihandle()
        else:
            with open(file_path, 'rb') as FILE:
                data['file[0]'] = (
                    file_name,
                    FILE,
                    'application/vnd.ms-excel',
                )
                json, returned_json = apihandle()

        return Dataset(self.__api, json, returned_json)

    @Helper.try_catch
    def create_from_dataframe(self, name, dataframe, description='', modalities=2,
                              continuous_threshold=0.95, missing_threshold=0.95,
                              metadata=None, discreteDict=None, keepVariableName=None):
        """
        Create a Dataset from a Pandas DataFrame
        Args:
            name (str): The name of the dataset
            dataframe (pandas.DataFrame): The dataframe to import
            description (str): The dataset description, default is ''
            modalities (int): Modality threshold for discrete variables, default is 2
            continuous_threshold (float): % of continuous values threshold for continuous variables ,default is 0.95
            missing_threshold (float): % of missing values threshold for ignored variables, default is 0.95
        Returns:
            Dataset
        """
        project_id = self.__project_id
        file_name = '{}.csv'.format(uuid.uuid4())
        metadata_file_name = '{}.json'.format(uuid.uuid4())
        discreteDict_file_name = '{}.json'.format(uuid.uuid4())
        DECIMAL = "."
        SEPARATOR = "semicolon"
        ENCODING = "utf-8"

        sep = self.char_delimiters[self.string_delimiters.index(SEPARATOR)]
        stream_df = io.StringIO(dataframe.to_csv(sep=sep, index=False))
        if metadata:
            import json
            stream_metadata = io.StringIO()
            json.dump(metadata, stream_metadata)
            if discreteDict:
                stream_discreteDict = io.StringIO()
                json.dump(discreteDict, stream_discreteDict)

        data = {
            'name': name,
            'fileName': file_name,
            'decimalDelimiter': DECIMAL,
            'delimiter': SEPARATOR,
            'separator': SEPARATOR,
            'encoding': ENCODING,
            'usePython': description,
            'useSpark': 'False',
            'sourceFileName': file_name,
            'description': description,
            'size': '{}'.format(sys.getsizeof(dataframe)),
            'nbModalitiesThreshold': str(modalities),
            'percentageContinuousThreshold': str(continuous_threshold),
            'percentageMissingThreshold': str(missing_threshold)
        }

        if keepVariableName:
            data['keepVariableName'] = keepVariableName

        data['file[0]'] = (
            file_name,
            stream_df,
            'application/vnd.ms-excel',
        )
        if metadata:
            data['metadataFileName'] = metadata_file_name
            data['file[1]'] = (
                metadata_file_name,
                stream_metadata,
                'application/json',
            )
            if discreteDict:
                data['discreteDictFileName'] = discreteDict_file_name
                data['file[2]'] = (
                    discreteDict_file_name,
                    stream_discreteDict,
                    'application/json',
                )
        json_ = {'project_ID': project_id, 'data': data, 'streaming': True}

        creation_json = self.__api.Datasets.uploaddatasets(**json_)
        try:
            self.__api.handle_work_states(project_id, work_type='datasetValidation', query={"datasetId": creation_json.get('_id')})
        except Exception as E:
            raise ApiException('Unable to get the dataset validation status', str(E))
        try:
            self.__api.handle_work_states(project_id, work_type='datasetDescription', query={"datasetId": creation_json.get('_id')})
        except Exception as E:
            raise ApiException('Unable to get the dataset description status', str(E))
        returned_json = self.__api.Datasets.getadataset(project_ID=project_id, dataset_ID=creation_json.get('_id'))

        return Dataset(self.__api, json_, returned_json)

    @Helper.try_catch
    def create_from_sql(self, name, connection_string, query, description='', modalities=2,
                        continuous_threshold=0.95, missing_threshold=0.95):
        """
        Create a Dataset from a sql database.
        Supported systems : PostgreSql
        Args:
            name (str): The name of the dataset
            connection_string (str): The connection string to the database (format : 'postgresql://username:password@host:port/database')
            query : The query to execute to fetch the data (example : 'SELECT * FROM data_table')
            description (str): The dataset description, default is ''
            modalities (int): Modality threshold for discrete variables, default is 2
            continuous_threshold (float): % of continuous values threshold for continuous variables, default is 0.95
            missing_threshold (float): % of missing values threshold for ignored variables, default is 0.95
        Returns:
            Dataset
        """
        project_id = self.__project_id
        SEPARATOR = "semicolon"
        ENCODING = "utf-8"

        dataset_data = {
            'datasetName': name,
            'description': description,
            'cached': True,
            'separator': SEPARATOR,
            'encoding': ENCODING,
            'type': 'dbAccess',
            'dbSystem': 'pgsql',
            'query': query,
            'connectionString': connection_string
        }
        json = {'project_ID': project_id, 'json': dataset_data}
        creation_json = self.__api.Datasets.createdataset(**json)

        try:
            self.__api.handle_work_states(project_id, work_type='datasetValidation', query={"datasetId": creation_json.get('_id')})
        except Exception as E:
            raise ApiException('Unable to get the dataset validation status', str(E))
        try:
            self.__api.handle_work_states(project_id, work_type='datasetDescription', query={"datasetId": creation_json.get('_id')})
        except Exception as E:
            raise ApiException('Unable to get the dataset description status', str(E))

        returned_json = self.__api.Datasets.getadataset(project_ID=project_id, dataset_ID=creation_json.get('_id'))

        return Dataset(self.__api, json, returned_json)

    @Helper.try_catch
    def filter(self):
        """
        Get all datasets. Returns a list of datasets in the selected project.
        Returns:
            list of Dataset
        """
        json = {'project_ID': self.__project_id}
        return list(map(lambda x: Dataset(self.__api, json, x), self.__api.Datasets.datasets(**json)))

    @Helper.try_catch
    def get(self, name):
        """
        Returns a dataset found by name or None if no match.
        Args:
            name (str): The name of the dataset
        Returns:
            Dataset or None
        """
        datasets = list(filter(lambda x: x.name == name, self.filter()))
        if datasets:
            return datasets[0]
        return None

    @Helper.try_catch
    def get_by_id(self, id):
        """
        Returns a dataset found by ID or None if no match.
        Args:
            id (str): The ID of the dataset
        Returns:
            Dataset or None
        """
        json = {'project_ID': self.__project_id, 'dataset_ID': id}
        return Dataset(self.__api, json, self.__api.Datasets.getadataset(**json))

    @Helper.try_catch
    def get_default(self):
        """
        Returns the default dataset in this project.
        Returns:
            Dataset
        """
        if self.__api.session.version >= self.__api.session.version.__class__('3.6'):
            project_json = self.__api.Projects.getaproject(project_ID=self.__project_id)
            return self.get_by_id(project_json.get('defaultDatasetId'))
        else:
            datasets = list(filter(lambda x: x.is_default is True, self.filter()))
            if datasets:
                return datasets[0]
        return None

    @Helper.try_catch
    def get_or_create(self, name, file_path, decimal='.', delimiter=';', encoding='UTF-8', selectedSheet=1,
                      description='', modalities=2, continuous_threshold=0.95, missing_threshold=0.95):
        """
        Returns an existing dataset matching the given name. If no match, create a new dataset from a file (csv, Excel).
        Args:
            name (str): The name of the dataset
            file_path (str): The origin path of the file
            decimal (str): Decimal separator - csv files only, default is '.'
            delimiter (str): The csv field delimiter - csv files only, default is ';'
            encoding (str): The file encoding - csv files only, default is 'UTF-8'
            selectedSheet (int): The worksheet to use (starts at 1 like in Hypercube User Interface) - Excel files only, default is 1
            description (str): The dataset description, default is ''
            modalities (int): Modality threshold for discrete variables, default is 2
            continuous_threshold (float): % of continuous values threshold for continuous variables, default is 0.95
            missing_threshold (float): % of missing values threshold for ignored variables, default is 0.95
        Returns:
            Dataset
        """
        return self.get(name) or self.create(name=name,
                                             file_path=file_path,
                                             decimal=decimal,
                                             delimiter=delimiter,
                                             encoding=encoding,
                                             selectedSheet=selectedSheet,
                                             description=description,
                                             modalities=modalities,
                                             continuous_threshold=continuous_threshold,
                                             missing_threshold=missing_threshold)


class Dataset(Base):
    """
    """
    def __init__(self, api, json_sent, json_return):
        self.__api = api
        self.__json_sent = json_sent
        self.__json_returned = json_return
        self._is_deleted = False
        self.__Xray = XrayFactory(self.__api, self.project_id)
        self.__Ruleset = RulesetFactory(self.__api, self.project_id)
        self.__Variable = VariableFactory(self.__api, self.project_id, self.dataset_id)

    def __repr__(self):
        return """\n{} : {} <{}>\n""".format(
            self.__class__.__name__,
            self.name,
            self.dataset_id
        ) + ("\t<This is the default Dataset>\n" if self.is_default else "") + \
            ("\t<! This dataset has been deleted>\n" if self._is_deleted else "") + \
            """\t- Description : {}\n\t- Size : {} bytes\n\t- Created on : {}\n\t- Modified on : {}\n""".format(
            self.description,
            self.size,
            self.created.strftime('%Y-%m-%d %H:%M:%S UTC') if self.created is not None else "N/A",
            self.modified.strftime('%Y-%m-%d %H:%M:%S UTC') if self.modified is not None else "N/A")

    # Factory part
    @property
    def Variable(self):
        """
        This object includes utilities for retrieving and interacting with variables on this dataset.
        Returns:
            An object of type VariableFactory
        """
        return self.__Variable

    # Property part
    @property
    def _json(self):
        """
        This object includes utilities for retrieving and interacting with variables on this dataset.
        Returns:
            An object of type VariableFactory
        """
        return self.__json_returned

    @property
    def _discretizations(self):
        discretizations = {}
        continuous_variables = list(filter(lambda x: x.is_discrete is False, self.variables))
        discretized_continuous_variables = list(filter(lambda x: x.discretization is not None, continuous_variables))
        for var in discretized_continuous_variables:
            discretizations[var.name] = {"type": "custom"}
        return discretizations

    @property
    def dataset_id(self):
        """
        Returns dataset ID.
        """
        return self.__json_returned.get('_id')

    @property
    def name(self):
        """
        The dataset name.
        """
        return self.__json_returned.get('datasetName')

    @property
    def description(self):
        """
        Returns all descriptions in this dataset.
        """
        return self.__json_returned.get('description')

    @property
    def size(self):
        return self.__json_returned.get('size')

    @property
    def created(self):
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

    @property
    def modified(self):
        return self.str2date(self.__json_returned.get('modified'), '%Y-%m-%dT%H:%M:%S.%fZ')

    @property
    def project_id(self):
        return self.__json_returned.get('projectId')

    @property
    def is_default(self):
        return self.__json_returned.get('selected')

    @property
    def source_file_name(self):
        return self.__json_returned.get('sourceFileName')

    @property
    def separator(self):
        return self.__json_returned.get('separator')

    @property
    def delimiter(self):
        return self.__json_returned.get('delimiter')

    @property
    def xrays(self):
        return list(filter(lambda x: x.dataset_id == self.dataset_id, self.__Xray.filter()))

    @property
    def rulesets(self):
        return list(filter(lambda x: x.dataset_id == self.dataset_id, self.__Ruleset.filter()))

    @property
    def variables(self):
        return list(self.__Variable.filter())

    # Method part
    @Helper.try_catch
    def delete(self):
        """
        Delete this dataset.
        """
        if not self._is_deleted:
            json = {'project_ID': self.project_id, 'dataset_ID': self.dataset_id}
            self.__api.Datasets.deletedataset(**json)
            self._is_deleted = True
        return self

    @Helper.try_catch
    def set_as_default(self):
        """
        Set this dataset as default.
        """
        if not self._is_deleted:
            if self.__api.session.version >= self.__api.session.version.__class__('3.6'):
                self.__json_sent = {'defaultDatasetId': self.dataset_id}
                self.__api.Projects.update(project_ID=self.project_id, json=self.__json_sent)
            else:
                self.__json_sent = {'project_ID': self.project_id, 'dataset_ID': self.dataset_id}
                self.__api.Datasets.defaultdataset(**self.__json_sent)
            self.__json_returned = DatasetFactory(self.__api, self.project_id).get_by_id(self.dataset_id).__json_returned
        return self

    @Helper.try_catch
    def split(self, train_ratio=0.7, random_state=42, keep_proportion_variable=None, train_dataset_name=None,
              train_dataset_desc=None, test_dataset_name=None, test_dataset_desc=None):
        """
        Split the dataset into two subsets for training and testing models.
        Args:
            train_ratio (float): ratio between training set size and original data set size
            random_state (int): seed used by the random number generator
            keep_proportion_variable (Variable): discrete variable which modalities
                keep similar proportions in training and test sets
            train_dataset_name (str): name of the training set
            train_dataset_desc (str): description of the training set
            test_dataset_name (str): name of the test set
            test_dataset_desc (str): description of the test set
        Returns:
            The new training and test datasets
        """
        if not self._is_deleted:
            if not 0 < train_ratio < 1:
                raise ApiException('train_ratio must be greater than 0 and lower than 1')

            if not 0 < random_state < 1001:
                raise ApiException('random_state must be greater than 0 and lower than 1001')

            if keep_proportion_variable and not keep_proportion_variable.is_discrete:
                raise ApiException('keep_proportion_variable must be a discrete variable')

            train_name = train_dataset_name or self.name + '_train'
            test_name = test_dataset_name or self.name + '_test'
            train_name, test_name = self.__get_unique_names(train_name, test_name)

            data = {
                'charactInvalidTest': '',
                'charactInvalidTrain': '',
                'dataset': self.__json_returned,
                'datasetId': self.dataset_id,
                'projectId': self.project_id,
                'randomState': random_state,
                'target': keep_proportion_variable._json if keep_proportion_variable else '',
                'testDescription': test_dataset_desc or 'Test set of dataset ' + self.name,
                'testName': test_name,
                'train': train_ratio,
                'trainDescription': train_dataset_desc or 'Train set of dataset ' + self.name,
                'trainName': train_name
            }
            json = {'project_ID': self.project_id, 'dataset_ID': self.dataset_id, 'json': data}
            split_json = self.__api.Datasets.split(**json)

            try:
                self.__api.handle_work_states(self.project_id, work_type='datasetSplit', work_id=split_json.get('id'))
            except Exception as E:
                raise ApiException('Unable to get the split status', str(E))

            factory = DatasetFactory(self.__api, self.project_id)
            return factory.get(train_name), factory.get(test_name)

    def __get_unique_names(self, train_name, test_name):
        set_names = [set.name for set in DatasetFactory(self.__api, self.project_id).filter()]
        if train_name not in set_names and test_name not in set_names:
            return train_name, test_name

        for i in range(500):
            new_train_name = "{}_{}".format(train_name, i)
            new_test_name = "{}_{}".format(test_name, i)
            if new_train_name not in set_names and new_test_name not in set_names:
                return new_train_name, new_test_name

        # last chance scenario
        suffix = str(uuid.uuid4())[:8]
        return "{}_{}".format(train_name, suffix), "{}_{}".format(test_name, suffix)

    @Helper.try_catch
    def _export(self):
        json = {
            "format": "csv",
            "useFileStream": True,
            "projectId": self.project_id,
            "datasetId": self.dataset_id,
            "limit": -1,
            "reload": True,
            "rawData": True,
            "returnHeaders": True,
            "params": {},
            "refilter": 0,
            "filename": self.name,
        }
        _filter_task = self.__api.Datasets.filteredgrid(project_ID=self.project_id,
                                                        dataset_ID=self.dataset_id,
                                                        json=json)
        _task_id = _filter_task.get('_id')
        self.__api.handle_work_states(self.project_id, work_type='dataGrid', work_id=_task_id)

        _exported = io.StringIO()
        _exported = self.__api.Datasets.exportcsv(project_ID=self.project_id,
                                                  dataset_ID=self.dataset_id,
                                                  params={"task_id": _task_id})
        return _exported

    @Helper.try_catch
    def export_csv(self, path):
        """
        Export the dataset to a csv file
        Args:
            path (str): The destination path for the resulting csv
        """
        if not self._is_deleted:
            with open(path, 'wb') as FILE_OUT:
                FILE_OUT.write(self._export())

    @Helper.try_catch
    def export_dataframe(self):
        """
        Export the dataset to a Pandas DataFrame
        Returns:
            DataFrame
        """
        if not self._is_deleted:
            pd = get_required_module('pandas')
            _data = io.StringIO(self._export().decode('utf-8'))

            # Create a dictionnary giving the string dtype for all discrete variables
            _forced_types = dict((_v.name, str) for _v in self.variables if _v.is_discrete)

            # Reading the stream with forced datatypes
            # _forced_types can be replaced with {'name_of_the_variable': str} to force specific variables
            return pd.read_csv(_data, sep=";", encoding="utf-8", dtype=_forced_types)

    @Helper.try_catch
    def get_metadata(self):
        """
        Get dataset metadata
        """
        if not self._is_deleted:
            return self.__api.Datasets.exportmetadata(project_ID=self.project_id,
                                                      dataset_ID=self.dataset_id)

    @Helper.try_catch
    def get_discreteDict(self):
        """
        Get dataset DiscreteDict
        """
        if not self._is_deleted:
            return self.__api.Datasets.exportdiscretedict(project_ID=self.project_id,
                                                          dataset_ID=self.dataset_id)

    @Helper.try_catch
    def encode_dataframe(self, name, dataframe, description='', modalities=2,
                         continuous_threshold=0.95, missing_threshold=0.95):
        '''
        Create a new dataset from a dataframe with the same encoding than the current dataset
        Args:
            name (str): The name of the dataset
            dataframe (pandas.DataFrame): The dataframe to import
            description (str): The dataset description, default is ''
            modalities (int): Modality threshold for discrete variables, default is 2
            continuous_threshold (float): % of continuous values threshold for continuous variables ,default is 0.95
            missing_threshold (float): % of missing values threshold for ignored variables, default is 0.95
            
        Returns:
            Dataset
        '''
        metadata = self.get_metadata()
        oldNames = set([
            str(var.get("varName", '')).strip().replace("\n", "")
            for var in metadata.get("variables")
        ])
        newNames = set([
            str(var).strip().replace("\n", "")
            for var in dataframe.columns
        ])
        keepVariableName = 'true' if newNames <= oldNames else 'false'
        discreteDict = self.get_discreteDict()
        dataset = DatasetFactory(self.__api, self.project_id).create_from_dataframe(name, dataframe,
                                                                                    description=description, modalities=modalities,
                                                                                    continuous_threshold=continuous_threshold, missing_threshold=missing_threshold,
                                                                                    metadata=metadata, discreteDict=discreteDict, keepVariableName=keepVariableName)
        return dataset
