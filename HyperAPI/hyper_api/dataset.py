from os.path import getsize, split
import sys
import uuid
import io
from HyperAPI.util import Helper
from HyperAPI.utils.exceptions import ApiException
from HyperAPI.utils.imports import get_required_module
from HyperAPI.hyper_api.base import Base
from HyperAPI.hyper_api.variable import Variable, VariableFactory
from HyperAPI.hyper_api.xray import Xray, XrayFactory
from HyperAPI.hyper_api.ruleset import Ruleset, RulesetFactory


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
               delimiter='semicolon', encoding='UTF-8', selectedSheet=1,
               description='', modalities=2, continuous_threshold=0.95, missing_threshold=0.95,
               metadata_file_path=None, discreteDict_file_path=None, keepVariableName=None):
        """
        create(name, file_path, decimal='.',delimiter='semicolon', encoding='UTF-8', selectedSheet=1, description='', modalities=2, continuous_threshold=0.95, missing_threshold=0.95, metadata_file_path=None, discreteDict_file_path=None, keepVariableName=None)

        Create a Dataset from a file (csv, Excel)

        Args:
            name (str): The name of the dataset
            file_path (str): The origin path of the file
            decimal (str): Decimal separator - csv files only, default is '.'
            delimiter (str): The csv field delimiter - csv files only, default is 'semicolon'
            encoding (str): The file encoding - csv files only, default is 'UTF-8'
            selectedSheet (int): The worksheet to use (starts at 1 like in Hypercube User Interface) - Excel files only, default is 1
            description (str): The dataset description, default is ''
            modalities (int): Modality threshold for discrete variables, default is 2
            continuous_threshold (float): % of continuous values threshold for continuous variables, default is 0.95
            missing_threshold (float): % of missing values threshold for ignored variables, default is 0.95

        Returns:
            Dataset: Created dataset
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
        create_from_dataframe(name, dataframe, description='', modalities=2, continuous_threshold=0.95, missing_threshold=0.95, metadata=None, discreteDict=None, keepVariableName=None)
            
        Create a Dataset from a Pandas DataFrame

        Args:
            name (str): The name of the dataset
            dataframe (pandas.DataFrame): The dataframe to import
            description (str): The dataset description, default is ''
            modalities (int): Modality threshold for discrete variables, default is 2
            continuous_threshold (float): % of continuous values threshold for continuous variables ,default is 0.95
            missing_threshold (float): % of missing values threshold for ignored variables, default is 0.95

        Returns:
            Dataset: Created dataset
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
        create_from_sql(name, connection_string, query, description='', modalities=2, continuous_threshold=0.95, missing_threshold=0.95)

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
            Dataset: created Dataset
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
        filter()
        Get all datasets. Returns a list of datasets in the selected project.

        Returns:
            list(Dataset): All datasets belonging to the project
        """
        json = {'project_ID': self.__project_id}
        return list(map(lambda x: Dataset(self.__api, json, x), self.__api.Datasets.datasets(**json)))

    @Helper.try_catch
    def get(self, name):
        """
        get(name)
        Returns a dataset found by name or None if no match.

        Args:
            name (str): The name of the dataset

        Returns:
            Dataset or None: Retrieved dataset
        """
        datasets = list(filter(lambda x: x.name == name, self.filter()))
        if datasets:
            return datasets[0]
        return None

    @Helper.try_catch
    def get_by_id(self, id):
        """
        get_by_id(id)
        Returns a dataset found by ID or None if no match.

        Args:
            id (str): The ID of the dataset

        Returns:
            Dataset or None: Retrieved dataset
        """
        json = {'project_ID': self.__project_id, 'dataset_ID': id}
        return Dataset(self.__api, json, self.__api.Datasets.getadataset(**json))

    @Helper.try_catch
    def get_default(self):
        """
        get_default()
        Get the default dataset of this project.

        Returns:
            Dataset: Default dataset
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
        get_or_create(name, file_path, decimal='.', delimiter=';', encoding='UTF-8', selectedSheet=1, description='', modalities=2, continuous_threshold=0.95, missing_threshold=0.95)

        Get an existing dataset matching the given name. If no match, create a new dataset from a file (csv, Excel).
        
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
            Dataset: Retrieved or created dataset
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
    Dataset()
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
        VariableFactory: Tool class for creating and retrieving existing variables in this project
        """
        return self.__Variable

    # Property part
    @property
    def _json(self):
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
        str: Dataset ID
        """
        return self.__json_returned.get('_id')

    @property
    def name(self):
        """
        str: Dataset name
        """
        return self.__json_returned.get('datasetName')

    @property
    def description(self):
        """
        str: Dataset description
        """
        return self.__json_returned.get('description')

    @property
    def size(self):
        """
        int: Size in bytes
        """
        return self.__json_returned.get('size')

    @property
    def created(self):
        """
        datetime: Created date
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

    @property
    def modified(self):
        """
        datetime: last modification date
        """
        return self.str2date(self.__json_returned.get('modified'), '%Y-%m-%dT%H:%M:%S.%fZ')

    @property
    def source_file_name(self):
        """
        str: source file name
        """
        return self.__json_returned.get('sourceFileName')

    @property
    def project_id(self):
        """
        str: project ID
        """
        return self.__json_returned.get('projectId')

    @property
    def is_default(self):
        """
        Boolean: indicating if this project is the default project.
        """
        if self._is_deleted:
            return False
        json = {'project_ID': self.project_id}
        json_returned = self.__api.Projects.getaproject(**json)
        if self.__api.session.version >= self.__api.session.version.__class__('6.0.1'):
            return self.dataset_id == json_returned.get('defaultDatasetId')
        else:
            return next(
                (
                    _dataset.get('selected') 
                    for _dataset in json_returned.get('Datasets') 
                    if _dataset.get('_id') == self.dataset_id
                ), 
                False
            )

    @property
    def separator(self):
        """
        str: field separator in the source file.
        """
        return self.__json_returned.get('separator')

    @property
    def delimiter(self):
        """
        str: decimal delimiter in the source file.
        """
        return self.__json_returned.get('delimiter')

    @property
    def xrays(self):
        """
        list(Xray): All XRays related on the current dataset.
        """
        return list(filter(lambda x: x.dataset_id == self.dataset_id, self.__Xray.filter()))

    @property
    def rulesets(self):
        """
        list(Ruleset): All rulesets related on the current dataset.
        """
        return list(filter(lambda x: x.dataset_id == self.dataset_id, self.__Ruleset.filter()))

    @property
    def variables(self):
        """
        list(Variable): All variables related on the current dataset.
        """
        return list(self.__Variable.filter())

    # Method part
    @Helper.try_catch
    def delete(self):
        """
        delete()
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
        set_as_default()
        Set this dataset as default.

        Returns:
            Dataset: the dataset itself
        """
        if not self._is_deleted:
            if self.__api.session.version >= self.__api.session.version.__class__('3.6'):
                self.__json_sent = {'project_ID': self.project_id, 'json': {'defaultDatasetId': self.dataset_id}}
                self.__api.Projects.updateproject(**self.__json_sent)
            else:
                self.__json_sent = {'project_ID': self.project_id, 'dataset_ID': self.dataset_id}
                self.__api.Datasets.defaultdataset(**self.__json_sent)
            self.__json_returned = DatasetFactory(self.__api, self.project_id).get_by_id(self.dataset_id).__json_returned
        return self

    @Helper.try_catch
    def split(self, train_ratio=0.7, random_state=42, keep_proportion_variable=None, train_dataset_name=None,
              train_dataset_desc=None, test_dataset_name=None, test_dataset_desc=None):
        """
        split(train_ratio=0.7, random_state=42, keep_proportion_variable=None, train_dataset_name=None, train_dataset_desc=None, test_dataset_name=None, test_dataset_desc=None)
        
        Split the dataset into two subsets for training and testing models.

        Args:
            train_ratio (float): ratio between training set size and original data set size, default = 0.7
            random_state (int): seed used by the random number generator, default = 42
            keep_proportion_variable (Variable): discrete variable which modalities
                keep similar proportions in training and test sets, default = None
            train_dataset_name (str): name of the training set, default = None
            train_dataset_desc (str): description of the training set, default = None
            test_dataset_name (str): name of the test set, default = None
            test_dataset_desc (str): description of the test set, default = None

        Returns:
            (Dataset, Dataset): The new training and test datasets
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
    def __export(self):
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
        export_csv(path)
        Export the dataset to a csv file

        Args:
            path (str): The destination path for the resulting csv
        """
        if not self._is_deleted:
            with open(path, 'wb') as FILE_OUT:
                FILE_OUT.write(self.__export())

    @Helper.try_catch
    def export_dataframe(self):
        """
        export_dataframe()
        Export the dataset to a Pandas DataFrame

        Returns:
            pd.DataFrame: exported dataset
        """
        if not self._is_deleted:
            pd = get_required_module('pandas')
            _data = io.StringIO(self.__export().decode('utf-8'))

            # Create a dictionnary giving the string dtype for all discrete variables
            _forced_types = dict((_v.name, str) for _v in self.variables if _v.is_discrete)

            # Reading the stream with forced datatypes
            # _forced_types can be replaced with {'name_of_the_variable': str} to force specific variables
            return pd.read_csv(_data, sep=";", encoding="utf-8", dtype=_forced_types)

    @Helper.try_catch
    def get_metadata(self):
        """
        get_metadata()
        Get dataset metadata

        Returns:
            Dict: metadata
        """
        if not self._is_deleted:
            return self.__api.Datasets.exportmetadata(project_ID=self.project_id,
                                                      dataset_ID=self.dataset_id)

    @Helper.try_catch
    def _get_discreteDict(self):
        """
        Get dataset DiscreteDict

        Returns:
            Dict: discreteDict
        """
        if not hasattr(self.__api.Datasets, "exportdiscretedict"):
            raise NotImplementedError('The feature is not available on this platform')

        if not self._is_deleted:
            return self.__api.Datasets.exportdiscretedict(project_ID=self.project_id,
                                                          dataset_ID=self.dataset_id)

    @Helper.try_catch
    def encode_dataframe(self, name, dataframe, description='', modalities=2,
                         continuous_threshold=0.95, missing_threshold=0.95):
        '''
        encode_dataframe(name, dataframe, description='', modalities=2, continuous_threshold=0.95, missing_threshold=0.95)

        Create a new dataset from a dataframe with the same encoding than the current dataset

        Args:
            name (str): The name of the dataset
            dataframe (pandas.DataFrame): The dataframe to import
            description (str): The dataset description, default is ''
            modalities (int): Modality threshold for discrete variables, default is 2
            continuous_threshold (float): % of continuous values threshold for continuous variables ,default is 0.95
            missing_threshold (float): % of missing values threshold for ignored variables, default is 0.95

        Returns:
            Dataset: Encoded dataset
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
        discreteDict = self._get_discreteDict()
        dataset = DatasetFactory(self.__api, self.project_id).create_from_dataframe(name, dataframe,
                                                                                    description=description, modalities=modalities,
                                                                                    continuous_threshold=continuous_threshold, missing_threshold=missing_threshold,
                                                                                    metadata=metadata, discreteDict=discreteDict, keepVariableName=keepVariableName)
        return dataset
