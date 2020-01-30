from HyperAPI.util import Helper
from HyperAPI.hyper_api.base import Base
from HyperAPI.utils.exceptions import ApiException


class ModelTypes:
    '''
    '''
    HYPERCUBE = 'HyperCube'
    LOGISTICREGRESSION = 'LogisticRegression'
    DECISIONTREE = 'DecisionTree'
    RANDOMFOREST = 'RandomForest'
    GRADIENTBOOSTING = 'GradientBoosting'
    GRADIENTBOOSTINGREGRESSOR = 'GradientBoostingRegressor'
    XGBREGRESSOR = 'XGBRegressor'
    LASSO = 'Lasso'
    PERCEPTRON = 'Perceptron'
    LIST = [HYPERCUBE, LOGISTICREGRESSION, DECISIONTREE, RANDOMFOREST,
            GRADIENTBOOSTING, GRADIENTBOOSTINGREGRESSOR, XGBREGRESSOR,
            LASSO, PERCEPTRON, 'All']
    REGRESSORLIST = [GRADIENTBOOSTINGREGRESSOR, XGBREGRESSOR, LASSO]


class AlgoTypes:
    LIST = ['Bayesian', 'Random', None]


class AutomatedModelFactory:
    """
    """
    _PURITY = 'Purity'
    _COVERAGE = 'Coverage'
    _LIFT = 'Lift'

    _INDICATOR_DISCRETE_WITH_MODALITY = "Discrete variable with a modality"

    def __init__(self, api, project_id):
        self.__api = api
        self.__project_id = project_id

    @Helper.try_catch
    def filter(self):
        """
        Get all models.

        Returns:
            List of models
        """
        project_id = self.__project_id

        json = {'project_ID': project_id}
        return [AutomatedModel(self.__api, x) for x in self.__api.AutomatedPrediction.getautomatedmodels(**json)]

    @Helper.try_catch
    def get_all(self):
        """
        Get a model matching the given projectId or None if there is no match.

        Args:
            name (str): The name of the dataset

        Returns:
            The Model or None
        """
        return self.__api.AutomatedPrediction.getautomatedmodels(project_ID=self.__project_id)

    @Helper.try_catch
    def get_by_id(self, id):
        """
        Get the model matching the given ID or None if there is no match.

        Args:
            id (str): The id of the Model

        Returns:
            The Model or None
        """
        return self.__api.AutomatedPrediction.getautomatedmodel(project_ID=self.__project_id, model_ID=id)

    @Helper.try_catch
    def create(self, dataset, target, params):
        """
        Private method. Create a classifier or regressor Scikit-learn model

        Args:
            dataset (Dataset): Dataset the model is fitted on
            target (Target): Target used to generate the model
            params (dict): parameters used by the HyperWorker
        Returns:
            the created model
        """
        if params['modelType'] not in ModelTypes.LIST:
            print('Unexpected model name : {}, valid options are : {}'.format(params['modelType'], ', '.join(ModelTypes.LIST)))
            return
        if params['algoType'] not in AlgoTypes.LIST:
            print('Unexpected model name : {}, valid options are : {}'.format(params['algoType'], ', '.join(AlgoTypes.LIST)))
            return

        if target.indicator_type == self._INDICATOR_DISCRETE_WITH_MODALITY:
            variable = next(variable for variable in dataset.variables if variable.name == target.variable_name)
            index = variable.modalities.index(target.modality)
            datasetPurity = variable.purities[index]
            score_purity_min = round(datasetPurity, 3)
            coverage_min = 10 if (variable.frequencies[index] < 1000) else 0.01

        scores = []
        for score_id, score_type in zip(target.score_ids, target.scores):
            score = {
                'deleted': False,
                'kpiFamily': target.indicator_family,
                'kpiName': target.name,
                'kpiType': target.indicator_type,
                'output': target.variable_name,
                'projectId': target.project_id,
                'scoreType': score_type,
                '_id': score_id
            }
            if target.indicator_type == self._INDICATOR_DISCRETE_WITH_MODALITY:
                score['omodality'] = target.modality
                if score_type == self._PURITY:
                    score['minValue'] = score_purity_min
                elif score_type == self._COVERAGE:
                    score['minValue'] = coverage_min
                elif score_type == self._LIFT:
                    score['minValue'] = 1
                scores.append(score)
                scores = [score for score in scores if score['scoreType'] == self._PURITY or score['scoreType'] == self._COVERAGE]
            else:
                scores.append(score)

        kpisel = {
            'kpiFamily': target.indicator_family,
            'kpiName': target.name,
            'kpiType': target.indicator_type,
            'projectId': target.project_id,
            'selectedBy': 'target'
        }
        if target.indicator_type == self._INDICATOR_DISCRETE_WITH_MODALITY:
            kpisel['datasetPurity'] = datasetPurity

        params['sourceFileName'] = dataset.source_file_name
        params['target'] = scores
        data = {
            'datasetId': dataset.dataset_id,
            'datasetName': dataset.name,
            'kpi': kpisel,
            'modelName': params['modelName'],
            'params': params,
            'projectId': dataset.project_id,
            'selectedDataset': dataset._json,
            'type': 'automatedModels',
            'validTarget': True,
        }
        new_automodel = self.__api.AutomatedPrediction.createautomatedmodel(project_ID=self.__project_id, json=data)
        try:
            self.__api.handle_work_states(self.__project_id, work_type='automatedModels', work_id=new_automodel.get('workId'))
        except Exception as E:
            raise ApiException('Unable to get the automated model status', str(E))

        return AutomatedModel(self.__api, new_automodel)

    @Helper.try_catch
    def publish(self, autoModel_ID, model_ID):
        """
        Publish the best model in the model page.

        Args:
            autoModel_ID: The id of the automated Model
            model_ID: The id of the Model
        """

        return self.__api.AutomatedPrediction.publishautomatedmodel(project_ID=self.__project_id, model_ID=autoModel_ID, generatedmodel_ID=model_ID)

    @Helper.try_catch
    def get_generated_models(self, model_ID):
        """
        Get all the models generated by the automated optimization

        Args:
            model_ID: The id of the automated Model
        """
        return self.__api.AutomatedPrediction.getgeneratedmodels(project_ID=self.__project_id, model_ID=model_ID)


class AutomatedModel(Base):
    """
    """
    def __init__(self, api, json_return):
        self.__api = api
        self.__json_returned = json_return
        self._is_deleted = False

    def __repr__(self):
        return """\n{} : {} <{}>\n""".format(
            'Model',
            self.name,
            self.id
        ) + ("\t<! This model has been deleted>\n" if self._is_deleted else "") + \
            """\t- Dataset : {}\n\t- Target : {}\n\t- Created on : {}\n""".format(
            self.dataset_name,
            self.kpi_name,
            self.created.strftime('%Y-%m-%d %H:%M:%S UTC'))

    @property
    def _json(self):
        return self.__json_returned

    @property
    def algoType(self):
        return self.__json_returned.get('algoType')

    @property
    def dataset_name(self):
        return self.__json_returned.get('datasetName')

    @property
    def dataset_id(self):
        return self.__json_returned.get('datasetId')

    @property
    def id(self):
        return self.__json_returned.get('_id')

    @property
    def kpi_name(self):
        return self.__json_returned.get('kpiName')

    @property
    def project_id(self):
        return self.__json_returned.get('projectId')

    @property
    def name(self):
        """
        The model name.
        """
        return self.__json_returned.get('modelName')

    @property
    @Helper.try_catch
    def created(self):
        return self.str2date(self.__json_returned.get('createdAt'), '%Y-%m-%dT%H:%M:%S.%fZ')
