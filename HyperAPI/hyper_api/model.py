from HyperAPI.util import Helper
from HyperAPI.hyper_api.base import Base
from HyperAPI.utils.exceptions import ApiException
from HyperAPI.utils.imports import get_required_module
from datetime import datetime
from io import StringIO
from json import dump, dumps


class AlgoTypes:
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
    TIMESERIESFORECASTER= 'TimeSeriesForecaster' 

    LIST = [HYPERCUBE, LOGISTICREGRESSION, DECISIONTREE, RANDOMFOREST,
            GRADIENTBOOSTING, GRADIENTBOOSTINGREGRESSOR, XGBREGRESSOR,
            LASSO, PERCEPTRON, TIMESERIESFORECASTER]
    REGRESSORLIST = [GRADIENTBOOSTINGREGRESSOR, XGBREGRESSOR, LASSO, TIMESERIESFORECASTER]


class Curves:
    '''
    '''
    ROC = "ROC curve"
    GAIN = "Gain curve"
    LIFT = "Lift curve"
    PURITY = "Purity curve"
    PRECISIONRECALL = "Precision Recall"
    LIST = [ROC, GAIN, LIFT, PURITY, PRECISIONRECALL]


class ExportFormats:
    '''
    '''
    PYTHON = "Python"
    CSV = "csv"
    JSON = "JSON"
    R = "R"
    SCALA = "Scala"
    JAVA = "Java"
    JAVASCRIPT = "JavaScript"
    MYSQL = "MySQL"
    PLSQL = "PLSQL"
    TSQL = "TSQL"
    LIST = [PYTHON, CSV, JSON, R, SCALA, JAVA, JAVASCRIPT, MYSQL, TSQL]


class ModelFactory:
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

        data = {
            'projectId': project_id,
            'type': {'$in': ["prediction", "applyOtherModel", "applyHypercubeModel",
                             "predictionRuleset", "otherPrediction", "hypercubePrediction"]
                     }
        }
        json = {'project_ID': project_id, 'json': data}
        json_returned = self.__api.Task.task(**json)
        return [HyperCube(self.__api, model_json) if model_json.get('algoType') == AlgoTypes.HYPERCUBE
                else RegressorModel(self.__api, model_json) if model_json.get('algoType') in AlgoTypes.REGRESSORLIST
                else ClassifierModel(self.__api, model_json) for model_json in json_returned]

    @Helper.try_catch
    def get(self, name):
        """
        Get a model matching the given name or None if there is no match.

        Args:
            name (str): The name of the dataset

        Returns:
            The Model or None
        """
        models = list(filter(lambda x: x.name == name, self.filter()))
        if models:
            return models[0]
        return None

    @Helper.try_catch
    def get_by_id(self, id):
        """
        Get the model matching the given ID or None if there is no match.

        Args:
            id (str): The id of the Model

        Returns:
            The Model or None
        """
        models = [model for model in self.filter() if model.id == id]
        if models:
            return models[0]
        return None

    @Helper.try_catch
    def predict_from_ruleset(self, dataset_source, dataset_predict, rulesetname, name, target, nb_minimizations=1, coverage_increment=0.01):
        """
        Create a HyperCube classifier model from a ruleset

        Args:
            dataset_source (Dataset): Dataset where the ruleset was performed
            dataset_predict (Dataset): Dataset used to make these predictions
            rulesetname (str): Name of the ruleset you want to use to create the model
            name (str): Name of the new model
            target (Target): Target used to generate the model
            nb_minimizations (int): Number of minimizations to perform on the ruleset, default is 1
            coverage_increment (float): Percentage increment of target samples that a new rule must bring to be added to the minimized ruleset,
                default is 0.01

        Returns:
            The Model
        """
        kpiData = []
        for _id, _type in zip(target.score_ids, target.scores):
            kpiData.append({'kpiId': _id,
                            'type': _type,
                            'kpiFamily': target.indicator_family,
                            'scoreType': _type,
                            'kpiType': target.indicator_type,
                            'output': target.variable_name,
                            'kpiName': target.name,
                            'omodality': target.modality
                            })

        json = {'projectId': dataset_predict.project_id,
                'datasetId': dataset_predict.dataset_id,
                'datasetName': dataset_predict.name,
                'type': 'predictionRuleset',
                'params': {'modelName': name,
                           'sourceDatasetId': dataset_source.dataset_id,
                           'sourceFileName': dataset_predict.source_file_name,
                           'rulesetName': rulesetname,
                           'delimiter': dataset_source.delimiter,
                           'algoType': 'HyperCube',
                           'target': kpiData,
                           'nbMinimizations': nb_minimizations,
                           'coverageIncrement': coverage_increment
                           }
                }

        model = self.__api.Task.createtask(project_ID=self.__project_id, json=json)
        self.__api.handle_work_states(self.__project_id, work_type='predictionRuleset', work_id=model.get('_id'))
        return HyperCube(self.__api, model)

    @Helper.try_catch
    def create_hypercube(self, dataset, name, target, purity_min=None, coverage_min=None, rule_complexity=2, quantiles=10, min_marginal_contribution=None,
                         max_complexity=3, nb_minimizations=1, coverage_increment=0.01, split_ratio=0.7, nb_iterations=1,
                         purity_tolerance=0.1, enable_custom_discretizations=True, save_all_rules=False):
        """
        Create a HyperCube classifier model

        Args:
            dataset (Dataset): Dataset the model is fitted on
            name (str): Name of the new model
            target (Target): Target used to generate the model
            purity_min (float): Minimum purity of rules, default is the entire dataset purity
            coverage_min (int): Minimum coverage of the target population for each rule, default is 10
            rule_complexity (int): Maximum number of variables in rules, default is 2
            quantiles (int): Number of bins for all continuous numeric variables during quantization, default is 10
            min_marginal_contribution (float): a new rule R', created by adding a new constraint to an existing rule R (and thus increasing its complexity),
                is added to the ruleset if and only if it increases the original purity of R by the minimum marginal contribution or more. Default is 0.1
            max_complexity (int): maximum number of variables contained in the rules created during the local complexity increase phase. Default is 3
            nb_minimizations (int): Number of minimizations to perform on the ruleset, default is 1
            coverage_increment (float): Percentage increment of target samples that a new rule must bring to be added to the minimized ruleset,
                default is 0.01
            split_ratio (float): the first step in the model generation is the random split of the original dataset into a learning (or train) dataset
                representing by default 70% of the original dataset, and a validation (or test) dataset containing the remaining 30%. Default is 0.7
            nb_iterations (int): The final model is the result of several models based on different splits of the original dataset, using a bootstrap method.
                The parameter "Number of iterations" corresponds to the number of these splits that are made. Default is 1
            purity_tolerance (float): maximum spread between the purities of the rules applied to the learning and validation datasets
            enable_custom_discretizations (boolean): when ticked use the custom discretization(s) link to the selected dataset,
                eventually use "Quantiles" parameter for remaining variables. Default is True
            save_all_rules (boolean): save all generated rules in a new ruleset. Default is False

        Returns:
            the created model
        """
        variable = next(variable for variable in dataset.variables if variable.name == target.variable_name)
        index = variable.modalities.index(target.modality)
        datasetPurity = variable.purities[index]
        score_purity_min = purity_min or round(datasetPurity, 3)

        if min_marginal_contribution is None:
            if score_purity_min > 0.99:
                min_marginal_contribution = round(1 / score_purity_min - 1, 3)
            elif score_purity_min > 0.9:
                min_marginal_contribution = round(0.99 / score_purity_min - 1, 3)
            else:
                min_marginal_contribution = 0.1

        coverage_min = coverage_min or 10 if (variable.frequencies[index] < 1000) else 0.01

        if enable_custom_discretizations is True:
            discretizations = dataset._discretizations
        else:
            discretizations = {}

        scores = []
        for score_id, score_type in zip(target.score_ids, target.scores):
            score = {
                'deleted': False,
                'kpiFamily': target.indicator_family,
                'kpiName': target.name,
                'kpiType': target.indicator_type,
                'omodality': target.modality,
                'output': target.variable_name,
                'projectId': target.project_id,
                'scoreType': score_type,
                '_id': score_id
            }
            if score_type == self._PURITY:
                score['minValue'] = score_purity_min
            elif score_type == self._COVERAGE:
                score['minValue'] = coverage_min
            elif score_type == self._LIFT:
                score['minValue'] = 1
            scores.append(score)

        kpisel = {
            'datasetPurity': datasetPurity,
            'kpiFamily': target.indicator_family,
            'kpiName': target.name,
            'kpiType': target.indicator_type,
            'projectId': target.project_id,
            'scores': scores,
            'selectedBy': 'target'
        }

        params = {
            'algoType': 'HyperCube',
            'complexityExhaustive': rule_complexity,
            'countQuantiles': quantiles,
            'coverageIncrement': coverage_increment,
            'coverageThreshold': 10,
            'delimiter': 'semicolon',
            'discretizations': discretizations,
            'dtmaxdepth': 4,
            'elasticNetParam': 0,
            'enableCustomDiscretizations': enable_custom_discretizations,
            'featureSubsetStrategy': 'sqrt',
            'gbmaxdepth': 3,
            'gbn_estimators': 100,
            'kpis': [],
            'learning_rate': 0.1,
            'lrCost': 1,
            'maxComplexity': max_complexity,
            'maxDepth': 3,
            'maxIter': 100,
            'minInfoGain': 0,
            'minInstance': 1,
            'minMarginalContribution': min_marginal_contribution,
            'minObservation': 3,
            'missingValues': 0.1,
            'modelName': name,
            'nbMaxModality': 50,
            'nbMinObservation': 10,
            'nbMinimizations': nb_minimizations,
            'nbModels': nb_iterations,
            'numTrees': 10,
            'percentageSplit': split_ratio,
            'purityThreshold': score_purity_min,
            'purityTolerance': purity_tolerance,
            'regParam': 0,
            'replaceMissingValues': 'Median',
            'rfmaxdepth': 2,
            'rfn_estimators': 100,
            'saveAllRules': 1 if save_all_rules else 0,
            'sourceFileName': dataset.source_file_name,
            'splitRatio': split_ratio,
            'stepSize': 0.1,
            'subsamplingRate': 1,
            'target': [score for score in scores if score['scoreType'] == self._PURITY or score['scoreType'] == self._COVERAGE],
            'tol': 0.000001
        }

        data = {
            'algo': 'HyperCube',
            'algolist': AlgoTypes.LIST,
            'datasetId': dataset.dataset_id,
            'datasetName': dataset.name,
            'dtcr': 'gini',
            'enableCustomDiscretizations': enable_custom_discretizations,
            'gbloss': 'deviance',
            'keyIndicators': [],
            'kpisel': kpisel,
            'lrPenalty': 'l2',
            'lrSolver': 'liblinear',
            'modelFile': '',
            'modelName': name,
            'params': params,
            'projectId': dataset.project_id,
            'rfcr': 'gini',
            'saveAllRules': save_all_rules,
            'selectedDataset': dataset._json,
            'sourceFileName': '',
            'spark': False,
            'type': 'hypercubePrediction',
            'validTarget': True
        }

        json = {'project_ID': dataset.project_id, 'json': data}
        json_returned = self.__api.Task.createtask(**json)

        try:
            self.__api.handle_work_states(dataset.project_id, work_type=json_returned.get('type'), work_id=json_returned.get('_id'))
        except Exception as E:
            raise ApiException('Unable to create the HyperCube model ' + name, str(E))

        return HyperCube(self.__api, json_returned)

    @Helper.try_catch
    def get_or_create_hypercube(self, dataset, name, target=None, purity_min=None, coverage_min=None, rule_complexity=2, quantiles=10,
                                min_marginal_contribution=None, max_complexity=3, nb_minimizations=1, coverage_increment=0.01, split_ratio=0.7,
                                nb_iterations=1, purity_tolerance=0.1, enable_custom_discretizations=True, save_all_rules=False):
        """
        Get or Create a classifier or regressor model

        Args:
            dataset (Dataset): Dataset the model is fitted on
            name (str): Name of the model
            target (Target): Target used to generate the model
            purity_min (float): Minimum purity of rules, default is the entire dataset purity
            coverage_min (int): Minimum coverage of the target population for each rule, default is 10
            rule_complexity (int): Maximum number of variables in rules, default is 2
            quantiles (int): Number of bins for all continuous numeric variables during quantization, default is 10
            min_marginal_contribution (float): a new rule R', created by adding a new constraint to an existing rule R (and thus increasing its complexity),
                is added to the ruleset if and only if it increases the original purity of R by the minimum marginal contribution or more. Default is 0.1
            max_complexity (int): maximum number of variables contained in the rules created during the local complexity increase phase. Default is 3
            nb_minimizations (int): Number of minimizations to perform on the ruleset, default is 1
            coverage_increment (float): Percentage increment of target samples that a new rule must bring to be added to the minimized ruleset,
                default is 0.01
            split_ratio (float): the first step in the model generation is the random split of the original dataset into a learning (or train) dataset
                representing by default 70% of the original dataset, and a validation (or test) dataset containing the remaining 30%. Default is 0.7
            nb_iterations (int): The final model is the result of several models based on different splits of the original dataset, using a bootstrap method.
                The parameter "Number of iterations" corresponds to the number of these splits that are made. Default is 1
            purity_tolerance (float): maximum spread between the purities of the rules applied to the learning and validation datasets
            enable_custom_discretizations (boolean): when ticked use the custom discretization(s) link to the selected dataset,
                eventually use "Quantiles" parameter for remaining variables. Default is True
            save_all_rules (boolean): save all generated rules in a new ruleset. Default is False

        Returns:
            the created model or existing model matching name parameter
        """

        model = self.get(name)
        if model is not None:
            return model

        return self.create_hypercube(dataset, name, target, purity_min, coverage_min, rule_complexity, quantiles, min_marginal_contribution,
                                     max_complexity, nb_minimizations, coverage_increment, split_ratio, nb_iterations, purity_tolerance,
                                     enable_custom_discretizations, save_all_rules)

    @Helper.try_catch
    def __create_skModel(self, dataset, target, params):
        """
        Private method. Create a classifier or regressor Scikit-learn model

        Args:
            dataset (Dataset): Dataset the model is fitted on
            target (Target): Target used to generate the model
            params (dict): parameters used by the HyperWorker
        Returns:
            the created model
        """
        if params['algoType'] not in AlgoTypes.LIST:
            print('Unexpected algorithm type : {}, valid options are : {}'.format(params['algoType'], ', '.join(AlgoTypes.LIST)))
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
            'algo': params['algoType'],
            'algolist': AlgoTypes.LIST,
            'datasetId': dataset.dataset_id,
            'datasetName': dataset.name,
            'kpisel': kpisel,
            'modelName': params['modelName'],
            'params': params,
            'projectId': dataset.project_id,
            'selectedDataset': dataset._json,
            'type': 'otherPrediction',
            'validTarget': True,
        }

        json = {'project_ID': dataset.project_id, 'json': data}
        json_returned = self.__api.Task.createtask(**json)
        try:
            self.__api.handle_work_states(dataset.project_id, work_type=json_returned.get('type'), work_id=json_returned.get('_id'))
        except Exception as E:
            raise ApiException('Unable to create the model ' + params.get('modelName'), str(E))

        if params['algoType'] in AlgoTypes.REGRESSORLIST:
            return RegressorModel(self.__api, json_returned)
        else:
            return ClassifierModel(self.__api, json_returned)

    @Helper.try_catch
    def create_DecisionTree(self, dataset, name, target, max_depth=4, criterion='gini', split_ratio=0.7, enable_custom_discretizations=True,
                            nbMaxModality=50, nbMinObservation=10, replaceMissingValues='Median'):
        """
        Create a decision tree classifier model

        Args:
            dataset (Dataset): Dataset the model is fitted on
            name (str): Name of the new model
            target (Target): Target used to generate the model
            max_depth (int): The maximum depth of the tree. Default is 4
            criterion (str): The function to measure the quality of a split. Supported criteria are “gini” for the Gini impurity and “entropy” for the
                information gain.Default is 'gini'
            split_ratio (float): the first step in the model generation is the random split of the original dataset into a learning (or train) dataset
                representing by default 70% of the original dataset, and a validation (or test) dataset containing the remaining 30%. Default is 0.7
            enable_custom_discretizations (boolean): when ticked use the custom discretization(s) link to the selected dataset. Default is True
            nbMaxModality (int): Maximum number of modalities per variable. Default is 50
            nbMinObservation (int): Modalities with a number of observations lower will be ignored. Default is 10
            replaceMissingValues (str): Method to replace missing values. Available methods are 'Median', 'Mean' and 'Delete'. Default is 'Median'
        Returns:
            the created model
        """
        hyperParameters = {'max_depth': max_depth, 'criterion': criterion}

        discretizations = {}
        if enable_custom_discretizations is True:
            discretizations = dataset._discretizations
        params = {
            'nbMaxModality': nbMaxModality,
            'nbMinObservation': nbMinObservation,
            'replaceMissingValues': replaceMissingValues,
            'paramsSk': dumps(hyperParameters),
            'algoType': AlgoTypes.DECISIONTREE,
            'modelName': name,
            'enable_custom_discretizations': enable_custom_discretizations,
            'discretizations': discretizations,
        }
        return self.__create_skModel(dataset, target, params)

    @Helper.try_catch
    def create_LogisticRegression(self, dataset, name, target, penalty='l2', C=1, solver='liblinear', split_ratio=0.7, enable_custom_discretizations=True,
                                  nbMaxModality=50, nbMinObservation=10, replaceMissingValues='Median'):
        """
        Create a logistic regression model

        Args:
            dataset (Dataset): Dataset the model is fitted on
            name (str): Name of the new model
            target (Target): Target used to generate the model
            penalty (str): Nnorm used in the penalization. Supported penalties are 'l1' and 'l2'. The ‘newton-cg’, ‘sag’ and ‘lbfgs’ solvers support
                only l2 penalties. Default is 'l2'
            C (float): Inverse of regularization strength; must be a positive float. Like in support vector machines, smaller values specify stronger
                regularization. Greater or equal to 1. Default is 1.
            solver (str): Compatible solvers are 'newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'. Default is 'liblinear'
            split_ratio (float): the first step in the model generation is the random split of the original dataset into a learning (or train) dataset
                representing by default 70% of the original dataset, and a validation (or test) dataset containing the remaining 30%. Default is 0.7
            enable_custom_discretizations (boolean): when ticked use the custom discretization(s) link to the selected dataset. Default is True
            nbMaxModality (int): Maximum number of modalities per variable. Default is 50
            nbMinObservation (int): Modalities with a number of observations lower will be ignored. Default is 10
            replaceMissingValues (str): Method to replace missing values. Available methods are 'Median', 'Mean' and 'Delete'. Default is 'Delete'
        Returns:
            the created model
        """
        hyperParameters = {'penalty': penalty, 'C': C, 'solver': solver}
        discretizations = {}
        if enable_custom_discretizations is True:
            discretizations = dataset._discretizations
        params = {
            'nbMaxModality': nbMaxModality,
            'nbMinObservation': nbMinObservation,
            'replaceMissingValues': replaceMissingValues,
            'paramsSk': dumps(hyperParameters),
            'algoType': AlgoTypes.LOGISTICREGRESSION,
            'modelName': name,
            'enable_custom_discretizations': enable_custom_discretizations,
            'discretizations': discretizations,
        }
        return self.__create_skModel(dataset, target, params)

    @Helper.try_catch
    def create_RandomForest(self, dataset, name, target, n_estimators=100, max_depth=2, criterion='gini', split_ratio=0.7, enable_custom_discretizations=True,
                            nbMaxModality=50, nbMinObservation=10, replaceMissingValues='Median'):
        """
        Create a random forest model

        Args:
            dataset (Dataset): Dataset the model is fitted on
            name (str): Name of the new model
            target (Target): Target used to generate the model
            n_estimators (int): Tnumber of trees in the forest. Default is 100
            max_depth (int): The maximum depth of the tree. Default is 2
            criterion (str): The function to measure the quality of a split. Supported criteria are “gini” for the Gini impurity and “entropy” for the
                information gain.Default is 'gini'
            split_ratio (float): the first step in the model generation is the random split of the original dataset into a learning (or train) dataset
                representing by default 70% of the original dataset, and a validation (or test) dataset containing the remaining 30%. Default is 0.7
            enable_custom_discretizations (boolean): when ticked use the custom discretization(s) link to the selected dataset. Default is True
            nbMaxModality (int): Maximum number of modalities per variable. Default is 50
            nbMinObservation (int): Modalities with a number of observations lower will be ignored. Default is 10
            replaceMissingValues (str): Method to replace missing values. Available methods are 'Median', 'Mean' and 'Delete'. Default is 'Median'
        Returns:
            the created model
        """
        hyperParameters = {'n_estimators': n_estimators, 'max_depth': max_depth, 'criterion': criterion}
        discretizations = {}
        if enable_custom_discretizations is True:
            discretizations = dataset._discretizations
        params = {
            'nbMaxModality': nbMaxModality,
            'nbMinObservation': nbMinObservation,
            'replaceMissingValues': replaceMissingValues,
            'paramsSk': dumps(hyperParameters),
            'algoType': AlgoTypes.RANDOMFOREST,
            'modelName': name,
            'enable_custom_discretizations': enable_custom_discretizations,
            'discretizations': discretizations,
        }
        return self.__create_skModel(dataset, target, params)

    @Helper.try_catch
    def create_GradientBoosting(self, dataset, name, target, loss='deviance', n_estimators=100, maxdepth=3, split_ratio=0.7,
                                enable_custom_discretizations=True, nbMaxModality=50, nbMinObservation=10, replaceMissingValues='Median'):
        """
        Create a Gradient Boosting classifier

        Args:
            dataset (Dataset): Dataset the model is fitted on
            name (str): Name of the new model
            target (Target): Target used to generate the model
            loss (str): loss function to be optimized. 'deviance' refers to deviance (= logistic regression) for classification with probabilistic
                outputs. For loss 'exponential' gradient boosting recovers the AdaBoost algorithm.Dafault is 'deviance')
            n_estimators (int): The number  of boosting stages to perform. Default is 100
            max_depth (int): The maximum depth of the individual regression estimators. Default is 3
            split_ratio (float): the first step in the model generation is the random split of the original dataset into a learning (or train) dataset
                representing by default 70% of the original dataset, and a validation (or test) dataset containing the remaining 30%. Default is 0.7
            enable_custom_discretizations (boolean): when ticked use the custom discretization(s) link to the selected dataset. Default is True
            nbMaxModality (int): Maximum number of modalities per variable. Default is 50
            nbMinObservation (int): Modalities with a number of observations lower will be ignored. Default is 10
            replaceMissingValues (str): Method to replace missing values. Available methods are 'Median', 'Mean' and 'Delete'. Default is 'Median'
        Returns:
            the created model
        """
        hyperParameters = {'loss': loss, 'learning_rate': 0.1, 'n_estimators': n_estimators, 'max_depth': maxdepth}
        discretizations = {}
        if enable_custom_discretizations is True:
            discretizations = dataset._discretizations
        params = {
            'nbMaxModality': nbMaxModality,
            'nbMinObservation': nbMinObservation,
            'replaceMissingValues': replaceMissingValues,
            'paramsSk': dumps(hyperParameters),
            'algoType': AlgoTypes.GRADIENTBOOSTING,
            'modelName': name,
            'enable_custom_discretizations': enable_custom_discretizations,
            'discretizations': discretizations,
        }
        return self.__create_skModel(dataset, target, params)

    @Helper.try_catch
    def create_GradientBoostingRegressor(self, dataset, name, target, n_estimators=100, maxdepth=3, split_ratio=0.7, enable_custom_discretizations=True,
                                         nbMaxModality=50, nbMinObservation=10, replaceMissingValues='Median'):
        """
        Create a Gradient Boosting regressor

        Args:
            dataset (Dataset): Dataset the model is fitted on
            name (str): Name of the new model
            target (Target): Target used to generate the model
            n_estimators (int): The number  of boosting stages to perform. Default is 100
            max_depth (int): The maximum depth of the individual regression estimators. Default is 3
            split_ratio (float): the first step in the model generation is the random split of the original dataset into a learning (or train) dataset
                representing by default 70% of the original dataset, and a validation (or test) dataset containing the remaining 30%. Default is 0.7
            enable_custom_discretizations (boolean): when ticked use the custom discretization(s) link to the selected dataset. Default is True
            nbMaxModality (int): Maximum number of modalities per variable. Default is 50
            nbMinObservation (int): Modalities with a number of observations lower will be ignored. Default is 10
            replaceMissingValues (str): Method to replace missing values. Available methods are 'Median', 'Mean' and 'Delete'. Default is 'Median'
        Returns:
            the created model
        """
        hyperParameters = {'learning_rate': 0.1, 'n_estimators': n_estimators, 'max_depth': maxdepth}
        discretizations = {}
        if enable_custom_discretizations is True:
            discretizations = dataset._discretizations
        params = {
            'nbMaxModality': nbMaxModality,
            'nbMinObservation': nbMinObservation,
            'replaceMissingValues': replaceMissingValues,
            'paramsSk': dumps(hyperParameters),
            'algoType': AlgoTypes.GRADIENTBOOSTINGREGRESSOR,
            'modelName': name,
            'enable_custom_discretizations': enable_custom_discretizations,
            'discretizations': discretizations,
        }
        return self.__create_skModel(dataset, target, params)

    @Helper.try_catch
    def create_XGBRegressor(self, dataset, name, target, n_estimators=100, maxdepth=3, split_ratio=0.7, enable_custom_discretizations=True,
                            nbMaxModality=50, nbMinObservation=10, replaceMissingValues='Median'):
        """
        Create a eXtreme Gradient Boosting (XGBoost) regressor

        Args:
            dataset (Dataset): Dataset the model is fitted on
            name (str): Name of the new model
            target (Target): Target used to generate the model
            n_estimators (int): The number  of boosting stages to perform. Default is 100
            max_depth (int): The maximum depth of the individual regression estimators. Default is 3
            split_ratio (float): the first step in the model generation is the random split of the original dataset into a learning (or train) dataset
                representing by default 70% of the original dataset, and a validation (or test) dataset containing the remaining 30%. Default is 0.7
            enable_custom_discretizations (boolean): when ticked use the custom discretization(s) link to the selected dataset. Default is True
            nbMaxModality (int): Maximum number of modalities per variable. Default is 50
            nbMinObservation (int): Modalities with a number of observations lower will be ignored. Default is 10
        Returns:
            the created model
        """

        if self.__api.session.version < self.__api.session.version.__class__('4.2.10'):
            raise NotImplementedError('The feature is not available on this platform')

        hyperParameters = {'n_estimators': n_estimators, 'learning_rate': 0.1, 'max_depth': maxdepth}
        discretizations = {}
        if enable_custom_discretizations is True:
            discretizations = dataset._discretizations
        params = {
            'nbMaxModality': nbMaxModality,
            'nbMinObservation': nbMinObservation,
            'replaceMissingValues': replaceMissingValues,
            'paramsSk': dumps(hyperParameters),
            'algoType': AlgoTypes.XGBREGRESSOR,
            'modelName': name,
            'enable_custom_discretizations': enable_custom_discretizations,
            'discretizations': discretizations,
        }
        return self.__create_skModel(dataset, target, params)

    @Helper.try_catch
    def create_Lasso(self, dataset, name, target,  max_iter=1000, tol=0.0001, alpha=1.0, fit_intercept=True, normalize=False,
                     split_ratio=0.7, enable_custom_discretizations=True, nbMaxModality=50, nbMinObservation=10, replaceMissingValues='Median'):
        """
        Linear Model trained with L1 prior as regularizer (Lasso)

        Args:
            dataset (Dataset): Dataset the model is fitted on
            name (str): Name of the new model
            target (Target): Target used to generate the model
            max_iter (int): The maximum number of iterations. Default is 1000
            tol (float): The tolerance for the optimization. Default is 0.0001
            alpha (float) : Constant that multiplies the L1 term. Default is 1.0
            fit_intercept (boolean) : Whether to calculate the intercept for this model. Default is True
            normalize (boolean) : if True, the regressors X will be normalized before regression by 
                subtracting the mean and dividing by the l2-norm. Default is False
            split_ratio (float): the first step in the model generation is the random split of the original dataset into a learning (or train) dataset
                representing by default 70% of the original dataset, and a validation (or test) dataset containing the remaining 30%. Default is 0.7
            enable_custom_discretizations (boolean): when ticked use the custom discretization(s) link to the selected dataset. Default is True
            nbMaxModality (int): Maximum number of modalities per variable. Default is 50
            nbMinObservation (int): Modalities with a number of observations lower will be ignored. Default is 10
        Returns:
            the created model
        """

        if self.__api.session.version < self.__api.session.version.__class__('4.2.10'):
            raise NotImplementedError('The feature is not available on this platform')

        hyperParameters = {'max_iter': max_iter, 'tol': tol, 'alpha': alpha, 'fit_intercept': fit_intercept, 
                           'normalize': normalize}
        discretizations = {}
        if enable_custom_discretizations is True:
            discretizations = dataset._discretizations
        params = {
            'nbMaxModality': nbMaxModality,
            'nbMinObservation': nbMinObservation,
            'replaceMissingValues': replaceMissingValues,
            'paramsSk': dumps(hyperParameters),
            'algoType': AlgoTypes.LASSO,
            'modelName': name,
            'enable_custom_discretizations': enable_custom_discretizations,
            'discretizations': discretizations,
        }
        return self.__create_skModel(dataset, target, params)

    @Helper.try_catch
    def create_Perceptron(self, dataset, name, target, penalty=None, alpha=1e-4,fit_intercept=True, max_iter=1000, tol=1e-3,
                          class_weight='balanced', method='isotonic', split_ratio=0.7, enable_custom_discretizations=True,
                          nbMaxModality=50, nbMinObservation=10, replaceMissingValues='Median'):
        """
        Create a Perceptron classifier model

        Args:
            dataset (Dataset): Dataset the model is fitted on
            name (str): Name of the new model
            target (Target): Target used to generate the model
            penalty (str): The penalty (aka regularization term) to be used. Defaults to None.
            alpha (float): Constant that multiplies the regularization term if regularization is used. Defaults to 0.0001
            fit_intercept (boolean): Whether the intercept should be estimated or not. If False, the data is assumed to be already centered.
                Defaults to True.
            max_iter (int): The maximum number of passes over the training data (aka epochs). Defaults to 1000
            tol (float): The stopping criterion. Defaults to 0.001
            class_weight (str): The “balanced” mode uses the values of y to automatically adjust weights inversely proportional to class frequencies 
                in the input data, if None, all classes are supposed to have weight one. Defaults to "balanced"
            method (str): The method to use for calibration. Can be ‘sigmoid’ which corresponds to Platt’s method or ‘isotonic’ which is a non-parametric 
                approach. It is not advised to use isotonic calibration with too few calibration samples (<<1000) since it tends to overfit. Use sigmoids 
                (Platt’s calibration) in this case. Defaults to 'isotonic'
            split_ratio (float): the first step in the model generation is the random split of the original dataset into a learning (or train) dataset
                representing by default 70% of the original dataset, and a validation (or test) dataset containing the remaining 30%. Default is 0.7
            enable_custom_discretizations (boolean): when ticked use the custom discretization(s) link to the selected dataset. Default is True
            nbMaxModality (int): Maximum number of modalities per variable. Default is 50
            nbMinObservation (int): Modalities with a number of observations lower will be ignored. Default is 10
            replaceMissingValues (str): Method to replace missing values. Available methods are 'Median', 'Mean' and 'Delete'. Default is 'Median'
        Returns:
            the created model
        """

        if self.__api.session.version < self.__api.session.version.__class__('4.2.10'):
            raise NotImplementedError('The feature is not available on this platform')

        hyperParameters = {'penalty': penalty, 'alpha': alpha, 'max_iter': max_iter, 'tol': tol, 'class_weight': class_weight, 
                           'fit_intercept': fit_intercept, 'method': method}

        discretizations = {}
        if enable_custom_discretizations is True:
            discretizations = dataset._discretizations
        params = {
            'nbMaxModality': nbMaxModality,
            'nbMinObservation': nbMinObservation,
            'replaceMissingValues': replaceMissingValues,
            'paramsSk': dumps(hyperParameters),
            'algoType': AlgoTypes.PERCEPTRON,
            'modelName': name,
            'enable_custom_discretizations': enable_custom_discretizations,
            'discretizations': discretizations,
        }
        return self.__create_skModel(dataset, target, params)


    @Helper.try_catch
    def _create_TimeSeriesForecaster(self, dataset, name, target, freq='1T', seasonalities_choice=[], targets_to_enrich='', targets_history=[],
                                    n_lags_max="", lags_to_include=[], one_hot_encode_seasons=False, remove_seasons =False, dropna=False,
                                    scaler_target='', scaler_X_var='', grid_search=False, rolling_enrich_features=[], test_ratio=0.3, 
                                    enable_custom_discretizations=True, nbMaxModality=50, nbMinObservation=10, replaceMissingValues='Median', 
                                    regressor= { "model_name":"elasticNet",
                                   "params": {"l1_ratio": [.1, .5, .7, .9, .95, .99, .995, 1],
                                        "eps": 0.001, "n_alphas": 100, "fit_intercept": True,
                                        "normalize": True, "precompute": 'auto', "max_iter": 2000,
                                        "tol": 0.0001, "cv": 5, "copy_X": True, "verbose": 0,
                                        "n_jobs": -1, "positive": False, "random_state": None, "selection": 'cyclic'}
                                 }, deseason_signal=False, detrend_signal=False):
        """
        Create a Time Series forecasting model based on regressor

        Args:
            dataset (Dataset): Dataset the model is fitted on
            name (str): Name of the new model
            target (Target): Target used to generate the model
            freq (str): Frequency of the time series
            seasonalities_choice (list): possible seasonalities to consider
            targets_history (list): variables to consider when adding lags
            n_lags_max (int): maximum lags to include
            lags_to_include (list): specific lags to include
            one_hot_encode_seasons (bool): One hot encode the seasonalities [categorical features]
            remove_seasons (bool): remove seasonalities for features
            dropna (bool): remove NaNs values
            scaler_target (str): name of scaler for time series
            scaler_X_var (str): name of scaler for X values (features)
            grid_search (bool): apply grid search for XGboost Regressor
            rolling_enrich_features (list): features to include additional rolling statistics
            test_ratio (float): test size 
            regressor (dict): defines the regressor algorithm to use and its hyperparameters
            deseason_signal (bool)
            detrend_signal (bool)
            example:

                Three possible algorithms [ XGBOOST, LSTM and ElaticNet]
                "regressor":{
                    "model_name":"lstm", "params":{"n_units": 100, "activation": "tanh",
                    "dropout": 0.4, "return_sequences": true,
                    "loss": "mae", "metrics": "mse", "optimizer_name": "sgd",
                    "optimizer_params": {"lr": 0.01, "decay": 1e-6, "momentum": 0.9, "nesterov": true},
                    "callbacks": {"monitor": "val_loss", "mode": "min", "patience": 5},
                    "epochs": 100, "batch_size": 128, "validation_split": 0.3,
                    "shuffle": false, "verbose": 0}
                                    }

                    "elasticNet_params" : { "model_name":"elasticNet",
                                   "params": {"l1_ratio": [.1, .5, .7, .9, .95, .99, .995, 1],
                                        "eps": 0.001, "n_alphas": 100, "fit_intercept": True,
                                        "normalize": True, "precompute": 'auto', "max_iter": 2000,
                                        "tol": 0.0001, "cv": 5, "copy_X": True, "verbose": 0,
                                        "n_jobs": -1, "positive": False, "random_state": None, "selection": 'cyclic'}
                                 }


                    "xgboost_params" ={'target_name': target_name,
                     'freq':freq, 
                     'seasonalities_choice':seasons_choice,
                     'targets_to_enrich': targets_in, 
                     'targets_history': targets_history_in,
                     'n_lags_max': n_lags_maxi,
                     'lags_to_include':lags_to_include_in,  
                     'one_hot_encode_seasons': one_hot_encode_seasons,
                     'remove_seasons':remove_seasons,
                     'dropna':dropna,
                     'scaler_target':scaler_target,
                     'scaler_X_var':scaler_X_var,
                     
                     'rolling_enrich_features':rolling_enrich_features,
                    
                    'grid_search':grid_search,
                    "regressor": { "model_name":"xgboost",
                                   "params": {
                                        "xgb_model_params":{
                                            "n_estimators": 200,
                                            "max_depth": 3,
                                            "learning_rate": 0.05},
                                       "grid_params":{
                                            "max_depth": [3, 5, 7],
                                            "n_estimators": [100, 150, 200],
                                            "learning_rate": [0.1, 0.05, 0.01, 0.2]
                                               }
                                             }
                                 }
                    }



            enable_custom_discretizations (boolean): when ticked use the custom discretization(s) link to the selected dataset. Default is True
            nbMaxModality (int): Maximum number of modalities per variable. Default is 50
            nbMinObservation (int): Modalities with a number of observations lower will be ignored. Default is 10
            replaceMissingValues (str): Method to replace missing values. Available methods are 'Median', 'Mean' and 'Delete'. Default is 'Median'
            
        Returns:
            the created model
        """

        if self.__api.session.version < self.__api.session.version.__class__('4.2.10'):
            raise NotImplementedError('The feature is not available on this platform')

        # case empty regressor entry
        try: 
            assert(len(regressor)!=0)
        except Exception as E:
            print('You have inserted an empty Regressor choice')
            return E
        
        target_name = target.name
        targets_history.append(target_name)
        hyperParameters = { 
                "target_name": target_name,
                "freq": freq,
                "seasonalities_choice": seasonalities_choice,
                "targets_to_enrich": targets_to_enrich,
                "targets_history": targets_history,
                "n_lags_max": n_lags_max,
                "lags_to_include": lags_to_include,
                "one_hot_encode_seasons": one_hot_encode_seasons,
                "remove_seasons": remove_seasons,
                "dropna": dropna,
                "scaler_target": scaler_target,
                "scaler_X_var": scaler_X_var,
                "grid_search": grid_search,
                "rolling_enrich_features": rolling_enrich_features,
                "test_ratio": test_ratio,
                "regressor": regressor,
                "deseason_signal": deseason_signal,
                "detrend_signal": detrend_signal
         }
        discretizations = {}
        if enable_custom_discretizations is True:
            discretizations = dataset._discretizations
        params = {
            'nbMaxModality': nbMaxModality,
            'nbMinObservation': nbMinObservation,
            'replaceMissingValues': replaceMissingValues,
            'paramsSk': dumps(hyperParameters),
            'algoType': AlgoTypes.TIMESERIESFORECASTER,
            'modelName': name,
            'enable_custom_discretizations': enable_custom_discretizations,
            'discretizations': discretizations,
        }
        return self.__create_skModel(dataset, target, params)


class Model(Base):
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
            """\t- Algo : {}\n\t- Dataset : {}\n\t- Target : {}\n\t- Created on : {}\n""".format(
            self.algoType,
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

    @Helper.try_catch
    def delete(self):
        if not self._is_deleted:
            json = {'project_ID': self.project_id, 'task_ID': self.id}
            self.__api.Task.deletetask(**json)
            self._is_deleted = True
        return self


class ClassifierModel(Model):
    """
    """
    def __init__(self, api, json_return):
        self.__api = api
        self.__json_returned = json_return
        self.__json_confusion_matrix = None
        super().__init__(api, json_return)

    @Helper.try_catch
    def apply(self, dataset, applied_model_name, add_score_to_dataset=False, score_column_name=None):
        """
        Apply the classifier model on a selected data set

        Args:
            dataset (Dataset): Dataset the model is applied on
            applied_model_name (str): Name of the new applied model
            add_score_to_dataset (boolean): if set to True a new column containing the scores is added to the dataset.
                Default is False.
            score_column_name (str): name of the score column, used only if add_score_to_dataset is set to True

        Returns:
            the applied Model
        """
        params = dict(self.__json_returned)
        params['modelName'] = applied_model_name

        data = {
            'datasetId': dataset.dataset_id,
            'datasetName': dataset.name,
            'fromDatasetId': self.__json_returned.get('datasetId'),
            'modelId': self.__json_returned.get('_id'),
            'params': params,
            'projectId': dataset.project_id,
            'spark': False,
            'type': 'applyPrediction'
        }

        if add_score_to_dataset:
            data['saveScore'] = score_column_name or 'score_' + applied_model_name

        json = {'project_ID': dataset.project_id, 'json': data}
        json_returned = self.__api.Task.createtask(**json)
        try:
            self.__api.handle_work_states(dataset.project_id, work_type=json_returned.get('type'), work_id=json_returned.get('_id'))
        except Exception as E:
            raise ApiException('Unable to create the applied model ' + applied_model_name, str(E))
        return HyperCube(self.__api, json_returned) if self.algoType == AlgoTypes.HYPERCUBE else ClassifierModel(self.__api, json_returned)

    @Helper.try_catch
    def preprocess_data(self, dataset):
        """
        Return a preprocessed dataframe for Scikit-learn models

        Args:
            dataset (Dataset): Dataset the model is applied on

        Returns:
            preprocessed dataframe
        """
        
        if not hasattr(self.__api.Prediction, "exportpreprocesseddata"):
            raise NotImplementedError('The feature is not available on this platform')

        pd = get_required_module('pandas')

        if self.algoType == AlgoTypes.HYPERCUBE:
            raise ApiException('Preprocessing is not available for Hypercube models')
        applied_model = self.apply(dataset, self.name + '_applied')
        json = {'project_ID': self.project_id, 'model_ID': applied_model.id}
        url = self.__api.Prediction.exportpreprocesseddata(**json)
        df = pd.read_csv(StringIO(url.decode('utf-8')))
        applied_model.delete()
        return df

    @Helper.try_catch
    def export_scores(self, path, variables=None):
        """
        Export the scores of this model in a csv file

        Args:
            path (str): the destination path for the exported scores
            variables (list of Variable): the variables of the dataset to add in the file. Default is None
        """
        data = {
            'datasetId': self.__json_returned.get('datasetId'),
            'columns': [variable.name for variable in variables] if variables else []
        }
        json = {'project_ID': self.project_id, 'model_ID': self.id, 'json': data}
        json_returned = self.__api.Prediction.postexportscores(**json)

        try:
            self.__api.handle_work_states(self.project_id, work_type=json_returned.get('type'), work_id=json_returned.get('_id'))
        except Exception as E:
            raise ApiException('Unable to export the model scores for ' + self.name, str(E))

        outputFile = json_returned.get('workParams').get('outputFile').split("_")[1]
        data = {
            'outputFile': outputFile
        }
        json = {'project_ID': self.project_id, 'model_ID': self.id, 'params': data}
        to_export = self.__api.Prediction.getexportscores(**json)

        with open(path, 'wb') as FILE_OUT:
            FILE_OUT.write(to_export)

    @Helper.try_catch
    def predict_scores(self, dataset, keep_applied_model=False):
        """
        Predict target scores for input dataset

        Args:
            dataset (Dataset): the dataset containing the input samples.
            keep_applied_model (boolean): An applied model is temporarily created to compute these scores,
                set this parameter to True if you want this model to be persisted. Default is False.

        Returns:
            a NumPy array of shape [n_samples,] where n_samples is the number of samples in the input dataset
        """
        pd = get_required_module('pandas')

        applied_model = self.apply(dataset, '{}_applied_{}'.format(self.name, datetime.now().strftime("%Y-%m-%d_%H-%M-%S")))
        data = {
            'datasetId': dataset.dataset_id,
            'columns': []
        }
        json = {'project_ID': applied_model.project_id, 'model_ID': applied_model.id, 'json': data}
        json_returned = self.__api.Prediction.postexportscores(**json)

        try:
            self.__api.handle_work_states(applied_model.project_id, work_type=json_returned.get('type'), work_id=json_returned.get('_id'))
        except Exception as E:
            raise ApiException('Unable to get the model scores for {}'.format(self.name), str(E))

        outputFile = json_returned.get('workParams').get('outputFile').split("_")[1]
        data = {
            'outputFile': outputFile
        }
        json = {'project_ID': applied_model.project_id, 'model_ID': applied_model.id, 'params': data}
        scores = self.__api.Prediction.getexportscores(**json)
        scoreIO = StringIO(scores.decode('utf-8'))

        try:
            df = pd.read_csv(scoreIO, sep=';', usecols=[1])
        except Exception as E:
            raise ApiException('Unable to read the model scores for {}'.format(self.name), str(E))

        if not keep_applied_model:
            applied_model.delete()

        return pd.np.reshape(df.values, (df.values.shape[0]))

    def __load_confusion_matrix(self):
        if not self.__json_confusion_matrix:
            json = {'project_ID': self.project_id, 'model_ID': self.id, 'dataset_ID': self.__json_returned.get('datasetId')}
            self.__json_confusion_matrix = self.__api.Prediction.getconfusionmatrix(**json)

    @Helper.try_catch
    def get_confusion_matrix(self, top_score_ratio=0.1):
        """
        Get the Confusion Matrix

        Args:
            top_score_ratio (float): must be greater or equal to 0 and lower or equal to 1, default=0.1.

        Returns:
            ConfusionMatrix
        """
        if not 0 <= top_score_ratio <= 1:
            raise ApiException('top_score_ratio must be greater or equal to 0 and lower or equal to 1')

        self.__load_confusion_matrix()
        index = self.__get_index(top_score_ratio)
        values = self.__json_confusion_matrix[Curves.LIFT][index]
        return ConfusionMatrix(true_positives=values['TP'], false_positives=values['FP'],
                               true_negatives=values['TN'], false_negatives=values['FN'])

    def __get_index(self, top_score_ratio):
        if top_score_ratio == 0:
            return 0
        else:
            length = len(self.__json_confusion_matrix[Curves.LIFT])
            return max(0, round(length * top_score_ratio) - 1)

    @property
    @Helper.try_catch
    def area_under_roc(self):
        self.__load_confusion_matrix()
        return self.__json_confusion_matrix[Curves.ROC][-1]['auc']

    def __get_x_y_info(self, curve):
        if curve == Curves.ROC:
            x = [point['FPR'] for point in self.__json_confusion_matrix[curve]]
            y = [point['Sensitivity'] for point in self.__json_confusion_matrix[curve]]
            x_name = 'False Positive Rate'
            y_name = 'True Positive Rate'
        elif curve == Curves.GAIN:
            x = [point['TopScore'] for point in self.__json_confusion_matrix[curve]]
            y = [point['Sensitivity'] for point in self.__json_confusion_matrix[curve]]
            x_name = 'Top score percentages'
            y_name = 'Recall'
        elif curve == Curves.LIFT:
            x = [point['TopScore'] for point in self.__json_confusion_matrix[curve]]
            y = [point['Lift'] for point in self.__json_confusion_matrix[curve]]
            x_name = 'Top score percentages'
            y_name = 'Target lift'
        elif curve == Curves.PURITY:
            x = [point['TopScore'] for point in self.__json_confusion_matrix[curve]]
            y = [point['Purity'] for point in self.__json_confusion_matrix[curve]]
            x_name = 'Top score percentages'
            y_name = 'Precision'
        elif curve == Curves.PRECISIONRECALL:
            x = [point['Sensitivity'] for point in self.__json_confusion_matrix[curve]]
            y = [point['Purity'] for point in self.__json_confusion_matrix[curve]]
            x_name = 'Recall'
            y_name = 'Precision'
        return x, y, x_name, y_name

    @Helper.try_catch
    def display_curve(self, curve=Curves.ROC, title=None, model_line=None, random_line=None, legend=None):
        """
        Plot the selected curve of this model

        Args:
            curve (str or None): curve to be diplayed, options are 'ROC curve', 'Gain curve', 'Lift curve', 'Purity curve' and
                'Precision Recall'. If None is provided, ROC curve will be displayed. Default is ROC curve.
            title (str): Title of the diagram. Default is a custom model name
            model_line (dict): display options of model line, ex: dict(color=('rgb(205, 12, 24)'), dash='dash', width=1).
                Default is a blue line. see https://plot.ly/python/line-and-scatter/
            random_line (dict): display options of random line. Default is a red dash line.
            legend (dict): legend options, ex: dict(orientation="h") or dict(x=-.1, y=1.2).
                Default is at the right of the diagram. see https://plot.ly/python/legend/

        Returns:
            plot of the curve
        """

        try:
            import plotly.graph_objs as go
            import plotly.offline as py
            from plotly.offline import init_notebook_mode
        except ImportError as E:
            raise ApiException('Plotly external package is required for this operation, please execute "!pip install plotly" and restart the kernel', str(E))

        if curve is None:
            curve = Curves.ROC
        elif curve not in Curves.LIST:
            print('Unexpected curve type : {}, valid options are : {}'.format(curve, ', '.join(Curves.LIST)))
            return

        self.__load_confusion_matrix()

        x, y, x_name, y_name = self.__get_x_y_info(curve)

        init_notebook_mode(connected=False)
        if model_line:
            roc = go.Scatter(x=x, y=y, name='{}'.format(self.name), mode='lines', line=model_line)
        else:
            roc = go.Scatter(x=x, y=y, name='{}'.format(self.name), mode='lines')

        data = [roc]
        random_line_arg = random_line or dict(color=('rgb(205, 12, 24)'), dash='dash', width=1)
        if curve == Curves.ROC or curve == Curves.GAIN:
            random = go.Scatter(x=[0, 1], y=[0, 1], name='Random', mode='lines', line=random_line_arg)
        elif curve == Curves.LIFT:
            random = go.Scatter(x=[0, 1], y=[1, 1], name='Random', mode='lines', line=random_line_arg)
        else:
            random = None

        if random:
            data.append(random)

        default_title = '{} of {}'.format(curve, self.name)
        if curve == Curves.ROC or curve == Curves.PRECISIONRECALL:
            default_title = '{} (AUC = {:0.2f})'.format(default_title,
                                                        self.__json_confusion_matrix[curve][-1]['auc'])
        curve_title = title or default_title

        layout = dict(title=curve_title, xaxis=dict(title=x_name, range=[0, 1]),
                      yaxis=dict(title=y_name, range=[0, max(y) + 0.05]),)
        if legend:
            layout['legend'] = legend

        fig = dict(data=data, layout=layout)
        py.iplot(fig, validate=False)

    @Helper.try_catch
    def export_model(self, path):
        """
        Export this model in a local file

        Args:
            path (str): the destination path for the exported model (zipped Pickle)
        """
        json = {'project_ID': self.project_id, 'dataset_ID': self.dataset_id, 'prediction_ID': self.id}
        res = self.__api.Prediction.exportscikit(**json)
        if not path.split('.')[-1] == 'zip':
            path += '.zip'
        if isinstance(res, bytes):
            with open(path, 'wb') as FILE_OUT:
                FILE_OUT.write(res)
        else:
            with open(path, 'w') as FILE_OUT:
                dump(res, FILE_OUT)


class HyperCube(ClassifierModel):
    """
    """
    def __init__(self, api, json_return):
        self.__api = api
        self.__json_returned = json_return
        self.__export_formats = ExportFormats()
        super().__init__(api, json_return)

    @Helper.try_catch
    def export_model(self, path, format=ExportFormats.PYTHON):
        """
        Export this model in a local file

        Args:
            path (str): the destination path for the exported model
            format (str or None): {'Python', 'csv', 'JSON', 'R', 'Scala', 'Java', 'JavaScript', 'MySQL', 'PLSQL', 'TSQL'}.
                If None, the format is 'Python'. Default is 'Python'
        """
        if format is None:
            format = ExportFormats.PYTHON
        elif format not in ExportFormats.LIST:
            print('Unexpected export format : {}, valid options are : {}'.format(format, ', '.join(ExportFormats.LIST)))
            return

        URL_PREFIX = 'api/v1/'
        data = {
            'format': format
        }
        json = {'project_ID': self.project_id, 'model_ID': self.id, 'params': data}
        url = self.__api.Prediction.exportrules(**json)
        url = url.decode("utf-8").replace(URL_PREFIX, '')
        to_export = self.__api.session.request(url=url, method='get')
        if isinstance(to_export, bytes):
            with open(path, 'wb') as FILE_OUT:
                FILE_OUT.write(to_export)
        else:
            with open(path, 'w') as FILE_OUT:
                dump(to_export, FILE_OUT)


class RegressorModel(Model):
    """
    """
    def __init__(self, api, json_return):
        self.__api = api
        self.__json_returned = json_return
        super().__init__(api, json_return)

    @Helper.try_catch
    def apply(self, dataset, applied_model_name):
        """
        Apply the model on a selected data set

        Args:
            dataset (Dataset): Dataset the model is applied on
            applied_model_name (str): Name of the new applied model

        Returns:
            the applied Model
        """
        params = dict(self.__json_returned)
        params['modelName'] = applied_model_name

        data = {
            'datasetId': dataset.dataset_id,
            'datasetName': dataset.name,
            'fromDatasetId': self.__json_returned.get('datasetId'),
            'modelId': self.__json_returned.get('_id'),
            'params': params,
            'projectId': dataset.project_id,
            'spark': False,
            'type': 'applyPrediction'
        }

        json = {'project_ID': dataset.project_id, 'json': data}
        json_returned = self.__api.Task.createtask(**json)
        try:
            self.__api.handle_work_states(dataset.project_id, work_type=json_returned.get('type'), work_id=json_returned.get('_id'))
        except Exception as E:
            raise ApiException('Unable to create the applied model ' + applied_model_name, str(E))

        return RegressorModel(self.__api, json_returned)

    @Helper.try_catch
    def preprocess_data(self, dataset):
        """
        Return a preprocessed dataframe for Scikit-learn models

        Args:
            dataset (Dataset): Dataset the model is applied on

        Returns:
            preprocessed dataframe
        """

        if not hasattr(self.__api.Prediction, "exportpreprocesseddata"):
            raise NotImplementedError('The feature is not available on this platform')

        pd = get_required_module('pandas')

        applied_model = self.apply(dataset, self.name + '_applied')
        json = {'project_ID': self.project_id, 'model_ID': applied_model.id}
        url = self.__api.Prediction.exportpreprocesseddata(**json)
        df = pd.read_csv(StringIO(url.decode('utf-8')))
        applied_model.delete()
        return df

    @Helper.try_catch
    def export_prediction(self, path, variables=None):
        """
        Export prediction of this model in a csv file

        Args:
            path (str): the destination path for the exported prediction
            variables (list of Variable): the variables of the dataset to add in the file. Default is None
        """
        data = {
            'datasetId': self.__json_returned.get('datasetId'),
            'columns': [variable.name for variable in variables] if variables else []
        }
        json = {'project_ID': self.project_id, 'model_ID': self.id, 'json': data}
        json_returned = self.__api.Prediction.postexportscores(**json)

        try:
            self.__api.handle_work_states(self.project_id, work_type=json_returned.get('type'), work_id=json_returned.get('_id'))
        except Exception as E:
            raise ApiException('Unable to export the model scores for ' + self.name, str(E))

        outputFile = json_returned.get('workParams').get('outputFile').split("_")[1]
        data = {
            'outputFile': outputFile
        }
        json = {'project_ID': self.project_id, 'model_ID': self.id, 'params': data}
        to_export = self.__api.Prediction.getexportscores(**json)

        with open(path, 'wb') as FILE_OUT:
            FILE_OUT.write(to_export)

    @Helper.try_catch
    def predict(self, dataset, keep_applied_model=False):
        """
        Target prediction for input dataset

        Args:
            dataset (Dataset): the dataset containing the input samples.
            keep_applied_model (boolean): An applied model is temporarily created to compute these scores,
                set this parameter to True if you want this model to be persisted. Default is False.

        Returns:
            a NumPy array of shape [n_samples,] where n_samples is the number of samples in the input dataset
        """
        pd = get_required_module('pandas')

        applied_model = self.apply(dataset, '{}_applied_{}'.format(self.name, datetime.now().strftime("%Y-%m-%d_%H-%M-%S")))
        data = {
            'datasetId': dataset.dataset_id,
            'columns': []
        }
        json = {'project_ID': applied_model.project_id, 'model_ID': applied_model.id, 'json': data}
        json_returned = self.__api.Prediction.postexportscores(**json)

        try:
            self.__api.handle_work_states(applied_model.project_id, work_type=json_returned.get('type'), work_id=json_returned.get('_id'))
        except Exception as E:
            raise ApiException('Unable to get the model scores for {}'.format(self.name), str(E))

        outputFile = json_returned.get('workParams').get('outputFile').split("_")[1]
        data = {
            'outputFile': outputFile
        }
        json = {'project_ID': applied_model.project_id, 'model_ID': applied_model.id, 'params': data}
        scores = self.__api.Prediction.getexportscores(**json)
        scoreIO = StringIO(scores.decode('utf-8'))

        try:
            df = pd.read_csv(scoreIO, sep=';', usecols=[1])
        except Exception as E:
            raise ApiException('Unable to read the model scores for {}'.format(self.name), str(E))

        if not keep_applied_model:
            applied_model.delete()

        return pd.np.reshape(df.values, (df.values.shape[0]))

    @Helper.try_catch
    def export_model(self, path):
        """
        Export this model in a local file

        Args:
            path (str): the destination path for the exported model (zipped Pickle)
        """
        json = {'project_ID': self.project_id, 'dataset_ID': self.dataset_id, 'prediction_ID': self.id}
        res = self.__api.Prediction.exportscikit(**json)
        if not path.split('.')[-1] == 'zip':
            path += '.zip'
        if isinstance(res, bytes):
            with open(path, 'wb') as FILE_OUT:
                FILE_OUT.write(res)
        else:
            with open(path, 'w') as FILE_OUT:
                dump(res, FILE_OUT)


class ConfusionMatrix:

    def __init__(self, true_positives, false_positives, true_negatives, false_negatives):
        self.true_positives = true_positives
        self.false_positives = false_positives
        self.true_negatives = true_negatives
        self.false_negatives = false_negatives

    def __repr__(self):
        _rep = {
            "True Positives": self.true_positives,
            "True Negatives": self.true_negatives,
            "False Positives": self.false_positives,
            "False Negatives": self.false_negatives
        }
        return "\n".join(f"{_k}: {_v}" for _k,_v in _rep.items())
