from HyperAPI.util import Helper
from HyperAPI.hyper_api.target import Description
from HyperAPI.hyper_api.base import Base
from HyperAPI.hyper_api.model import Model, ModelFactory
from HyperAPI.hyper_api.rule import Rules, decode_kpiname_to_id
from HyperAPI.utils.exceptions import ApiException

import urllib.parse


class KeyIndicatorOption:
    def __init__(self, target, purity_min=None, purity_max=None, coverage_min=None, coverage_max=None,
                 lift_min=None, lift_max=None, zscore_min=None, zscore_max=None,
                 average_value_min=None, average_value_max=None, standard_deviation_min=None, standard_deviation_max=None,
                 shift_min=None, shift_max=None):
        self.target = target
        self.purity_min = purity_min
        self.purity_max = purity_max
        self.coverage_min = coverage_min
        self.coverage_max = coverage_max
        self.lift_min = lift_min
        self.lift_max = lift_max
        self.zscore_min = zscore_min
        self.zscore_max = zscore_max
        self.average_value_min = average_value_min
        self.average_value_max = average_value_max
        self.standard_deviation_min = standard_deviation_min
        self.standard_deviation_max = standard_deviation_max
        self.shift_min = shift_min
        self.shift_max = shift_max

    def __repr__(self):
        return "\n{} :".format(self.__class__.__name__) + \
               "{}".format(self.target) + \
               ("\t- Purity min: {}\n".format(self.purity_min) if self.purity_min is not None else "") + \
               ("\t- Purity max: {}\n".format(self.purity_max) if self.purity_max is not None else "") + \
               ("\t- Coverage min: {}\n".format(self.coverage_min) if self.coverage_min is not None else "") + \
               ("\t- Coverage max: {}\n".format(self.coverage_max) if self.coverage_max is not None else "") + \
               ("\t- Lift min: {}\n".format(self.lift_min) if self.lift_min is not None else "") + \
               ("\t- Lift max: {}\n".format(self.lift_max) if self.lift_max is not None else "") + \
               ("\t- Z-score min: {}\n".format(self.zscore_min) if self.zscore_min is not None else "") + \
               ("\t- Z-score max: {}\n".format(self.zscore_max) if self.zscore_max is not None else "") + \
               ("\t- Average value min: {}\n".format(self.average_value_min) if self.average_value_min is not None else "") + \
               ("\t- Average value max: {}\n".format(self.average_value_max) if self.average_value_max is not None else "") + \
               ("\t- Standard deviation min: {}\n".format(self.standard_deviation_min) if self.standard_deviation_min is not None else "") + \
               ("\t- Standard deviation max: {}\n".format(self.standard_deviation_max) if self.standard_deviation_max is not None else "") + \
               ("\t- Shift min: {}\n".format(self.shift_min) if self.shift_min is not None else "") + \
               ("\t- Shift max: {}\n".format(self.shift_max) if self.shift_max is not None else "")


class RulesetFactory:
    """
    """
    _PURITY = 'Purity'
    _COVERAGE = 'Coverage'
    _LIFT = 'Lift'
    _ZSCORE = 'Z-score'
    _AVERAGE_VALUE = 'Average value'
    _STANDARD_DEVIATION = 'Standard deviation'
    _SHIFT = 'Shift'
    _DISCRETE_MODALITY = "Discrete variable with a modality"
    _DISCRETE = "Discrete variable"
    _CONTINUOUS = "Continuous variable"

    def __init__(self, api, project_id):
        self.__api = api
        self.__project_id = project_id

    @Helper.try_catch
    def create_kpi_option(self, target, purity_min=None, purity_max=None, coverage_min=None, coverage_max=None,
                          lift_min=None, lift_max=None, zscore_min=None, zscore_max=None,
                          average_value_min=None, average_value_max=None, standard_deviation_min=None, standard_deviation_max=None,
                          shift_min=None, shift_max=None):
        """
        create_kpi_option(target, purity_min=None, purity_max=None, coverage_min=None, coverage_max=None, lift_min=None, lift_max=None, zscore_min=None, zscore_max=None, average_value_min=None, average_value_max=None, standard_deviation_min=None, standard_deviation_max=None, shift_min=None, shift_max=None)

        Create an additional key indicator

        Args:
            target (Target): Target to generate the key indicator
            purity_min (float):  Minimum value for purity score
            purity_max (float):  Maximum value for purity score
            coverage_min (float):  Minimum value for coverage score
            coverage_max (float):  Maximum value for coverage score
            lift_min (float):  Maximum value for lift score
            lift_max (float):  Maximum value for lift score
            zscore_min (float):  Minimum value for zscore score
            zscore_max (float):  Maximum value for zscore score
            average_value_min (float):  Minimum value for average value score
            average_value_max (float):  Maximum value for average value score
            standard_deviation_min (float):  Minimum value for standard deviation score
            standard_deviation_max (float):  Maximum value for standard deviation score
            shift_min (float):  Minimum value for shift score
            shift_max (float):  Maximum value for shift score

        Returns:
            key indicator
        """
        return KeyIndicatorOption(target=target, purity_min=purity_min, coverage_min=coverage_min, coverage_max=coverage_max,
                                  lift_min=lift_min, lift_max=lift_max, zscore_min=zscore_min, zscore_max=zscore_max,
                                  average_value_min=average_value_min, average_value_max=average_value_max,
                                  standard_deviation_min=standard_deviation_min, standard_deviation_max=standard_deviation_max,
                                  shift_min=shift_min, shift_max=shift_max)

    @Helper.try_catch
    def create(self, dataset, name, target, purity_min=None, coverage_min=None, lift_min=None, zscore_min=None, average_value_min=None,
               standard_deviation_max=None, shift_min=None, rule_complexity=2, quantiles=10,
               enable_custom_discretizations=True, min_marginal_contribution=None, compute_other_key_indicators=None,
               locally_increase_complexity=False, max_complexity=3, nb_minimizations=1, coverage_increment=0.01,
               validate_stability=False, split_ratio=0.7, nb_iterations=1, purity_tolerance=0.1):
        """
        create(dataset, name, target, purity_min=None, coverage_min=None, lift_min=None, zscore_min=None, average_value_min=None, standard_deviation_max=None, shift_min=None, rule_complexity=2, quantiles=10, enable_custom_discretizations=True, min_marginal_contribution=None, compute_other_key_indicators=None, locally_increase_complexity=False, max_complexity=3, nb_minimizations=1, coverage_increment=0.01, validate_stability=False, split_ratio=0.7, nb_iterations=1, purity_tolerance=0.1)

        Create a new ruleset

        Args:
            dataset (Dataset): Dataset used to generate the ruleset
            name (str): Name of the new ruleset
            target (Target): Target to generate the ruleset
            purity_min (float): Minimum purity of rules, default is the entire dataset purity (discrete target only)
            coverage_min (int): Minimum coverage of the target population for each rule, default is 10 (discrete target only)
            lift_min (float): Minimum lift, default is 1 (discrete target only)
            zscore_min (float): Minimum Z-score, default is None (discrete target only)
            average_value_min (float): Minimum average value, default is average value of the target on the whole dataset (continuous target only)
            standard_deviation_max (float) : Maximum standard deviation, default is None (continuous target only)
            shift_min (float): Minimum shift, default is None (continuous target only)
            rule_complexity (int): Maximum number of variables in rules, default is 2
            quantiles (int): Number of intervals the continuous variables are quantized in, default is 10
            enable_custom_discretizations (boolean): use custom discretizations, eventually use "quantiles" parameter for remaining variables, default is True
            min_marginal_contribution (float): a new rule R', created by adding a new constraint to an existing rule R (and thus increasing its complexity),
                is added to the ruleset if and only if it increases the original purity of R by the minimum marginal contribution or more. Default is 0.1
            compute_other_key_indicators (list of KeyIndicatorOption): Compute other Key Indicators.
            locally_increase_complexity (bool): Enable the locally increase complexity when set as true. Default is False
            max_complexity (int): Maximum numbers of features per rule. Default is 3
            nb_minimizations (int):Interate the minimization process. Default is 1
            coverage_increment (float): Percentage increment of target samples that a new rule must bring to be added to the minimization ruleset.
                Default is 0.01
            validate_stability (bool): Enable to split your dataset, add iteration and set a purity tolerance when set as true. Default is False
            split_ratio (float): The percentage for the split (Between 0 and 1). Default is 0.7
            nb_iterations (int): Number of iterations wanted. Default is 1
            purity_tolerance (float): Purity tolerence allowed (Between 0 and 1). Default is 0.1

        Returns:
            Ruleset
        """
        variable = next(variable for variable in dataset.variables if variable.name == target.variable_name)
        score_purity_min = None
        if (variable.is_discrete):
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
        else:
            min_marginal_contribution = 0.1

        if enable_custom_discretizations is True:
            discretizations = dataset._discretizations
        else:
            discretizations = {}

        if not compute_other_key_indicators:
            compute_other_key_indicators = []

        if not target:
            raise ApiException('You need a target to create a ruleset')
        if isinstance(target, Description):
            raise ApiException('Cannot perform a ruleset with a description kpi')

        data = {
            "projectId": self.__project_id,
            "task": {
                "type": "learning",
                "datasetId": dataset.dataset_id,
                "projectId": self.__project_id,
                "params": {
                    "learningName": name,
                    "datasetName": dataset.name,
                    "buildPredictiveModel": 0,
                    "sourceFileName": dataset.source_file_name,
                    "delimiter": dataset.separator,
                    "complexityExhaustive": rule_complexity,
                    "countQuantiles": quantiles,
                    "discretizations": discretizations,
                    "minMarginalContribution": min_marginal_contribution,
                    "target": [],
                    "kpis": []
                }
            }
        }

        for _id, _type in zip(target.score_ids, target.scores):
            _kpiData = {
                "kpiId": _id,
                "type": _type,
                "kpiFamily": target.indicator_family,
                "scoreType": _type,
                "kpiType": target.indicator_type,
                "output": target.variable_name,
                "kpiName": target.name,
                "omodality": target.modality
            }
            if _type == self._PURITY and score_purity_min is not None:
                _kpiData['minValue'] = score_purity_min
            elif _type == self._COVERAGE and coverage_min is not None:
                _kpiData['minValue'] = coverage_min
            elif _type == self._LIFT and lift_min is not None:
                _kpiData['minValue'] = lift_min
            elif _type == self._ZSCORE and zscore_min is not None:
                _kpiData['minValue'] = zscore_min
            elif _type == self._AVERAGE_VALUE and average_value_min is not None:
                _kpiData['minValue'] = average_value_min
            elif _type == self._STANDARD_DEVIATION and standard_deviation_max is not None:
                _kpiData['maxValue'] = standard_deviation_max
            elif _type == self._SHIFT and shift_min is not None:
                _kpiData['minValue'] = shift_min
            data['task']['params']['target'].append(_kpiData)

        msg = "Ruleset settings: \n\t- Target: {}".format(target.name) + \
              ("\n\t- Min Purity: {}".format(score_purity_min) if score_purity_min is not None else "") + \
              ("\n\t- Min Coverage: {}".format(coverage_min) if coverage_min is not None else "") + \
              ("\n\t- Min Lift: {}".format(lift_min) if lift_min is not None else "") + \
              ("\n\t- Min Z-score: {}".format(zscore_min) if zscore_min is not None else "") + \
              ("\n\t- Min Average value: {}".format(average_value_min) if average_value_min is not None else "") + \
              ("\n\t- Max Standard deviation: {}".format(standard_deviation_max) if standard_deviation_max is not None else "") + \
              ("\n\t- Min Shift: {}".format(shift_min) if shift_min is not None else "") + \
              "\n\t- Rule Complexity: {}\n\t- Default Number of Bins: {} \n\t- Enable custom discretizations: {}  \n\t- Min Marginal contribution: \
{}".format(rule_complexity, quantiles, enable_custom_discretizations, min_marginal_contribution)

        if (len(compute_other_key_indicators) > 0):
            for key_indicator in compute_other_key_indicators:
                for _id, _type in zip(key_indicator.target.score_ids, key_indicator.target.scores):
                    _kpiKI = {
                        "kpiId": _id,
                        "type": _type,
                        "kpiFamily": key_indicator.target.indicator_family,
                        "scoreType": _type,
                        "kpiType": key_indicator.target.indicator_type,
                        "output": key_indicator.target.variable_name,
                        "kpiName": key_indicator.target.name,
                        "omodality": key_indicator.target.modality
                    }
                    if (key_indicator.target.indicator_type == self._DISCRETE_MODALITY or key_indicator.target.indicator_type == self._DISCRETE):
                        if _type == self._PURITY:
                            if key_indicator.purity_min is not None:
                                _kpiKI['minValue'] = key_indicator.purity_min
                            if key_indicator.purity_max is not None:
                                _kpiKI['maxValue'] = key_indicator.purity_max
                        elif _type == self._COVERAGE:
                            if key_indicator.coverage_min is not None:
                                _kpiKI['minValue'] = key_indicator.coverage_min
                            if key_indicator.coverage_max is not None:
                                _kpiKI['maxValue'] = key_indicator.coverage_max
                        elif _type == self._LIFT:
                            if key_indicator.lift_min is not None:
                                _kpiKI['minValue'] = key_indicator.lift_min
                            if key_indicator.lift_max is not None:
                                _kpiKI['maxValue'] = key_indicator.lift_max
                        elif _type == self._ZSCORE:
                            if key_indicator.zscore_min is not None:
                                _kpiKI['minValue'] = key_indicator.zscore_min
                            if key_indicator.zscore_max is not None:
                                _kpiKI['maxValue'] = key_indicator.zscore_max
                    else:
                        if _type == self._AVERAGE_VALUE:
                            if key_indicator.average_value_min is not None:
                                _kpiKI['minValue'] = key_indicator.average_value_min
                            if key_indicator.average_value_max is not None:
                                _kpiKI['maxValue'] = key_indicator.average_value_max
                        elif _type == self._STANDARD_DEVIATION:
                            if key_indicator.standard_deviation_min is not None:
                                _kpiKI['minValue'] = key_indicator.standard_deviation_min
                            if key_indicator.standard_deviation_max is not None:
                                _kpiKI['maxValue'] = key_indicator.standard_deviation_max
                        elif _type == self._SHIFT:
                            if key_indicator.shift_min is not None:
                                _kpiKI['minValue'] = key_indicator.shift_min
                            if key_indicator.shift_max is not None:
                                _kpiKI['maxValue'] = key_indicator.shift_max
                    if 'kpis' not in data['task']['params']:
                        data['task']['params']['kpis'] = []
                    data['task']['params']['kpis'].append(_kpiKI)

        if (locally_increase_complexity):
            data['task']['params']['maxComplexity'] = max_complexity
            data['task']['params']['nbMinimizations'] = nb_minimizations
            data['task']['params']['coverageIncrement'] = coverage_increment
            msg += "\n\t- Max complexity: {} \n\t- Number of Minimizations: {} \n\t- Minimization \
            Coverage Increment: {}".format(max_complexity, nb_minimizations, coverage_increment)
        if (validate_stability):
            data['task']['params']['percentageSplit'] = split_ratio
            data['task']['params']['nbModels'] = nb_iterations
            data['task']['params']['purityTolerance'] = purity_tolerance
            msg += "\n\t- Percentage split: {} \n\t- Number of Iterations: {} \n\t- Purity Tolerance: {}".format(split_ratio, nb_iterations, purity_tolerance)

        print(msg)
        _ruleset = self.__api.Task.createtask(project_ID=self.__project_id, json=data)
        self.__api.handle_work_states(self.__project_id, work_type='learning', work_id=_ruleset.get('_id'))
        return self.get(name)

    @Helper.try_catch
    def filter(self):
        """
        filter()

        Get all the rulesets of the project.

        Returns:
            list(Ruleset): all the rulesets
        """
        from HyperAPI.hyper_api.dataset import DatasetFactory
        factory = DatasetFactory(self.__api, self.__project_id)
        ruleset_project = self.__api.Rules.getlearnings(project_ID=self.__project_id)
        return [Ruleset(self, self.__api, factory.get_by_id(ruleset.get('datasetId')), ruleset) for ruleset in ruleset_project]

    @Helper.try_catch
    def minimize(self, ruleset, minimization_name, score_to_minimize='Purity', increment_threshold=0.01):
        """
        minimize(ruleset, minimization_name, score_to_minimize='Purity', increment_threshold=0.01)

        Perform a minimzation on a given ruleset.

        Args:
            ruleset (Ruleset): Ruleset to minimize
            minimization_name (str): Name of the new ruleset
            score_to_minimize (str): Score to apply the minimization, default is 'Purity'
            increment_threshold (float): Percentage increment of target samples that a new rule must bring to be added to the minimized ruleset, default is 0.01

        Return:
            Ruleset: Minimized ruleset
        """
        json = {
            "type": "minimization",
            "datasetId": ruleset.dataset_id,
            "projectId": ruleset.project_id,
            "params": {
                "query": "tagsfilter={}".format(urllib.parse.quote(ruleset.name)),
                "taglist": [ruleset.name],
                "incrementThreshold": increment_threshold,
                "tag": minimization_name
            }
        }
        _kpiId = decode_kpiname_to_id(ruleset.kpis, score_to_minimize)
        if _kpiId != score_to_minimize:
            json['params']['kpiId'] = _kpiId

        _kpi_corr = self.__api.Kpi.getkpicorrelation(project_ID=ruleset.project_id)
        json['params']['kpisList'] = _kpi_corr

        _ruleset = self.__api.Task.createtask(project_ID=ruleset.project_id, json=json)
        self.__api.handle_work_states(ruleset.project_id, work_type='minimization', work_id=_ruleset.get('_id'))
        return self.get(minimization_name)

    def get(self, name):
        """
        get(name)

        Get a ruleset by name

        Args:
            name (str): Name of the ruleset

        Returns:
            Ruleset: Retrieved ruleset
        """
        try:
            return [ruleset for ruleset in self.filter() if ruleset.name == name][0]
        except IndexError:
            return []

    @Helper.try_catch
    def get_by_id(self, id):
        """
        get_by_id(id)

        Get the ruleset matching the given ID or None if there is no match

        Args:
            id (str): ID of the ruleset

        Returns:
            Ruleset or None: retrieved ruleset
        """
        rulesets = [ruleset for ruleset in self.filter() if ruleset.id == id]
        if rulesets:
            return rulesets[0]
        return None

    def get_or_create(self, dataset, name, target=None, purity_min=None, coverage_min=None, lift_min=None, zscore_min=None, average_value_min=None,
                      standard_deviation_max=None, shift_min=None, rule_complexity=2, quantiles=10,
                      enable_custom_discretizations=True, min_marginal_contribution=None, compute_other_key_indicators=None,
                      locally_increase_complexity=False, max_complexity=3, nb_minimizations=1, coverage_increment=0.01,
                      validate_stability=False, split_ratio=0.7, nb_iterations=1, purity_tolerance=0.1):
        """
        get_or_create(dataset, name, target=None, purity_min=None, coverage_min=None, lift_min=None, zscore_min=None, average_value_min=None, standard_deviation_max=None, shift_min=None, rule_complexity=2, quantiles=10, enable_custom_discretizations=True, min_marginal_contribution=None, compute_other_key_indicators=None, locally_increase_complexity=False, max_complexity=3, nb_minimizations=1, coverage_increment=0.01, validate_stability=False, split_ratio=0.7, nb_iterations=1, purity_tolerance=0.1)

        Get or create a ruleset, if the ruleset exists, only the name is mandatory

        Args:
            dataset (Dataset): Dataset used to generate the ruleset
            name (str): Name of the new ruleset
            target (Target): Target to generate the ruleset
            purity_min (float): Minimum purity of rules, default is the entire dataset purity (discrete target only)
            coverage_min (int): Minimum coverage of the target population for each rule, default is 10 (discrete target only)
            lift_min (float): Minimum lift, default is 1 (discrete target only)
            zscore_min (float): Minimum Z-score, default is None (discrete target only)
            average_value_min (float): Minimum average value, default is average value of the target on the whole dataset (continuous target only)
            standard_deviation_max (float) : Maximum standard deviation, default is None (continuous target only)
            shift_min (float): Minimum shift, default is None (continuous target only)
            rule_complexity (int): Maximum number of variables in rules, default is 2
            quantiles (int): Number of intervals the continuous variables are quantized in, default is 10
            enable_custom_discretizations (boolean): use custom discretizations, eventually use "quantiles" parameter for remaining variables, default is True
            min_marginal_contribution (float): a new rule R', created by adding a new constraint to an existing rule R (and thus increasing its complexity),
                is added to the ruleset if and only if it increases the original purity of R by the minimum marginal contribution or more. Default is 0.1
            compute_other_key_indicators (list of KeyIndicatorOption): Compute other Key Indicators.
            locally_increase_complexity (bool): Enable the locally increase complexity when set as true. Default is False
            max_complexity (int): Maximum numbers of features per rule. Default is 3
            nb_minimizations (int):Interate the minimization process. Default is 1
            coverage_increment (float): Percentage increment of target samples that a new rule must bring to be added to the minimization ruleset.
                Default is 0.01
            validate_stability (bool): Enable to split your dataset, add iteration and set a purity tolerance when set as true. Default is False
            split_ratio (float): The percentage for the split (Between 0 and 1). Default is 0.7
            nb_iterations (int): Number of iterations wanted. Default is 1
            purity_tolerance (float): Purity tolerence allowed (Between 0 and 1). Default is 0.1

        Returns:
            Ruleset: Retrieved or created ruleset
        """

        for ruleset in dataset.rulesets:
            if (ruleset.name == name) and (ruleset.dataset_id == dataset.dataset_id):
                return ruleset

        return self.create(dataset, name, target, purity_min, coverage_min, lift_min, zscore_min, average_value_min, standard_deviation_max, shift_min,
                           rule_complexity, quantiles, enable_custom_discretizations,
                           min_marginal_contribution, compute_other_key_indicators, locally_increase_complexity, max_complexity,
                           nb_minimizations, coverage_increment, validate_stability, split_ratio, nb_iterations, purity_tolerance)


class Ruleset(Base):
    """
    Ruleset()
    """
    def __init__(self, factory, api, dataset, json_return):
        self.__api = api
        self.__factory = factory
        self.__json_returned = json_return
        self.__dataset = dataset
        self._is_deleted = False
        self._is_in_error = self.__json_returned.get('status', '').lower() == "error"

    def __repr__(self):
        return """\n{} : {} <{}>\n""".format(
            self.__class__.__name__,
            self.name,
            self.id
        ) + ("\t<! This ruleset has been deleted>\n" if self._is_deleted else "") + \
            ("\t<! This ruleset is in error>\n" if self._is_in_error else "") + \
            """\t- Dataset : {}\n\t- Rules count : {}\n\t- Created on : {}\n""".format(
            self.dataset_name,
            self.rules_count,
            self.created.strftime('%Y-%m-%d %H:%M:%S UTC'))

    # Property part
    @property
    def _json(self):
        return self.__json_returned

    @property
    def dataset_name(self):
        return self.__json_returned.get('datasetName')

    @property
    def name(self):
        """
        str: Ruleset name
        """
        if self.__json_returned.get('tag') is not None:
            if type(self.__json_returned.get('tag')) == str:
                return self.__json_returned.get('tag')
            else:
                return self.__json_returned.get('tag').get('tagName', '')
        else:
            return ''

    @property
    def kpis(self):
        """
        list(dict): Kpis of the ruleset
        """
        if self._is_in_error:
            return None
        return self.__json_returned.get('tag').get('kpis')

    @property
    def rules_count(self):
        """
        int: Number of rules in the ruleset
        """
        return self.__json_returned.get('rulesCount', None)

    @property
    def dataset_id(self):
        """
        str: Dataset ID
        """
        return self.__json_returned.get('datasetId')

    @property
    def project_id(self):
        """
        str: Project ID
        """
        return self.__json_returned.get('projectId')

    @property
    def created(self):
        """
        datetime: Created date
        """
        createdAt = self.__json_returned.get('lastChangeAt', self.__json_returned.get('createdAt'))
        if createdAt.find('.') > 0:
            return self.str2date(createdAt, '%Y-%m-%dT%H:%M:%S.%fZ')
        return self.str2date(createdAt, '%Y-%m-%dT%H:%M:%S')

    @property
    def id(self):
        """
        str: Ruleset ID
        """
        return self.__json_returned.get('_id')

    # Method part
    @Helper.try_catch
    def _get_params(self):
        if not self._is_deleted:
            return NotImplemented

    @Helper.try_catch
    def _export(self):
        if not self._is_deleted:
            return NotImplemented

    @Helper.try_catch
    def delete(self):
        """
        delete()

        Delete the ruleset
        """
        if not self._is_deleted:
            json = {
                '_id': self.id,
                'status': self.__json_returned.get('status', 'done').lower() if
                type(self.__json_returned.get('tag')) == str else 'done',
                'tagName': self.name
            }
            self.__api.Rules.removealearning(project_ID=self.project_id, dataset_ID=self.dataset_id, json=json)
            if RulesetFactory(self.__api, self.project_id).get_by_id(self.id) is None:
                self._is_deleted = True
        return self

    @Helper.try_catch
    def minimize(self, minimization_name, score_to_minimize='Purity', increment_threshold=0.01):
        """
        minimize(minimization_name, score_to_minimize='Purity', increment_threshold=0.01)

        Function to apply a minimization on a ruleset

        Args:
            minimization_name (str): Name of the new ruleset
            score_to_minimize (str): Score to apply the minimization, default is purity
            increment_threshold (float): Percentage increment of target samples that a new rule must bring to be added to the minimized ruleset. Default is 0.01

        Returns:
            Ruleset or None: minimized ruleset of None if ruleset is deleted / in error
        """
        if not self._is_deleted and not self._is_in_error:
            return self.__factory.minimize(self, minimization_name, score_to_minimize, increment_threshold)
        return None

    @Helper.try_catch
    def get_rules(self, skip=0, limit=100, sort=None, min_scores=[], max_scores=[], include_variables=[], exclude_variables=[], include_tags=[], exclude_tags=[]):
        """
        get_rules(skip=0, limit=100, sort=None, min_scores=[], max_scores=[], include_variables=[], exclude_variables=[], include_tags=[], exclude_tags=[])

        Args:
            skip (int): Number of rules to skip, default is 0
            limit (int): Number of rules to get, default is 100
            sort (dict): Sorting the rules on a score in a given order
                Example {'score': 'size', 'asc': true}
                score takes one of the following values: ["Purity", "Coverage", "size", "complexity"]
                if kpi has additionnal scores, it can also be used ("Lift, Z-score") 
            min_scores (array): Array of minimum value for a score. 
                (logical AND between several values)
                Example: [{'score': 'Purity', 'value': 0,563}, {'score': 'Coverage', 'value': 20}]
            max_scores (array): Array of maximum value for a score. 
                (logical AND between several values)
                Example: [{'score': 'Purity', 'value': 0,9}, {'score': 'Coverage', 'value': 300}]
            include_variables (array): Array of variable names to include
            exclude_variables (array): Array of variables names to exclude 
            include_tags (array): Array of variables tags to include 
            exclude_tags (array): Array of variables tags to exclude 

        Returns:
            list(rules) or None: All the rules of None if Ruleset is deleted / in error
        """
        if not self._is_deleted and not self._is_in_error:
            # _params is used in HDP version elder than 6.0 whereas _query is used in later versions
            _params = {'skip': skip,
                    'limit': limit,
                    'tagsfilter': self.name
                    }

            _query = {
                'params': {
                    'rulesetNames': [self.name], 
                    'skip': skip,
                    'limit': limit 
                }
            }
            
            kpis_param = []
            for kpi in self.kpis:
                kpis_param.append({
                    "filterId": kpi["kpiId"],
                    "minValue": "",
                    "maxValue": "",
                    "sortby": 0
                })
            for kpiId in ["size", "complexity"]:
                kpi = {
                    "filterId": kpiId, 
                    "minValue": "",
                    "maxValue": "",
                    "sortby": 0
                }
                if kpi not in kpis_param:
                    kpis_param.append(kpi)
            
            if sort:
                if 'score' in sort and 'asc' in sort:
                    kpi_id = decode_kpiname_to_id(self.kpis, sort['score'])
                    if sort['asc']:
                        _params['sortasc'] = kpi_id
                    else:
                        _params['sortdesc'] = kpi_id

                    kpi_index = -1
                    for i, item in enumerate(kpis_param):
                        if item["filterId"] == kpi_id:
                            kpi_index = i
                    if kpi_index > -1:    
                        if sort['asc']:
                            kpis_param[kpi_index]["sortby"] = 2
                        else:
                            kpis_param[kpi_index]["sortby"] = 1

                else:
                    raise ValueError('Wrong sort parameter, please follow this syntax : {\'score\': String, \'asc\': Boolean}')

            if min_scores:
                if isinstance(min_scores, dict):
                    min_scores = [min_scores]
                for score in min_scores:
                    if 'score' in score and 'value' in score:
                        kpi_id = decode_kpiname_to_id(self.kpis, score['score'])
                        _params['min ' + kpi_id] = score['value']
                        kpi_index = -1
                        for i, item in enumerate(kpis_param):
                            if item["filterId"] == kpi_id:
                                kpi_index = i
                        if kpi_index > -1:
                            kpis_param[kpi_index]["minValue"] = score['value']
                    else:
                        raise ValueError('Wrong min_score parameter, please follow this syntax : {\'score\': String, \'value\': Number}')

            if max_scores:
                if isinstance(max_scores, dict):
                    max_scores = [max_scores]
                for score in max_scores:
                    if 'score' in score and 'value' in score:
                        kpi_id = decode_kpiname_to_id(self.kpis, score['score'])
                        _params['max ' + kpi_id] = score['value']
                        kpi_index = -1
                        for i, item in enumerate(kpis_param):
                            if item["filterId"] == kpi_id:
                                kpi_index = i
                        if kpi_index > -1:
                            kpis_param[kpi_index]["maxValue"] = score['value']
                    else:
                        raise ValueError('Wrong max_score parameter, please follow this syntax : {\'score\': String, \'value\': Number}')

            _query["params"]["kpis"] = kpis_param

            variables_or_tags_param = {
                "include_lists": {
                    "variables": [],
                    "tags": [],
                    "metatypeTags": []
                },
                "exclude_lists": {
                    "variables": [],
                    "tags": [],
                    "metatypeTags": []
                }
            }

            if include_variables:
                if isinstance(include_variables, str):
                    include_variables = [include_variables]
                _params['varinclus'] = urllib.parse.quote(','.join(include_variables), safe='~()*!.\'')
                variables_or_tags_param["include_lists"]["variables"] = include_variables

            if exclude_variables:
                if isinstance(exclude_variables, str):
                    exclude_variables = [exclude_variables]
                _params['varexclus'] = urllib.parse.quote(','.join(exclude_variables), safe='~()*!.\'')
                variables_or_tags_param["exclude_lists"]["variables"] = exclude_variables

            if include_tags:
                if isinstance(include_tags, str):
                    include_tags = [include_tags]
                _params['varTagInclus'] = urllib.parse.quote(','.join(include_tags), safe='~()*!.\'')
                variables_or_tags_param["include_lists"]["tags"] = include_tags

            if exclude_tags:
                if isinstance(exclude_tags, str):
                    exclude_tags = [exclude_tags]
                _params['varTagExclus'] = urllib.parse.quote(','.join(exclude_tags), safe='~()*!.\'')
                variables_or_tags_param["exclude_lists"]["tags"] = exclude_tags

            _query["params"]["variablesOrTags"] = variables_or_tags_param

            if self.__api.session.version >= self.__api.session.version.__class__('6.0.1'):
                json_returned = self.__api.Rules.getrules(project_ID=self.project_id, dataset_ID=self.dataset_id, json=_query).get('rules')
            else:
                json_returned = self.__api.Rules.getrules(project_ID=self.project_id, dataset_ID=self.dataset_id, params=_params).get('rules')
            return Rules(self.__api, json_returned, self.kpis, self.project_id, self.dataset_id)
        return None

    @Helper.try_catch
    def predict(self, dataset, name, target, nb_minimizations=1, coverage_increment=0.01):
        """
        predict(dataset, name, target, nb_minimizations=1, coverage_increment=0.01)

        Create a prediction model from the ruleset

        Args:
            dataset (Dataset): Dataset to apply the prediction
            name (str): Name of the new model
            target (Target): Target used to generate the model
            nb_minimizations (int): Number of minimizations to perform on the ruleset, default is 1
            coverage_increment (float): Percentage increment of target samples that a new rule must bring to be added to the minimized ruleset,
                default is 0.01

        Returns:
            Model or None: Model or None if the ruleset is deleted / in error
        """
        if not self._is_deleted and not self._is_in_error:
            return ModelFactory(self.__api, self.project_id).predict_from_ruleset(self.__dataset, dataset, self.name, name, target,
                                                                                  nb_minimizations, coverage_increment)
        return None
