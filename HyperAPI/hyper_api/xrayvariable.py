from HyperAPI.util import Helper
from HyperAPI.hyper_api.base import Base


class XRayVariableFactory:
    """
    """
    def __init__(self, api, xray):
        self.__api = api
        self.__xray = xray

    @Helper.try_catch
    def filter(self):
        """
        filter()

        Returns:
            list(XRayVariable): list of variables in the current Xray
        """
        json = {'project_ID': self.__xray.project_id, 'dataset_ID': self.__xray.dataset_id, 'simpleLift_ID': self.__xray.id}
        return [XRayVariable(self.__api, self.__xray, x) for x in self.__api.SimpleLift.getsimplelift(**json).get('variables')]

    @Helper.try_catch
    def get(self, name):
        """
        get(name)

        Returns:
            XRayVariable: Xray variable found by name
        """
        xray_variables = list(filter(lambda x: x.name == name, self.filter()))
        if xray_variables:
            return xray_variables[0]
        return None

    @Helper.try_catch
    def get_by_column_id(self, column_id):
        """
        get_by_column_id(column_id)

        Returns:
            XRayVariable: XRay variable found by column_id
        """
        xray_variables = list(filter(lambda x: x.column_id == column_id, self.filter()))
        if xray_variables:
            return xray_variables[0]
        return None

    @Helper.try_catch
    def sort(self, contrast_rate=None, reverse=True):
        """
        sort(contrast_rate=None, reverse=True)

        sort Xray variables (default is by decreasing constrast rate on first target)

        Args:
            contrast_rate: name of the variable on which sorting by contrast rate will be done
            reverse (bool): order to reverse (default: True => decreasing)

        Returns:
            XRayVariable[]: list of variables in the current Xray
        """
        xray_variables = self.filter()

        if contrast_rate is None:
            return sorted(xray_variables, key=lambda x: list(x.contrast_rates.values())[0], reverse=reverse)
        if len(xray_variables) == 0 or not xray_variables[0].contrast_rates or xray_variables[0].contrast_rates.get(contrast_rate) is None:
            return xray_variables
        return sorted(xray_variables, key=lambda x: x.contrast_rates.get(contrast_rate), reverse=reverse)


class XRayVariable(Base):
    """
    XRayVariable()
    """
    def __init__(self, api, xray, json_return):
        self.__api = api
        self.__xray = xray
        self.__details = None
        self.__json_returned = json_return

    def __repr__(self):
        return "\n{} : {}\n".format(
            self.__class__.__name__,
            self.name
        ) + ("\t<! This variable has been ignored>\n" if self.is_ignored else "") + \
            """\t- Discrete: {}\n""".format(
            self.is_discrete
        ) + ("""\t- Contrast rates: {}\n""".format(self.contrast_rates) if self.contrast_rates else "")

    @property
    def _json(self):
        return self.__json_returned

    @property
    def name(self):
        """
        str: XRay variable name
        """
        return self.__json_returned.get('name')

    @property
    def is_discrete(self):
        """
        Boolean: Indicating if this XRay variable is discrete
        """
        return self.__json_returned.get('type') == 'D'

    @property
    def is_ignored(self):
        """
        Boolean: Indicating if this XRay variable is ignored
        """
        return self.__json_returned.get('ignored')

    @property
    def column_id(self):
        """
        str: Indicating if this XRay variable is ignored
        """
        return self.__json_returned.get('column')

    @property
    def contrast_rates(self):
        """
        Dict: key is variable name (str) and value is contrast_rates (float)
        """
        contrastRates = self.__json_returned.get('contrastRates')
        return contrastRates

    @property
    def coverage(self):
        self.__get_details()
        return self.__details.get('coverage')

    @property
    def xValues(self):
        self.__get_details()
        return self.__details.get('xValues')

    @property
    def yValues(self):
        self.__get_details()
        return self.__details.get('yValues')

    @property
    def size(self):
        self.__get_details()
        return self.__details.get('size')

    @property
    def value_range(self):
        self.__get_details()
        return self.__details.get('value_range')

    @property
    def outputs(self):
        self.__get_details()
        return self.__details.get('outputs')

    @Helper.try_catch
    def __get_details(self):
        if self.__details is None:
            json = {'project_ID': self.__xray.project_id, 'dataset_ID': self.__xray.dataset_id, 'task_ID': self.__xray.id, 'column_ID': self.column_id}
            self.__details = self.__api.SimpleLift.getvariablejsonfile(**json)
        return self.__details
