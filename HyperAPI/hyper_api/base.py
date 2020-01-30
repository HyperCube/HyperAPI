from datetime import datetime


class Base(object):
    def __repr__(self):
        return "{} : {} => {}".format(self.__class__.__name__, self.__getattribute__('name'), [x for x in dir(self) if "__" not in x])

    @staticmethod
    def timestamp2date(timestamp):
        if timestamp is None:
            return None
        return datetime.fromtimestamp(timestamp / 1000)

    @staticmethod
    def str2date(string, date_format):
        if string is None or date_format is None:
            return None
        return datetime.strptime(string, date_format)
