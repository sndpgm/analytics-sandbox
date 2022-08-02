"""Utility tool functions models code."""


class Bunch(dict):
    """
    Returns a dict-like object with keys accessible via attribute lookup.
    Parameters
    ----------
    *args
        Arguments passed to dict constructor, tuples (key, value).
    **kwargs
        Keyword argument passed to dict constructor, key=value.
    """

    def __init__(self, *args, **kwargs):
        super(Bunch, self).__init__(*args, **kwargs)
        self.__dict__ = self
