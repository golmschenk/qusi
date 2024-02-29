from collections.abc import Iterable
from unittest.mock import MagicMock


class IterableMock(MagicMock, Iterable):
    """
    A workaround class to fix inspections where MagicMock was not considered iterable for type checking.
    """

    def __iter__(self):
        return super().__iter__()
