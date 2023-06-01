from abc import ABC
from typing import Iterable, Iterator
from unittest.mock import MagicMock


class IterableMock(MagicMock, Iterable):
    def __iter__(self):
        return super().__iter__()
