from unittest.mock import Mock


class PicklableMock(Mock):
    """Makes the Mock picklable for use in multiprocessing methods."""

    def __reduce__(self):
        return Mock, ()
