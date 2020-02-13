"""Tests for the TessSyntheticInjectedDatabase class."""
from pathlib import Path
from unittest.mock import Mock

import pytest

from ramjet.photometric_database.tess_synthetic_injected_database import TessSyntheticInjectedDatabase


class TestTessSyntheticInjectedDatabase:
    """Tests for the TessSyntheticInjectedDatabase class."""
    @pytest.fixture
    def database(self) -> TessSyntheticInjectedDatabase:
        """
        Sets up the database for use in a test.

        :return: The database.
        """
        return TessSyntheticInjectedDatabase()

    @pytest.mark.functional
    def test_can_generate_training_and_validation_datasets(self, database):
        # Mock and initialize dataset components for simple testing.
        batch_size = 10
        database.batch_size = batch_size
        time_steps_per_example = 20
        database.time_steps_per_example = time_steps_per_example
        database.lightcurve_directory = Mock(glob=Mock(return_value=(Path(f'{index}.fits') for index in range(30))))
        database.synthetic_signal_directory = Mock(glob=Mock(return_value=(Path(f'{index}.feather')
                                                                           for index in range(40))))
        # Generate the datasets.
        training_dataset, validation_dataset = database.generate_datasets()
        # Test the datasets look right.
        training_iterator = iter(training_dataset)
        training_batch0 = next(training_iterator)
        training_batch1 = next(training_iterator)
        validation_iterator = iter(validation_dataset)
        validation_batch0 = next(validation_iterator)
        validation_batch1 = next(validation_iterator)
        assert training_batch0[0].shape == (batch_size, time_steps_per_example, 1)  # Batch examples shape
        assert training_batch1[0].shape == (batch_size, time_steps_per_example, 1)
        assert training_batch1[1].shape == (batch_size, 1)  # Batch labels shape
        assert training_batch1[1].sum == batch_size // 2  # Half the labels are positive.
        assert validation_batch0[0].shape == (batch_size, time_steps_per_example, 1)
        assert validation_batch1[0].shape == (batch_size, time_steps_per_example, 1)
