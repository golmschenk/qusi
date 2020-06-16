import pytest
import numpy as np
from pathlib import Path

from ramjet.photometric_database.lightcurve_collection import LightcurveCollection
from ramjet.photometric_database.standard_and_injected_lightcurve_database import StandardAndInjectedLightcurveDatabase


class TestStandardAndInjectedLightcurveDatabase:
    def test_database_has_lightcurve_collection_properties(self):
        database = StandardAndInjectedLightcurveDatabase()
        assert hasattr(database, 'training_standard_lightcurve_collections')
        assert hasattr(database, 'training_injectee_lightcurve_collection')
        assert hasattr(database, 'training_injectable_lightcurve_collections')
        assert hasattr(database, 'validation_standard_lightcurve_collections')
        assert hasattr(database, 'validation_injectee_lightcurve_collection')
        assert hasattr(database, 'validation_injectable_lightcurve_collections')

    @pytest.mark.functional
    def test_database_can_generate_training_and_validation_datasets(self):
        database = StandardAndInjectedLightcurveDatabase()
        # Setup mock lightcurve collections.
        standard_lightcurve_collection0 = LightcurveCollection(
            lambda: [Path('standard_path0.ext')], lambda path: (np.array([10, 20, 30]), np.array([0, 1, 2])), 0)
        standard_lightcurve_collection1 = LightcurveCollection(
            lambda: [Path('standard_path1.ext')], lambda path: (np.array([20, 30, 40]), np.array([1, 2, 3])), 0)
        injectee_lightcurve_collection = LightcurveCollection(
            lambda: [Path('injectee_path.ext')], lambda path: (np.array([30, 40, 50]), np.array([2, 3, 4])), 0)
        injectable_lightcurve_collection0 = LightcurveCollection(
            lambda: [Path('injectable_path0.ext')], lambda path: (np.array([0, 10, 20]), np.array([0.5, 1, 1.5])), 0)
        injectable_lightcurve_collection1 = LightcurveCollection(
            lambda: [Path('injectable_path1.ext')], lambda path: (np.array([0, 10, 20, 30]), np.array([0, 1, 1, 0])), 0)
        database.training_standard_lightcurve_collections = [standard_lightcurve_collection0,
                                                             standard_lightcurve_collection1]
        database.training_injectee_lightcurve_collection = injectee_lightcurve_collection
        database.training_injectable_lightcurve_collections = [injectable_lightcurve_collection0,
                                                               injectable_lightcurve_collection1]
        database.validation_standard_lightcurve_collections = [standard_lightcurve_collection1]
        database.validation_injectee_lightcurve_collection = injectee_lightcurve_collection
        database.validation_injectable_lightcurve_collections = [injectable_lightcurve_collection1]
        # Setup simplified database settings
        database.batch_size = 4
        database.time_steps_per_example = 3
        database.normalize = lambda fluxes: fluxes  # Don't normalize values to keep it simple.
        training_dataset, validation_dataset = database.generate_datasets()
        training_batch = next(training_dataset)
        assert np.array_equal(training_batch[0].numpy(), [0, 1, 2])  # Standard lightcurve 0.
        assert np.array_equal(training_batch[1].numpy(), [1, 2, 3])  # Standard lightcurve 1.
        assert np.array_equal(training_batch[2].numpy(), [1, 3, 6])  # Injected lightcurve 0.
        assert np.array_equal(training_batch[3].numpy(), [0, 3, 4])  # Injected lightcurve 1, with injectable clipped.
        validation_batch = next(validation_dataset)
        assert np.array_equal(validation_batch[0].numpy(), [1, 2, 3])  # Standard lightcurve 1.
        assert np.array_equal(validation_batch[1].numpy(), [0, 3, 4])  # Injected lightcurve 1, with injectable clipped.
        assert np.array_equal(validation_batch[2].numpy(), [1, 2, 3])  # Standard lightcurve 1.
        assert np.array_equal(validation_batch[3].numpy(), [0, 3, 4])  # Injected lightcurve 1, with injectable clipped.
