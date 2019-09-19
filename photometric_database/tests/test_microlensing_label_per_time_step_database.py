"""Tests for the MicrolensingLabelPerTimeStepDatabase class."""
from photometric_database.microlensing_label_per_time_step_database import MicrolensingLabelPerTimeStepDatabase


class TestMicrolensingLabelPerTimeStepDatabase:
    """Tests for the MicrolensingLabelPerTimeStepDatabase class."""
    def test_can_read_microlensing_meta_data_file(self):
        database = MicrolensingLabelPerTimeStepDatabase()
        meta_data_file_path = 'photometric_database/tests/resources/shortened_candlist_RADec.dat.txt'
        meta_data_frame = database.load_microlensing_meta_data(meta_data_file_path)
        assert meta_data_frame['tE'].iloc[1] == 1.1946342164440846e+02
