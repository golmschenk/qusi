from pathlib import Path
from unittest.mock import Mock, patch

import pytest

import ramjet.data_interface.tess_two_minute_cadence_light_curve_metadata_manager as module
from ramjet.data_interface.tess_two_minute_cadence_light_curve_metadata_manager import (
    TessTwoMinuteCadenceLightCurveMetadataManger,
)


class TestTessTwoMinuteCadenceLightCurveMetadataManger:
    @pytest.fixture
    def metadata_manger(self) -> TessTwoMinuteCadenceLightCurveMetadataManger:
        """
        The metadata manager class instance under test.

        :return: The metadata manager.
        """
        return TessTwoMinuteCadenceLightCurveMetadataManger()

    @patch.object(module.TessTwoMinuteCadenceLightCurveMetadata, "insert_many")
    def test_can_insert_multiple_sql_database_rows_from_paths(
        self, mock_insert_many, metadata_manger
    ):
        with patch.object(module, "metadatabase"):
            light_curve_path0 = Path(
                "light_curves/tess2019169103026-s0013-0000000382068171-0146-s_lc.fits"
            )
            light_curve_path1 = Path(
                "light_curves/tess2019112060037-s0011-0000000280909647-0143-s_lc.fits"
            )
            uuid0 = "mock-uuid-output0"
            uuid1 = "mock-uuid-output1"
            with (
                patch.object(module, "metadatabase_uuid") as mock_metadatabase_uuid,
                patch.object(
                    module, "dataset_split_from_uuid"
                ) as mock_dataset_split_generator,
            ):
                mock_dataset_split_generator.side_effect = [2, 3]
                mock_metadatabase_uuid.side_effect = [uuid0, uuid1]
                metadata_manger.insert_multiple_rows_from_paths_into_database(
                    light_curve_paths=[light_curve_path0, light_curve_path1]
                )
            expected_insert = [
                {
                    "path": str(light_curve_path0),
                    "tic_id": 382068171,
                    "sector": 13,
                    "dataset_split": 2,
                },
                {
                    "path": str(light_curve_path1),
                    "tic_id": 280909647,
                    "sector": 11,
                    "dataset_split": 3,
                },
            ]
            assert mock_insert_many.call_args[0][0] == expected_insert

    @patch.object(Path, "glob")
    def test_can_populate_sql_dataset(self, mock_glob, metadata_manger):
        path_list = [
            metadata_manger.light_curve_root_directory_path.joinpath(f"{index}.fits")
            for index in range(20)
        ]
        mock_glob.return_value = path_list
        mock_insert = Mock()
        metadata_manger.insert_multiple_rows_from_paths_into_database = mock_insert
        metadata_manger.populate_sql_database()
        assert mock_insert.call_args[0][0] == [
            Path(f"{index}.fits") for index in range(20)
        ]
