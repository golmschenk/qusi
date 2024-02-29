from pathlib import Path
from unittest.mock import Mock, patch

import pytest

import ramjet.data_interface.tess_ffi_light_curve_metadata_manager as module
from ramjet.data_interface.tess_ffi_light_curve_metadata_manager import (
    TessFfiLightCurveMetadataManager,
)


class TestTessFfiLightCurveMetadataManager:
    @pytest.fixture
    def metadata_manger(self) -> TessFfiLightCurveMetadataManager:
        """
        The metadata manager class instance under test.

        :return: The metadata manager.
        """
        return TessFfiLightCurveMetadataManager()

    @patch.object(module, "dataset_split_from_uuid")
    @patch.object(module, "metadatabase_uuid")
    @patch.object(module.TessFfiLightCurve, "get_magnitude_from_file")
    @patch.object(module.TessFfiLightCurveMetadata, "insert_many")
    def test_can_insert_multiple_sql_database_rows_from_paths(
        self,
        mock_insert_many,
        mock_get_magnitude_from_file,
        mock_metadatabase_uuid,
        mock_dataset_split_generator,
        metadata_manger,
    ):
        light_curve_path0 = Path(
            "tesslcs_sector_1_104/tesslcs_tmag_7_8/tesslc_1111.pkl"
        )
        light_curve_path1 = Path(
            "tesslcs_sector_12_104/tesslcs_tmag_14_15/tesslc_1234567.pkl"
        )
        light_curve_paths = [light_curve_path0, light_curve_path1]
        light_curve_paths = list(
            map(
                metadata_manger.light_curve_root_directory_path.joinpath,
                light_curve_paths,
            )
        )
        mock_get_magnitude_from_file.side_effect = [4.5, 5.5]
        mock_dataset_split_generator.side_effect = [2, 3]
        mock_metadatabase_uuid.side_effect = ["uuid0", "uuid1"]
        metadata_manger.insert_multiple_rows_from_paths_into_database(
            light_curve_paths=light_curve_paths
        )
        expected_insert = [
            {
                "path": str(light_curve_path0),
                "tic_id": 1111,
                "sector": 1,
                "dataset_split": 2,
                "magnitude": 7,
            },
            {
                "path": str(light_curve_path1),
                "tic_id": 1234567,
                "sector": 12,
                "dataset_split": 3,
                "magnitude": 14,
            },
        ]
        assert mock_insert_many.call_args[0][0] == expected_insert

    @patch.object(Path, "glob")
    def test_can_populate_sql_dataset(self, mock_glob, metadata_manger):
        path_list = [
            metadata_manger.light_curve_root_directory_path.joinpath(f"{index}.fits")
            for index in range(20)
        ]
        x = False

        def mock_glob_side_effect(_path):
            nonlocal x
            if not x:
                x = True
                return (path for path in path_list)
            return (path for path in [])

        mock_glob.side_effect = mock_glob_side_effect
        mock_insert = Mock()
        metadata_manger.insert_multiple_rows_from_paths_into_database = mock_insert
        metadata_manger.populate_sql_database()
        assert mock_insert.call_args[0][0] == path_list
