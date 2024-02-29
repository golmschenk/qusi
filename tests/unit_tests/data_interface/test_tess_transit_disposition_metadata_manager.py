from unittest.mock import PropertyMock, patch

import pandas as pd

import ramjet.data_interface.tess_transit_metadata_manager as module
from ramjet.data_interface.tess_toi_data_interface import ToiColumns
from ramjet.data_interface.tess_transit_metadata_manager import (
    Disposition,
    TessTransitMetadataManager,
)


class TestTessTransitMetadata:
    @patch.object(module, "TessTransitMetadata")
    def test_table_building_creates_rows_based_on_toi_dispositions(
        self,
        mock_tess_target_transit_disposition,
    ):
        tess_transit_disposition_metadata_manager = TessTransitMetadataManager()
        toi_dispositions = pd.DataFrame(
            {
                ToiColumns.tic_id.value: [1, 2, 3],
                ToiColumns.disposition.value: ["KP", "", "FP"],
            }
        )
        ctoi_dispositions = pd.DataFrame(
            {ToiColumns.tic_id.value: [], ToiColumns.disposition.value: []}
        )
        with (
            patch.object(
                module.TessToiDataInterface,
                "toi_dispositions",
                new_callable=PropertyMock,
            ) as mock_toi_dispositions,
            patch.object(
                module.TessToiDataInterface,
                "ctoi_dispositions",
                new_callable=PropertyMock,
            ) as mock_ctoi_dispositions,
        ):
            mock_toi_dispositions.return_value = toi_dispositions
            mock_ctoi_dispositions.return_value = ctoi_dispositions
            tess_transit_disposition_metadata_manager.build_table()
        call_args_list = mock_tess_target_transit_disposition.call_args_list
        assert len(call_args_list) == 3
        assert call_args_list[0][1] == {
            "tic_id": 1,
            "disposition": Disposition.CONFIRMED.value,
        }
        assert call_args_list[1][1] == {
            "tic_id": 2,
            "disposition": Disposition.CANDIDATE.value,
        }
        assert call_args_list[2][1] == {
            "tic_id": 3,
            "disposition": Disposition.FALSE_POSITIVE.value,
        }
