import pytest

from ramjet.photometric_database.tess_light_curve import MissingTicRow, TessLightCurve


class TestTessLightCurve:
    @pytest.mark.slow
    @pytest.mark.external
    def test_setting_tic_rows_from_mast_for_list(self):
        light_curve0 = TessLightCurve()
        light_curve0.tic_id = 266980320
        light_curve1 = TessLightCurve()
        light_curve1.tic_id = 231663901
        assert light_curve0._tic_row is None  # noqa SLF001
        assert light_curve1._tic_row is None  # noqa SLF001
        TessLightCurve.load_tic_rows_from_mast_for_list([light_curve0, light_curve1])
        assert light_curve0._tic_row is not None  # noqa SLF001
        assert light_curve1._tic_row is not None  # noqa SLF001
        assert float(light_curve0._tic_row["Tmag"]) == pytest.approx(9.179, rel=1e-3)  # noqa SLF001

    @pytest.mark.slow
    @pytest.mark.external
    def test_setting_tic_rows_from_mast_for_list_notes_missing_row_for_tic_ids_not_in_tic(
        self,
    ):
        light_curve0 = TessLightCurve()
        light_curve0.tic_id = 99999999999999999
        assert light_curve0._tic_row is None  # noqa SLF001
        TessLightCurve.load_tic_rows_from_mast_for_list([light_curve0])
        assert light_curve0._tic_row is MissingTicRow  # noqa SLF001
