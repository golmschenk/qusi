import pytest

from ramjet.analysis.convert_column_name_to_display_name import (
    convert_column_name_to_display_name,
)


@pytest.mark.parametrize(
    ("column_name", "expected_display_name"),
    [
        ("simple_name", "Simple name"),
        ("name_with__units", "Name with (units)"),
        ("time__btjd", "Time (BTJD)"),
        ("sap", "SAP"),
        ("relative_sap_flux", "Relative SAP flux"),
    ],
)
def test_converting_column_name_to_display_name(column_name, expected_display_name):
    display_name = convert_column_name_to_display_name(column_name)
    assert display_name == expected_display_name
