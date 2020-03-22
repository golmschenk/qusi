"""Tests for deprecated package/module locations."""

import pytest


class TestPackageImportDeprecations:
    """Tests for deprecated package/module locations."""

    def test_tess_data_interface_from_photometric_database_subpackage_gives_deprecation_warning(self):
        with pytest.deprecated_call():
            from ramjet.photometric_database import tess_data_interface
        with pytest.deprecated_call():
            import ramjet.photometric_database.tess_data_interface
