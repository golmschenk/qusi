from ramjet.photometric_database.standard_and_injected_lightcurve_database import StandardAndInjectedLightcurveDatabase


def test_database_has_lightcurve_collection_properties():
    database = StandardAndInjectedLightcurveDatabase()
    assert hasattr(database, 'standard_lightcurve_collections')
    assert hasattr(database, 'injectee_lightcurve_collection')
    assert hasattr(database, 'injectable_lightcurve_collections')
