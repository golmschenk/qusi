from qusi.light_curve_dataset import LightCurveDataset
from qusi.train_session import TrainSession
from ramjet.photometric_database.derived.toy_light_curve_collection import ToyFlatLightCurveCollection, \
    ToySineWaveLightCurveCollection

toy_flat_light_curve_collection = ToyFlatLightCurveCollection()
toy_sine_wave_light_curve_collection = ToySineWaveLightCurveCollection()
light_curve_dataset = LightCurveDataset.new(standard_light_curve_collections=[toy_flat_light_curve_collection,
                                                                              toy_sine_wave_light_curve_collection])
train_run = TrainSession(train_datasets=[light_curve_dataset], validation_datasets=[light_curve_dataset])
train_run.run()
