from qusi.light_curve_dataset import LightCurveDataset
from qusi.toy_light_curve_collection import toy_flat_light_curve_collection, toy_sine_wave_light_curve_collection
from qusi.train_session import TrainSession

light_curve_dataset = LightCurveDataset.new(standard_light_curve_collections=[toy_flat_light_curve_collection,
                                                                              toy_sine_wave_light_curve_collection])
train_session = TrainSession.new(train_datasets=[light_curve_dataset], validation_datasets=[light_curve_dataset],
                                 batch_size=103, train_steps_per_epoch=500, validation_steps_per_epoch=500)
train_session.run()
