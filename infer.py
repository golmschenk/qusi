"""Code for inference on the contents of a directory."""

import tensorflow as tf

from lightcurve_database import LightcurveDatabase
from models import SimpleLightcurveCnn

tf.keras.backend.set_learning_phase(True)

print('Preprocessing data...')
database = LightcurveDatabase()
example_paths, inference_dataset = database.generate_inference_dataset('data/inference')

model = SimpleLightcurveCnn()
model.load_weights('logs/baseline/model.ckpt')

print('Inferring...')
dataset_tensor = tf.convert_to_tensor(inference_dataset)
predictions = model.predict(dataset_tensor)
for index in range(len(predictions)):
    print(f'{example_paths[index]}: {predictions[index]}')
