"""Code for inference on the contents of a directory."""

import tensorflow as tf

from models import SimpleLightcurveCnn
from photometric_database.microlensing_label_per_time_step_database import MicrolensingLabelPerTimeStepDatabase

print('Preprocessing data...')
database = MicrolensingLabelPerTimeStepDatabase()
example_paths, inference_dataset = database.generate_inference_dataset('data/inference')

model = SimpleLightcurveCnn()
model.load_weights('logs/baseline/model.ckpt')

print('Inferring...')
dataset_tensor = tf.convert_to_tensor(inference_dataset)
predictions = model.predict(dataset_tensor)
for index in range(len(predictions)):
    print(f'{example_paths[index]}: {predictions[index]}')
