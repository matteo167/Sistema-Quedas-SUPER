import tensorflow as tf
from tensorflow import keras
import numpy as np
import csv
from keras_nlp.layers import TransformerEncoder, SinePositionEncoding, PositionEmbedding

path_to_model = '../models/trained_model_keras_nlp.keras'
path_to_dataset = '../dataset/key_quedas_v2/S001C001P001R002A043_rgb.csv'

loaded_model = keras.models.load_model(path_to_model)
loaded_model.input.set_shape((1,) + loaded_model.input.shape[1:])

def representative_dataset_generator():
    with open(path_to_dataset, newline='') as csvfile:
        reader = csv.reader(csvfile)
        batch = []
        for row in reader:
            data = np.array(row, dtype=np.float32)
            batch.append(data)
            if len(batch) == 44:  
                batch = np.array(batch).reshape(1, 44, 132)  
                yield [batch]
                batch = []  

converter = tf.lite.TFLiteConverter.from_keras_model(loaded_model)
converter.optimizations = [tf.lite.Optimize.DEFAULT] # quantize all fixed parameters
converter.representative_dataset = representative_dataset_generator
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS,
  tf.lite.OpsSet.SELECT_TF_OPS] # operations supported at conversion
converter.inference_input_type = tf.float32
converter.inference_output_type = tf.float32
tflite_model = converter.convert()



# Save the model.
with open('../models/model.tflite', 'wb') as f:
  f.write(tflite_model)