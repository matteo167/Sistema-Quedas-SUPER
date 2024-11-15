import os
import pandas as pd
import tensorflow as tf
from tensorflow import keras
import numpy as np
from keras_nlp.layers import TransformerEncoder, SinePositionEncoding, PositionEmbedding


path_to_model = '../models/trained_model_FNetEncoder.keras'

loaded_model = keras.models.load_model(path_to_model)
loaded_model.input.set_shape((1,) + loaded_model.input.shape[1:])

def representative_dataset_generator(*csv_directories):
    for directory in csv_directories:
        print(directory)
        # Encontra todos os arquivos CSV no diretório
        csv_files = [f for f in os.listdir(directory) if f.endswith('.csv')]
        print(len(csv_files))
        print()
        
        for csv_file in csv_files:
            file_path = os.path.join(directory, csv_file)
            
            # Lê o arquivo CSV usando pandas
            df = pd.read_csv(file_path)
            
            # Converte o DataFrame para um array NumPy de float32
            data_array = df.to_numpy(dtype=np.float32)
            
            # Verifica se o número de linhas é 44
            if data_array.shape[0] == 44:
                # Reshape para (1, 44, N)
                # Data_array is a 3D array now
                batch = data_array.reshape(1, 44, -1)
                
                # Verifica e ajusta o número de colunas para 132
                if batch.shape[2] != 132:
                    batch = np.pad(batch, ((0,0), (0,0), (0, 132 - batch.shape[2])), 
                                   mode='constant', constant_values=0)
                
                yield [batch]
            else:
                print(f"Warning: File {csv_file} has not 44 lines. Skipping.")

converter = tf.lite.TFLiteConverter.from_keras_model(loaded_model)

converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.target_spec.supported_ops = [tf.lite.OpsSet.EXPERIMENTAL_TFLITE_BUILTINS_ACTIVATIONS_INT16_WEIGHTS_INT8,
                                       tf.lite.OpsSet.TFLITE_BUILTINS,
                                       tf.lite.OpsSet.SELECT_TF_OPS]
converter.representative_dataset = lambda: representative_dataset_generator('../dataset/key_not_quedas_v2', '../dataset/key_quedas_v2')



converter.inference_input_type = tf.float32
converter.inference_output_type = tf.float32
tflite_model = converter.convert()

tf.lite.experimental.Analyzer.analyze(model_content=tflite_model, gpu_compatibility=True)

# Save the model.
with open('../models/model1.tflite', 'wb') as f:
  f.write(tflite_model)
