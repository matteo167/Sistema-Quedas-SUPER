import argparse

import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from keras import layers
from keras_nlp.layers import TransformerEncoder, SinePositionEncoding, FNetEncoder
from keras.callbacks import EarlyStopping
import tensorflow_addons as tfa


parser = argparse.ArgumentParser(description="Runs multiple Python subprograms with specific arguments.")
parser.add_argument("--inversion", choices=["nI", "I", "bothI"], required=True, help="Inversion type: nI, I, bothI")
parser.add_argument("--orientation", choices=["N", "W"], required=True, help="Orientation: N or W")
parser.add_argument("--extractor", choices=["F", "L", "H", "LF", "LH", "FH", "LFH"], required=True, help="Feature extractor")
parser.add_argument("--encoder", choices=["Reg", "Fnet", "Mlp"], required=True, help="Encoding type")

args = parser.parse_args()

# Defina uma semente aleatória para reproduzir os mesmos resultados
seed = 42
np.random.seed(seed)
tf.random.set_seed(seed)
tf.config.experimental.enable_op_determinism()


from tensorflow.keras.saving import register_keras_serializable

@register_keras_serializable()
class MLPMixerLayer(layers.Layer):
    def __init__(self, num_patches, intermediate_dim, dropout_rate, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.mlp1 = keras.Sequential(
            [
                layers.Dense(units=num_patches),
                tfa.layers.GELU(),
                layers.Dense(units=num_patches),
                layers.Dropout(rate=dropout_rate),
            ]
        )
        self.mlp2 = keras.Sequential(
            [
                layers.Dense(units=num_patches),
                tfa.layers.GELU(),
                layers.Dense(units=intermediate_dim),
                layers.Dropout(rate=dropout_rate),
            ]
        )
        self.normalize = layers.LayerNormalization(epsilon=1e-6)

    def call(self, inputs):
        # Apply layer normalization.
        x = self.normalize(inputs)
        # Transpose inputs from [num_batches, num_patches, hidden_units] to [num_batches, hidden_units, num_patches].
        x_channels = tf.linalg.matrix_transpose(x)
        # Apply mlp1 on each channel independently.
        mlp1_outputs = self.mlp1(x_channels)
        # Transpose mlp1_outputs from [num_batches, hidden_dim, num_patches] to [num_batches, num_patches, hidden_units].
        mlp1_outputs = tf.linalg.matrix_transpose(mlp1_outputs)
        # Add skip connection.
        x = mlp1_outputs + inputs
        # Apply layer normalization.
        x_patches = self.normalize(x)
        # Apply mlp2 on each patch independtenly.
        mlp2_outputs = self.mlp2(x_patches)
        # Add skip connection.
        x = x + mlp2_outputs
        return x

keras.utils.get_custom_objects()['MLPMixerLayer'] = MLPMixerLayer

def geraNp(pastas):
    # Recebe uma lista de pastas
    dados = []
    
    for pasta in pastas:
        arquivos_csv = [arquivo for arquivo in os.listdir(pasta) if arquivo.endswith('')]
        exemplo_arquivo = pd.read_csv(os.path.join(pasta, arquivos_csv[0]))
        x = len(arquivos_csv)
        y, z = exemplo_arquivo.shape

        # Cria uma lista para armazenar os dados de cada arquivo CSV na pasta
        pasta_dados = np.empty((x, y, z))

        for i, arquivo in enumerate(arquivos_csv):
            caminho_arquivo = os.path.join(pasta, arquivo)
            pasta_dados[i] = pd.read_csv(caminho_arquivo).to_numpy()
        
        # Concatenando os dados de todas as pastas
        dados.append(pasta_dados)
    
    # Concatenar os dados de todas as pastas
    return np.concatenate(dados, axis=0)


# Carregar os dados de treino e validação
pastas_non_quedas_treino = []
pastas_quedas_treino = []

pastas_non_quedas_val = []
pastas_quedas_val = []


for letra in str(args.extractor):
    if (letra == "L"):
        pastas_non_quedas_treino.append("../dataset/keypoints/not_quedas/@/$/lite/train")
        pastas_quedas_treino.append("../dataset/keypoints/quedas/@/$/lite/train")
        pastas_non_quedas_val.append("../dataset/keypoints/not_quedas/@/$/lite/val")
        pastas_quedas_val.append("../dataset/keypoints/quedas/@/$/lite/val")
    if (letra == "F"):
        pastas_non_quedas_treino.append("../dataset/keypoints/not_quedas/@/$/full/train")
        pastas_quedas_treino.append("../dataset/keypoints/quedas/@/$/full/train")
        pastas_non_quedas_val.append("../dataset/keypoints/not_quedas/@/$/full/val")
        pastas_quedas_val.append("../dataset/keypoints/quedas/@/$/full/val")
    if (letra == "H"):
        pastas_non_quedas_treino.append("../dataset/keypoints/not_quedas/@/$/heavy/train")
        pastas_quedas_treino.append("../dataset/keypoints/quedas/@/$/heavy/train")
        pastas_non_quedas_val.append("../dataset/keypoints/not_quedas/@/$/heavy/val")
        pastas_quedas_val.append("../dataset/keypoints/quedas/@/$/heavy/val")

if(args.inversion == "nI"):
    for i in range(len(pastas_non_quedas_treino)):
        pastas_non_quedas_treino[i] = pastas_non_quedas_treino[i].replace('@', 'not_inverted')
    for i in range(len(pastas_quedas_treino)):
        pastas_quedas_treino[i] = pastas_quedas_treino[i].replace('@', 'not_inverted')
    for i in range(len(pastas_non_quedas_val)):
        pastas_non_quedas_val[i] = pastas_non_quedas_val[i].replace('@', 'not_inverted')
    for i in range(len(pastas_quedas_val)):
        pastas_quedas_val[i] = pastas_quedas_val[i].replace('@', 'not_inverted')

if(args.inversion == "I"):
    for i in range(len(pastas_non_quedas_treino)):
        pastas_non_quedas_treino[i] = pastas_non_quedas_treino[i].replace('@', 'inverted')
    for i in range(len(pastas_quedas_treino)):
        pastas_quedas_treino[i] = pastas_quedas_treino[i].replace('@', 'inverted')
    for i in range(len(pastas_non_quedas_val)):
        pastas_non_quedas_val[i] = pastas_non_quedas_val[i].replace('@', 'inverted')
    for i in range(len(pastas_quedas_val)):
        pastas_quedas_val[i] = pastas_quedas_val[i].replace('@', 'inverted')

if(args.inversion == "bothI"):
    for pasta in pastas_non_quedas_treino[:]:
        pastas_non_quedas_treino.append(pasta.replace('@', 'not_inverted'))

    for i in range(len(pastas_non_quedas_treino)):
        pastas_non_quedas_treino[i] = pastas_non_quedas_treino[i].replace('@', 'inverted')
    

if(args.orientation == "N"):
    for i in range(len(pastas_non_quedas_treino)):
        pastas_non_quedas_treino[i] = pastas_non_quedas_treino[i].replace('$', 'normalized')
    for i in range(len(pastas_quedas_treino)):
        pastas_quedas_treino[i] = pastas_quedas_treino[i].replace('$', 'normalized')
    for i in range(len(pastas_non_quedas_val)):
        pastas_non_quedas_val[i] = pastas_non_quedas_val[i].replace('$', 'normalized')
    for i in range(len(pastas_quedas_val)):
        pastas_quedas_val[i] = pastas_quedas_val[i].replace('$', 'normalized')
if(args.orientation == "W"):
    for i in range(len(pastas_non_quedas_treino)):
        pastas_non_quedas_treino[i] = pastas_non_quedas_treino[i].replace('$', 'world')
    for i in range(len(pastas_quedas_treino)):
        pastas_quedas_treino[i] = pastas_quedas_treino[i].replace('$', 'world')
    for i in range(len(pastas_non_quedas_val)):
        pastas_non_quedas_val[i] = pastas_non_quedas_val[i].replace('$', 'world')
    for i in range(len(pastas_quedas_val)):
        pastas_quedas_val[i] = pastas_quedas_val[i].replace('$', 'world')




train_quedas = geraNp(pastas_quedas_treino)
train_non_quedas = geraNp(pastas_non_quedas_treino)

val_quedas = geraNp(pastas_quedas_val)
val_non_quedas = geraNp(pastas_non_quedas_val)

# Concatenar os dados de treino e validação
train_dados = np.concatenate((train_quedas, train_non_quedas), axis=0)
val_dados = np.concatenate((val_quedas, val_non_quedas), axis=0)

# Criar os rótulos para cada conjunto
train_rotulos = np.array([0] * len(train_quedas) + [1] * len(train_non_quedas))
val_rotulos = np.array([0] * len(val_quedas) + [1] * len(val_non_quedas))

# Criar o modelo
input = keras.Input(shape=(44, 132))
position_embeddings = SinePositionEncoding()(input)
input_position = input + position_embeddings

if(args.encoder == "Reg"):
    x = TransformerEncoder(intermediate_dim=64, num_heads=8)(input_position)
if(args.encoder == "Fnet"):
    x = FNetEncoder(intermediate_dim=64)(input_position)
if(args.encoder == "Mlp"):
    x = MLPMixerLayer(44, intermediate_dim=132, dropout_rate=0)(input_position)

x = layers.Permute((2, 1))(x)
x = layers.GlobalAveragePooling1D()(x)
x = layers.Dense(22, activation="relu")(x)
output = layers.Dense(1, activation='sigmoid')(x)

model = keras.Model(inputs=input, outputs=output)

model.compile(
    optimizer='adam', 
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# Early stopping
early_stopping = EarlyStopping(monitor='val_loss',  
                               patience=30,  
                               restore_best_weights=True)

# Treinamento
history = model.fit(train_dados, train_rotulos, epochs=1000, batch_size=32, validation_data=(val_dados, val_rotulos), callbacks=[early_stopping])

# Salvar o modelo treinado
model.save("../models/" + args.inversion + "_" + args.orientation + "_" + args.extractor + "_" + args.encoder + ".keras")
