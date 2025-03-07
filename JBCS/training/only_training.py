import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from keras import layers
from keras_nlp.layers import SinePositionEncoding, FNetEncoder
from keras.callbacks import EarlyStopping
import tensorflow_addons as tfa

# Defina uma semente aleatória para reproduzir os mesmos resultados
seed = 42
np.random.seed(seed)
tf.random.set_seed(seed)
tf.config.experimental.enable_op_determinism()

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
pastas_quedas_treino = ["../dataset/keypoints/not_quedas/inverted/normalized/full/train", 
                 "../dataset/keypoints/not_quedas/not_inverted/normalized/full/train"]
pastas_non_quedas_treino = ["../dataset/keypoints/quedas/inverted/normalized/full/train", 
                     "../dataset/keypoints/quedas/not_inverted/normalized/full/train"]

train_quedas = geraNp(pastas_quedas_treino)
train_non_quedas = geraNp(pastas_non_quedas_treino)

pastas_quedas_val = ["../dataset/keypoints/not_quedas/inverted/normalized/full/val", 
                     "../dataset/keypoints/not_quedas/not_inverted/normalized/full/val"]
pastas_non_quedas_val = ["../dataset/keypoints/quedas/inverted/normalized/full/val", 
                         "../dataset/keypoints/quedas/not_inverted/normalized/full/val"]

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

x = FNetEncoder(intermediate_dim=64)(input_position)
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
model.save('../models/trained_model.keras')
