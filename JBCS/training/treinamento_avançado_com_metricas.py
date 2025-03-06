import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from keras import layers
from keras_nlp.layers import TransformerEncoder, SinePositionEncoding, FNetEncoder
from keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
import tensorflow_addons as tfa
import matplotlib.pyplot as plt

# Defina uma semente aleatória para reproduzir os mesmos resultados
seed = 42
np.random.seed(seed)
tf.random.set_seed(seed)
tf.config.experimental.enable_op_determinism()

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


# Carregar os dados das diferentes pastas
pastas_quedas_treino = ["../dataset/keypoints/not_quedas/inverted/normalized/full/train", 
                 "../dataset/keypoints/not_quedas/not_inverted/normalized/full/train"]
pastas_non_quedas_treino = ["../dataset/keypoints/quedas/inverted/normalized/full/train", 
                     "../dataset/keypoints/quedas/not_inverted/normalized/full/train"]

train_quedas = geraNp(pastas_quedas_treino)
train_non_quedas = geraNp(pastas_non_quedas_treino)

# Carregar os dados de validação e teste da mesma forma
pastas_quedas_val = ["../dataset/keypoints/not_quedas/inverted/normalized/full/val", 
                     "../dataset/keypoints/not_quedas/not_inverted/normalized/full/val"]
pastas_non_quedas_val = ["../dataset/keypoints/quedas/inverted/normalized/full/val", 
                         "../dataset/keypoints/quedas/not_inverted/normalized/full/val"]

val_quedas = geraNp(pastas_quedas_val)
val_non_quedas = geraNp(pastas_non_quedas_val)

pastas_quedas_test = ["../dataset/keypoints/not_quedas/inverted/normalized/full/test", 
                      "../dataset/keypoints/not_quedas/not_inverted/normalized/full/test"]
pastas_non_quedas_test = ["../dataset/keypoints/quedas/inverted/normalized/full/test", 
                          "../dataset/keypoints/quedas/not_inverted/normalized/full/test"]

test_quedas = geraNp(pastas_quedas_test)
test_non_quedas = geraNp(pastas_non_quedas_test)

# Concatenar os dados de cada conjunto (treinamento, validação e teste)
train_dados = np.concatenate((train_quedas, train_non_quedas), axis=0)
val_dados = np.concatenate((val_quedas, val_non_quedas), axis=0)
test_dados = np.concatenate((test_quedas, test_non_quedas), axis=0)

# Criar os rótulos para cada conjunto
train_rotulos = np.array([0] * len(train_quedas) + [1] * len(train_non_quedas))
val_rotulos = np.array([0] * len(val_quedas) + [1] * len(val_non_quedas))
test_rotulos = np.array([0] * len(test_quedas) + [1] * len(test_non_quedas))

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

print('///////////////////////////')
model.summary()

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

# Plot training and validation accuracy
plt.figure(figsize=(12, 4))
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

# Plot training and validation loss
plt.figure(figsize=(12, 4))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()

# Predições no conjunto de teste
y_pred = model.predict(test_dados)

# Converter probabilidades para previsões binárias
y_pred_binary = (y_pred > 0.5).astype(int)

# Relatório de classificação
print("Classification Report:")
print(classification_report(test_rotulos, y_pred_binary))

# Matriz de confusão
conf_matrix = confusion_matrix(test_rotulos, y_pred_binary)
print("Confusion Matrix:")
print(conf_matrix)

# Plot da curva ROC
fpr, tpr, _ = roc_curve(test_rotulos, y_pred)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(6, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = {:.2f})'.format(roc_auc))
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc='lower right')
plt.show()

# Número de parâmetros no modelo
num_params = model.count_params()
print(f'Number of parameters in the saved model: {num_params}')

# Salvar o modelo treinado
model.save('../models/trained_model.keras')
