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

def geraNp (pasta):
    arquivos_csv = [arquivo for arquivo in os.listdir(pasta)]

    exemplo_arquivo = pd.read_csv(os.path.join(pasta, arquivos_csv[0]))
    x = len(arquivos_csv)
    y, z = exemplo_arquivo.shape

    dados = np.empty((x, y, z))

    for i, arquivo in enumerate(arquivos_csv):
        caminho_arquivo = os.path.join(pasta, arquivo)
        dados[i] = pd.read_csv(caminho_arquivo).to_numpy()
    
    return dados

quedas = geraNp("../dataset/keypoints/not_quedas/inverted/normalized/full/test")
non_quedas = geraNp("../dataset/keypoints/quedas/inverted/normalized/full/test")

dados = np.concatenate((quedas, non_quedas), axis=0)
rotulos = np.array([0] * len(quedas) + [1] * len(non_quedas))

all_indices = list(range(dados.shape[0]))

train_indices, test_indices = train_test_split(all_indices, test_size=0.2)
X_train = dados[train_indices]
y_train = rotulos[train_indices]
X_test = dados[test_indices]
y_test = rotulos[test_indices]


# Create a simple model containing the encoder.
input = keras.Input(shape=(44, 132))
position_embeddings = SinePositionEncoding()(input)


# position_embeddings = PositionEmbedding(sequence_length=44)(input)
input_position = input + position_embeddings


# x = TransformerEncoder(intermediate_dim=64, num_heads=8)(input_position)
x = FNetEncoder(intermediate_dim=64)(input_position)
# x = MLPMixerLayer(44, intermediate_dim=132, dropout_rate=0)(input_position)


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


early_stopping = EarlyStopping(monitor='val_loss',  
                               patience=30,  
                               restore_best_weights=True)

history = model.fit(X_train, y_train, epochs=1000, batch_size=32, validation_split=0.2, callbacks=[early_stopping])

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


# Make predictions on the test set
y_pred = model.predict(X_test)

# Convert probabilities to binary predictions
y_pred_binary = (y_pred > 0.5).astype(int)

'''

'''
# Print classification report
print("Classification Report:")
print(classification_report(y_test, y_pred_binary))

# Print confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred_binary)
print("Confusion Matrix:")
print(conf_matrix)

# Plot ROC curve
fpr, tpr, _ = roc_curve(y_test, y_pred)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(6, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = {:.2f})'.format(roc_auc))
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc='lower right')
plt.show()

# Print number of parameters in the saved model
num_params = model.count_params()
print(f'Number of parameters in the saved model: {num_params}')

model.save('../models/trained_model.keras')

