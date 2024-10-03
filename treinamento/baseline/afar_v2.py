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

# Define a random seed for reproducibility
seed = 42
np.random.seed(seed)
tf.random.set_seed(seed)

def geraNp (pasta):
    arquivos_csv = [arquivo for arquivo in os.listdir(pasta) if arquivo.endswith('.csv')]

    exemplo_arquivo = pd.read_csv(os.path.join(pasta, arquivos_csv[0]))
    x = len(arquivos_csv)
    y, z = exemplo_arquivo.shape

    dados = np.empty((x, y, z))

    for i, arquivo in enumerate(arquivos_csv):
        caminho_arquivo = os.path.join(pasta, arquivo)
        dados[i] = pd.read_csv(caminho_arquivo).to_numpy()
    
    return dados

quedas = geraNp("key_not_quedas_v2")
non_quedas = geraNp("key_quedas_v2")

dados = np.concatenate((quedas, non_quedas), axis=0)
rotulos = np.array([0] * len(quedas) + [1] * len(non_quedas))

all_indices = list(range(dados.shape[0]))
train_indices, test_indices = train_test_split(all_indices, test_size=0.2, random_state=seed)
X_train = dados[train_indices]
y_train = rotulos[train_indices]
X_test = dados[test_indices]
y_test = rotulos[test_indices]

# Define the model architecture
input_layer = keras.Input(shape=(44, 132))

# 1D Convolutional Layer 1
x = layers.Conv1D(filters=512, kernel_size=4, activation='relu')(input_layer)
x = layers.Dropout(0.2)(x)

# 1D Convolutional Layer 2
x = layers.Conv1D(filters=512, kernel_size=4, activation='relu')(x)
x = layers.Dropout(0.2)(x)

# Global Average Pooling Layer
x = layers.GlobalAveragePooling1D()(x)

# Fully Connected Layer 1
x = layers.Dense(512, activation='relu')(x)
x = layers.Dropout(0.2)(x)

# Fully Connected Layer 2
x = layers.Dense(256, activation='relu')(x)

# Output Layer
output_layer = layers.Dense(1, activation='sigmoid')(x)

model = keras.Model(inputs=input_layer, outputs=output_layer)

print('///////////////////////////')
model.summary()

# Compile the model
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-5),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# Define early stopping callback
early_stopping = EarlyStopping(monitor='val_loss', patience=30, restore_best_weights=True)

# Train the model
history = model.fit(X_train, y_train, epochs=100, batch_size=64, validation_split=0.2, callbacks=[early_stopping])











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

model.save('trained_model.keras')