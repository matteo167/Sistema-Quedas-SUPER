import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from keras import layers
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc, accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import sys


from keras_nlp.layers import TransformerEncoder, SinePositionEncoding, PositionEmbedding


# Defina uma semente aleatória para reproduzir os mesmos resultados
seed = 42
np.random.seed(seed)
tf.random.set_seed(seed)

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



if len(sys.argv) < 3:
    print("Uso: python script.py <nome_do_modelo> <pasta_de_teste>")
    sys.exit(1)

model_name = sys.argv[1]
test_folder = sys.argv[2]

# Carregar os dados de teste
pastas_quedas_test = ["../dataset/keypoints/not_quedas/inverted/" + test_folder, 
                      "../dataset/keypoints/not_quedas/not_inverted/" + test_folder]
pastas_non_quedas_test = ["../dataset/keypoints/quedas/inverted/" + test_folder, 
                          "../dataset/keypoints/quedas/not_inverted/" + test_folder]

test_quedas = geraNp(pastas_quedas_test)
test_non_quedas = geraNp(pastas_non_quedas_test)

# Concatenar os dados de teste
test_dados = np.concatenate((test_quedas, test_non_quedas), axis=0)

# Criar os rótulos para o conjunto de teste
test_rotulos = np.array([1] * len(test_quedas) + [0] * len(test_non_quedas))





from tensorflow.keras.saving import register_keras_serializable
import tensorflow_addons as tfa


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
    
# Carregar o modelo treinado
model = keras.models.load_model("../models/" + model_name + ".keras", custom_objects={'MLPMixerLayer': MLPMixerLayer})

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



# Predições no conjunto de teste
y_pred = model.predict(test_dados)

# Converter probabilidades para previsões binárias
y_pred_binary = (y_pred > 0.5).astype(int)

# Cálculo das métricas corretas
accuracy = accuracy_score(test_rotulos, y_pred_binary)
precision = precision_score(test_rotulos, y_pred_binary)
recall = recall_score(test_rotulos, y_pred_binary)
f1 = f1_score(test_rotulos, y_pred_binary)


result_file = f'../results/resultados.txt'

with open(result_file, 'a') as f:
    f.write(f'\nModelo: {model_name}\n')
    f.write(f'Conj. teste: {test_folder}\n')
    f.write(f'Precisão: {precision:.4f}\n')
    f.write(f'Revocação: {recall:.4f}\n')
    f.write(f'Acurácia: {accuracy:.4f}\n')
    f.write(f'F1-Score: {f1:.4f}\n')
    f.write(f'Nº de parâmetros: {num_params}\n')
    f.write('-' * 40 + '\n')
