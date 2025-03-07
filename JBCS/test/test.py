import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
import matplotlib.pyplot as plt


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


# Carregar os dados de teste
pastas_quedas_test = ["../dataset/keypoints/not_quedas/inverted/normalized/full/test", 
                      "../dataset/keypoints/not_quedas/not_inverted/normalized/full/test"]
pastas_non_quedas_test = ["../dataset/keypoints/quedas/inverted/normalized/full/test", 
                          "../dataset/keypoints/quedas/not_inverted/normalized/full/test"]

test_quedas = geraNp(pastas_quedas_test)
test_non_quedas = geraNp(pastas_non_quedas_test)

# Concatenar os dados de teste
test_dados = np.concatenate((test_quedas, test_non_quedas), axis=0)

# Criar os rótulos para o conjunto de teste
test_rotulos = np.array([0] * len(test_quedas) + [1] * len(test_non_quedas))

# Carregar o modelo treinado
model = keras.models.load_model('../models/trained_model.keras')

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
