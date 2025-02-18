import os
import pandas as pd

# Caminho para a pasta principal
pasta_principal = 'keypoints_processados'

# Lista para armazenar informações sobre os arquivos CSV
arquivos_info = []

# Percorre todas as pastas e subpastas
for root, _, files in os.walk(pasta_principal):
    for arquivo in files:
        if arquivo.endswith('.csv'):
            # Caminho completo para o arquivo
            caminho_arquivo = os.path.join(root, arquivo)
            
            # Lê o arquivo CSV e conta as linhas
            df = pd.read_csv(caminho_arquivo)
            num_linhas = len(df)
            
            # Armazena informações sobre o arquivo
            arquivos_info.append((caminho_arquivo, num_linhas))

# Ordena a lista de arquivos pelo número de linhas
arquivos_info.sort(key=lambda x: x[1])

# Pega os 50 arquivos com menor número de linhas
arquivos_menor_num_linhas = arquivos_info[:50]

# Pega os 100 arquivos com maior número de linhas
arquivos_maior_num_linhas = arquivos_info[-100:]

# Exclui os arquivos com menor número de linhas
for caminho_arquivo, _ in arquivos_menor_num_linhas:
    os.remove(caminho_arquivo)
    print(f"Arquivo {caminho_arquivo} com menor número de linhas foi excluído.")

# Exclui os arquivos com maior número de linhas
for caminho_arquivo, _ in arquivos_maior_num_linhas:
    os.remove(caminho_arquivo)
    print(f"Arquivo {caminho_arquivo} com maior número de linhas foi excluído.")

