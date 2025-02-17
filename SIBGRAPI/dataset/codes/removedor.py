import os
import pandas as pd

# Caminho para a pasta com os arquivos CSV
pasta = 'key_not_quedas_v2'

# Lista para armazenar informações sobre os arquivos CSV
arquivos_info = []

# Percorre todos os arquivos na pasta
for arquivo in os.listdir(pasta):
    if arquivo.endswith('.csv'):
        # Caminho completo para o arquivo
        caminho_arquivo = os.path.join(pasta, arquivo)
        
        # Lê o arquivo CSV e conta as linhas
        df = pd.read_csv(caminho_arquivo)
        num_linhas = len(df)
        
        # Armazena informações sobre o arquivo
        arquivos_info.append((arquivo, num_linhas))

# Ordena a lista de arquivos pelo número de linhas
arquivos_info.sort(key=lambda x: x[1])

# Pega os 50 arquivos com menor número de linhas
arquivos_menor_num_linhas = arquivos_info[:50]

# Pega os 100 arquivos com maior número de linhas
arquivos_maior_num_linhas = arquivos_info[-100:]

# Exclui os arquivos com menor número de linhas
for arquivo, _ in arquivos_menor_num_linhas:
    caminho_arquivo = os.path.join(pasta, arquivo)
    os.remove(caminho_arquivo)
    print(f"Arquivo {arquivo} com menor número de linhas foi excluído.")

# Exclui os arquivos com maior número de linhas
for arquivo, _ in arquivos_maior_num_linhas:
    caminho_arquivo = os.path.join(pasta, arquivo)
    os.remove(caminho_arquivo)
    print(f"Arquivo {arquivo} com maior número de linhas foi excluído.")