import os
import pandas as pd

# Caminho para a pasta com os arquivos CSV
pasta = 'key_not_quedas_v2'

# Lista de arquivos CSV na pasta
arquivos_csv = [arquivo for arquivo in os.listdir(pasta) if arquivo.endswith('.csv')]

# Função para verificar e remover a última linha de um arquivo CSV
def remover_ultima_linha(caminho_arquivo):
    df = pd.read_csv(caminho_arquivo)
    num_linhas = len(df)
    
    # Verifica se o arquivo tem 46 linhas
    if num_linhas == 45:
        df = df.iloc[:-1]  # Remove a última linha
    
        # Salva as alterações no arquivo CSV
        df.to_csv(caminho_arquivo, index=False)

# Itera sobre os arquivos CSV e verifica/remove a última linha, se necessário
for arquivo in arquivos_csv:
    caminho_arquivo = os.path.join(pasta, arquivo)
    remover_ultima_linha(caminho_arquivo)
    print(f"Verificado e, se necessário, removida a última linha do arquivo {arquivo}.")

print("Processo concluído.")