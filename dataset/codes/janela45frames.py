import os
import pandas as pd

# Caminho para a pasta com os arquivos CSV
pasta = 'key_not_quedas_v2'

# Lista de arquivos CSV na pasta
arquivos_csv = [arquivo for arquivo in os.listdir(pasta) if arquivo.endswith('.csv')]

# Função para manter apenas as 45 linhas do meio de um arquivo CSV
def manter_linhas_do_meio(caminho_arquivo):
    df = pd.read_csv(caminho_arquivo)
    num_linhas = len(df)
    
    # Verifica se há pelo menos 45 linhas no arquivo
    if num_linhas > 45:
        inicio = (num_linhas - 45) // 2
        fim = inicio + 45
        df = df.iloc[inicio:fim]
    
    # Salva as 45 linhas do meio de volta no arquivo CSV
    df.to_csv(caminho_arquivo, index=False)

# Itera sobre os arquivos CSV e mantém apenas as 45 linhas do meio
for arquivo in arquivos_csv:
    caminho_arquivo = os.path.join(pasta, arquivo)
    manter_linhas_do_meio(caminho_arquivo)
    print(f"Arquivo {arquivo}: Mantidas apenas as 45 linhas do meio.")

print("Processo concluído.")