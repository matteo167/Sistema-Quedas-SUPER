import os
import pandas as pd

# Caminho para a pasta base com os arquivos CSV
pasta_base = 'keypoints_processados'

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

# Percorre todas as pastas e subpastas
for root, _, files in os.walk(pasta_base):
    for file in files:
        if file.endswith('.csv'):
            caminho_arquivo = os.path.join(root, file)
            manter_linhas_do_meio(caminho_arquivo)
            # print(f"Arquivo {caminho_arquivo}: Mantidas apenas as 45 linhas do meio.")

print("Processo concluído.")

