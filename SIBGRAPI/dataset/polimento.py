import os
import pandas as pd

# Caminho para a pasta principal
pasta_principal = 'keypoints_processados'

# Função para verificar e remover a última linha de um arquivo CSV
def remover_ultima_linha(caminho_arquivo):
    df = pd.read_csv(caminho_arquivo)
    num_linhas = len(df)

    # Verifica se o arquivo tem 45 linhas
    if num_linhas == 45:
        df = df.iloc[:-1]  # Remove a última linha
        
        # Salva as alterações no arquivo CSV
        df.to_csv(caminho_arquivo, index=False)
        # print(f"Arquivo {caminho_arquivo}: Última linha removida.")

# Percorre todas as pastas e subpastas
for root, _, files in os.walk(pasta_principal):
    for arquivo in files:
        if arquivo.endswith('.csv'):
            caminho_arquivo = os.path.join(root, arquivo)
            remover_ultima_linha(caminho_arquivo)

print("Processo concluído.")

