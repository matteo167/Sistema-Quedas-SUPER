import os
import pandas as pd

# Caminho para a pasta principal
pasta_principal = 'keypoints_processados'

# Contadores para arquivos removidos
arquivos_grandes_removidos = 0
arquivos_pequenos_removidos = 0

# Percorre todas as pastas e subpastas
for root, _, files in os.walk(pasta_principal):
    for arquivo in files:
        if arquivo.endswith('.csv'):
            # Caminho completo para o arquivo
            caminho_arquivo = os.path.join(root, arquivo)
            
            try:
                # Lê o arquivo CSV e conta as linhas
                df = pd.read_csv(caminho_arquivo)
                num_linhas = len(df)
                
                # Verifica e remove arquivos conforme a condição
                if num_linhas > 200:
                    os.remove(caminho_arquivo)
                    arquivos_grandes_removidos += 1
                    print(f"Arquivo removido (muito grande): {caminho_arquivo} ({num_linhas} linhas)")
                elif num_linhas < 45:
                    os.remove(caminho_arquivo)
                    arquivos_pequenos_removidos += 1
                    print(f"Arquivo removido (muito pequeno): {caminho_arquivo} ({num_linhas} linhas)")
            except Exception as e:
                print(f"Erro ao processar {caminho_arquivo}: {e}")

# Exibir total de arquivos removidos
print(f"Total de arquivos muito grandes removidos: {arquivos_grandes_removidos}")
print(f"Total de arquivos muito pequenos removidos: {arquivos_pequenos_removidos}")

