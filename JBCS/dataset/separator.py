#objetivo do código, separar pontos chave nos conjuntos de treino, teste e validação.

import os
import random

def separar_arquivos(diretorio_origem, proporcoes=(0.6, 0.2, 0.2)):
    """
    Separa os arquivos do diretório de origem em três pastas: treino, teste e validação.
    :param diretorio_origem: Caminho da pasta onde os arquivos estão localizados.
    :param proporcoes: Tuple com a proporção para treino, teste e validação, respectivamente.
    """
    # Criar diretórios de destino
    treino_dir = os.path.join(diretorio_origem, "treino")
    teste_dir = os.path.join(diretorio_origem, "teste")
    validacao_dir = os.path.join(diretorio_origem, "validacao")
    os.makedirs(treino_dir, exist_ok=True)
    os.makedirs(teste_dir, exist_ok=True)
    os.makedirs(validacao_dir, exist_ok=True)
    
    # Obter lista de arquivos
    arquivos = [f for f in os.listdir(diretorio_origem) if os.path.isfile(os.path.join(diretorio_origem, f))]
    random.shuffle(arquivos)  # Embaralha os arquivos para distribuição aleatória
    
    # Definir quantidades
    total_arquivos = len(arquivos)
    n_treino = int(total_arquivos * proporcoes[0])
    n_teste = int(total_arquivos * proporcoes[1])
    
    # Distribuir arquivos
    for i, arquivo in enumerate(arquivos):
        origem = os.path.join(diretorio_origem, arquivo)
        if i < n_treino:
            destino = os.path.join(treino_dir, arquivo)
        elif i < n_treino + n_teste:
            destino = os.path.join(teste_dir, arquivo)
        else:
            destino = os.path.join(validacao_dir, arquivo)
        os.rename(origem, destino)  # Move o arquivo renomeando seu caminho
    
    print(f"Arquivos distribuídos: {n_treino} treino, {n_teste} teste, {total_arquivos - n_treino - n_teste} validação.")

# Exemplo de uso
separar_arquivos("pasta_teste")
