import os
import shutil
import random
import sys

def split_dataset(directory, train_ratio=0.6, test_ratio=0.2, val_ratio=0.2, seed=42):
    assert train_ratio + test_ratio + val_ratio == 1, "As proporções devem somar 1"
    
    # Configurar a semente aleatória para garantir a mesma ordem de embaralhamento
    random.seed(seed)
    
    # Criando diretórios de saída dentro do diretório de origem
    train_dir = os.path.join(directory, 'train')
    test_dir = os.path.join(directory, 'test')
    val_dir = os.path.join(directory, 'val')
    
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)
    
    # Pegando todos os arquivos do diretório de origem, excluindo as pastas criadas
    files = [f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))]
    random.shuffle(files)  # Embaralha os arquivos
    
    # Definindo quantidades para cada conjunto
    total_files = len(files)
    train_count = int(total_files * train_ratio)
    test_count = int(total_files * test_ratio)
    val_count = total_files - train_count - test_count  # Garantir que todos os arquivos sejam utilizados
    
    # Separando os arquivos
    train_files = files[:train_count]
    test_files = files[train_count:train_count + test_count]
    val_files = files[train_count + test_count:]
    
    # Movendo arquivos para as pastas correspondentes
    for f in train_files:
        shutil.move(os.path.join(directory, f), os.path.join(train_dir, f))
    for f in test_files:
        shutil.move(os.path.join(directory, f), os.path.join(test_dir, f))
    for f in val_files:
        shutil.move(os.path.join(directory, f), os.path.join(val_dir, f))
    
    print(f"Arquivos distribuídos: {len(train_files)} treino, {len(test_files)} teste, {len(val_files)} validação.")

# Exemplo de uso
if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Uso: python script.py <caminho_do_diretorio>")
        sys.exit(1)
    
    directory = sys.argv[1]
    split_dataset(directory)