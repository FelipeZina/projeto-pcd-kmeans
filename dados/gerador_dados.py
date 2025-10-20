
import numpy as np
import pandas as pd
import os

TIPO_DATASET = 'grande'
## Usei esse código para gerar os dados medios e grandes
if TIPO_DATASET == 'medio':
    N = 100000 
    K = 8       
    NOME_BASE = 'medio'
elif TIPO_DATASET == 'grande':
    N = 1000000 
    K = 16    
    NOME_BASE = 'grande'
else:
    raise ValueError("TIPO_DATASET deve ser 'medio' ou 'grande'")

    
ARQUIVO_DADOS = f'{NOME_BASE}_dados.csv'
ARQUIVO_CENTROIDES = f'{NOME_BASE}_centroides.csv'
# Define os centros dos clusters. O número de centros deve ser igual a K.
# Isso cria uma sequência como: 100, 300, 500, 700... para os centros
cluster_centers = np.arange(100, 100 + K * 200, 200) 
cluster_std_dev = 25 # Desvio padrão (quão "espalhados" são os pontos em torno do centro)

print(f"--- Gerando dataset '{NOME_BASE}' ---")
print(f"N = {N}, K = {K}")



points_per_cluster = N // K
all_points = []

print("Gerando pontos para os clusters...")
for center in cluster_centers:
    points = np.random.normal(loc=center, scale=cluster_std_dev, size=points_per_cluster)
    all_points.extend(points)

np.random.shuffle(all_points)


print("Selecionando centroides iniciais...")
initial_centroids = np.random.choice(all_points, size=K, replace=False)

print(f"Salvando pontos em '{ARQUIVO_DADOS}'...")
pd.DataFrame(all_points).to_csv(ARQUIVO_DADOS, header=False, index=False)

print(f"Salvando centroides em '{ARQUIVO_CENTROIDES}'...")
pd.DataFrame(initial_centroids).to_csv(ARQUIVO_CENTROIDES, header=False, index=False)

print("\nArquivos gerados com sucesso!")