# Versão Sequencial (Baseline) (`/serial`)

Esta pasta contém o código-fonte da implementação sequencial (ou *naive*) do algoritmo K-Means, conforme fornecido.

Este código serve como **baseline** (Etapa 0) para a avaliação de desempenho. O tempo de execução deste programa é a referência (`Tempo_Serial`) usada para calcular o *speedup* das versões paralelas (OpenMP, CUDA, MPI).

O código original `kmeans_1d_naive.c` utiliza `double` (precisão de 64 bits) para todos os cálculos.

## Como Compilar

Para compilar o código, use o GCC. A flag `-O2` é recomendada para uma otimização justa de baseline, e `-lm` é necessária para a biblioteca matemática (`math.h`).

```bash
gcc -O2 -std=c99 kmeans_1d_naive.c -o kmeans_1d_naive -lm
```

## Como Executar

O programa espera pelo menos os dois arquivos de entrada. Os argumentos seguintes são opcionais.

**Formato:**
`./kmeans_1d_naive <arq_dados> <arq_centroides> [max_iter] [eps] [outAssign] [outCentroid]`

**Exemplo:**
```bash
./kmeans_1d_naive ../dados/dados.csv ../dados/centroides_iniciais.csv 50 0.0001
```
