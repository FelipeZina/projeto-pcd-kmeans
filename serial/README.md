# Versão Sequencial (Baseline) (`/serial`)

Esta pasta contém os códigos-fonte das implementações sequenciais do K-Means, que servem como **baseline** (ponto de referência) para a avaliação de desempenho das versões paralelas.

Existem duas versões do baseline para garantir uma comparação de desempenho justa ("maçãs com maçãs").

## 1. `kmeans_1d_naive.c` (Baseline para OpenMP)

Este é o código-fonte original fornecido (Etapa 0), que utiliza `double` (precisão de 64 bits) para todos os cálculos.

Este código é o baseline oficial para a **Etapa 1 (OpenMP)**, que também foi implementada usando `double`.

### Como Compilar
```bash
gcc -O2 -std=c99 kmeans_1d_naive.c -o serial_baseline_double -lm
```
### Como Executar
```bash
./serial_baseline_double ../dados/grande_dados.csv ../dados/grande_centroides.csv 50 0.0001
```

---

## 2. `kmeans_1d_serial_float.c` (Baseline para CUDA)

Esta é uma versão modificada do código sequencial, convertida para usar `float` (precisão de 32 bits).

Como a implementação da **Etapa 2 (CUDA)** foi otimizada para `float` (para melhor desempenho e compatibilidade com a GPU), este arquivo serve como o **baseline correto** para uma comparação justa de desempenho (`float` vs `float`).

### Como Compilar
```bash
gcc -O2 -std=c99 kmeans_1d_serial_float.c -o serial_baseline_float -lm
```
### Como Executar
```bash
./serial_baseline_float ../dados/grande_dados.csv ../dados/grande_centroides.csv 50 0.0001
```
