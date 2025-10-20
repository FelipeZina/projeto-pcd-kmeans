# Versão Paralela com OpenMP (`/openmp`)

Esta pasta contém a implementação paralela do algoritmo K-Means, utilizando a API OpenMP. O objetivo é otimizar a execução para CPUs com múltiplos núcleos em um ambiente de memória compartilhada.

A paralelização foi focada nos dois laços computacionalmente mais intensivos do algoritmo: o passo de atribuição (*assignment*) e o passo de atualização (*update*).

## Estratégias de Paralelização

-   **`assignment_step_1d`**: O laço principal desta função foi paralelizado com a diretiva `#pragma omp parallel for`. Para garantir a corretude da soma da variável `sse` (Sum of Squared Errors), foi utilizada a cláusula `reduction(+:sse)`, que evita condições de corrida de forma eficiente.

-   **`update_step_1d`**: Para o passo de atualização, duas estratégias foram implementadas e comparadas, conforme solicitado no enunciado:
    1.  **`#pragma omp critical`**: Uma abordagem mais simples onde o acesso às variáveis compartilhadas de soma e contagem é protegido, permitindo que apenas uma thread as modifique por vez.
    2.  **Acumuladores Locais (Opção A)**: Uma abordagem mais escalável e performática, onde cada thread trabalha em uma cópia local dos vetores de soma e contagem. Ao final do laço, os resultados locais de cada thread são somados aos vetores globais dentro de uma seção crítica otimizada.

A versão final presente no código utiliza a abordagem de **acumuladores locais** por ser a mais eficiente.

## Como Compilar

Para compilar o código, é essencial utilizar a flag `-fopenmp` para que o compilador GCC inclua as bibliotecas do OpenMP.

```bash
gcc -O2 -fopenmp -std=c99 kmeans_1d_omp.c -o kmeans_1d_omp -lm
```

## Como Executar

O programa espera os arquivos de entrada, seguidos de parâmetros opcionais e os arquivos de saída.

### Execução Básica

O formato completo é:
`./kmeans_1d_omp.exe <arq_dados> <arq_centroides> [max_iter] [eps] [arq_saida_assign] [arq_saida_centroids]`

**Exemplo com o dataset "grande":**
```bash
./kmeans_1d_omp.exe ../dados/grande_dados.csv ../dados/grande_centroides.csv 50 0.0001 grande_assign.csv grande_centroids.csv
```

### Controlando o Desempenho (Número de Threads)

O número de threads utilizados pelo OpenMP é controlado pela variável de ambiente `OMP_NUM_THREADS`.

**No Windows (PowerShell):**
```powershell
# Para executar com 8 threads
$env:OMP_NUM_THREADS=8
./kmeans_1d_omp.exe ../dados/grande_dados.csv ../dados/grande_centroides.csv
```

**No Linux ou macOS:**
```bash
# Para executar com 8 threads
export OMP_NUM_THREADS=8
./kmeans_1d_omp.exe ../dados/grande_dados.csv ../dados/grande_centroides.csv
```
