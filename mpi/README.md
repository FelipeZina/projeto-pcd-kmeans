# Versão Distribuída com MPI (`/mpi`)

Esta pasta contém a implementação da Etapa 3 do projeto, utilizando a biblioteca **MPI (Message Passing Interface)** para computação distribuída.

O algoritmo divide o conjunto de dados `N` entre `P` processos. A cada iteração, os processos calculam a atribuição de clusters localmente e utilizam primitivas de comunicação coletiva (`MPI_Allreduce`) para atualizar os centroides globais.

## Como Compilar

É necessário ter uma implementação de MPI instalada (como OpenMPI ou MPICH).

```bash
mpicc -O2 -std=c99 kmeans_1d_mpi.c -o kmeans_1d_mpi -lm
