# Versão Híbrida (CUDA + OpenMP) (`/cuda`)

Esta pasta contém a implementação da Etapa 2 do projeto, que paralisa o K-Means usando CUDA.

## Estratégia de Implementação (Opção A)

Seguindo a "Opção A" do enunciado ("copiar `assign` para CPU e calcular médias no host"), esta implementação é um **programa híbrido**:

1.  **Assignment (GPU/CUDA):** O passo de atribuição (`assignment_step`), que é massivamente paralelo (N pontos), é executado em um kernel CUDA. Cada thread da GPU processa um ponto.
2.  **Update (CPU/OpenMP):** O array `assign` é copiado de volta para a CPU, e o passo de atualização (`update_step`) é executado no host usando a função `update_step_1d_omp` (a mesma versão otimizada com OpenMP da Etapa 1).

### Conversão para `float`

O código foi inteiramente refatorado de `double` (64 bits) para `float` (32 bits). Isso foi essencial por duas razões:
1.  **Compatibilidade e Correção:** O bug `SSE=0.0` foi resolvido, pois as GPUs (especialmente no Colab) são otimizadas para `float` e podem falhar silenciosamente com cálculos em `double`.
2.  **Desempenho:** Cálculos em `float` são significativamente mais rápidos em hardware de GPU.

## Como Compilar e Executar (no Google Colab)

Este código **requer um ambiente CUDA específico** para ser compilado, devido a incompatibilidades entre o `nvcc` padrão do Colab (v12+) e o `gcc` necessário para o OpenMP.

A célula de notebook abaixo instala o **CUDA 11.8** (uma versão compatível) e compila o programa corretamente:

```bash
# 1. Instala o CUDA Toolkit 11.8 (uma versão estável e compatível)
print("Iniciando a instalação do CUDA 11.8...")
!wget [https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-ubuntu2204.pin](https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-ubuntu2204.pin) -O /tmp/cuda.pin > /dev/null 2>&1
!sudo mv /tmp/cuda.pin /etc/apt/preferences.d/cuda-repository-pin-600
!wget [https://developer.download.nvidia.com/compute/cuda/11.8.0/local_installers/cuda-repo-ubuntu2204-11-8-local_11.8.0-520.61.05-1_amd64.deb](https://developer.download.nvidia.com/compute/cuda/11.8.0/local_installers/cuda-repo-ubuntu2204-11-8-local_11.8.0-520.61.05-1_amd64.deb) -O /tmp/cuda.deb > /dev/null 2>&1
!sudo dpkg -i /tmp/cuda.deb > /dev/null 2>&1
!sudo cp /var/cuda-repo-ubuntu2204-11-8-local/cuda-*-keyring.gpg /usr/share/keyrings/
!sudo apt-get update > /dev/null 2>&1
!sudo apt-get -y install cuda-toolkit-11-8 > /dev/null 2>&1
print("CUDA 11.8 instalado.")

# 2. Compila o código, usando o nvcc 11.8 e ativando o OpenMP
print("\nCompilando o código híbrido...")
!/usr/local/cuda-11.8/bin/nvcc -O2 -o kmeans_cuda_final cuda/kmeans_1d_cuda.cu -lm -Xcompiler -fopenmp
print("Compilação concluída com sucesso!")
```

## Como Executar (Testes de Desempenho)

O programa aceita o `blockSize` (Tamanho de Bloco) como o terceiro argumento da linha de comando.

O formato é:
`./kmeans_cuda_final <arq_dados> <arq_centroides> [blockSize] [max_iter] [eps]`

**Exemplo de execução para o dataset "grande" com `blockSize=256`:**
```bash
!./kmeans_cuda_final dados/grande_dados.csv dados/grande_centroides.csv 256
```
