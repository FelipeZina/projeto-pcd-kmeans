
Esta pasta contém os conjuntos de dados de entrada utilizados para testar e avaliar o desempenho do algoritmo K-Means, bem como o script para gerá-los.

## Conteúdo dos Arquivos

-   **`*_dados.csv`**: Arquivos que contêm N pontos (valores numéricos de uma dimensão), um por linha.
-   **`*_centroides.csv`**: Arquivos que contêm K centroides iniciais, um por linha.
-   **`gerador_dados.py`**: Script em Python responsável por gerar os datasets "médio" e "grande" de forma programática.

## Como Gerar os Datasets

Os datasets "médio" e "grande" foram gerados usando o script `gerador_dados.py`. Para reproduzir ou gerar novos dados, siga os passos abaixo.

### Pré-requisitos

É necessário ter Python instalado, junto com as bibliotecas `numpy` e `pandas`. Para instalá-las, use o pip:

```bash
pip install numpy pandas
```

### Execução

1.  **Abra o script `gerador_dados.py`** em um editor de texto.
2.  **Escolha o dataset a ser gerado** alterando a variável `TIPO_DATASET` no topo do arquivo para `'medio'` ou `'grande'`.
3.  Abra um terminal, navegue até esta pasta (`/dados`).
4.  Execute o script com o seguinte comando:

```bash
python gerador_dados.py
```

O script criará (ou substituirá) os arquivos `medio_dados.csv` e `medio_centroides.csv` (ou os arquivos `grande_...`) dentro desta mesma pasta.

