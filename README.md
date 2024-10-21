# Point Cloud Autoencoder

Este projeto é uma implementação de um Autoencoder de Nuvem de Pontos utilizando PyTorch. O modelo codifica e decodifica dados de nuvem de pontos 3D, utilizando Chamfer Distance como função de perda. O projeto divide os dados em conjuntos de treino e teste e salva periodicamente os resultados, incluindo gráficos de perda e reconstruções de amostra.

## Estrutura do Projeto

```bash
. 
├── config.py           # Arquivo de configuração com hiperparâmetros
├── data/               # Diretório contendo os dados de entrada (.npy)
├── output/             # Diretório onde os resultados serão salvos
├── model.py            # Implementação do modelo Autoencoder de Nuvem de Pontos
├── train.py            # Funções para treinamento, teste e salvamento de resultados
├── utils.py            # Funções utilitárias (ex.: para plotar nuvens de pontos)
└── main.py             # Script principal para rodar o treinamento
```

## Dependências Certifique-se de ter as seguintes dependências instaladas

- **Python 3.x**
- **PyTorch**
- **numpy**
- **matplotlib**
- **pytorch3d**

Você pode instalá-las utilizando o seguinte comando:

```bash
pip install torch numpy matplotlib pytorch3d
```

## Executando o Projeto

#### 1. Preparar os Dados Coloque o seu dataset de nuvem de pontos na pasta `data/` no formato `.npy`. Atualize o caminho dos dados (`data_path`) no arquivo `config.py` para apontar para o seu dataset.

#### 2. Treinar o Modelo Para iniciar o treinamento do autoencoder, execute o seguinte comando

```bash
python main.py
```

O processo de treinamento começará, utilizando as configurações definidas no arquivo
`config.py`:

- **`batch_size`**: Tamanho dos lotes de dados durante o treinamento.
- **`latent_size`**: Tamanho do espaço latente para codificação.
- **`learning_rate`**: Taxa de aprendizado para o otimizador.
- **`num_epochs`**: Número de épocas para treinar o modelo.
- **`use_GPU`**: Defina como `True` se uma GPU estiver disponível; caso contrário, a CPU será usada.

#### 3. Monitorar o Treinamento Durante o treinamento, o script exibirá a perda para os conjuntos de treino e teste a cada época. Os resultados serão salvos na pasta `output/`, incluindo

- Gráficos de perda a cada 50 épocas. - Amostras de reconstruções de entrada/saída.

#### 4. Resultados

- Gráficos de perda são salvos como `loss_epoch_X.png` na pasta `output/`.

- Amostras de nuvens de pontos comparando os dados de entrada e os reconstruídos também são salvas periodicamente na pasta `output/`.

### Configuração Ajuste os hiperparâmetros no arquivo `config.py` conforme necessário

``` python config = {
    'data_path': 'data/chair_set.npy',  # Caminho para o arquivo .npy com dados de nuvem de pontos 
    'output_folder': 'output/', # Pasta para salvar os resultados
    'batch_size': 32, # Tamanho do lote para o treinamento
    'latent_size': 128, # Tamanho do espaço latente
    'learning_rate': 0.0005, # Taxa de aprendizado para o otimizador
    'use_GPU': True, # Utilizar GPU para o treinamento
    'num_epochs': 1001, # Número de épocas de treinamento
    'save_results': True # Salvar resultados periodicamente } 
```
