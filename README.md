# Classificadores HBN: Langseth e Gohari

Implementação experimental de classificadores *Hierarchical Bayesian Network*
(HBN) desenvolvida em um projeto de iniciação científica. O repositório contém
duas estratégias para criação de variáveis latentes:

- **Langseth:** busca e seleção incremental de estruturas hierárquicas;
- **Gohari (NB-BLCA):** agrupamento de atributos e estimação das variáveis
  latentes categóricas com StepMix.

Os experimentos utilizam atributos extraídos por DenseNet121, EfficientNetB0,
MobileNetV2 e VGG16 em quatro conjuntos de imagens.

## Requisitos

- Python 3.10 ou superior;
- NumPy;
- pandas;
- SciPy;
- scikit-learn;
- pgmpy;
- StepMix.

Instalação sugerida no Windows:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Dados de entrada

As bases devem estar em `bases/` e seguir, por padrão, o padrão de nome
`Features_BIN_*.csv`. Os arquivos incluídos no repositório já contêm atributos
discretizados. O código apenas recodifica os estados existentes para inteiros
consecutivos exigidos pelo pgmpy; ele não calcula os intervalos de
discretização.

A coluna-alvo é detectada pelo programa. A coluna `path`, quando presente, é
tratada apenas como identificador e removida dos atributos preditivos.

## Execução interativa

```powershell
python Main_menu.py
```

O menu permite executar Langseth, Gohari ou o processamento automático das
bases.

## Execução automática

Para processar todas as bases com as configurações padrão:

```powershell
python run_batch.py
```

Também é possível executar `rodar_todas_bases.bat` no Windows.

Configurações padrão do lote:

| Método | Parâmetros principais |
| --- | --- |
| Langseth | `kappa=6`, `max_iter=5`, `EM=20` |
| Gohari | `group_size=6`, componentes de 2 a 3, `n_init=5` |
| Avaliação | holdout estratificado 80/20 e 1.000 repetições bootstrap |

Exemplos:

```powershell
# Executar somente o Gohari
python run_batch.py --methods gohari

# Executar somente o Langseth
python run_batch.py --methods langseth --kappa 6 --langseth-max-iter 5 --langseth-em-iter 20

# Refazer resultados já existentes
python run_batch.py --force
```

Consulte todas as opções com:

```powershell
python run_batch.py --help
```

## Validação cruzada de 10 folds

A validação cruzada usada na comparação estatística pode ser executada com:

```powershell
python run_cross_validation.py
```

Por padrão, são usados 10 folds estratificados, semente 42 e os dois métodos.
Para executar apenas um deles:

```powershell
python run_cross_validation.py --methods gohari
python run_cross_validation.py --methods langseth
```

Cada fold preserva seus índices de treino e teste para permitir que outros
classificadores sejam avaliados exatamente nas mesmas divisões.

## Protocolos e métricas

### Holdout com bootstrap

O protocolo padrão divide os dados de forma estratificada em 80% para treino e
20% para teste. O modelo é treinado uma vez. Em seguida, os pares de classe real
e predição do conjunto de teste são reamostrados por bootstrap para calcular a
acurácia média e o desvio-padrão.

### Validação cruzada

No protocolo de 10 folds, cada modelo é treinado novamente em nove folds e
avaliado no fold restante. A média e o desvio-padrão são calculados a partir das
dez acurácias. Esse protocolo é o indicado para comparações pareadas entre os
classificadores e para os testes estatísticos do projeto.

### Score final

O `score_final` é calculado para o modelo ajustado com a base completa. Ele mede
o ajuste aos dados de treinamento e não deve ser interpretado como estimativa de
desempenho em dados novos. Por isso, pode ser maior que a acurácia de holdout ou
de validação cruzada.

## Arquivos principais

| Arquivo | Responsabilidade |
| --- | --- |
| `Main_menu.py` | Interface interativa |
| `run_batch.py` | Execução automática das bases |
| `run_cross_validation.py` | Validação cruzada estratificada |
| `HBNvalidation.py` | Holdout, bootstrap e validação cruzada |
| `HBNBuilder.py` | Leitura, limpeza e codificação dos estados |
| `HBNalgorithm.py` | Operações de construção das HBNs |
| `HBNpipeline.py` | Fluxo do método Langseth |
| `HBNgohari.py` | Fluxo do método Gohari com StepMix |
| `HBNLearner.py` | Estimação de parâmetros por EM |
| `HBNstatecollapse.py` | Colapso dos estados latentes |
| `HBNreport.py` | Geração dos relatórios e modelos finais |

## Resultados

As execuções criam diretórios como `resultados/` e
`resultados_cv_10fold/`. Eles podem conter CSVs, logs e modelos BIF. Esses
artefatos são reproduzíveis e estão ignorados pelo Git para evitar commits
grandes e alterações geradas automaticamente.

Antes de uma execução longa, desative temporariamente a suspensão automática
do Windows. A tela pode ser desligada sem interromper o processamento.
