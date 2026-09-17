import numpy as np
import pandas as pd

from pgmpy.models import DiscreteBayesianNetwork
from pgmpy.factors.discrete import TabularCPD


NON_FEATURE_COLUMNS = {"path"}


def drop_non_feature_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove colunas de identificação que não são atributos preditivos.

    ``path`` possui praticamente um valor diferente por imagem nas bases
    Features_BIN. Tratá-la como variável categórica cria estados que não
    existem no treino e invalida a avaliação do conjunto de teste.
    """
    columns_to_drop = [
        col for col in df.columns
        if col.strip().lower() in NON_FEATURE_COLUMNS
    ]
    return df.drop(columns=columns_to_drop)


def fit_state_mappings(df: pd.DataFrame) -> dict:
    """Aprende nos dados de treino o código inteiro de cada estado."""
    mappings = {}

    for col in df.columns:
        serie = df[col]
        fill_value = serie.mode().iloc[0] if serie.isnull().any() else None

        if fill_value is not None:
            serie = serie.fillna(fill_value)

        if (
            serie.dtype == "object"
            or isinstance(serie.dtype, pd.CategoricalDtype)
        ):
            values = pd.Categorical(serie).categories.tolist()
        else:
            values = sorted(serie.dropna().unique().tolist())

        mappings[col] = {
            "fill_value": fill_value,
            "mapping": {value: idx for idx, value in enumerate(values)}
        }

    return mappings


def transform_dataframe_with_state_mappings(
    df: pd.DataFrame,
    state_mappings: dict
) -> pd.DataFrame:
    """Aplica a um dataframe o mapeamento aprendido no treino."""
    encoded = pd.DataFrame(index=df.index)

    for col in df.columns:
        if col not in state_mappings:
            continue

        serie = df[col]
        fill_value = state_mappings[col]["fill_value"]

        if fill_value is not None:
            serie = serie.fillna(fill_value)

        encoded[col] = serie.map(state_mappings[col]["mapping"])

    return encoded


def encode_dataframe_for_pgmpy(df: pd.DataFrame) -> pd.DataFrame:
    """
    Codifica todas as colunas para estados discretos 0, 1, 2, ...

    Isso evita erros do pgmpy quando uma coluna vem com estados como:
    1, 2, 3, ..., 20

    O pgmpy espera índices começando em 0:
    0, 1, 2, ..., 19
    """

    df = drop_non_feature_columns(df)
    mappings = fit_state_mappings(df)
    return transform_dataframe_with_state_mappings(
        df,
        mappings
    ).astype(int)

# ============================================================
# Builder: carrega CSV + encode + cria NB inicial
# ============================================================
class HBNBuilder:
    """
    Builder que:
      1) lê CSV
      2) aplica encode_dataframe_for_pgmpy (Parte 2)
      3) cria BN Naive Bayes (Parte 3): class -> atributos
      4) cria CPDs uniformes iniciais
    """

    def __init__(self, class_column: str = "class", debug: bool = True):
        self.class_column = class_column
        self.debug = debug

    def build(self, file: str) -> tuple[DiscreteBayesianNetwork, pd.DataFrame]:
        # 1) Ler CSV
        df_raw = pd.read_csv(file)
        df_raw.columns = df_raw.columns.str.strip()
        
        if self.class_column not in df_raw.columns:
            raise ValueError(
                f"Coluna de classe '{self.class_column}' não encontrada no CSV. "
                f"Colunas disponíveis: {list(df_raw.columns)}"
            )

        # 2) Encode
        df = encode_dataframe_for_pgmpy(df_raw)

        # 3) Criar rede NB
        bn = DiscreteBayesianNetwork()
        bn.add_nodes_from(df.columns)

        # Arestas: class -> atributos
        for col in df.columns:
            if col != self.class_column:
                bn.add_edge(self.class_column, col)

        # 4) CPD da classe (uniforme)
        class_card = df[self.class_column].dropna().nunique()
        if class_card < 2:
            raise ValueError(
                "A coluna de classe tem cardinalidade < 2. Não dá para treinar/classificar.")

        cpd_class = TabularCPD(
            variable=self.class_column,
            variable_card=class_card,
            values=np.array([1.0 / class_card] *
                            class_card).reshape(class_card, 1)
        )
        bn.add_cpds(cpd_class)

        # 5) CPDs dos atributos dado class (uniformes)
        for col in df.columns:
            if col == self.class_column:
                continue

            var_card = df[col].dropna().nunique()
            if var_card < 2:
                if self.debug:
                    print(
                        f"[HBNBuilder] Ignorando '{col}' (variável constante: card={var_card})")
                if col in bn.nodes():
                    bn.remove_node(col)    
                continue
            
            values = np.full((var_card, class_card), 1.0 / var_card).tolist()
            cpd = TabularCPD(
                variable=col,
                variable_card=var_card,
                values=values,
                evidence=[self.class_column],
                evidence_card=[class_card]
            )
            bn.add_cpds(cpd)

        if self.debug:
            print("[HBNBuilder] Rede NB criada.")
            print("  Nós:", list(bn.nodes()))
            print("  Arestas:", list(bn.edges()))
            print("  CPDs:", [cpd.variable for cpd in bn.cpds])

        return bn, df



