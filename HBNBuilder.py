import numpy as np
import pandas as pd

from pgmpy.models import DiscreteBayesianNetwork
from pgmpy.factors.discrete import TabularCPD

def encode_dataframe_for_pgmpy(df: pd.DataFrame) -> pd.DataFrame:
    df_encoded = df.copy()

    for col in df_encoded.columns:
        # Categóricas (string/categorical) -> códigos inteiros
        if df_encoded[col].dtype == "object" or pd.api.types.is_categorical_dtype(df_encoded[col]):
            codes = df_encoded[col].astype("category").cat.codes

            if codes.min() == -1:
                df_encoded[col] = codes.astype(pd.Int64Dtype())
            else:
                df_encoded[col] = codes.astype(int)

        # Numéricas float mas discretas -> converte somente se forem inteiras
        elif pd.api.types.is_float_dtype(df_encoded[col]):
            values = df_encoded[col].dropna()

            if values.apply(float.is_integer).all():
                df_encoded[col] = df_encoded[col].astype(pd.Int64Dtype())

    return df_encoded

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



    