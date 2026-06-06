import pandas as pd

from pgmpy.models import DiscreteBayesianNetwork
from pgmpy.estimators import ExpectationMaximization

from HBNBuilder import encode_dataframe_for_pgmpy


# ============================================================
# classe ParameterLearner
# ============================================================


class ParameterLearner:
    """
    Implementa EM:
      - recebe um nome de arquivo CSV + uma rede bayesiana
      - faz uso da Parte 2 (carregar + encode)
      - executa Parte 5 (treinar CPDs via EM)
    """

    def __init__(self, debug: bool = True):
        self.debug = debug

    def EM(self, file: str, bn: DiscreteBayesianNetwork, latent_nodes: list[str],
           latent_cardinality: int = 2, fillna_mode: bool = True, data: pd.DataFrame | None = None, **kwargs):

        # Parte 2: carregar + encode
        if data is None:
            df_raw = pd.read_csv(file)
            df_raw.columns = df_raw.columns.str.strip()
            df_encoded = encode_dataframe_for_pgmpy(df_raw)
        else:
            df_encoded = data.copy()

        # Parte 5: tratamento de NaN
        if fillna_mode:
            for col in df_encoded.columns:
                if df_encoded[col].isnull().any():
                    mode_val = df_encoded[col].mode().iloc[0]
                    df_encoded[col] = df_encoded[col].fillna(mode_val)

        df_final = df_encoded.astype(int)
        

        if self.debug:
            print("[ParameterLearner] Rodando EM...")
            print("  Linhas:", len(df_final),
                  "| Colunas:", list(df_final.columns))

        estimator = ExpectationMaximization(bn, df_final)

        # cardinalidades dos latentes
        latents = {}
        for node in latent_nodes:
            if node not in bn.nodes():
                raise ValueError(f"Nó latente '{node}' não existe na rede.")
            cpd_node = bn.get_cpds(node)
            if cpd_node is not None:
                latents[node] = int(cpd_node.variable_card)
            else:
                latents[node] = latent_cardinality

        params = {
            "latent_card": latents,
            "n_jobs": 1,
            "atol": 0.001,
            "max_iter": 100
        }
        params.update(kwargs)

        estimated_cpds = estimator.get_parameters(**params)

        # Atualiza CPDs do modelo
        bn.remove_cpds(*bn.cpds)
        bn.add_cpds(*estimated_cpds)

        if self.debug:
            print("[ParameterLearner] EM concluído.")
            print("  CPDs:", [cpd.variable for cpd in bn.cpds])

        return bn, df_final
