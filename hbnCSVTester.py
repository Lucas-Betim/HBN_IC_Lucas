import pandas as pd
from pgmpy.inference import VariableElimination

from HBNBuilder import encode_dataframe_for_pgmpy


def inferir_classe_csv(
    modelo,
    arquivo_csv: str,
    class_column: str = "class"
):
    """
    Realiza inferência para cada linha de um CSV usando uma HBN treinada.

    Retorna uma lista com a classe prevista para cada linha.
    """

    df_raw = pd.read_csv(arquivo_csv)
    df_raw.columns = df_raw.columns.str.strip()

    if class_column not in df_raw.columns:
        raise ValueError(
            f"O arquivo CSV deve conter uma coluna chamada '{class_column}'"
        )

    df_encoded = encode_dataframe_for_pgmpy(df_raw)

    infer = VariableElimination(modelo)

    resultados = []

    for _, row in df_encoded.iterrows():
        evidencia = row.drop(class_column).dropna().to_dict()

        resultado_query = infer.query(
            variables=[class_column],
            evidence=evidencia,
        )

        valor_mais_provavel = resultado_query.values.argmax()

        estado_previsto = resultado_query.state_names[class_column][
            valor_mais_provavel
        ]

        resultados.append(estado_previsto)

    return resultados