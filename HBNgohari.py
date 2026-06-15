import pandas as pd
from stepmix.stepmix import StepMix
from HBNBuilder import encode_dataframe_for_pgmpy
import numpy as np
from pgmpy.factors.discrete import TabularCPD
from HNB import HNB
from HBNLearner import ParameterLearner
from HBNpipeline import score_hnb_model_accuracy


def get_cardinality_from_data(df: pd.DataFrame, col: str) -> int:
    """
    Retorna cardinalidade usando max+1.
    Evita erro quando existem estados como:
    [0, 2] -> cardinalidade correta = 3
    """

    values = df[col].dropna().astype(int)

    if len(values) == 0:
        raise ValueError(
            f"Coluna '{col}' não possui valores válidos."
        )

    return int(values.max()) + 1


def split_attributes_into_groups(
    attributes: list[str],
    group_size: int
) -> list[list[str]]:
    """
    Divide os atributos observados em grupos sequenciais.

    Exemplo:
        atributos = [A, B, C, D, E]
        group_size = 2

        retorna:
        [[A, B], [C, D], [E]]
    """

    if group_size <= 0:
        raise ValueError("group_size deve ser maior que zero.")

    groups = []

    for i in range(0, len(attributes), group_size):
        group = attributes[i:i + group_size]

        if len(group) > 0:
            groups.append(group)

    return groups


def create_gohari_latent_variables_with_stepmix(
    csv_file: str,
    class_column: str = "class",
    group_size: int = 2,
    n_components: int = 2,
    measurement: str = "categorical",
    random_state: int = 42,
    debug: bool = True
):
    df_raw = pd.read_csv(csv_file)
    df_raw.columns = df_raw.columns.str.strip()

    if class_column not in df_raw.columns:
        raise ValueError(
            f"Coluna de classe '{class_column}' não encontrada no CSV."
        )

    df_encoded = encode_dataframe_for_pgmpy(df_raw)

    for col in df_encoded.columns:
        if df_encoded[col].isnull().any():
            mode_val = df_encoded[col].mode().iloc[0]
            df_encoded[col] = df_encoded[col].fillna(mode_val)

    df_encoded = df_encoded.astype(int)

    attributes = []

    for col in df_encoded.columns:
        if col == class_column:
            continue

        if df_encoded[col].nunique() < 2:
            if debug:
                print(f"[Gohari-StepMix] Ignorando atributo constante: {col}")
            continue

        attributes.append(col)

    groups = split_attributes_into_groups(
        attributes=attributes,
        group_size=group_size
    )

    df_with_latents = df_encoded.copy()
    latent_groups = {}

    lv_count = 1

    for group in groups:
        if len(group) < 2:
            if debug:
                print(
                    f"[Gohari-StepMix] Grupo ignorado por ter apenas 1 atributo: {group}"
                )
            continue

        latent_name = f"LV_{lv_count}"
        lv_count += 1

        X_group = df_encoded[group].astype(int)

        stepmix = StepMix(
            n_components=n_components,
            measurement=measurement,
            random_state=random_state,
            verbose=0
        )

        stepmix.fit(X_group)

        latent_states = stepmix.predict(X_group)

        if debug:
            print(
                f"{latent_name} estados encontrados:",
                sorted(set(latent_states))
            )

        df_with_latents[latent_name] = latent_states.astype(int)
        latent_groups[latent_name] = group

        if debug:
            print(
                f"[Gohari-StepMix] {latent_name} criado "
                f"com grupo={group} | n_components={n_components}"
            )

    return df_with_latents, latent_groups


def build_gohari_hnb_structure(
    df_with_latents: pd.DataFrame,
    latent_groups: dict,
    class_column: str = "class",
    debug: bool = True
):
    def get_cardinality_from_data(df: pd.DataFrame, col: str) -> int:
        values = df[col].dropna().astype(int)

        if len(values) == 0:
            raise ValueError(
                f"Coluna '{col}' não possui valores válidos."
            )

        return int(values.max()) + 1

    if class_column not in df_with_latents.columns:
        raise ValueError(
            f"Coluna de classe '{class_column}' não encontrada no dataframe."
        )

    hbn = HNB()

    latent_nodes = list(latent_groups.keys())

    observed_attributes = []

    for col in df_with_latents.columns:
        if col == class_column:
            continue

        if col in latent_nodes:
            continue

        if df_with_latents[col].nunique() < 2:
            if debug:
                print(f"[Gohari-HNB] Ignorando atributo constante: {col}")
            continue

        observed_attributes.append(col)

    hbn.add_node(class_column)

    for latent in latent_nodes:
        hbn.add_node(latent)
        hbn.latents.add(latent)

    for attr in observed_attributes:
        hbn.add_node(attr)

    grouped_attrs = set()

    for attrs in latent_groups.values():
        grouped_attrs.update(attrs)

    direct_attrs = [
        attr for attr in observed_attributes
        if attr not in grouped_attrs
    ]

    for latent, group_attrs in latent_groups.items():
        hbn.add_edge(class_column, latent)

        for attr in group_attrs:
            if attr in observed_attributes:
                hbn.add_edge(latent, attr)

    for attr in direct_attrs:
        hbn.add_edge(class_column, attr)

    class_card = get_cardinality_from_data(
        df_with_latents,
        class_column
    )

    cpd_class = TabularCPD(
        variable=class_column,
        variable_card=class_card,
        values=np.ones((class_card, 1)) / class_card
    )

    hbn.add_cpds(cpd_class)

    for latent in latent_nodes:
        latent_card = get_cardinality_from_data(
            df_with_latents,
            latent
        )

        values = np.ones((latent_card, class_card)) / latent_card

        cpd_latent = TabularCPD(
            variable=latent,
            variable_card=latent_card,
            values=values,
            evidence=[class_column],
            evidence_card=[class_card]
        )

        hbn.add_cpds(cpd_latent)

    for latent, group_attrs in latent_groups.items():
        latent_card = get_cardinality_from_data(
            df_with_latents,
            latent
        )

        for attr in group_attrs:
            if attr not in observed_attributes:
                continue

            attr_card = get_cardinality_from_data(
                df_with_latents,
                attr
            )

            values = np.ones((attr_card, latent_card)) / attr_card

            cpd_attr = TabularCPD(
                variable=attr,
                variable_card=attr_card,
                values=values,
                evidence=[latent],
                evidence_card=[latent_card]
            )

            hbn.add_cpds(cpd_attr)

    for attr in direct_attrs:
        attr_card = get_cardinality_from_data(
            df_with_latents,
            attr
        )

        values = np.ones((attr_card, class_card)) / attr_card

        cpd_attr = TabularCPD(
            variable=attr,
            variable_card=attr_card,
            values=values,
            evidence=[class_column],
            evidence_card=[class_card]
        )

        hbn.add_cpds(cpd_attr)

    if debug:
        print("\n[Gohari-HNB] Estrutura criada.")
        print("Nós:", list(hbn.nodes()))
        print("Arestas:", list(hbn.edges()))
        print("Latentes:", list(hbn.latents))

    hbn.check_model()

    return hbn, latent_nodes


def train_gohari_hnb_with_em(
    hbn,
    df_with_latents: pd.DataFrame,
    latent_nodes: list[str],
    max_iter: int = 10,
    debug: bool = True
):
    """
    Treina a HBN Gohari.

    No método Gohari com StepMix, as variáveis LV_i já foram estimadas
    e estão presentes no dataframe. Então, durante o treinamento dos
    parâmetros, tratamos as LV_i como observadas para evitar que o pgmpy
    tente inferi-las novamente como ocultas.

    Depois do treinamento, restauramos hbn.latents para manter a semântica
    HNB do modelo.
    """

    learner = ParameterLearner(debug=debug)

    # Guarda as latentes originais
    original_latents = set(hbn.latents)

    # IMPORTANTE:
    # Para o treinamento, as LV_i criadas pelo StepMix serão tratadas como observadas
    hbn.latents = set()

    hbn_trained, df_final = learner.EM(
        file="",
        bn=hbn,
        latent_nodes=[],
        data=df_with_latents,
        max_iter=max_iter
    )

    # Restaura as latentes depois do treinamento
    hbn_trained.latents = original_latents

    if debug:
        print("\n[Gohari-EM] Treinamento concluído.")
        print("CPDs treinadas:", [cpd.variable for cpd in hbn_trained.cpds])
        print("Latentes restauradas:", list(hbn_trained.latents))

    return hbn_trained, df_final


def run_gohari_once(
    csv_file: str,
    class_column: str = "class",
    group_size: int = 2,
    n_components: int = 2,
    measurement: str = "categorical",
    max_iter_em: int = 10,
    random_state: int = 42,
    debug: bool = True
):
    """
    Executa uma rodada completa do método Gohari:
      1) StepMix cria latentes
      2) HBN é montada
      3) HBN é treinada
      4) Acurácia é calculada
    """

    df_gohari, latent_groups = create_gohari_latent_variables_with_stepmix(
        csv_file=csv_file,
        class_column=class_column,
        group_size=group_size,
        n_components=n_components,
        measurement=measurement,
        random_state=random_state,
        debug=debug
    )

    hbn_gohari, latent_nodes = build_gohari_hnb_structure(
        df_with_latents=df_gohari,
        latent_groups=latent_groups,
        class_column=class_column,
        debug=debug
    )

    hbn_gohari_trained, df_final = train_gohari_hnb_with_em(
        hbn=hbn_gohari,
        df_with_latents=df_gohari,
        latent_nodes=latent_nodes,
        max_iter=max_iter_em,
        debug=debug
    )

    score = score_hnb_model_accuracy(
        model=hbn_gohari_trained,
        csv_file=csv_file,
        class_node=class_column
    )

    result = {
        "n_components": n_components,
        "group_size": group_size,
        "score": score,
        "num_latents": len(latent_nodes),
        "latent_nodes": latent_nodes,
        "latent_groups": latent_groups,
        "model": hbn_gohari_trained,
        "df_final": df_final
    }

    return result


def gohari_elbow_accuracy(
    csv_file: str,
    class_column: str = "class",
    group_size: int = 2,
    components_range=range(2, 7),
    measurement: str = "categorical",
    max_iter_em: int = 10,
    random_state: int = 42,
    debug: bool = True
):
    results = []

    for n_components in components_range:
        if debug:
            print("\n" + "=" * 60)
            print(f"[GOHARI-ELBOW] Testando n_components={n_components}")
            print("=" * 60)

        try:
            result = run_gohari_once(
                csv_file=csv_file,
                class_column=class_column,
                group_size=group_size,
                n_components=n_components,
                measurement=measurement,
                max_iter_em=max_iter_em,
                random_state=random_state,
                debug=debug
            )

            results.append(result)

            if debug:
                print(
                    f"[GOHARI-ELBOW] n_components={n_components} | "
                    f"score={result['score']:.6f}"
                )

        except ValueError as e:
            if debug:
                print(
                    f"[GOHARI-ELBOW] n_components={n_components} ignorado. "
                    f"Motivo: {e}"
                )
            continue

    if len(results) == 0:
        raise ValueError(
            "Nenhum modelo Gohari válido foi encontrado no elbow.")

    best_result = max(results, key=lambda r: r["score"])

    resumo = pd.DataFrame([
        {
            "n_components": r["n_components"],
            "group_size": r["group_size"],
            "score": r["score"],
            "num_latents": r["num_latents"]
        }
        for r in results
    ])

    resumo.to_csv("resultado_gohari_elbow.csv", index=False)

    return results, best_result, resumo

from pgmpy.readwrite import BIFWriter
def gerar_relatorios_gohari(
    best_result: dict,
    resumo: pd.DataFrame,
    output_prefix: str = "gohari"
):
    """
    Gera arquivos finais do método Gohari:
      - resultado_gohari.csv
      - historico_gohari_elbow.csv
      - modelo_final_gohari.bif
    """

    model = best_result["model"]

    resultado = {
        "melhor_n_components": best_result["n_components"],
        "group_size": best_result["group_size"],
        "score_final": best_result["score"],
        "numero_latentes": best_result["num_latents"],
        "latentes": "; ".join(best_result["latent_nodes"])
    }

    df_resultado = pd.DataFrame([resultado])

    arquivo_resultado = f"resultado_{output_prefix}.csv"
    arquivo_historico = f"historico_{output_prefix}_elbow.csv"
    arquivo_bif = f"modelo_final_{output_prefix}.bif"

    df_resultado.to_csv(arquivo_resultado, index=False)
    resumo.to_csv(arquivo_historico, index=False)

    writer = BIFWriter(model)
    writer.write_bif(arquivo_bif)

    return {
        "resultado_csv": arquivo_resultado,
        "historico_csv": arquivo_historico,
        "modelo_bif": arquivo_bif
    }