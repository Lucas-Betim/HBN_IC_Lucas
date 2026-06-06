import copy
import pandas as pd

from HNB import HNB
from HBNLearner import ParameterLearner
from HBNalgorithm import (
    create_overlapping_subsets,
    select_candidate_latent_variable_v2,
    get_max_latent_cardinality,
    add_latent_column_from_children
)
from HBNstatecollapse import collapse_latent_states_by_mdl
from HBNBuilder import encode_dataframe_for_pgmpy

# =========================
# Setup comum
# =========================


def build_and_train(csv_file: str):
    """
    Pipeline padrão:
      1) cria NB a partir do CSV
      2) insere latente HL1 entre class e [precip,temp,wind]
      3) treina parâmetros via EM
    """
    hnb = HNB()
    _df_encoded = hnb.create_network_topology_from_data(
        csv_file, class_column="class", debug=False)

    hnb.change_bn_topology(
        "class", ["precip", "temp", "wind"], "HL1", debug=False)

    learner = ParameterLearner(debug=False)
    hnb_trained, _df_final = learner.EM(
        csv_file, hnb, latent_nodes=["HL1"]
    )

    return hnb_trained, _df_final


def score_hnb_model_accuracy(
    model,
    csv_file: str,
    class_node: str = "class"
) -> float:
    """
    Calcula Score(H | DN).

    Nesta versão, o score é a acurácia do modelo no conjunto DN.
    Depois podemos evoluir para wrapper/cross-validation, como no artigo.
    """

    df_raw = pd.read_csv(csv_file)
    df_raw.columns = df_raw.columns.str.strip()

    df_encoded = encode_dataframe_for_pgmpy(df_raw)

    for col in df_encoded.columns:
        if df_encoded[col].isnull().any():
            mode_val = df_encoded[col].mode().iloc[0]
            df_encoded[col] = df_encoded[col].fillna(mode_val)

    df_final = df_encoded.astype(int)

    if class_node not in df_final.columns:
        raise ValueError(f"Coluna de classe '{class_node}' não encontrada.")

    X = df_final.drop(columns=[class_node])
    y_true = df_final[class_node]

    # Remove do dataframe colunas que não existem no modelo
    model_nodes = set(model.nodes())
    X = X[[col for col in X.columns if col in model_nodes]]

    predictions = model.predict(X, n_jobs=1)

    if class_node not in predictions.columns:
        raise ValueError(
            f"A predição não retornou a coluna de classe '{class_node}'."
        )

    y_pred = predictions[class_node]

    accuracy = (y_pred.values == y_true.values).mean()

    return float(accuracy)


def get_candidate_latents_from_original_nb(csv_file: str):
    """
    Roda:
      - 3.1: seleção de candidato
      - construção da latente com cardinalidade máxima
      - 3.2: colapso por Delta MDL
    """

    hnb = HNB()
    df_encoded = hnb.create_network_topology_from_data(
        csv_file,
        class_column="class",
        debug=False
    )

    subsets, subset_indices = create_overlapping_subsets(
        df_encoded.fillna(0).astype(int),
        kappa=5,
        random_state=42,
        debug=False
    )

    candidate_latents = []

    for i, subset in enumerate(subsets, start=1):
        cand = select_candidate_latent_variable_v2(
            hnb,
            subset,
            class_node="class",
            debug=False
        )

        if cand is None:
            continue

        (
            hbn_reduced,
            df_with_latent,
            initial_cardinality,
            final_cardinality,
            collapse_history
        ) = build_candidate_model_with_max_latent(
            current_model=hnb,
            csv_file=csv_file,
            candidate=cand,
            df_subset=subset,
            class_node="class"
        )

        score = score_hnb_model_accuracy(
            model=hbn_reduced,
            csv_file=csv_file,
            class_node="class"
        )

        cand["subset_id"] = i
        cand["initial_cardinality"] = int(initial_cardinality)
        cand["latent_cardinality"] = int(final_cardinality)
        cand["collapse_history"] = collapse_history
        cand["model"] = hbn_reduced
        cand["score"] = score

        candidate_latents.append(cand)

    if len(candidate_latents) == 0:
        return [], None

    best_candidate = max(
        candidate_latents,
        key=lambda cand: cand["score"]
    )

    return candidate_latents, best_candidate


def build_candidate_model_with_max_latent(
    current_model: HNB,
    csv_file: str,
    candidate: dict,
    df_subset,
    class_node: str = "class"
):
    """
    Passo 3.a.iii do artigo:
    define H(i) incluindo L(i) no modelo atual Hk.

    Fluxo:
      1) recebe Hk
      2) copia Hk para formar H(i)
      3) calcula cardinalidade máxima inicial de L(i)
      4) insere L(i) em H(i)
      5) treina parâmetros via EM
      6) colapsa estados via Delta MDL
      7) retorna H(i) final
    """

    # H(i) começa como uma cópia de Hk
    h_i = copy.deepcopy(current_model)

    # Garante que as latentes já existentes em Hk sejam preservadas
    if hasattr(current_model, "latents"):
        h_i.latents = set(current_model.latents)

    pair = candidate["pair"]
    latent_name = candidate["latent_name"]

    latent_cardinality = get_max_latent_cardinality(
        df_subset=df_subset,
        child_vars=pair
    )

    # 3.a.iii: define H(i) incluindo L(i) em Hk
    h_i.change_bn_topology(
        fromnode=class_node,
        toList=list(pair),
        hnode=latent_name,
        hnode_cardinality=latent_cardinality,
        debug=False
    )

    learner = ParameterLearner(debug=False)

    all_latents = list(h_i.latents)

    h_i_trained, df_final = learner.EM(
        csv_file,
        h_i,
        latent_nodes=all_latents,
        latent_cardinality=latent_cardinality,
        data=df_subset
    )

    df_with_latent = add_latent_column_from_children(
        df_subset=df_subset,
        child_vars=pair,
        latent_name=latent_name
    )

    h_i_reduced, collapse_history = collapse_latent_states_by_mdl(
        model=h_i_trained,
        df_with_latent=df_with_latent,
        latent_node=latent_name,
        class_var=class_node,
        debug=False
    )

    final_cardinality = h_i_reduced.get_cpds(
        latent_name
    ).variable_card

    return (
        h_i_reduced,
        df_with_latent,
        latent_cardinality,
        final_cardinality,
        collapse_history
    )


def learn_hnb_classifier(
    csv_file: str,
    class_node: str = "class",
    kappa: int = 5,
    max_iter: int | None = None,
    debug: bool = True
):
    """
    Implementa o loop principal do Algoritmo 3:

      H0 = NB inicial

      Para k:
        1) cria subsets D(i)
        2) gera candidatos H(i)
        3) escolhe H' = arg max Score(H(i) | DN)
        4) se Score(H') > Score(Hk), aceita H'
        5) senão, para

    Retorna:
      - melhor modelo final Hk
      - histórico das iterações
    """

    # H0: modelo NB inicial
    current_model = HNB()
    df_encoded = current_model.create_network_topology_from_data(
        csv_file,
        class_column=class_node,
        debug=False
    )
    current_df = df_encoded.fillna(0).astype(int).copy()

    current_score = score_hnb_model_accuracy(
        model=current_model,
        csv_file=csv_file,
        class_node=class_node
    )

    initial_score = current_score

    if max_iter is None:
        # no máximo n - 1 inserções, como no artigo
        max_iter = max(1, len(df_encoded.columns) - 2)

    history = []

    if debug:
        print("\n===== INÍCIO DO ALGORITMO HNB =====")
        print(f"Score H0: {current_score:.6f}")

    for k in range(max_iter):

        subsets, subset_indices = create_overlapping_subsets(
            current_df,
            kappa=kappa,
            random_state=42 + k,
            debug=False
        )
        candidate_latents = []

        for i, subset in enumerate(subsets, start=1):
            cand = select_candidate_latent_variable_v2(
                current_model,
                subset,
                class_node=class_node,
                debug=False
            )

            if cand is None:
                continue

            (
                h_i_reduced,
                df_with_latent,
                initial_cardinality,
                final_cardinality,
                collapse_history
            ) = build_candidate_model_with_max_latent(
                current_model=current_model,
                csv_file=csv_file,
                candidate=cand,
                df_subset=subset,
                class_node=class_node
            )

            score = score_hnb_model_accuracy(
                model=h_i_reduced,
                csv_file=csv_file,
                class_node=class_node
            )

            cand["iteration"] = k
            cand["subset_id"] = i
            cand["initial_cardinality"] = int(initial_cardinality)
            cand["latent_cardinality"] = int(final_cardinality)
            cand["collapse_history"] = collapse_history
            cand["model"] = h_i_reduced
            cand["score"] = float(score)
            cand["df_with_latent"] = df_with_latent

            candidate_latents.append(cand)

        if len(candidate_latents) == 0:
            if debug:
                print(f"[Iteração {k}] Nenhum candidato encontrado. Parando.")
            break

        # Passo 3.b: H' = arg max Score(H(i) | DN)
        best_candidate = max(
            candidate_latents,
            key=lambda cand: cand["score"]
        )

        best_score = best_candidate["score"]

        if debug:
            print(f"\n[Iteração {k}] Melhor candidato:")
            print({
                "subset_id": best_candidate["subset_id"],
                "latent_name": best_candidate["latent_name"],
                "pair": best_candidate["pair"],
                "score": best_score,
                "current_score": current_score,
                "initial_cardinality": best_candidate["initial_cardinality"],
                "latent_cardinality": best_candidate["latent_cardinality"]
            })

        # Passo 3.c
        if best_score > current_score:
            current_model = best_candidate["model"]
            current_score = best_score
            current_df = best_candidate["df_with_latent"].copy()

            history.append({
                "iteration": k,
                "accepted": True,
                "latent_name": best_candidate["latent_name"],
                "pair": best_candidate["pair"],
                "score": best_score,
                "initial_cardinality": best_candidate["initial_cardinality"],
                "latent_cardinality": best_candidate["latent_cardinality"],
                "collapse_history": best_candidate["collapse_history"]
            })

            if debug:
                print(
                    f"[Iteração {k}] H' aceito. "
                    f"Novo score: {current_score:.6f}"
                )

        else:
            history.append({
                "iteration": k,
                "accepted": False,
                "best_candidate_score": best_score,
                "current_score": current_score
            })

            if debug:
                print(
                    f"[Iteração {k}] H' rejeitado. "
                    f"Score candidato={best_score:.6f} <= score atual={current_score:.6f}"
                )

            break

    if debug:
        print("\n===== FIM DO ALGORITMO HNB =====")
        print(f"Score final: {current_score:.6f}")
        print("Latentes finais:", list(current_model.latents))

    return (
    current_model,
    initial_score,
    current_score,
    history
    )