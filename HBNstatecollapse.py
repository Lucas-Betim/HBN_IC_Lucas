import numpy as np
import itertools
import pandas as pd

from DeltaMDL import deltaMDL
from pgmpy.factors.discrete import TabularCPD

def colapsar_estados_especificos_no(model, node_escolhido, idx_i, idx_j):
    """
    Colapsa dois estados específicos de um nó da rede.

    Parâmetros:
    - model: rede bayesiana pgmpy
    - node_escolhido: nó cujo estado será colapsado
    - idx_i: índice do primeiro estado
    - idx_j: índice do segundo estado

    Retorna:
    - model: modelo atualizado
    - node_escolhido
    - li_name: nome do estado i
    - lj_name: nome do estado j
    """

    if node_escolhido not in model.nodes():
        raise ValueError(f"Nó '{node_escolhido}' não existe no modelo.")

    cpd_antigo = model.get_cpds(node_escolhido)

    if cpd_antigo is None:
        raise ValueError(f"Nó '{node_escolhido}' não possui CPD.")

    variable_card = int(cpd_antigo.variable_card)

    if variable_card <= 2:
        raise ValueError(
            f"Nó '{node_escolhido}' possui {variable_card} estados. "
            "Não é possível colapsar abaixo de 2 estados."
        )

    if idx_i == idx_j:
        raise ValueError("idx_i e idx_j devem ser diferentes.")

    if idx_i < 0 or idx_j < 0 or idx_i >= variable_card or idx_j >= variable_card:
        raise ValueError(
            f"Índices inválidos: idx_i={idx_i}, idx_j={idx_j}, "
            f"cardinalidade={variable_card}."
        )

    # garante ordem estável
    idx_i, idx_j = sorted([idx_i, idx_j])

    # nomes dos estados
    state_names = cpd_antigo.state_names.get(
        node_escolhido,
        list(range(variable_card))
    )

    li_name = state_names[idx_i]
    lj_name = state_names[idx_j]
    novo_nome_estado = f"{li_name}_ou_{lj_name}"

    # ==========================
    # 1. CPD do nó escolhido
    # ==========================

    evidencias_antigas = cpd_antigo.variables[1:]
    evidencias_card_antigas = cpd_antigo.cardinality[1:].tolist()

    valores_atuais = cpd_antigo.get_values()

    linha_colapsada = valores_atuais[idx_i, :] + valores_atuais[idx_j, :]

    indices_estados = list(range(variable_card))
    indices_restantes = [
        idx for idx in indices_estados
        if idx not in (idx_i, idx_j)
    ]

    novos_valores_matriz = valores_atuais[indices_restantes, :]
    novos_valores_matriz = np.vstack(
        [novos_valores_matriz, linha_colapsada]
    )

    # Normaliza as colunas da CPD do nó escolhido
    soma_colunas = novos_valores_matriz.sum(axis=0, keepdims=True)
    novos_valores_matriz = novos_valores_matriz / soma_colunas

    novos_state_names = [
        state_names[idx] for idx in indices_restantes
    ] + [novo_nome_estado]

    novo_dicionario_states = {
        node_escolhido: novos_state_names
    }

    if evidencias_antigas:
        for pai in evidencias_antigas:
            if pai in cpd_antigo.state_names:
                novo_dicionario_states[pai] = cpd_antigo.state_names[pai]

    novo_cpd = TabularCPD(
        variable=node_escolhido,
        variable_card=variable_card - 1,
        values=novos_valores_matriz,
        evidence=evidencias_antigas,
        evidence_card=evidencias_card_antigas,
        state_names=novo_dicionario_states
    )

    model.remove_cpds(cpd_antigo)
    model.add_cpds(novo_cpd)

    # ==========================
    # 2. CPDs dos filhos
    # ==========================

    for filho in model.get_children(node_escolhido):
        cpd_filho = model.get_cpds(filho)

        if cpd_filho is None:
            continue

        valores_filho = cpd_filho.get_values()

        evidencias_filho = cpd_filho.variables[1:]
        evidencias_card_filho = cpd_filho.cardinality[1:].tolist()

        if node_escolhido not in evidencias_filho:
            continue

        shape_nd = [cpd_filho.variable_card] + evidencias_card_filho
        valores_nd = valores_filho.reshape(shape_nd)

        eixo_pai = evidencias_filho.index(node_escolhido) + 1

        fatias_i = [slice(None)] * len(shape_nd)
        fatias_j = [slice(None)] * len(shape_nd)

        fatias_i[eixo_pai] = idx_i
        fatias_j[eixo_pai] = idx_j

        tensor_i = valores_nd[tuple(fatias_i)]
        tensor_j = valores_nd[tuple(fatias_j)]

        tensor_colapsado = (tensor_i + tensor_j) / 2.0

        valores_nd = np.delete(
            valores_nd,
            [idx_i, idx_j],
            axis=eixo_pai
        )

        valores_nd = np.append(
            valores_nd,
            np.expand_dims(tensor_colapsado, axis=eixo_pai),
            axis=eixo_pai
        )

        nova_evidence_card = list(evidencias_card_filho)
        nova_evidence_card[eixo_pai - 1] -= 1

        dicionario_states_filho = cpd_filho.state_names.copy()
        dicionario_states_filho[node_escolhido] = novos_state_names

        valores_reshape = valores_nd.reshape(cpd_filho.variable_card, -1)

        # Normaliza as colunas da CPD do filho
        soma_colunas = valores_reshape.sum(axis=0, keepdims=True)
        valores_reshape = valores_reshape / soma_colunas

        novo_cpd_filho = TabularCPD(
            variable=filho,
            variable_card=cpd_filho.variable_card,
            values=valores_reshape,
            evidence=evidencias_filho,
            evidence_card=nova_evidence_card,
            state_names=dicionario_states_filho
        )

        model.remove_cpds(cpd_filho)
        model.add_cpds(novo_cpd_filho)

    return model, node_escolhido, li_name, lj_name


def find_best_mdl_state_pair(
    model,
    df_with_latent: pd.DataFrame,
    latent_node: str,
    class_var: str = "class"
):
    """
    Testa todos os pares de estados de uma latente real no modelo
    e retorna o par com maior Delta MDL.
    """

    cpd_latent = model.get_cpds(latent_node)

    if cpd_latent is None:
        raise ValueError(f"Nó latente '{latent_node}' não possui CPD.")

    latent_cardinality = int(cpd_latent.variable_card)

    if latent_cardinality <= 2:
        return None, float("-inf")

    best_pair = None
    best_delta = float("-inf")

    for state_i, state_j in itertools.combinations(range(latent_cardinality), 2):
        model_colapsado = model.copy()

        model_colapsado, node, li_name, lj_name = colapsar_estados_especificos_no(
            model_colapsado,
            latent_node,
            state_i,
            state_j
        )

        delta = deltaMDL(
            model,
            model_colapsado,
            df_with_latent,
            class_var,
            latent_node,
            state_i,
            state_j
        )

        if delta > best_delta:
            best_delta = delta
            best_pair = (state_i, state_j, li_name, lj_name)

    return best_pair, float(best_delta)

def collapse_latent_states_by_mdl(
    model,
    df_with_latent,
    latent_node: str,
    class_var: str = "class",
    debug: bool = False
):
    """
    Executa o passo 3.2 usando Delta MDL.

    Repete:
      1) encontra o melhor par de estados da latente
      2) calcula Delta MDL
      3) se Delta > 0, colapsa
      4) senão, para
    """

    history = []

    while True:
        best_pair, best_delta = find_best_mdl_state_pair(
            model=model,
            df_with_latent=df_with_latent,
            latent_node=latent_node,
            class_var=class_var
        )

        if best_pair is None:
            break

        state_i, state_j, li_name, lj_name = best_pair

        if best_delta <= 0:
            if debug:
                print(
                    f"[DeltaMDL] Parada: melhor delta={best_delta:.6f} "
                    f"para ({li_name}, {lj_name})"
                )
            break

        if debug:
            print(
                f"[DeltaMDL] Colapsando {latent_node}: "
                f"{li_name} + {lj_name} | delta={best_delta:.6f}"
            )
         
        old_cardinality = int(
            model.get_cpds(latent_node).variable_card
        )

        model, node, li_name, lj_name = colapsar_estados_especificos_no(
            model,
            latent_node,
            state_i,
            state_j
        )

        df_with_latent = atualizar_coluna_latente_apos_colapso(
            df_with_latent=df_with_latent,
            latent_node=latent_node,
            idx_i=state_i,
            idx_j=state_j,
            old_cardinality=old_cardinality
        )

        history.append({
            "latent_node": latent_node,
            "merged_pair": (li_name, lj_name),
            "delta": float(best_delta),
            "cardinality_after": int(model.get_cpds(latent_node).variable_card)
        })

        if model.get_cpds(latent_node).variable_card <= 2:
            break

    return model, history


def atualizar_coluna_latente_apos_colapso(
    df_with_latent,
    latent_node: str,
    idx_i: int,
    idx_j: int,
    old_cardinality: int
):
    """
    Atualiza a coluna da latente no dataframe após colapsar dois estados.

    A função colapsar_estados_especificos_no remove idx_i e idx_j
    e adiciona o novo estado colapsado no final.

    Exemplo:
      estados antigos: 0, 1, 2, 3
      colapsa 0 e 3
      estados novos:
        antigo 1 -> novo 0
        antigo 2 -> novo 1
        antigo 0/3 -> novo 2
    """

    idx_i, idx_j = sorted([idx_i, idx_j])

    remaining_states = [
        idx for idx in range(old_cardinality)
        if idx not in (idx_i, idx_j)
    ]

    new_collapsed_state = len(remaining_states)

    mapping = {}

    for new_idx, old_idx in enumerate(remaining_states):
        mapping[old_idx] = new_idx

    mapping[idx_i] = new_collapsed_state
    mapping[idx_j] = new_collapsed_state

    df_new = df_with_latent.copy()
    df_new[latent_node] = df_new[latent_node].map(mapping)

    if df_new[latent_node].isnull().any():
        raise ValueError(
            f"Erro ao atualizar coluna '{latent_node}': "
            "existem estados antigos sem mapeamento."
        )

    df_new[latent_node] = df_new[latent_node].astype(int)

    return df_new