import copy
import itertools
import pandas as pd

from sklearn.model_selection import train_test_split

from hbnscores import order_variables_by_cmi_approximation

# Passo 2: criar subconjuntos


def create_overlapping_subsets(
    df: pd.DataFrame,
    kappa: int,
    random_state: int = 42,
    debug: bool = False
):
    """
    Gera kappa subconjuntos parcialmente sobrepostos
    e também guarda os índices originais de cada subconjunto.
    """

    if kappa < 2:
        raise ValueError("kappa deve ser >= 2")

    if len(df) == 0:
        raise ValueError("Dataset vazio")

    subsets = []
    subset_indices = []

    test_fraction = 1.0 / kappa

    for i in range(kappa):
        train_subset, _ = train_test_split(
            df,
            test_size=test_fraction,
            shuffle=True,
            random_state=random_state + i
        )

        # guarda os índices originais antes do reset_index
        subset_indices.append(set(train_subset.index))

        # subset limpo para usar no resto do código
        subsets.append(train_subset.reset_index(drop=True))

        if debug:
            print(f"[Passo 2] D^({i+1}) criado com {len(train_subset)} linhas")

    if debug:
        print(f"[Passo 2] Total de subconjuntos: {len(subsets)}")
        print(
            f"[Passo 2] Cada subconjunto ≈ {(kappa-1)/kappa:.2f} do dataset original")

    return subsets, subset_indices

# Passo 3.1

def select_candidate_latent_variable_v2(model, df_subset, class_node="class", debug=False):
    """
    Passo 3.1:
    Seleciona um par {X, Y} dentro de ch(C) no modelo atual Hk.

    Agora aceita tanto atributos observados quanto latentes já existentes,
    desde que a variável exista como coluna em df_subset.
    """

    class_children = list(model.get_children(class_node))

    candidates = []

    for node in class_children:
        parents = model.get_parents(node)

        if len(parents) == 1 and parents[0] == class_node:
            if node in df_subset.columns:
                candidates.append(node)

    if len(candidates) < 2:
        return None

    ordered_pairs = order_variables_by_cmi_approximation(
        df_subset,
        candidates,
        class_node
    )

    if len(ordered_pairs) == 0:
        return None

    best = ordered_pairs[0]

    if debug:
        print(
            f"[3.1] Melhor par: ({best['X_var']}, {best['Y_var']}) | p={best['pval']}"
        )

    return {
        "latent_name": f"L_{best['X_var']}_{best['Y_var']}",
        "pair": (best["X_var"], best["Y_var"]),
        "q_value": float(best["pval"])
    }

# Passo 3.2


def initialize_latent_state_map(df_subset: pd.DataFrame, child_vars: tuple[str, ...]):
    """
    Passo 3.2 - Algoritmo 2, linha 1:
    Inicializa |sp(L)| como o produto dos espaços dos filhos.

    Retorna:
    - state_map: dict[int, tuple]
      Ex.: {0: (0,0), 1: (0,1), 2: (1,0), ...}
    - combo_to_state: dict[tuple, int]
    """
    state_spaces = []
    for var in child_vars:
        values = sorted(df_subset[var].dropna().unique().tolist())
        state_spaces.append(values)

    combinations = list(itertools.product(*state_spaces))

    state_map = {idx: combo for idx, combo in enumerate(combinations)}
    combo_to_state = {combo: idx for idx, combo in state_map.items()}

    return state_map, combo_to_state


def get_max_latent_cardinality(
    df_subset: pd.DataFrame,
    child_vars: tuple[str, ...]
) -> int:
    """
    Calcula a cardinalidade máxima inicial da variável latente L.

    A cardinalidade máxima é:

        |sp(L)| = Π |sp(X_i)|

    onde X_i são os filhos da latente.

    Exemplo:
        precip = 2 estados
        wind   = 2 estados

        => |L| = 2 * 2 = 4
    """

    if len(child_vars) == 0:
        raise ValueError("child_vars não pode ser vazio.")

    cardinality = 1

    for var in child_vars:
        if var not in df_subset.columns:
            raise ValueError(
                f"Variável '{var}' não encontrada no dataframe."
            )

        n_states = df_subset[var].dropna().nunique()

        if n_states <= 0:
            raise ValueError(
                f"Variável '{var}' possui cardinalidade inválida: {n_states}"
            )

        cardinality *= int(n_states)

    return int(cardinality)


def assign_latent_states(
    df_subset: pd.DataFrame,
    child_vars: tuple[str, ...],
    combo_to_state: dict[tuple, int],
    latent_col: str = "__L__"
):
    """
    Passo 3.2 - Algoritmo 2, linha 1:
    Rotula cada linha com o estado inicial de L
    correspondente à combinação dos filhos.
    """
    df_tmp = df_subset.copy()

    def map_row_to_state(row):
        combo = tuple(row[var] for var in child_vars)
        return combo_to_state[combo]

    df_tmp[latent_col] = df_tmp.apply(map_row_to_state, axis=1)
    return df_tmp


def generic_delta(
    df_with_latent: pd.DataFrame,
    state_i: int,
    state_j: int,
    latent_col: str = "__L__",
    class_var: str = "class"
) -> float:
    """
    Delta genérico e leve:
    compara a distribuição da classe nos estados i e j.

    Quanto mais parecidas as distribuições de classe,
    maior o delta.
    """
    df_i = df_with_latent[df_with_latent[latent_col] == state_i]
    df_j = df_with_latent[df_with_latent[latent_col] == state_j]

    if len(df_i) == 0 or len(df_j) == 0:
        return float("-inf")

    p_i = df_i[class_var].value_counts(normalize=True)
    p_j = df_j[class_var].value_counts(normalize=True)

    all_classes = sorted(set(p_i.index).union(set(p_j.index)))

    l1_distance = 0.0
    for c in all_classes:
        l1_distance += abs(p_i.get(c, 0.0) - p_j.get(c, 0.0))

    # Similaridade simples: quanto menor a distância, maior o delta
    delta = 1.0 - l1_distance
    return float(delta)


def collapse_best_state_pair(
    df_with_latent: pd.DataFrame,
    state_map: dict[int, tuple],
    latent_col: str = "__L__",
    class_var: str = "class",
    debug: bool = False,
    ):
    """
    Passo 3.2 - Algoritmo 2, linhas 2, 3 e 4:
    testa todos os pares de estados atuais, escolhe o melhor delta
    e colapsa se delta > 0.
    """
    current_states = sorted(state_map.keys())

    if len(current_states) < 2:
        return False, df_with_latent, state_map, None, float("-inf")

    best_pair = None
    best_delta = float("-inf")

    for s_i, s_j in itertools.combinations(current_states, 2):

        delta = generic_delta(
            df_with_latent,
            s_i,
            s_j,
            latent_col=latent_col,
            class_var=class_var
    )
    
        if delta > best_delta:
            best_delta = delta
            best_pair = (s_i, s_j)

    if best_pair is None or best_delta <= 0:
        return False, df_with_latent, state_map, best_pair, best_delta

    s_keep, s_remove = best_pair

    # Atualiza a coluna latente: estados removidos passam a ser o estado mantido
    df_new = df_with_latent.copy()
    df_new[latent_col] = df_new[latent_col].replace(s_remove, s_keep)

    # Atualiza o state_map:
    # o estado mantido passa a representar a união dos dois
    new_state_map = copy.deepcopy(state_map)
    combo_keep = new_state_map[s_keep]
    combo_remove = new_state_map[s_remove]

    if (
        isinstance(combo_keep, tuple)
        and len(combo_keep) > 0
        and isinstance(combo_keep[0], tuple)
    ):
        merged_repr = combo_keep + (combo_remove,)
    else:
        merged_repr = (combo_keep, combo_remove)

    new_state_map[s_keep] = merged_repr
    del new_state_map[s_remove]

    if debug:
        print(
            f"[3.2] Colapsando estados {s_keep} e {s_remove} | delta={best_delta:.6f}")

    return True, df_new, new_state_map, best_pair, best_delta


def determine_latent_state_space(
    df_subset: pd.DataFrame,
    child_vars: tuple[str, ...],
    class_var: str = "class",
    debug: bool = False
):
    """
    Implementa o passo 3.2 do artigo com delta genérico.
    """
    # 1) Inicializa espaço de estados
    state_map, combo_to_state = initialize_latent_state_map(
        df_subset, child_vars)

    # 2) Rotula estados iniciais
    df_tmp = assign_latent_states(
        df_subset,
        child_vars,
        combo_to_state,
        latent_col="__L__"
    )

    history = []

    # 3) Loop de colapsos
    while True:
        improved, df_tmp, state_map, best_pair, best_delta = collapse_best_state_pair(
            df_tmp,
            state_map,
            latent_col="__L__",
            class_var=class_var,
            debug=debug
        )

        if not improved:
            break

        history.append({
            "merged_pair": best_pair,
            "delta": best_delta,
            "n_states_after": len(state_map)
        })

    return {
        "latent_cardinality": len(state_map),
        "state_map": state_map,
        "history": history,
        "data_with_latent": df_tmp
    }


def propose_latent_variable_with_state_space(
    model,
    df_subset: pd.DataFrame,
    class_node: str = "class",
    debug: bool = False
):
    """
    Roda 3.1 e 3.2 em sequência no mesmo subset.
    """
    cand = select_candidate_latent_variable_v2(
        model,
        df_subset,
        class_node=class_node,
        debug=debug
    )

    if cand is None:
        return None

    state_info = determine_latent_state_space(
        df_subset=df_subset,
        child_vars=cand["pair"],
        class_var=class_node,
        debug=debug
    )

    cand["latent_cardinality"] = state_info["latent_cardinality"]
    cand["state_map"] = state_info["state_map"]
    cand["state_history"] = state_info["history"]

    return cand


def add_latent_column_from_children(
    df_subset: pd.DataFrame,
    child_vars: tuple[str, ...],
    latent_name: str
) -> pd.DataFrame:
    """
    Cria uma coluna da latente no dataframe a partir das combinações dos filhos.

    Exemplo:
        child_vars = ("precip", "wind")

        precip wind  ->  L_precip_wind
          0     0    ->  0
          0     1    ->  1
          1     0    ->  2
          1     1    ->  3
    """
    df_tmp = df_subset.copy()

    state_spaces = []
    for var in child_vars:
        values = sorted(df_tmp[var].dropna().unique().tolist())
        state_spaces.append(values)

    combinations = list(itertools.product(*state_spaces))
    combo_to_state = {combo: idx for idx, combo in enumerate(combinations)}

    def map_row_to_state(row):
        combo = tuple(row[var] for var in child_vars)
        return combo_to_state[combo]

    df_tmp[latent_name] = df_tmp.apply(map_row_to_state, axis=1)

    return df_tmp