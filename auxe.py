def select_candidate_latent_variable_v2(model, df_subset, class_node="class", debug=False):
    
    class_children = list(model.get_children(class_node))

    candidates = []
    
    for node in class_children:
        if hasattr(model, "is_latent") and model.is_latent(node):
            continue

        parents = model.get_parents(node)
        if len(parents) == 1 and parents[0] == class_node:
            candidates.append(node)

    if len(candidates) < 2:
        return None

    # Função do professor
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
            f"[3.1] Melhor par: ({best['X_var']}, {best['Y_var']}) | p={best['pval']}")

    return {
        "latent_name": f"L_{best['X_var']}_{best['Y_var']}",
        "pair": (best["X_var"], best["Y_var"]),
        "q_value": float(best["pval"])
    }



 def EM(self, file: str, bn: DiscreteBayesianNetwork, latent_nodes: list[str],
           latent_cardinality: int = 2, fillna_mode: bool = True, **kwargs) -> tuple[DiscreteBayesianNetwork, pd.DataFrame]: