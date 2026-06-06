import numpy as np
import pandas as pd
from math import log
from scipy import stats
from pgmpy.metrics import structure_score

def get_mdl_score(model, bd):
    # Compute the MDL score (Equation 5)
    # 'bic' corresponds to f(N) = 0.5 * log(N)
    # https://jmlr.org/papers/volume7/decampos06a/decampos06a.pdf
    mdl_value = structure_score(model, bd, scoring_method='bic')
    return mdl_value


def get_mdl_pretrained(model, data):
    """
    Computes g_MDL = Log-Likelihood - 0.5 * C(G) * log(N)
    for a model where parameters (CPDs) are already trained.
    """
    N = len(data)
    total_parameters = 0
    log_likelihood = 0

    # 1. Calculate Log-Likelihood for Observed Data
    # Since latent variables are present, we marginalize them out
    # by getting the probability of the observed row evidence.
    for _, instance in data.iterrows():
        # instance.to_dict() contains only observed variables
        prob_instance = model.get_state_probability(instance.to_dict())
        if prob_instance > 0:
            log_likelihood += np.log(prob_instance)

    # 2. Calculate Complexity C(G) from existing CPDs
    # This includes both observed and latent nodes
    for node in model.nodes():
        cpd = model.get_cpds(node)
        if cpd is None:
            raise ValueError(
                f"Node {node} is missing CPDs. Ensure model is fully trained.")

        r_i = cpd.variable_card
        parents = model.get_parents(node)

        if not parents:
            # Root node complexity
            total_parameters += (r_i - 1)
        else:
            # q_i is the product of parent states
            # cpd.cardinality[1:] contains the sizes of all parent variables
            q_i = np.prod(cpd.cardinality[1:])
            total_parameters += q_i * (r_i - 1)

    # 3. Apply MDL Formula
    g_mdl = log_likelihood - (0.5 * total_parameters * np.log(N))

    return g_mdl

# Usage:
# score, ll, params = compute_mdl_from_scratch(hbn, bd)
# print(f"MDL Score: {score}")


def conditional_mutual_information(df, x, y, c):
    n = len(df)
    if n == 0:
        return 0.0

    cmi = 0.0
    p_c = df[c].value_counts(normalize=True).to_dict()

    for c_val, pc in p_c.items():
        df_c = df[df[c] == c_val]
        n_c = len(df_c)
        if n_c == 0:
            continue

        p_x_given_c = df_c[x].value_counts(normalize=True).to_dict()
        p_y_given_c = df_c[y].value_counts(normalize=True).to_dict()
        joint_counts = df_c.groupby([x, y]).size()

        for (x_val, y_val), count in joint_counts.items():
            p_xy_given_c = count / n_c
            p_xc = p_x_given_c.get(x_val, 0.0)
            p_yc = p_y_given_c.get(y_val, 0.0)

            if p_xy_given_c > 0 and p_xc > 0 and p_yc > 0:
                cmi += pc * p_xy_given_c * log(p_xy_given_c / (p_xc * p_yc))

    return cmi


def compute_cmi_significance(data, X, Y, C):
    n_samples = data.shape[0]

    cmi_value = conditional_mutual_information(data, X, Y, C)

    chi2_stat = 2 * n_samples * cmi_value

    card_x = data[X].nunique()
    card_y = data[Y].nunique()
    card_c = data[C].nunique()

    dfreedom = card_c * (card_x - 1) * (card_y - 1)

    p_value = stats.chi2.sf(chi2_stat, dfreedom)

    return p_value


def order_variables_by_cmi_approximation(data, selectecols, classVar):
    list_of_col_pairs = []

    for i in range(len(selectecols)):
        for j in range(i + 1, len(selectecols)):

            X = selectecols[i]
            Y = selectecols[j]

            p_value = compute_cmi_significance(data, X, Y, classVar)

            Q = {
                "X_var": X,
                "Y_var": Y,
                "C_var": classVar,
                "pval": float(p_value)
            }

            list_of_col_pairs.append(Q)

    # Ordena pelo menor p-value (mais dependente)
    list_of_col_pairs.sort(key=lambda x: x['pval'])

    return list_of_col_pairs
