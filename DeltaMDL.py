import numpy as np
import pandas as pd
import math

def calcular_parametros_rede(model):
    """
    Calcula o número total de parâmetros livres (|Θ|) de uma Rede Bayesiana.
    """
    total_parametros = 0

    for node in model.nodes():
        cpd = model.get_cpds(node)

        # r: cardinalidade (número de estados) do próprio nó
        r = cpd.variable_card

        # q: produto das cardinalidades dos pais
        # cpd.cardinality[1:] contém os estados dos pais.
        # Se estiver vazio (nó raiz), np.prod retorna 1.0, o que é matematicamente perfeito.
        cardinalidades_pais = cpd.cardinality[1:]
        q = np.prod(cardinalidades_pais) if len(cardinalidades_pais) > 0 else 1

        # Parâmetros do nó = (r - 1) * q
        parametros_no = (r - 1) * q
        total_parametros += parametros_no

    return int(total_parametros)


def difThetaS(model_original, model_colapsado):
    """Calcula a expressão (|Θ_{B_S}| - |Θ_{B'_S}|).
    """
    # 1. Calcula |Θ| do modelo original (B_S)
    theta_original = calcular_parametros_rede(model_original)

    # 2. Calcula |Θ| do modelo após o colapso (B'_S)
    theta_colapsado = calcular_parametros_rede(model_colapsado)

    # 3. Subtrai os dois
    diferenca = theta_original - theta_colapsado

    # print(f"|Θ_{{B_S}}| (Original)   = {theta_original}")
    # print(f"|Θ_{{B'_S}}| (Colapsado) = {theta_colapsado}")
    # print(f"Diferença          = {diferenca}")

    return diferenca


def calcular_ganho_mdl_eq4(data, col_classe, col_atributo, estado_i, estado_j):
    """
    Calcula o ganho na Log-Verossimilhança Marginal ao fundir dois estados (Eq 4).

    :param data: pd.DataFrame com seus dados originais.
    :param col_classe: Nome da variável alvo (C).
    :param col_atributo: Nome da variável latente/atributo (L).
    :param estado_i, estado_j: Nomes dos dois estados de L a serem colapsados.
    :return: Valor do ganho (float). Valores mais próximos de 0 indicam que a fusão é boa.
    """
    # 1. Tabela de contingência (Classe nas linhas, Atributo nas colunas)
    ct = pd.crosstab(data[col_classe], data[col_atributo])

    # Prevenção de erro: verifica se os estados realmente existem nos dados
    if estado_i not in ct.columns or estado_j not in ct.columns:
        return 0.0

    # 2. Contagens N(c, l) para os dois estados
    n_c_li = ct[estado_i].values
    n_c_lj = ct[estado_j].values

    # 3. Totais marginais N(l_i) e N(l_j)
    n_li = n_c_li.sum()
    n_lj = n_c_lj.sum()

    # Se algum estado nunca ocorreu, o ganho/perda da fusão é zero
    if n_li == 0 or n_lj == 0:
        return 0.0

    # 4. Valores fundidos (i ∪ j)
    n_c_lij = n_c_li + n_c_lj
    n_lij = n_li + n_lj

    # 5. Função auxiliar para a entropia: x * log(x/y)
    def calcular_termo(numerador, denominador):
        # Ignora avisos do numpy para divisão por zero (trataremos os NaNs)
        with np.errstate(divide='ignore', invalid='ignore'):
            res = numerador * np.log(numerador / denominador)
        # Transforma eventuais NaNs (de 0 * log(0)) em 0
        return np.nan_to_num(res)

    # 6. Cálculo dos três blocos da equação
    # Termo do estado fundido
    termo_ij = np.sum(calcular_termo(n_c_lij, n_lij))

    # Termos dos estados originais (separados)
    termo_i = np.sum(calcular_termo(n_c_li, n_li))
    termo_j = np.sum(calcular_termo(n_c_lj, n_lj))

    # Resultado: log(A / (B*C)) = log(A) - log(B) - log(C)
    ganho_log = termo_ij - (termo_i + termo_j)

    return ganho_log

# Exemplo de chamada:
# ganho = calcular_ganho_fusao(data, 'class', 'latent_L', 'estado1', 'estado2')


def deltaMDL(model1, model2, data, col_classe, latent_L, estado_i, estado_j):
    """
    Calcula o Delta MDL entre duas redes bayesianas.
    """
    N = data.shape[0]
    # kl_same_topology = calculate_kl_same_topology(model1, model2)
    difComplex = difThetaS(model1, model2)
    sumprobs = calcular_ganho_mdl_eq4(
        data, col_classe, latent_L, estado_i, estado_j)
    deltamdl = math.log2(N) * difComplex + sumprobs
    return deltamdl


