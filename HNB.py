from pgmpy.models import DiscreteBayesianNetwork
from pgmpy.factors.discrete import TabularCPD

from HBNBuilder import HBNBuilder

# ============================================================
# Classe HNB
# ============================================================


class HNB(DiscreteBayesianNetwork):
    """
    Extensão da DiscreteBayesianNetwork com métodos para HNB.
    """

    def __init__(self, ebunch=None, latents=None, **kwargs):
        if latents is None:
            latents = set()
        super().__init__(ebunch=ebunch, latents=latents, **kwargs)
        self.latents = set(latents)

    def is_latent(self, node):
        return node in self.latents

# --- Estrutura ---
    def isRegular(self):
        for node in self.nodes():
            if not self.is_latent(node):
                 continue

            cpd_node = self.get_cpds(node)
            if cpd_node is None:
                raise ValueError(f"Nó {node} não possui CPD definida.")

            node_cardinality = int(cpd_node.cardinality[0])

            neighbors = self.get_parents(node) + self.get_children(node)

            if len(neighbors) == 0:
                return False

            neighbor_cards = []

            for neighbor in neighbors:
                cpd_neighbor = self.get_cpds(neighbor)
                if cpd_neighbor is None:
                    raise ValueError(f"Vizinho {neighbor} não possui CPD definida.")

                neighbor_cards.append(int(cpd_neighbor.cardinality[0]))

            product_cards = 1
            for card in neighbor_cards:
                product_cards *= card

            max_card = max(neighbor_cards)
            regular_limit = product_cards / max_card

            has_latent_neighbor = any(self.is_latent(neighbor) for neighbor in neighbors)

            if len(neighbors) == 2 and has_latent_neighbor:
                if not (node_cardinality < regular_limit):
                    return False
            else:
                if not (node_cardinality <= regular_limit):
                    return False

        return True

    def is_tree(self):
        for node in self.nodes():
            if len(self.get_parents(node)) > 1:
                return False
        return True

    def check_model(self):
        super().check_model()
        if not self.is_tree():
            raise ValueError("HNB inválida: estrutura não é uma árvore.")
        if not self.isRegular():
            raise ValueError("HNB inválida: condição de regularidade violada.")
        return True

    # --- Construção da topologia NB via Builder (arquivo CSV) ---

    def create_network_topology_from_data(self, file: str, class_column: str = "class", debug: bool = True):
        builder = HBNBuilder(class_column=class_column, debug=debug)
        bn_nb, df_encoded = builder.build(file)

        self.clear()
        self.add_nodes_from(bn_nb.nodes())
        self.add_edges_from(bn_nb.edges())
        self.add_cpds(*bn_nb.cpds)

        if debug:
            print("[HNB] Topologia NB carregada a partir do CSV.")
            print("  Nós:", list(self.nodes()))
            print("  Arestas:", list(self.edges()))
            print("  CPDs:", [cpd.variable for cpd in self.cpds])

        return df_encoded

# --- Operações ---

    def can_insert_latent(self, nomePai, listaFilhos):
        for nome_node in listaFilhos:
            parents = self.get_parents(nome_node)
            if len(parents) != 1:
                return False
            if parents[0] != nomePai:
                return False
        return True

    def change_bn_topology(self, fromnode, toList, hnode, hnode_cardinality=2, debug=True):
        fromnode_cpd = self.get_cpds(fromnode)
        if fromnode_cpd is None:
            raise ValueError(f"Nó pai '{fromnode}' não possui CPD definida.")

        # garante condição de inserção
        if not self.can_insert_latent(fromnode, toList):
            raise ValueError(
                "Não é possível inserir latente aqui: algum filho não tem pai único = fromnode.")

        if hnode not in self.nodes():
            self.add_node(hnode)

        if (fromnode, hnode) not in self.edges():
            self.add_edge(fromnode, hnode)

        parent_card = fromnode_cpd.variable_card
        values_list = [
            [1.0 / hnode_cardinality] * parent_card
            for _ in range(hnode_cardinality)
        ]

        hnode_cpd = TabularCPD(
            variable=hnode,
            variable_card=hnode_cardinality,
            values=values_list,
            evidence=[fromnode],
            evidence_card=[parent_card]
        )

        self.add_cpds(hnode_cpd)
        self.latents.add(hnode)

        if debug:
            print(
                f"[change_bn_topology] Nó latente '{hnode}' inserido entre '{fromnode}' e {toList}")

        for targetAtr in toList:
            original_cpd = self.get_cpds(targetAtr)
            if original_cpd is None:
                if debug:
                    print(
                        f"[change_bn_topology] Aviso: '{targetAtr}' sem CPD. Pulando.")
                continue

            varCardinality = original_cpd.variable_card

            cpt = [[1.0 / varCardinality] *
                   hnode_cardinality for _ in range(varCardinality)]
            newcpd = TabularCPD(
                variable=targetAtr,
                variable_card=varCardinality,
                values=cpt,
                evidence=[hnode],
                evidence_card=[hnode_cardinality]
            )

            if (fromnode, targetAtr) in self.edges():
                self.remove_edge(fromnode, targetAtr)

            self.remove_cpds(original_cpd)

            if (hnode, targetAtr) not in self.edges():
                self.add_edge(hnode, targetAtr)

            self.add_cpds(newcpd)

        return self
