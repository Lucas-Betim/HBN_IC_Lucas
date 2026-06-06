import os

from hbnCSVTester import inferir_classe_csv

from HBNalgorithm import create_overlapping_subsets

from HBNpipeline import (
    build_and_train,
    get_candidate_latents_from_original_nb
)

from TESTShbn import (
    TestRunner,
    test_1_positive_case,
    test_2_tree_violation_two_parents,
    test_3_cycle_violation,
    test_4_missing_cpd,
    test_5_can_insert_latent
)

from HBNpipeline import (
    build_and_train,
    get_candidate_latents_from_original_nb,
    learn_hnb_classifier
)


CSV_FILE = os.path.join(os.path.dirname(__file__), "teste.csv")

# =========================
# Main
# =========================

if __name__ == "__main__":
    runner = TestRunner()

    if not os.path.exists(CSV_FILE):
        print(f"[ERRO] CSV não encontrado: {CSV_FILE}")
        raise SystemExit(1)

    test_1_positive_case(runner, CSV_FILE)
    test_2_tree_violation_two_parents(runner, CSV_FILE)
    test_3_cycle_violation(runner, CSV_FILE)
    test_4_missing_cpd(runner, CSV_FILE)
    test_5_can_insert_latent(runner, CSV_FILE)

    model, df_final = build_and_train(CSV_FILE)
    subsets, subset_indices = create_overlapping_subsets(
    df_final.fillna(0).astype(int),
    kappa=5,
    random_state=42,
    debug=False
    )

    candidate_latents, best_candidate = get_candidate_latents_from_original_nb(CSV_FILE)

    predicoes = inferir_classe_csv(
        modelo=model,
        arquivo_csv=CSV_FILE,
        class_column="class"
    )

    print("\n===== INFERÊNCIA =====")
    print(predicoes)

    print("\n===== TESTE PASSO 2 =====")
    print("Número de subsets:", len(subsets))

    for i, s in enumerate(subsets):
        print(f"D^{i+1} tamanho:", len(s))

    intersec = len(subset_indices[0].intersection(subset_indices[1]))
    print("Interseção D1 ∩ D2:", intersec)
    print("D1 == D2 ?", subsets[0].equals(subsets[1]))

    print("\n===== TESTE PASSO 3.1 + 3.2 =====")

    if len(candidate_latents) == 0:
        print("Nenhum candidato encontrado.")
    else:
        for cand in candidate_latents:
            print({
                "subset_id": cand.get("subset_id"),
                "latent_name": cand["latent_name"],
                "pair": cand["pair"],
                "q_value": cand["q_value"],
                "score": cand["score"],
                "initial_cardinality": cand["initial_cardinality"],
                "latent_cardinality": cand["latent_cardinality"],
                "collapse_history": cand["collapse_history"]
            })

    print("\n===== MELHOR CANDIDATO H' - PASSO 3.b =====")

    if best_candidate is None:
        print("Nenhum melhor candidato encontrado.")
    else:
        print({
            "subset_id": best_candidate.get("subset_id"),
            "latent_name": best_candidate["latent_name"],
            "pair": best_candidate["pair"],
            "score": best_candidate["score"],
            "initial_cardinality": best_candidate["initial_cardinality"],
            "latent_cardinality": best_candidate["latent_cardinality"],
            "collapse_history": best_candidate["collapse_history"]
        })
    print("\n===== ALGORITMO HNB COMPLETO: 3.c + LOOP =====")

    final_model, final_score, history = learn_hnb_classifier(
        csv_file=CSV_FILE,
        class_node="class",
        kappa=5,
        debug=True
    )

    print("\nScore final:", final_score)
    print("Histórico:")
    for h in history:
        print(h)

    runner.summary()
