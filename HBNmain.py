import os

from HBNreport import gerar_relatorios_hbn

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


CSV_FILE = os.path.join(os.path.dirname(__file__), "mushrooms.csv")

# =========================
# Main
# =========================

if __name__ == "__main__":

    if not os.path.exists(CSV_FILE):
        print(f"[ERRO] CSV não encontrado: {CSV_FILE}")
        raise SystemExit(1)

    final_model, final_score, history = learn_hnb_classifier(
        csv_file=CSV_FILE,
        class_node="class",
        kappa=2,
        max_iter=2,
        debug=True  
    )

    initial_score = history[0].get("current_score", None)

    if initial_score is None:
        initial_score = final_score

    arquivos = gerar_relatorios_hbn(
        model=final_model,
        history=history,
        initial_score=initial_score,
        final_score=final_score,
        output_prefix="hbn"
    )

    print("\n✓ Execução concluída")
    print(f"Score inicial: {initial_score:.6f}")
    print(f"Score final:   {final_score:.6f}")

    print("\nLatentes finais:")
    for latente in final_model.latents:
        print(f"- {latente}")

    print("\nArquivos gerados:")
    print(f"- {arquivos['resultado_csv']}")
    print(f"- {arquivos['historico_csv']}")
    print(f"- {arquivos['modelo_bif']}")

    print("\n===== RESULTADO FINAL =====")
    print("Score final:", final_score)
    print("Latentes finais:", list(final_model.latents))

    print("\n===== HISTÓRICO =====")
    for h in history:
        print(h)

    predicoes = inferir_classe_csv(
        modelo=final_model,
        arquivo_csv=CSV_FILE,
        class_column="class"
    )

    print("\n===== INFERÊNCIA FINAL =====")
    print(predicoes[:20])
    print("Total de predições:", len(predicoes))