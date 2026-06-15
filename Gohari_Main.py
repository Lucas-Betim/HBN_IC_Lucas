import os

from HBNgohari import (
    gohari_elbow_accuracy,
    gerar_relatorios_gohari
)

CSV_FILE = os.path.join(
    os.path.dirname(__file__),
    "mushrooms.csv"
)

if __name__ == "__main__":

    results, best_result, resumo = gohari_elbow_accuracy(
        csv_file=CSV_FILE,
        class_column="class",
        group_size=2,
        components_range=range(2, 6),
        measurement="categorical",
        max_iter_em=10,
        debug=True
    )

    print("\n===== RESUMO GOHARI ELBOW =====")
    print(resumo)

    print("\n===== MELHOR RESULTADO GOHARI =====")
    print("n_components:", best_result["n_components"])
    print("group_size:", best_result["group_size"])
    print("score:", best_result["score"])
    print("num_latents:", best_result["num_latents"])
    print("latentes:", best_result["latent_nodes"])

    arquivos = gerar_relatorios_gohari(
    best_result=best_result,
    resumo=resumo,
    output_prefix="gohari"
    )

    print("\n✓ Execução Gohari concluída")
    print(f"Melhor n_components: {best_result['n_components']}")
    print(f"Score final:         {best_result['score']:.6f}")

    print("\nLatentes finais:")
    for latente in best_result["latent_nodes"]:
        print(f"- {latente}")

    print("\nArquivos gerados:")
    print(f"- {arquivos['resultado_csv']}")
    print(f"- {arquivos['historico_csv']}")
    print(f"- {arquivos['modelo_bif']}")

