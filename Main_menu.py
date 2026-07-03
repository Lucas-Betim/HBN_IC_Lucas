import os

from HBNpipeline import learn_hnb_classifier
from HBNreport import gerar_relatorios_hbn

from HBNgohari import (
    gohari_elbow_accuracy,
    gerar_relatorios_gohari
)


def pedir_csv():
    caminho = input("Digite o nome ou caminho do CSV: ").strip()

    if not os.path.isabs(caminho):
        caminho = os.path.join(os.path.dirname(__file__), caminho)

    if not os.path.exists(caminho):
        raise FileNotFoundError(f"CSV não encontrado: {caminho}")

    return caminho


def nome_base(csv_file):
    return os.path.splitext(os.path.basename(csv_file))[0]


def rodar_langseth():
    csv_file = pedir_csv()

    kappa = int(input("kappa [padrão 5]: ") or 5)
    max_iter = int(input("max_iter Langseth [padrão 5]: ") or 5)

    base = nome_base(csv_file)

    final_model, initial_score, final_score, history = learn_hnb_classifier(
        csv_file=csv_file,
        class_node="class",
        kappa=kappa,
        max_iter=max_iter,
        debug=True
    )

    arquivos = gerar_relatorios_hbn(
        model=final_model,
        history=history,
        initial_score=initial_score,
        final_score=final_score,
        output_prefix=f"hbn_{base}"
    )

    print("\n✓ Langseth concluído")
    print(f"Score inicial: {initial_score:.6f}")
    print(f"Score final:   {final_score:.6f}")
    print(f"Ganho:         {final_score - initial_score:+.6f}")
    print("\nArquivos gerados:")
    print(arquivos)


def rodar_gohari():
    csv_file = pedir_csv()

    group_size = int(input("group_size [padrão 2]: ") or 2)
    comp_ini = int(input("n_components inicial [padrão 2]: ") or 2)
    comp_fim = int(input("n_components final exclusivo [padrão 6]: ") or 6)
    max_iter_em = int(input("max_iter EM [padrão 10]: ") or 10)

    base = nome_base(csv_file)

    results, best_result, resumo = gohari_elbow_accuracy(
        csv_file=csv_file,
        class_column="class",
        group_size=group_size,
        components_range=range(comp_ini, comp_fim),
        measurement="categorical",
        max_iter_em=max_iter_em,
        debug=True
    )

    arquivos = gerar_relatorios_gohari(
        best_result=best_result,
        resumo=resumo,
        output_prefix=f"gohari_{base}"
    )

    print("\n✓ Gohari concluído")
    print(f"Melhor n_components: {best_result['n_components']}")
    print(f"Score final:         {best_result['score']:.6f}")
    print("\nArquivos gerados:")
    print(arquivos)


def main():
    print("\n===== HBN IC Lucas =====")
    print("1 - Rodar Langseth / HNB")
    print("2 - Rodar Gohari / NB-BLCA")
    print("0 - Sair")

    opcao = input("Escolha uma opção: ").strip()

    if opcao == "1":
        rodar_langseth()
    elif opcao == "2":
        rodar_gohari()
    elif opcao == "0":
        print("Saindo...")
    else:
        print("Opção inválida.")


if __name__ == "__main__":
    main()