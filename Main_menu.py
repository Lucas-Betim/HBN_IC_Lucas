import os

from HBNpipeline import learn_hnb_classifier
from HBNreport import gerar_relatorios_hbn

from HBNgohari import (
    gohari_elbow_accuracy,
    gerar_relatorios_gohari
)

from HBNutils import (
    choose_class_column,
    get_base_name,
    print_section,
    escolher_csv_em_pasta
)

def pedir_csv():
    return escolher_csv_em_pasta(default_folder="bases")


def nome_base(csv_file):
    return os.path.splitext(os.path.basename(csv_file))[0]


def rodar_langseth():
    csv_file = pedir_csv()
    class_column = choose_class_column(csv_file)

    kappa = int(input("kappa [padrão 5]: ") or 5)
    max_iter = int(input("max_iter Langseth [padrão 5]: ") or 5)
    max_iter_em = int(input("max_iter EM [padrão 20]: ") or 20)

    base = get_base_name(csv_file)

    print_section("Executando Langseth / HNB")

    final_model, initial_score, final_score, history = learn_hnb_classifier(
        csv_file=csv_file,
        class_node=class_column,
        kappa=kappa,
        max_iter=max_iter,
        max_iter_em=max_iter_em,
        debug=True
    )

    arquivos = gerar_relatorios_hbn(
        model=final_model,
        history=history,
        initial_score=initial_score,
        final_score=final_score,
        output_prefix=f"langseth_{base}"
    )

    print("\n✓ Langseth concluído")
    print(f"Base:          {base}")
    print(f"Classe:        {class_column}")
    print(f"Score inicial: {initial_score:.6f}")
    print(f"Score final:   {final_score:.6f}")
    print(f"Ganho:         {final_score - initial_score:+.6f}")

    print("\nArquivos gerados:")
    print(f"- {arquivos['resultado_csv']}")
    print(f"- {arquivos['historico_csv']}")
    print(f"- {arquivos['modelo_bif']}")


def rodar_gohari():
    csv_file = pedir_csv()
    class_column = choose_class_column(csv_file)

    group_size = int(input("group_size [padrão 2]: ") or 2)
    comp_ini = int(input("n_components inicial [padrão 2]: ") or 2)
    comp_fim = int(input("n_components final exclusivo [padrão 3]: ") or 3)
    max_iter_em = int(input("max_iter EM [padrão 10]: ") or 10)

    base = get_base_name(csv_file)

    print_section("Executando Gohari / NB-BLCA")

    results, best_result, resumo = gohari_elbow_accuracy(
        csv_file=csv_file,
        class_column=class_column,
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
    print(f"Base:                {base}")
    print(f"Classe:              {class_column}")
    print(f"Melhor n_components: {best_result['n_components']}")
    print(f"Score final:         {best_result['score']:.6f}")

    print("\nArquivos gerados:")
    print(f"- {arquivos['resultado_csv']}")
    print(f"- {arquivos['historico_csv']}")
    print(f"- {arquivos['modelo_bif']}")


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