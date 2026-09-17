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
from HBNvalidation import (
    bootstrap_gohari_accuracy,
    bootstrap_langseth_accuracy
)
from run_batch import run_batch

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
    test_size = float(input("proporção para teste [padrão 0.20]: ") or 0.20)
    n_bootstrap = int(input("repetições bootstrap [padrão 1000]: ") or 1000)

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

    bootstrap_result = bootstrap_langseth_accuracy(
        csv_file=csv_file,
        class_column=class_column,
        kappa=kappa,
        max_iter=max_iter,
        max_iter_em=max_iter_em,
        test_size=test_size,
        n_bootstrap=n_bootstrap,
        random_state=42,
        debug=True
    )

    arquivos = gerar_relatorios_hbn(
        model=final_model,
        history=history,
        initial_score=initial_score,
        final_score=final_score,
        bootstrap_result=bootstrap_result,
        output_prefix=f"langseth_{base}"
    )

    print("\n✓ Langseth concluído")
    print(f"Base:          {base}")
    print(f"Classe:        {class_column}")
    print(f"Score inicial: {initial_score:.6f}")
    print(f"Score final:   {final_score:.6f}")
    print(f"Ganho:         {final_score - initial_score:+.6f}")
    print(
        f"Acurácia holdout: {bootstrap_result['accuracy_holdout']:.6f}\n"
        f"Bootstrap:        {bootstrap_result['accuracy_mean']:.6f} "
        f"(DP {bootstrap_result['accuracy_std']:.6f})"
    )

    print("\nArquivos gerados:")
    print(f"- {arquivos['resultado_csv']}")
    print(f"- {arquivos['historico_csv']}")
    print(f"- {arquivos['bootstrap_csv']}")
    print(f"- {arquivos['modelo_bif']}")


def rodar_gohari():
    csv_file = pedir_csv()
    class_column = choose_class_column(csv_file)

    group_size = int(input("group_size [padrão 6]: ") or 6)
    comp_ini = int(input("n_components inicial [padrão 2]: ") or 2)
    comp_fim = int(input("n_components final exclusivo [padrão 4]: ") or 4)
    stepmix_n_init = int(input("StepMix n_init [padrão 5]: ") or 5)
    test_size = float(input("proporção para teste [padrão 0.20]: ") or 0.20)
    n_bootstrap = int(input("repetições bootstrap [padrão 1000]: ") or 1000)

    base = get_base_name(csv_file)

    print_section("Executando Gohari / NB-BLCA")

    results, best_result, resumo = gohari_elbow_accuracy(
        csv_file=csv_file,
        class_column=class_column,
        group_size=group_size,
        components_range=range(comp_ini, comp_fim),
        measurement="categorical",
        max_iter_em=10,
        n_init=stepmix_n_init,
        stepmix_max_iter=1000,
        debug=True
    )

    bootstrap_result = bootstrap_gohari_accuracy(
        csv_file=csv_file,
        class_column=class_column,
        group_size=group_size,
        components_range=range(comp_ini, comp_fim),
        measurement="categorical",
        max_iter_em=10,
        test_size=test_size,
        n_bootstrap=n_bootstrap,
        random_state=42,
        n_init=stepmix_n_init,
        stepmix_max_iter=1000,
        debug=True
    )

    arquivos = gerar_relatorios_gohari(
        best_result=best_result,
        resumo=resumo,
        bootstrap_result=bootstrap_result,
        output_prefix=f"gohari_{base}",
        run_config={
            "config_gohari_group_size": group_size,
            "config_components_start": comp_ini,
            "config_components_end_exclusivo": comp_fim,
            "config_gohari_em_iter": 10,
            "config_stepmix_n_init": stepmix_n_init,
            "config_stepmix_max_iter": 1000
        }
    )

    print("\n✓ Gohari concluído")
    print(f"Base:                {base}")
    print(f"Classe:              {class_column}")
    print(f"Melhor n_components: {best_result['n_components']}")
    print(f"Score final:         {best_result['score']:.6f}")
    print(
        f"Acurácia holdout:    {bootstrap_result['accuracy_holdout']:.6f}\n"
        f"Bootstrap:           {bootstrap_result['accuracy_mean']:.6f} "
        f"(DP {bootstrap_result['accuracy_std']:.6f})"
    )

    print("\nArquivos gerados:")
    print(f"- {arquivos['resultado_csv']}")
    print(f"- {arquivos['historico_csv']}")
    print(f"- {arquivos['bootstrap_csv']}")
    print(f"- {arquivos['modelo_bif']}")


def rodar_bases_automaticamente():
    print_section("Execução automática em lote")
    escolha = input(
        "Métodos: 1=ambos, 2=Langseth, 3=Gohari [padrão 1]: "
    ).strip() or "1"
    methods_by_choice = {
        "1": ("langseth", "gohari"),
        "2": ("langseth",),
        "3": ("gohari",)
    }
    if escolha not in methods_by_choice:
        raise ValueError("Escolha de métodos inválida.")

    test_size = float(input("proporção para teste [padrão 0.20]: ") or 0.20)
    n_bootstrap = int(input("repetições bootstrap [padrão 1000]: ") or 1000)
    pattern = input(
        "Padrão das bases [padrão Features_BIN_*.csv]: "
    ).strip() or "Features_BIN_*.csv"

    run_batch(
        bases_dir=os.path.join(os.path.dirname(__file__), "bases"),
        output_dir=os.path.join(os.path.dirname(__file__), "resultados"),
        pattern=pattern,
        methods=methods_by_choice[escolha],
        test_size=test_size,
        bootstrap_repetitions=n_bootstrap,
        resume=True
    )


def main():
    print("\n===== HBN IC Lucas =====")
    print("1 - Rodar Langseth / HNB")
    print("2 - Rodar Gohari / NB-BLCA")
    print("3 - Rodar todas as bases automaticamente")
    print("0 - Sair")

    opcao = input("Escolha uma opção: ").strip()

    if opcao == "1":
        rodar_langseth()
    elif opcao == "2":
        rodar_gohari()
    elif opcao == "3":
        rodar_bases_automaticamente()
    elif opcao == "0":
        print("Saindo...")
    else:
        print("Opção inválida.")


if __name__ == "__main__":
    main()
