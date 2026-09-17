"""Executa Langseth e Gohari automaticamente em várias bases."""

from __future__ import annotations

import argparse
import csv
import os
import sys
import time
import traceback
from contextlib import contextmanager, redirect_stderr, redirect_stdout
from datetime import datetime
from pathlib import Path

import pandas as pd

from HBNgohari import gohari_elbow_accuracy, gerar_relatorios_gohari
from HBNpipeline import learn_hnb_classifier
from HBNreport import gerar_relatorios_hbn
from HBNutils import detect_class_column
from HBNvalidation import (
    GOHARI_BOOTSTRAP_VERSION,
    LANGSETH_BOOTSTRAP_VERSION,
    bootstrap_gohari_accuracy,
    bootstrap_langseth_accuracy
)


PROJECT_DIR = Path(__file__).resolve().parent
PROGRESS_COLUMNS = (
    "timestamp",
    "base",
    "method",
    "status",
    "duration_seconds",
    "accuracy_mean",
    "accuracy_std",
    "message"
)


class Tee:
    """Escreve simultaneamente no terminal e no arquivo de log."""

    def __init__(self, *streams):
        self.streams = streams

    def write(self, text):
        for stream in self.streams:
            stream.write(text)
            stream.flush()
        return len(text)

    def flush(self):
        for stream in self.streams:
            stream.flush()

    def isatty(self):
        return False


@contextmanager
def working_directory(path: Path):
    previous = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(previous)


def result_is_complete(
    result_dir: Path,
    method: str,
    base_name: str,
    test_size: float = 0.20,
    bootstrap_repetitions: int = 1000,
    expected_model_config: dict | None = None
) -> bool:
    result_file = result_dir / f"resultado_{method}_{base_name}.csv"
    if not result_file.exists():
        return False

    try:
        result = pd.read_csv(result_file)
    except Exception:
        return False

    required = {
        "versao_avaliacao",
        "acuracia_holdout",
        "acuracia_media_bootstrap",
        "desvio_padrao_acuracia_bootstrap",
        "numero_repeticoes_bootstrap",
        "proporcao_teste"
    }
    expected_version = {
        "langseth": LANGSETH_BOOTSTRAP_VERSION,
        "gohari": GOHARI_BOOTSTRAP_VERSION
    }[method]

    complete = (
        not result.empty
        and required.issubset(result.columns)
        and result.loc[0, "versao_avaliacao"] == expected_version
    )
    if not complete:
        return False
    bootstrap_matches = (
        int(result.loc[0, "numero_repeticoes_bootstrap"])
        == bootstrap_repetitions
        and abs(float(result.loc[0, "proporcao_teste"]) - test_size)
        < 1e-12
    )
    if not bootstrap_matches:
        return False
    if expected_model_config is None:
        return True
    return all(
        key in result.columns
        and int(result.loc[0, key]) == int(value)
        for key, value in expected_model_config.items()
    )


def append_progress(progress_file: Path, row: dict) -> None:
    write_header = not progress_file.exists()

    with progress_file.open("a", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=PROGRESS_COLUMNS)
        if write_header:
            writer.writeheader()
        writer.writerow({column: row.get(column, "") for column in PROGRESS_COLUMNS})
        stream.flush()


def run_langseth(
    csv_file: Path,
    class_column: str,
    base_name: str,
    result_dir: Path,
    test_size: float,
    bootstrap_repetitions: int,
    kappa: int,
    max_iter: int,
    max_iter_em: int,
    verbose_models: bool
) -> dict:
    final_model, initial_score, final_score, history = learn_hnb_classifier(
        csv_file=str(csv_file),
        class_node=class_column,
        kappa=kappa,
        max_iter=max_iter,
        max_iter_em=max_iter_em,
        debug=verbose_models
    )

    bootstrap_result = bootstrap_langseth_accuracy(
        csv_file=str(csv_file),
        class_column=class_column,
        kappa=kappa,
        max_iter=max_iter,
        max_iter_em=max_iter_em,
        test_size=test_size,
        n_bootstrap=bootstrap_repetitions,
        random_state=42,
        debug=True
    )

    with working_directory(result_dir):
        gerar_relatorios_hbn(
            model=final_model,
            history=history,
            initial_score=initial_score,
            final_score=final_score,
            output_prefix=f"langseth_{base_name}",
            bootstrap_result=bootstrap_result,
            run_config={
                "config_kappa": kappa,
                "config_langseth_max_iter": max_iter,
                "config_langseth_em_iter": max_iter_em
            }
        )

    return bootstrap_result


def run_gohari(
    csv_file: Path,
    class_column: str,
    base_name: str,
    result_dir: Path,
    test_size: float,
    bootstrap_repetitions: int,
    group_size: int,
    components_start: int,
    components_end: int,
    max_iter_em: int,
    stepmix_n_init: int,
    stepmix_max_iter: int,
    verbose_models: bool
) -> dict:
    component_values = range(components_start, components_end)

    results, best_result, summary = gohari_elbow_accuracy(
        csv_file=str(csv_file),
        class_column=class_column,
        group_size=group_size,
        components_range=component_values,
        measurement="categorical",
        max_iter_em=max_iter_em,
        random_state=42,
        n_init=stepmix_n_init,
        stepmix_max_iter=stepmix_max_iter,
        debug=verbose_models,
        save_elbow_csv=False
    )

    bootstrap_result = bootstrap_gohari_accuracy(
        csv_file=str(csv_file),
        class_column=class_column,
        group_size=group_size,
        components_range=component_values,
        measurement="categorical",
        max_iter_em=max_iter_em,
        test_size=test_size,
        n_bootstrap=bootstrap_repetitions,
        random_state=42,
        n_init=stepmix_n_init,
        stepmix_max_iter=stepmix_max_iter,
        debug=True
    )

    with working_directory(result_dir):
        gerar_relatorios_gohari(
            best_result=best_result,
            resumo=summary,
            output_prefix=f"gohari_{base_name}",
            bootstrap_result=bootstrap_result,
            run_config={
                "config_gohari_group_size": group_size,
                "config_components_start": components_start,
                "config_components_end_exclusivo": components_end,
                "config_gohari_em_iter": max_iter_em,
                "config_stepmix_n_init": stepmix_n_init,
                "config_stepmix_max_iter": stepmix_max_iter
            }
        )

    return bootstrap_result


def run_batch(
    bases_dir: str | Path = PROJECT_DIR / "bases",
    output_dir: str | Path = PROJECT_DIR / "resultados",
    pattern: str = "Features_BIN_*.csv",
    methods: tuple[str, ...] = ("langseth", "gohari"),
    test_size: float = 0.20,
    bootstrap_repetitions: int = 1000,
    resume: bool = True,
    kappa: int = 6,
    langseth_max_iter: int = 5,
    langseth_em_iter: int = 20,
    gohari_group_size: int = 6,
    components_start: int = 2,
    components_end: int = 4,
    gohari_em_iter: int = 10,
    stepmix_n_init: int = 5,
    stepmix_max_iter: int = 1000,
    verbose_models: bool = False
) -> dict:
    bases_dir = Path(bases_dir).resolve()
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    progress_file = output_dir / "progresso_execucao_lote.csv"
    csv_files = sorted(bases_dir.glob(pattern))

    if not csv_files:
        raise FileNotFoundError(
            f"Nenhuma base corresponde a '{pattern}' em {bases_dir}."
        )
    if not 0 < test_size < 1:
        raise ValueError("test_size deve estar entre 0 e 1.")
    if bootstrap_repetitions < 2:
        raise ValueError("bootstrap_repetitions deve ser pelo menos 2.")

    invalid_methods = set(methods) - {"langseth", "gohari"}
    if invalid_methods:
        raise ValueError(f"Métodos inválidos: {sorted(invalid_methods)}")
    if components_end <= components_start:
        raise ValueError(
            "components_end deve ser maior que components_start."
        )
    if stepmix_n_init < 1:
        raise ValueError("stepmix_n_init deve ser pelo menos 1.")
    if stepmix_max_iter < 1:
        raise ValueError("stepmix_max_iter deve ser pelo menos 1.")

    totals = {"completed": 0, "failed": 0, "skipped": 0}
    jobs = [(csv_file, method) for csv_file in csv_files for method in methods]

    print("=" * 70)
    print("EXECUÇÃO AUTOMÁTICA EM LOTE")
    print(f"Bases: {len(csv_files)} | Métodos: {', '.join(methods)}")
    print(
        f"Total de tarefas: {len(jobs)} | Bootstrap: "
        f"{bootstrap_repetitions} | Teste: {test_size:.0%}"
    )
    print(f"Resultados: {output_dir}")
    print("=" * 70)

    for job_number, (csv_file, method) in enumerate(jobs, start=1):
        base_name = csv_file.stem
        result_dir = output_dir / base_name
        result_dir.mkdir(parents=True, exist_ok=True)

        print(
            f"\n[{job_number}/{len(jobs)}] "
            f"{base_name} | {method.upper()}"
        )

        if method == "langseth":
            model_config = {
                "config_kappa": kappa,
                "config_langseth_max_iter": langseth_max_iter,
                "config_langseth_em_iter": langseth_em_iter
            }
        else:
            model_config = {
                "config_gohari_group_size": gohari_group_size,
                "config_components_start": components_start,
                "config_components_end_exclusivo": components_end,
                "config_gohari_em_iter": gohari_em_iter,
                "config_stepmix_n_init": stepmix_n_init,
                "config_stepmix_max_iter": stepmix_max_iter
            }

        if resume and result_is_complete(
            result_dir,
            method,
            base_name,
            test_size,
            bootstrap_repetitions,
            model_config
        ):
            print("Resultado completo já encontrado. Tarefa ignorada.")
            totals["skipped"] += 1
            append_progress(progress_file, {
                "timestamp": datetime.now().isoformat(timespec="seconds"),
                "base": base_name,
                "method": method,
                "status": "ignorado",
                "message": "Resultado completo já existente."
            })
            continue

        started_at = time.monotonic()

        try:
            raw_data = pd.read_csv(csv_file)
            raw_data.columns = raw_data.columns.str.strip()
            class_column = detect_class_column(raw_data)

            if method == "langseth":
                bootstrap_result = run_langseth(
                    csv_file=csv_file,
                    class_column=class_column,
                    base_name=base_name,
                    result_dir=result_dir,
                    test_size=test_size,
                    bootstrap_repetitions=bootstrap_repetitions,
                    kappa=kappa,
                    max_iter=langseth_max_iter,
                    max_iter_em=langseth_em_iter,
                    verbose_models=verbose_models
                )
            else:
                bootstrap_result = run_gohari(
                    csv_file=csv_file,
                    class_column=class_column,
                    base_name=base_name,
                    result_dir=result_dir,
                    test_size=test_size,
                    bootstrap_repetitions=bootstrap_repetitions,
                    group_size=gohari_group_size,
                    components_start=components_start,
                    components_end=components_end,
                    max_iter_em=gohari_em_iter,
                    stepmix_n_init=stepmix_n_init,
                    stepmix_max_iter=stepmix_max_iter,
                    verbose_models=verbose_models
                )

            duration = time.monotonic() - started_at
            totals["completed"] += 1
            print(
                f"Concluído em {duration / 60:.1f} min: "
                f"{bootstrap_result['accuracy_mean']:.6f} "
                f"(DP {bootstrap_result['accuracy_std']:.6f})"
            )
            append_progress(progress_file, {
                "timestamp": datetime.now().isoformat(timespec="seconds"),
                "base": base_name,
                "method": method,
                "status": "concluido",
                "duration_seconds": f"{duration:.2f}",
                "accuracy_mean": bootstrap_result["accuracy_mean"],
                "accuracy_std": bootstrap_result["accuracy_std"]
            })

        except Exception as error:
            duration = time.monotonic() - started_at
            totals["failed"] += 1
            print(f"ERRO após {duration / 60:.1f} min: {error}")
            traceback.print_exc()
            append_progress(progress_file, {
                "timestamp": datetime.now().isoformat(timespec="seconds"),
                "base": base_name,
                "method": method,
                "status": "erro",
                "duration_seconds": f"{duration:.2f}",
                "message": f"{type(error).__name__}: {error}"
            })

    print("\n" + "=" * 70)
    print("LOTE FINALIZADO")
    print(
        f"Concluídas: {totals['completed']} | "
        f"Ignoradas: {totals['skipped']} | "
        f"Erros: {totals['failed']}"
    )
    print(f"Acompanhamento: {progress_file}")
    print("=" * 70)
    return totals


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Executa Langseth/Gohari automaticamente em todas as bases."
        )
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        choices=("langseth", "gohari"),
        default=("langseth", "gohari")
    )
    parser.add_argument("--test-size", type=float, default=0.20)
    parser.add_argument("--bootstrap-repetitions", type=int, default=1000)
    parser.add_argument("--pattern", default="Features_BIN_*.csv")
    parser.add_argument("--bases-dir", default=str(PROJECT_DIR / "bases"))
    parser.add_argument("--output-dir", default=str(PROJECT_DIR / "resultados"))
    parser.add_argument(
        "--force",
        action="store_true",
        help="Refaz inclusive resultados completos já existentes."
    )
    parser.add_argument("--kappa", type=int, default=6)
    parser.add_argument("--langseth-max-iter", type=int, default=5)
    parser.add_argument("--langseth-em-iter", type=int, default=20)
    parser.add_argument("--gohari-group-size", type=int, default=6)
    parser.add_argument("--components-start", type=int, default=2)
    parser.add_argument("--components-end", type=int, default=4)
    parser.add_argument("--gohari-em-iter", type=int, default=10)
    parser.add_argument("--stepmix-n-init", type=int, default=5)
    parser.add_argument("--stepmix-max-iter", type=int, default=1000)
    parser.add_argument("--verbose-models", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    log_file = output_dir / "execucao_lote.log"

    with log_file.open("a", encoding="utf-8") as log:
        stdout = Tee(sys.stdout, log)
        stderr = Tee(sys.stderr, log)
        with redirect_stdout(stdout), redirect_stderr(stderr):
            print(
                f"\nInício: "
                f"{datetime.now().isoformat(timespec='seconds')}"
            )
            totals = run_batch(
                bases_dir=args.bases_dir,
                output_dir=output_dir,
                pattern=args.pattern,
                methods=tuple(args.methods),
                test_size=args.test_size,
                bootstrap_repetitions=args.bootstrap_repetitions,
                resume=not args.force,
                kappa=args.kappa,
                langseth_max_iter=args.langseth_max_iter,
                langseth_em_iter=args.langseth_em_iter,
                gohari_group_size=args.gohari_group_size,
                components_start=args.components_start,
                components_end=args.components_end,
                gohari_em_iter=args.gohari_em_iter,
                stepmix_n_init=args.stepmix_n_init,
                stepmix_max_iter=args.stepmix_max_iter,
                verbose_models=args.verbose_models
            )
            print(
                f"Fim: {datetime.now().isoformat(timespec='seconds')}"
            )

    return 1 if totals["failed"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
