"""Executa validação cruzada estratificada dos modelos HBN em todas as bases."""

from __future__ import annotations

import argparse
import csv
import sys
import time
import traceback
from contextlib import redirect_stderr, redirect_stdout
from datetime import datetime
from pathlib import Path

import pandas as pd

from HBNutils import detect_class_column
from HBNvalidation import (
    cross_validate_gohari_accuracy,
    cross_validate_langseth_accuracy,
)


PROJECT_DIR = Path(__file__).resolve().parent
SUMMARY_COLUMNS = (
    "timestamp", "base", "method", "status", "accuracy_mean",
    "accuracy_std", "accuracy_pooled", "n_splits", "split_seed",
    "duration_seconds", "message"
)


class Tee:
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


def load_summary(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=SUMMARY_COLUMNS)
    return pd.read_csv(path, encoding="utf-8-sig")


def append_summary(path: Path, row: dict) -> None:
    write_header = not path.exists()
    with path.open("a", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=SUMMARY_COLUMNS)
        if write_header:
            writer.writeheader()
        writer.writerow({column: row.get(column, "") for column in SUMMARY_COLUMNS})
        stream.flush()


def is_complete(summary: pd.DataFrame, base: str, method: str, n_splits: int) -> bool:
    if summary.empty:
        return False
    matches = summary[
        (summary["base"] == base)
        & (summary["method"] == method)
        & (summary["status"] == "concluido")
        & (summary["n_splits"].astype(str) == str(n_splits))
    ]
    return not matches.empty


def run(args) -> dict:
    bases_dir = Path(args.bases_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_file = output_dir / "resumo_cv_10fold.csv"
    csv_files = sorted(bases_dir.glob(args.pattern))
    if len(csv_files) != 16:
        raise ValueError(f"Esperava 16 bases, encontrei {len(csv_files)}.")

    totals = {"completed": 0, "skipped": 0, "failed": 0}
    jobs = [(csv_file, method) for method in args.methods for csv_file in csv_files]
    print(f"Bases: {len(csv_files)} | tarefas: {len(jobs)} | folds: {args.n_splits}")
    print(f"Saída: {output_dir}")

    for job_number, (csv_file, method) in enumerate(jobs, start=1):
        base_name = csv_file.stem
        summary = load_summary(summary_file)
        print(f"\n[{job_number}/{len(jobs)}] {base_name} | {method.upper()}")
        if not args.force and is_complete(summary, base_name, method, args.n_splits):
            print("Resultado completo já existente. Ignorando.")
            totals["skipped"] += 1
            continue

        raw_data = pd.read_csv(csv_file)
        raw_data.columns = raw_data.columns.str.strip()
        class_column = detect_class_column(raw_data)
        method_dir = output_dir / base_name
        method_dir.mkdir(parents=True, exist_ok=True)
        folds_file = method_dir / f"folds_{method}_{base_name}.csv"
        started = time.monotonic()

        try:
            if method == "gohari":
                result = cross_validate_gohari_accuracy(
                    csv_file=str(csv_file),
                    class_column=class_column,
                    n_splits=args.n_splits,
                    group_size=6,
                    components_range=range(2, 4),
                    max_iter_em=10,
                    random_state=args.random_state,
                    n_init=5,
                    stepmix_max_iter=1000,
                    debug=True,
                    results_file=str(folds_file)
                )
            else:
                result = cross_validate_langseth_accuracy(
                    csv_file=str(csv_file),
                    class_column=class_column,
                    n_splits=args.n_splits,
                    kappa=5,
                    max_iter=5,
                    max_iter_em=20,
                    random_state=args.random_state,
                    debug=True,
                    results_file=str(folds_file)
                )

            duration = time.monotonic() - started
            append_summary(summary_file, {
                "timestamp": datetime.now().isoformat(timespec="seconds"),
                "base": base_name,
                "method": method,
                "status": "concluido",
                "accuracy_mean": result["accuracy_mean"],
                "accuracy_std": result["accuracy_std"],
                "accuracy_pooled": result["accuracy_pooled"],
                "n_splits": result["n_splits"],
                "split_seed": result["split_random_state"],
                "duration_seconds": f"{duration:.2f}"
            })
            totals["completed"] += 1
            print(
                f"Concluído: {result['accuracy_mean']:.6f} "
                f"(DP {result['accuracy_std']:.6f}) | {duration / 60:.1f} min"
            )
        except Exception as error:
            duration = time.monotonic() - started
            totals["failed"] += 1
            append_summary(summary_file, {
                "timestamp": datetime.now().isoformat(timespec="seconds"),
                "base": base_name,
                "method": method,
                "status": "erro",
                "n_splits": args.n_splits,
                "duration_seconds": f"{duration:.2f}",
                "message": f"{type(error).__name__}: {error}"
            })
            print(f"ERRO: {error}")
            traceback.print_exc()

    print(f"\nTotais: {totals}")
    return totals


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bases-dir", default=str(PROJECT_DIR / "bases"))
    parser.add_argument("--output-dir", default=str(PROJECT_DIR / "resultados_cv_10fold"))
    parser.add_argument("--pattern", default="Features_BIN_*.csv")
    parser.add_argument("--methods", nargs="+", choices=("gohari", "langseth"), default=("gohari", "langseth"))
    parser.add_argument("--n-splits", type=int, default=10)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--force", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    log_file = output_dir / "execucao_cv_10fold.log"
    with log_file.open("a", encoding="utf-8") as log:
        with redirect_stdout(Tee(sys.stdout, log)), redirect_stderr(Tee(sys.stderr, log)):
            print(f"Início: {datetime.now().isoformat(timespec='seconds')}")
            totals = run(args)
            print(f"Fim: {datetime.now().isoformat(timespec='seconds')}")
    return 1 if totals["failed"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
