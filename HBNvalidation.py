import os
import tempfile

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split

from HBNBuilder import fit_state_mappings, drop_non_feature_columns
from HBNgohari import gohari_elbow_accuracy
from HBNpipeline import (
    learn_hnb_classifier,
    predict_hnb_model
)


LANGSETH_BOOTSTRAP_VERSION = "4_holdout_bootstrap_strict"
GOHARI_BOOTSTRAP_VERSION = "3_holdout_bootstrap_strict"
LANGSETH_CV_VERSION = "1_stratified_10fold_strict"
GOHARI_CV_VERSION = "1_stratified_10fold_strict"


def _state_safe_holdout(
    df: pd.DataFrame,
    class_column: str,
    test_size: float,
    random_state: int,
    max_attempts: int = 200
) -> tuple[pd.DataFrame, pd.DataFrame, int]:
    """Cria holdout estratificado sem estados exclusivos no teste."""
    if not 0 < test_size < 1:
        raise ValueError("test_size deve estar entre 0 e 1.")

    usable = drop_non_feature_columns(df)
    for attempt in range(max_attempts):
        seed = random_state + attempt
        train_idx, test_idx = train_test_split(
            np.arange(len(df)),
            test_size=test_size,
            stratify=df[class_column],
            random_state=seed
        )
        train = usable.iloc[train_idx]
        test = usable.iloc[test_idx]
        known_states = all(
            set(test[col].dropna().unique()).issubset(
                set(train[col].dropna().unique())
            )
            for col in usable.columns
        )
        if known_states:
            return (
                df.iloc[train_idx].reset_index(drop=True),
                df.iloc[test_idx].reset_index(drop=True),
                seed
            )

    raise ValueError(
        "Não foi possível criar um holdout sem estados exclusivos no teste. "
        "A base pode conter atributos com estados muito raros."
    )


def _state_safe_stratified_folds(
    df: pd.DataFrame,
    class_column: str,
    n_splits: int,
    random_state: int,
    max_attempts: int = 200
) -> tuple[list[tuple[np.ndarray, np.ndarray]], int]:
    """Cria folds estratificados sem estados exclusivos nos testes."""
    if n_splits < 2:
        raise ValueError("n_splits deve ser pelo menos 2.")

    smallest_class = int(df[class_column].value_counts().min())
    if smallest_class < n_splits:
        raise ValueError(
            f"Validação cruzada com {n_splits} folds impossível: "
            f"a menor classe possui {smallest_class} amostras."
        )

    usable = drop_non_feature_columns(df)
    for attempt in range(max_attempts):
        seed = random_state + attempt
        splitter = StratifiedKFold(
            n_splits=n_splits,
            shuffle=True,
            random_state=seed
        )
        folds = list(splitter.split(df, df[class_column]))
        all_folds_are_safe = True

        for train_idx, test_idx in folds:
            train = usable.iloc[train_idx]
            test = usable.iloc[test_idx]
            known_states = all(
                set(test[col].dropna().unique()).issubset(
                    set(train[col].dropna().unique())
                )
                for col in usable.columns
            )
            if not known_states:
                all_folds_are_safe = False
                break

        if all_folds_are_safe:
            return folds, seed

    raise ValueError(
        f"Não foi possível criar {n_splits} folds sem estados exclusivos "
        "nos testes. A base pode conter estados observados apenas uma vez."
    )


def _summarize_cross_validation(
    fold_results: pd.DataFrame,
    evaluation_version: str,
    split_random_state: int
) -> dict:
    total_correct = int(fold_results["Correct"].sum())
    total_test = int(fold_results["NTest"].sum())
    return {
        "protocol": "stratified_kfold",
        "evaluation_version": evaluation_version,
        "accuracy_mean": float(fold_results["Accuracy"].mean()),
        "accuracy_std": float(fold_results["Accuracy"].std(ddof=1)),
        "accuracy_pooled": float(total_correct / total_test),
        "n_splits": int(len(fold_results)),
        "n_test_total": total_test,
        "split_random_state": int(split_random_state),
        "fold_results": fold_results
    }


def _load_partial_cv_results(
    results_file: str | None,
    evaluation_version: str,
    n_splits: int,
    split_random_state: int
) -> pd.DataFrame:
    if results_file is None or not os.path.exists(results_file):
        return pd.DataFrame()

    existing = pd.read_csv(results_file)
    required = {
        "EvaluationVersion", "SplitSeed", "NSplits", "Fold",
        "Accuracy", "Correct", "NTrain", "NTest"
    }
    if not required.issubset(existing.columns):
        raise ValueError(f"Arquivo parcial incompatível: {results_file}")
    if not (
        (existing["EvaluationVersion"] == evaluation_version).all()
        and (existing["SplitSeed"] == split_random_state).all()
        and (existing["NSplits"] == n_splits).all()
    ):
        raise ValueError(
            "O arquivo parcial pertence a outra configuração de validação: "
            f"{results_file}"
        )
    return existing


def _save_partial_cv_results(
    rows: list[dict],
    results_file: str | None
) -> None:
    if results_file is None:
        return
    output_dir = os.path.dirname(os.path.abspath(results_file))
    os.makedirs(output_dir, exist_ok=True)
    pd.DataFrame(rows).sort_values("Fold").to_csv(results_file, index=False)


def _paired_bootstrap_results(
    y_true: pd.Series,
    y_pred: pd.Series,
    n_bootstrap: int,
    random_state: int
) -> pd.DataFrame:
    """Reamostra, em conjunto, cada par (classe real, predição)."""
    if n_bootstrap < 2:
        raise ValueError("n_bootstrap deve ser pelo menos 2.")

    true_values = np.asarray(y_true)
    pred_values = np.asarray(y_pred)
    rng = np.random.default_rng(random_state)
    all_indices = np.arange(len(true_values))
    accuracies = np.empty(n_bootstrap, dtype=float)

    for repetition in range(n_bootstrap):
        sampled = rng.choice(
            all_indices, size=len(all_indices), replace=True
        )
        accuracies[repetition] = np.mean(
            true_values[sampled] == pred_values[sampled]
        )

    return pd.DataFrame({
        "Bootstrap": np.arange(1, n_bootstrap + 1),
        "Accuracy": accuracies
    })


def _summarize_bootstrap(
    y_true: pd.Series,
    y_pred: pd.Series,
    bootstrap_results: pd.DataFrame,
    evaluation_version: str,
    test_size: float,
    split_random_state: int
) -> dict:
    return {
        "protocol": "bootstrap_holdout",
        "evaluation_version": evaluation_version,
        "accuracy_holdout": float(np.mean(np.asarray(y_true) == np.asarray(y_pred))),
        "accuracy_mean": float(bootstrap_results["Accuracy"].mean()),
        "accuracy_std": float(bootstrap_results["Accuracy"].std(ddof=1)),
        "n_bootstrap": int(len(bootstrap_results)),
        "test_size": float(test_size),
        "n_test": int(len(y_true)),
        "split_random_state": int(split_random_state),
        "bootstrap_results": bootstrap_results
    }


def _load_and_validate_data(
    csv_file: str,
    class_column: str
) -> pd.DataFrame:
    df = pd.read_csv(csv_file)
    df.columns = df.columns.str.strip()

    if class_column not in df.columns:
        raise ValueError(
            f"Coluna de classe '{class_column}' não encontrada no CSV."
        )

    smallest_class = int(df[class_column].value_counts().min())
    if smallest_class < 2:
        raise ValueError(
            "Holdout estratificado impossível: a menor classe "
            f"possui apenas {smallest_class} amostras."
        )

    return df


def bootstrap_langseth_accuracy(
    csv_file: str,
    class_column: str = "class",
    kappa: int = 5,
    max_iter: int | None = None,
    max_iter_em: int = 20,
    test_size: float = 0.20,
    n_bootstrap: int = 1000,
    random_state: int = 42,
    debug: bool = True
) -> dict:
    """Treina Langseth uma vez no holdout e reamostra suas predições."""
    df = _load_and_validate_data(csv_file, class_column)
    train_df, test_df, split_seed = _state_safe_holdout(
        df, class_column, test_size, random_state
    )

    with tempfile.TemporaryDirectory(prefix="langseth_bootstrap_") as temp_dir:
        train_file = os.path.join(temp_dir, "train.csv")
        test_file = os.path.join(temp_dir, "test.csv")
        train_df.to_csv(train_file, index=False)
        test_df.to_csv(test_file, index=False)
        if debug:
            print(
                f"[Langseth-bootstrap] Treino={len(train_df)} | "
                f"teste={len(test_df)} | repetições={n_bootstrap}"
            )
        model, _, _, _ = learn_hnb_classifier(
            csv_file=train_file,
            class_node=class_column,
            kappa=kappa,
            max_iter=max_iter,
            max_iter_em=max_iter_em,
            debug=False,
            progress_label="Langseth-bootstrap" if debug else None
        )
        y_true, y_pred = predict_hnb_model(
            model=model,
            csv_file=test_file,
            class_node=class_column,
            state_mappings=fit_state_mappings(train_df),
            require_all_rows=True
        )

    results = _paired_bootstrap_results(
        y_true, y_pred, n_bootstrap, random_state
    )
    return _summarize_bootstrap(
        y_true, y_pred, results, LANGSETH_BOOTSTRAP_VERSION,
        test_size, split_seed
    )


def bootstrap_gohari_accuracy(
    csv_file: str,
    class_column: str = "class",
    group_size: int = 2,
    components_range=range(2, 7),
    measurement: str = "categorical",
    max_iter_em: int = 10,
    test_size: float = 0.20,
    n_bootstrap: int = 1000,
    random_state: int = 42,
    n_init: int = 1,
    stepmix_max_iter: int = 1000,
    debug: bool = True
) -> dict:
    """Treina Gohari uma vez no holdout e reamostra suas predições."""
    df = _load_and_validate_data(csv_file, class_column)
    train_df, test_df, split_seed = _state_safe_holdout(
        df, class_column, test_size, random_state
    )

    with tempfile.TemporaryDirectory(prefix="gohari_bootstrap_") as temp_dir:
        train_file = os.path.join(temp_dir, "train.csv")
        test_file = os.path.join(temp_dir, "test.csv")
        train_df.to_csv(train_file, index=False)
        test_df.to_csv(test_file, index=False)
        if debug:
            print(
                f"[Gohari-bootstrap] Treino={len(train_df)} | "
                f"teste={len(test_df)} | repetições={n_bootstrap}"
            )
        _, best_result, _ = gohari_elbow_accuracy(
            csv_file=train_file,
            class_column=class_column,
            group_size=group_size,
            components_range=components_range,
            measurement=measurement,
            max_iter_em=max_iter_em,
            random_state=random_state,
            n_init=n_init,
            stepmix_max_iter=stepmix_max_iter,
            debug=False,
            save_elbow_csv=False
        )
        y_true, y_pred = predict_hnb_model(
            model=best_result["model"],
            csv_file=test_file,
            class_node=class_column,
            state_mappings=fit_state_mappings(train_df),
            require_all_rows=True
        )

    results = _paired_bootstrap_results(
        y_true, y_pred, n_bootstrap, random_state
    )
    summary = _summarize_bootstrap(
        y_true, y_pred, results, GOHARI_BOOTSTRAP_VERSION,
        test_size, split_seed
    )
    summary.update({
        "selected_n_components": int(best_result["n_components"]),
        "group_size": int(group_size),
        "stepmix_n_init": int(n_init),
        "stepmix_max_iter": int(stepmix_max_iter)
    })
    return summary


def cross_validate_langseth_accuracy(
    csv_file: str,
    class_column: str = "class",
    n_splits: int = 10,
    kappa: int = 5,
    max_iter: int = 5,
    max_iter_em: int = 20,
    random_state: int = 42,
    debug: bool = True,
    results_file: str | None = None
) -> dict:
    """Retreina e avalia Langseth em cada fold estratificado."""
    df = _load_and_validate_data(csv_file, class_column)
    folds, split_seed = _state_safe_stratified_folds(
        df, class_column, n_splits, random_state
    )
    existing = _load_partial_cv_results(
        results_file, LANGSETH_CV_VERSION, n_splits, split_seed
    )
    rows = existing.to_dict("records")
    completed_folds = set(existing["Fold"].astype(int)) if not existing.empty else set()

    with tempfile.TemporaryDirectory(prefix="langseth_cv_") as temp_dir:
        train_file = os.path.join(temp_dir, "train.csv")
        test_file = os.path.join(temp_dir, "test.csv")

        for fold_number, (train_idx, test_idx) in enumerate(folds, start=1):
            if fold_number in completed_folds:
                if debug:
                    print(f"[Langseth-CV] Fold {fold_number}/{n_splits} já concluído.")
                continue

            train_df = df.iloc[train_idx].reset_index(drop=True)
            test_df = df.iloc[test_idx].reset_index(drop=True)
            train_df.to_csv(train_file, index=False)
            test_df.to_csv(test_file, index=False)
            if debug:
                print(
                    f"[Langseth-CV] Fold {fold_number}/{n_splits} | "
                    f"treino={len(train_df)} | teste={len(test_df)}"
                )

            model, _, _, _ = learn_hnb_classifier(
                csv_file=train_file,
                class_node=class_column,
                kappa=kappa,
                max_iter=max_iter,
                max_iter_em=max_iter_em,
                debug=False,
                progress_label=(
                    f"Langseth-CV fold {fold_number}" if debug else None
                )
            )
            y_true, y_pred = predict_hnb_model(
                model=model,
                csv_file=test_file,
                class_node=class_column,
                state_mappings=fit_state_mappings(train_df),
                require_all_rows=True
            )
            correct = int(np.sum(np.asarray(y_true) == np.asarray(y_pred)))
            rows.append({
                "EvaluationVersion": LANGSETH_CV_VERSION,
                "Protocol": "stratified_kfold",
                "SplitSeed": split_seed,
                "NSplits": n_splits,
                "Fold": fold_number,
                "Accuracy": float(correct / len(y_true)),
                "Correct": correct,
                "NTrain": len(train_df),
                "NTest": len(test_df),
                "TrainIndices": " ".join(map(str, train_idx.tolist())),
                "TestIndices": " ".join(map(str, test_idx.tolist())),
                "Kappa": kappa,
                "MaxIter": max_iter,
                "EMIter": max_iter_em
            })
            _save_partial_cv_results(rows, results_file)

    fold_results = pd.DataFrame(rows).sort_values("Fold").reset_index(drop=True)
    if len(fold_results) != n_splits:
        raise ValueError(
            f"Esperava {n_splits} resultados Langseth, encontrei {len(fold_results)}."
        )
    return _summarize_cross_validation(
        fold_results, LANGSETH_CV_VERSION, split_seed
    )


def cross_validate_gohari_accuracy(
    csv_file: str,
    class_column: str = "class",
    n_splits: int = 10,
    group_size: int = 6,
    components_range=range(2, 4),
    measurement: str = "categorical",
    max_iter_em: int = 10,
    random_state: int = 42,
    n_init: int = 5,
    stepmix_max_iter: int = 1000,
    debug: bool = True,
    results_file: str | None = None
) -> dict:
    """Retreina e avalia Gohari em cada fold estratificado."""
    df = _load_and_validate_data(csv_file, class_column)
    folds, split_seed = _state_safe_stratified_folds(
        df, class_column, n_splits, random_state
    )
    existing = _load_partial_cv_results(
        results_file, GOHARI_CV_VERSION, n_splits, split_seed
    )
    rows = existing.to_dict("records")
    completed_folds = set(existing["Fold"].astype(int)) if not existing.empty else set()

    with tempfile.TemporaryDirectory(prefix="gohari_cv_") as temp_dir:
        train_file = os.path.join(temp_dir, "train.csv")
        test_file = os.path.join(temp_dir, "test.csv")

        for fold_number, (train_idx, test_idx) in enumerate(folds, start=1):
            if fold_number in completed_folds:
                if debug:
                    print(f"[Gohari-CV] Fold {fold_number}/{n_splits} já concluído.")
                continue

            train_df = df.iloc[train_idx].reset_index(drop=True)
            test_df = df.iloc[test_idx].reset_index(drop=True)
            train_df.to_csv(train_file, index=False)
            test_df.to_csv(test_file, index=False)
            if debug:
                print(
                    f"[Gohari-CV] Fold {fold_number}/{n_splits} | "
                    f"treino={len(train_df)} | teste={len(test_df)}"
                )

            _, best_result, _ = gohari_elbow_accuracy(
                csv_file=train_file,
                class_column=class_column,
                group_size=group_size,
                components_range=components_range,
                measurement=measurement,
                max_iter_em=max_iter_em,
                random_state=random_state + fold_number,
                n_init=n_init,
                stepmix_max_iter=stepmix_max_iter,
                debug=False,
                save_elbow_csv=False
            )
            y_true, y_pred = predict_hnb_model(
                model=best_result["model"],
                csv_file=test_file,
                class_node=class_column,
                state_mappings=fit_state_mappings(train_df),
                require_all_rows=True
            )
            correct = int(np.sum(np.asarray(y_true) == np.asarray(y_pred)))
            rows.append({
                "EvaluationVersion": GOHARI_CV_VERSION,
                "Protocol": "stratified_kfold",
                "SplitSeed": split_seed,
                "NSplits": n_splits,
                "Fold": fold_number,
                "Accuracy": float(correct / len(y_true)),
                "Correct": correct,
                "NTrain": len(train_df),
                "NTest": len(test_df),
                "TrainIndices": " ".join(map(str, train_idx.tolist())),
                "TestIndices": " ".join(map(str, test_idx.tolist())),
                "GroupSize": group_size,
                "ComponentsStart": min(components_range),
                "ComponentsEndExclusive": max(components_range) + 1,
                "SelectedNComponents": int(best_result["n_components"]),
                "EMIter": max_iter_em,
                "StepMixNInit": n_init,
                "StepMixMaxIter": stepmix_max_iter
            })
            _save_partial_cv_results(rows, results_file)

    fold_results = pd.DataFrame(rows).sort_values("Fold").reset_index(drop=True)
    if len(fold_results) != n_splits:
        raise ValueError(
            f"Esperava {n_splits} resultados Gohari, encontrei {len(fold_results)}."
        )
    summary = _summarize_cross_validation(
        fold_results, GOHARI_CV_VERSION, split_seed
    )
    summary["selected_components_by_fold"] = fold_results[
        "SelectedNComponents"
    ].astype(int).tolist()
    return summary
