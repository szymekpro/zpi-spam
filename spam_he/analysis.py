from dataclasses import replace

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.metrics import roc_auc_score

from spam_he import classifier
from spam_he.config import Config

METRIC_LABELS = {
    "recall_spam": "Recall",
    "recall_default": "Recall(próg 0)",
    "precision_spam": "Precision",
    "f1_spam": "F1",
    "accuracy": "Accuracy",
    "roc_auc": "ROC-AUC",
}


def run_trial(dataset: classifier.Dataset, config: Config, seed: int, max_features: int,
              alpha_grid: tuple[float, ...]) -> dict:
    trial = replace(config, seed=seed, max_features=max_features, alpha_grid=alpha_grid)
    split = classifier.split_dataset(dataset, trial)
    trained = classifier.train_classifier(split, trial)
    metrics = classifier.evaluate(trained, split.test_texts, split.test_labels)
    margins = np.asarray(trained.margins(split.test_texts))
    is_spam = (split.test_labels == "spam").to_numpy()
    return {
        "seed": seed,
        "max_features": max_features,
        "alpha": trained.alpha,
        "threshold": trained.threshold,
        "accuracy": metrics["accuracy"],
        "precision_spam": metrics["precision_spam"],
        "recall_spam": metrics["recall_spam"],
        "recall_default": float(((margins >= 0) & is_spam).sum() / is_spam.sum()),
        "f1_spam": metrics["f1_spam"],
        "roc_auc": float(roc_auc_score(is_spam, margins)),
    }


def seeds_for(config: Config) -> range:
    return range(config.seed, config.seed + config.sensitivity_seeds)


def run_trials(dataset: classifier.Dataset, config: Config, settings: list[tuple]) -> pd.DataFrame:
    rows = Parallel(n_jobs=-1)(
        delayed(run_trial)(dataset, config, seed, max_features, alpha_grid)
        for seed, max_features, alpha_grid in settings
    )
    return pd.DataFrame(rows)


def feature_sensitivity(dataset: classifier.Dataset, config: Config) -> pd.DataFrame:
    settings = [
        (seed, size, config.alpha_grid)
        for size in config.feature_sizes
        for seed in seeds_for(config)
    ]
    return run_trials(dataset, config, settings)


def alpha_sensitivity(dataset: classifier.Dataset, config: Config) -> pd.DataFrame:
    settings = [
        (seed, config.max_features, (alpha,))
        for alpha in config.alpha_values
        for seed in seeds_for(config)
    ]
    return run_trials(dataset, config, settings)


def summary_lines(trials: pd.DataFrame, by: str) -> list[str]:
    lines = []
    for value, group in trials.groupby(by):
        parts = [
            f"{label} {group[column].mean():.3f} ± {group[column].std(ddof=0):.3f}"
            for column, label in METRIC_LABELS.items()
        ]
        lines.append(f"  {value:>6g}: " + " | ".join(parts))
    return lines


def paired_difference(trials: pd.DataFrame, baseline: int, candidate: int, metric: str) -> dict:
    table = trials.pivot(index="seed", columns="max_features", values=metric)
    difference = table[candidate] - table[baseline]
    return {
        "mean": float(difference.mean()),
        "std": float(difference.std(ddof=0)),
        "better": int((difference > 0).sum()),
        "equal": int((difference == 0).sum()),
        "worse": int((difference < 0).sum()),
        "total": len(difference),
    }
