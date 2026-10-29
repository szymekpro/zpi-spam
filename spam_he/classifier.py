import csv
import io
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (
    accuracy_score, confusion_matrix, f1_score, precision_score, recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB

from spam_he.config import DATA_SOURCES, Config

LABELS = ("ham", "spam")
MIN_EXPECTED_LINES = 5000
DOWNLOAD_TIMEOUT_S = 45
RECALL_TOLERANCE = 1e-12


@dataclass
class Dataset:
    messages: pd.DataFrame
    raw_count: int


@dataclass
class DataSplit:
    train_texts: pd.Series
    test_texts: pd.Series
    train_labels: pd.Series
    test_labels: pd.Series


@dataclass
class TrainedClassifier:
    vectorizer: TfidfVectorizer
    model: MultinomialNB
    alpha: float
    threshold: float
    validation: pd.DataFrame

    def margins(self, texts) -> np.ndarray:
        return score_margins(self.model, self.vectorizer.transform(texts))

    def predict(self, texts) -> np.ndarray:
        return np.where(self.margins(texts) >= self.threshold, "spam", "ham")


def obtain_dataset(config: Config) -> Path:
    path = config.data_path
    if path.exists():
        return path
    for url in DATA_SOURCES:
        try:
            print("Pobieranie:", url)
            request = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(request, timeout=DOWNLOAD_TIMEOUT_S) as response:
                body = response.read()
            if zipfile.is_zipfile(io.BytesIO(body)):
                with zipfile.ZipFile(io.BytesIO(body)) as archive:
                    body = archive.read("SMSSpamCollection")
            if body.count(b"\n") < MIN_EXPECTED_LINES:
                raise ValueError("Plik zawiera podejrzanie mało wierszy")
            path.write_bytes(body)
            return path
        except Exception as exc:
            print("Próba nieudana:", type(exc).__name__, str(exc)[:180])
    raise RuntimeError(
        f"Brak danych. Pobierz plik SMSSpamCollection ze strony UCI i umieść go jako {path}."
    )


def load_dataset(path: Path) -> Dataset:
    frame = pd.read_csv(
        path, sep="\t", header=None, names=["label", "message"],
        encoding="latin-1", quoting=csv.QUOTE_NONE, engine="python", on_bad_lines="error",
    )
    if not frame["label"].isin(LABELS).all():
        raise ValueError("Nieznane etykiety w zbiorze")
    if frame["message"].isna().any():
        raise ValueError("Brakujące wiadomości")
    raw_count = len(frame)
    conflicting = int(frame.groupby("message")["label"].nunique().gt(1).sum())
    if conflicting:
        raise ValueError(f"Znaleziono {conflicting} wiadomości z różnymi etykietami")
    messages = frame.drop_duplicates(subset="message").reset_index(drop=True)
    return Dataset(messages=messages, raw_count=raw_count)


def split_dataset(dataset: Dataset, config: Config) -> DataSplit:
    train_texts, test_texts, train_labels, test_labels = train_test_split(
        dataset.messages["message"], dataset.messages["label"],
        test_size=config.test_size, random_state=config.seed,
        stratify=dataset.messages["label"],
    )
    return DataSplit(train_texts, test_texts, train_labels, test_labels)


def split_validation(split: DataSplit, config: Config) -> DataSplit:
    fit_texts, validation_texts, fit_labels, validation_labels = train_test_split(
        split.train_texts, split.train_labels,
        test_size=config.validation_size, random_state=config.seed,
        stratify=split.train_labels,
    )
    return DataSplit(fit_texts, validation_texts, fit_labels, validation_labels)


def build_vectorizer(config: Config) -> TfidfVectorizer:
    return TfidfVectorizer(max_features=config.max_features, sublinear_tf=True)


def class_index(model: MultinomialNB, label: str) -> int:
    return int(np.where(model.classes_ == label)[0][0])


def scores_for_sms(model: MultinomialNB, matrix) -> np.ndarray:
    return np.asarray(matrix @ model.feature_log_prob_.T) + model.class_log_prior_


def score_margins(model: MultinomialNB, matrix) -> np.ndarray:
    scores = scores_for_sms(model, matrix)
    return scores[:, class_index(model, "spam")] - scores[:, class_index(model, "ham")]


def select_threshold(margins: np.ndarray, actual, minimum_recall: float) -> dict:
    values = np.unique(margins)
    candidates = np.r_[
        np.nextafter(values[0], -np.inf),
        (values[:-1] + values[1:]) / 2,
        np.nextafter(values[-1], np.inf),
    ]
    best = None
    for threshold in candidates:
        predicted = np.where(margins >= threshold, "spam", "ham")
        recall = recall_score(actual, predicted, pos_label="spam")
        if recall + RECALL_TOLERANCE < minimum_recall:
            continue
        f1 = f1_score(actual, predicted, pos_label="spam")
        precision = precision_score(actual, predicted, pos_label="spam", zero_division=0)
        rank = (f1, precision, threshold)
        if best is None or rank > best[0]:
            best = (rank, float(threshold), float(recall), float(precision), float(f1))
    if best is None:
        raise ValueError("Żaden próg nie osiąga docelowego recall")
    return {"threshold": best[1], "recall": best[2], "precision": best[3], "f1": best[4]}


def train_classifier(split: DataSplit, config: Config) -> TrainedClassifier:
    validation = split_validation(split, config)
    validation_vectorizer = build_vectorizer(config)
    x_fit = validation_vectorizer.fit_transform(validation.train_texts)
    x_validation = validation_vectorizer.transform(validation.test_texts)

    rows = []
    for alpha in config.alpha_grid:
        trial = MultinomialNB(alpha=alpha)
        trial.fit(x_fit, validation.train_labels)
        calibration = select_threshold(
            score_margins(trial, x_validation), validation.test_labels, config.target_recall
        )
        rows.append({"alpha": alpha, **calibration})
    table = (
        pd.DataFrame(rows)
        .sort_values(["f1", "precision"], ascending=False)
        .reset_index(drop=True)
    )
    chosen = table.iloc[0]

    vectorizer = build_vectorizer(config)
    x_train = vectorizer.fit_transform(split.train_texts)
    model = MultinomialNB(alpha=float(chosen["alpha"]))
    model.fit(x_train, split.train_labels)
    return TrainedClassifier(
        vectorizer=vectorizer,
        model=model,
        alpha=float(chosen["alpha"]),
        threshold=float(chosen["threshold"]),
        validation=table,
    )


def evaluate(classifier: TrainedClassifier, texts, labels) -> dict:
    predicted = classifier.predict(texts)
    matrix = confusion_matrix(labels, predicted, labels=list(LABELS))
    return {
        "accuracy": float(accuracy_score(labels, predicted)),
        "precision_spam": float(precision_score(labels, predicted, pos_label="spam", zero_division=0)),
        "recall_spam": float(recall_score(labels, predicted, pos_label="spam")),
        "f1_spam": float(f1_score(labels, predicted, pos_label="spam")),
        "confusion_matrix": matrix.tolist(),
    }


def stratified_sample(labels: pd.Series, size: int, seed: int) -> np.ndarray:
    """Wylosuj pozycje SMS-ów, zachowując w przybliżeniu proporcje klas."""
    if size < 1 or len(labels) == 0:
        raise ValueError("Próbka i zbiór testowy muszą być niepuste")
    size = min(size, len(labels))
    ham = np.flatnonzero(labels.to_numpy() == "ham")
    spam = np.flatnonzero(labels.to_numpy() == "spam")
    spam_count = round(size * len(spam) / len(labels))
    if size >= 2 and len(ham) > 0 and len(spam) > 0:
        spam_count = max(1, min(spam_count, size - 1))
    spam_count = min(spam_count, len(spam))
    ham_count = min(size - spam_count, len(ham))
    spam_count = size - ham_count

    rng = np.random.default_rng(seed)
    positions = np.concatenate([
        rng.choice(ham, ham_count, replace=False),
        rng.choice(spam, spam_count, replace=False),
    ])
    rng.shuffle(positions)
    return positions
