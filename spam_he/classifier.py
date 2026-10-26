from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

from spam_he.config import Config


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
        raise NotImplementedError

    def predict(self, texts) -> np.ndarray:
        raise NotImplementedError


def obtain_dataset(config: Config) -> Path:
    raise NotImplementedError


def load_dataset(path: Path) -> Dataset:
    raise NotImplementedError


def split_dataset(dataset: Dataset, config: Config) -> DataSplit:
    raise NotImplementedError


def scores_for_sms(model: MultinomialNB, matrix) -> np.ndarray:
    raise NotImplementedError


def score_margins(model: MultinomialNB, matrix) -> np.ndarray:
    raise NotImplementedError


def select_threshold(margins: np.ndarray, actual, minimum_recall: float) -> dict:
    raise NotImplementedError


def train_classifier(split: DataSplit, config: Config) -> TrainedClassifier:
    raise NotImplementedError


def evaluate(classifier: TrainedClassifier, texts, labels) -> dict:
    raise NotImplementedError


def stratified_sample(labels: pd.Series, size: int, seed: int) -> np.ndarray:
    raise NotImplementedError
