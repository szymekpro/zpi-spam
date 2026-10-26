from dataclasses import dataclass

import numpy as np
import pandas as pd

from spam_he.classifier import DataSplit, TrainedClassifier
from spam_he.config import Config


def encode_model(classifier: TrainedClassifier, config: Config) -> tuple[np.ndarray, np.ndarray]:
    raise NotImplementedError


class Client:
    def __init__(self, classifier: TrainedClassifier, config: Config):
        raise NotImplementedError

    def vectorize(self, message: str) -> np.ndarray:
        raise NotImplementedError

    def encrypt(self, features: np.ndarray) -> str:
        raise NotImplementedError

    def decide(self, response: str) -> tuple[str, np.ndarray]:
        raise NotImplementedError


class Server:
    def __init__(self, weights: np.ndarray, priors: np.ndarray):
        raise NotImplementedError

    def handle(self, request: str) -> str:
        raise NotImplementedError


@dataclass
class EncryptedResult:
    label: str
    times: dict[str, float]
    score_error: float


def classify_encrypted(client: Client, server: Server, message: str) -> EncryptedResult:
    raise NotImplementedError


def compare_on_sample(
    client: Client, server: Server, classifier: TrainedClassifier, split: DataSplit, config: Config
) -> pd.DataFrame:
    raise NotImplementedError


def summarize_comparison(comparison: pd.DataFrame) -> dict:
    raise NotImplementedError
