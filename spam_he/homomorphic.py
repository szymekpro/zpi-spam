import json
from dataclasses import dataclass

import numpy as np
import pandas as pd
from phe import paillier

from spam_he.classifier import DataSplit, TrainedClassifier
from spam_he.config import Config


def generate_keypair(key_bits: int) -> tuple[paillier.PaillierPublicKey, paillier.PaillierPrivateKey]:
    """Generate a Paillier key pair with a modulus of `key_bits` bits."""
    return paillier.generate_paillier_keypair(n_length=key_bits)


def to_fixed_point(values, scale: int) -> list[int]:
    """Encode floats as integers `round(value * scale)`.

    Paillier works on integers modulo n, so floating-point TF-IDF values are
    mapped to fixed-point integers. Plain Python ints are returned (not numpy
    scalars), so `phe` encrypts them with exponent 0.
    """
    array = np.asarray(values, dtype=float)
    if not np.isfinite(array).all():
        raise ValueError("Wartości cech muszą być skończone")
    return [int(value) for value in np.rint(array * scale)]


def from_fixed_point(values, scale: int) -> np.ndarray:
    """Decode fixed-point integers back to floats (inverse of `to_fixed_point`)."""
    return np.asarray([int(value) for value in values], dtype=float) / scale


def encode_model(classifier: TrainedClassifier, config: Config) -> tuple[np.ndarray, np.ndarray]:
    raise NotImplementedError


class Client:
    def __init__(self, classifier: TrainedClassifier, config: Config):
        self.vectorizer = classifier.vectorizer
        self.config = config
        self.public_key, self._private_key = generate_keypair(config.key_bits)

    def vectorize(self, message: str) -> np.ndarray:
        """Return the dense TF-IDF vector (always one value per feature, zeros included)."""
        return self.vectorizer.transform([message]).toarray()[0]

    def encrypt(self, features: np.ndarray) -> str:
        """Encrypt every feature with the public key and return a JSON request.

        The request contains only the public key modulus and the ciphertexts
        (decimal strings, implicit exponent 0), nothing in plaintext.
        """
        encoded = to_fixed_point(features, self.config.feature_scale)
        ciphertexts = []
        for value in encoded:
            number = self.public_key.encrypt(value)
            if number.exponent != 0:
                raise ValueError("Szyfrogram ma wykładnik różny od 0")
            ciphertexts.append(str(number.ciphertext()))
        return json.dumps({"public_key": {"n": str(self.public_key.n)}, "features": ciphertexts})

    def decrypt_features(self, request: str) -> np.ndarray:
        """Decrypt a request produced by `encrypt` (used only for self-checks on the client)."""
        payload = json.loads(request)
        if int(payload["public_key"]["n"]) != self.public_key.n:
            raise ValueError("Zapytanie zaszyfrowane innym kluczem")
        values = [
            self._private_key.decrypt(paillier.EncryptedNumber(self.public_key, int(text), 0))
            for text in payload["features"]
        ]
        return from_fixed_point(values, self.config.feature_scale)

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
