import json
import time
from dataclasses import dataclass

import numpy as np
import pandas as pd
from phe import paillier

from spam_he.classifier import LABELS, DataSplit, TrainedClassifier, class_index
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
    """Encode weights at weight_scale and priors at the dot-product scale.

    Scores always follow LABELS (ham, spam), regardless of model.classes_.
    Python integers avoid overflow in numpy's fixed-width integer arithmetic.
    """
    indices = [class_index(classifier.model, label) for label in LABELS]
    weights = np.asarray([
        to_fixed_point(classifier.model.feature_log_prob_[index], config.weight_scale)
        for index in indices
    ], dtype=object)
    priors = np.asarray(to_fixed_point(
        classifier.model.class_log_prior_[indices], config.feature_scale * config.weight_scale
    ), dtype=object)
    return weights, priors


class Client:
    def __init__(self, classifier: TrainedClassifier, config: Config):
        self.vectorizer = classifier.vectorizer
        self.config = config
        self.threshold = classifier.threshold
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
        """Decrypt class scores and apply the same margin threshold as the plain model."""
        payload = json.loads(response)
        if int(payload["public_key"]["n"]) != self.public_key.n:
            raise ValueError("Odpowiedź zaszyfrowana innym kluczem")
        if len(payload["scores"]) != len(LABELS):
            raise ValueError("Oczekiwano dwóch wyników: ham i spam")
        values = [
            self._private_key.decrypt(paillier.EncryptedNumber(self.public_key, int(text), 0))
            for text in payload["scores"]
        ]
        scores = from_fixed_point(values, self.config.feature_scale * self.config.weight_scale)
        label = "spam" if scores[1] - scores[0] >= self.threshold else "ham"
        return label, scores


class Server:
    def __init__(self, weights: np.ndarray, priors: np.ndarray):
        """Keep only the encoded public model; the private key stays with Client."""
        self.weights = np.asarray(weights, dtype=object)
        self.priors = np.asarray(priors, dtype=object)
        if (self.weights.ndim != 2 or self.weights.shape[0] != len(LABELS)
                or self.priors.shape != (len(LABELS),)):
            raise ValueError("Model musi zawierać wagi i priory dla ham i spam")

    def handle(self, request: str) -> str:
        """Compute E(prior_c + sum(x_i * weight_ci)) without decrypting features."""
        payload = json.loads(request)
        public_key = paillier.PaillierPublicKey(int(payload["public_key"]["n"]))
        if len(payload["features"]) != self.weights.shape[1]:
            raise ValueError("Liczba cech nie zgadza się z modelem")
        features = [
            paillier.EncryptedNumber(public_key, int(text), 0)
            for text in payload["features"]
        ]
        scores = []
        for weights, prior in zip(self.weights, self.priors):
            score = public_key.encrypt(int(prior))
            for feature, weight in zip(features, weights):
                score += feature * int(weight)
            # ciphertext() re-randomizes the result before sending it to the client.
            scores.append(str(score.ciphertext()))
        return json.dumps({"public_key": {"n": str(public_key.n)}, "scores": scores})


@dataclass
class EncryptedResult:
    label: str
    scores: np.ndarray
    times: dict[str, float]


def classify_encrypted(client: Client, server: Server, message: str) -> EncryptedResult:
    """Run one client -> server -> client exchange using serialized ciphertexts."""
    started = time.perf_counter()
    features = client.vectorize(message)
    vectorized = time.perf_counter()
    request = client.encrypt(features)
    encrypted = time.perf_counter()
    response = server.handle(request)
    classified = time.perf_counter()
    label, scores = client.decide(response)
    decrypted = time.perf_counter()
    return EncryptedResult(label, scores, {
        "vectorize": vectorized - started,
        "encrypt": encrypted - vectorized,
        "classify": classified - encrypted,
        "decrypt": decrypted - classified,
    })


def compare_on_sample(
    client: Client, server: Server, classifier: TrainedClassifier, split: DataSplit, config: Config
) -> pd.DataFrame:
    raise NotImplementedError


def summarize_comparison(comparison: pd.DataFrame) -> dict:
    raise NotImplementedError
