import json
import time
from dataclasses import dataclass

import numpy as np
import pandas as pd
from phe import paillier

from spam_he.classifier import LABELS, DataSplit, TrainedClassifier, class_index, stratified_sample
from spam_he.config import Config


def generate_keypair(key_bits: int) -> tuple[paillier.PaillierPublicKey, paillier.PaillierPrivateKey]:
    """Wygeneruj parę kluczy Paillier o module o długości `key_bits` bitów."""
    return paillier.generate_paillier_keypair(n_length=key_bits)


def to_fixed_point(values, scale: int) -> list[int]:
    """Koduj liczby zmiennoprzecinkowe jako całkowite `round(value * scale)`.

    Paillier działa na liczbach całkowitych modulo n, dlatego wartości TF-IDF
    są zamieniane na liczby stałoprzecinkowe. Zwracane są zwykłe liczby całkowite
    Pythona (nie skalary numpy), dzięki czemu `phe` szyfruje je z wykładnikiem 0.
    """
    array = np.asarray(values, dtype=float)
    if not np.isfinite(array).all():
        raise ValueError("Wartości cech muszą być skończone")
    return [int(value) for value in np.rint(array * scale)]


def from_fixed_point(values, scale: int) -> np.ndarray:
    """Zdekoduj liczby stałoprzecinkowe z powrotem na zmiennoprzecinkowe (odwrotność `to_fixed_point`)."""
    return np.asarray([int(value) for value in values], dtype=float) / scale


def encode_model(classifier: TrainedClassifier, config: Config) -> tuple[np.ndarray, np.ndarray]:
    """Zakoduj wagi w skali weight_scale, a priory w skali iloczynu skalarnego.

    Wyniki zawsze są zgodne z LABELS (ham, spam), niezależnie od model.classes_.
    Liczby całkowite Pythona unikają przepełnienia w arytmetyce numpy o stałej szerokości.
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
        """Zwróć gęsty wektor TF-IDF (zawsze jedna wartość na cechę, łącznie z zerami)."""
        return self.vectorizer.transform([message]).toarray()[0]

    def encrypt(self, features: np.ndarray) -> str:
        """Zaszyfruj każdą cechę kluczem publicznym i zwróć zapytanie w formacie JSON.

        Zapytanie zawiera wyłącznie moduł klucza publicznego i szyfrogramy
        (ciągi dziesiętne, domyślny wykładnik 0), bez żadnych danych jawnych.
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
        """Odszyfruj zapytanie wygenerowane przez `encrypt` (używane tylko do autotestów po stronie klienta)."""
        payload = json.loads(request)
        if int(payload["public_key"]["n"]) != self.public_key.n:
            raise ValueError("Zapytanie zaszyfrowane innym kluczem")
        values = [
            self._private_key.decrypt(paillier.EncryptedNumber(self.public_key, int(text), 0))
            for text in payload["features"]
        ]
        return from_fixed_point(values, self.config.feature_scale)

    def decide(self, response: str) -> tuple[str, np.ndarray]:
        """Odszyfruj wyniki klas i zastosuj ten sam próg marginesu co model jawny."""
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
        """Zachowaj tylko zakodowany model; klucz prywatny pozostaje w Client."""
        self.weights = np.asarray(weights, dtype=object)
        self.priors = np.asarray(priors, dtype=object)
        if (self.weights.ndim != 2 or self.weights.shape[0] != len(LABELS)
                or self.priors.shape != (len(LABELS),)):
            raise ValueError("Model musi zawierać wagi i priory dla ham i spam")

    def handle(self, request: str) -> str:
        """Oblicz E(prior_c + sum(x_i * weight_ci)) bez odszyfrowywania cech."""
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
    """Wykonaj jedną wymianę klient -> serwer -> klient z użyciem zserializowanych szyfrogramów."""
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
    positions = stratified_sample(split.test_labels, config.n_crypto_samples, config.seed)
    rows = []
    for position in positions:
        message = split.test_texts.iloc[position]
        started = time.perf_counter()
        plain_label = classifier.predict([message])[0]
        plain_time = time.perf_counter() - started
        result = classify_encrypted(client, server, message)
        rows.append({
            "test_position": int(position),
            "actual": split.test_labels.iloc[position],
            "plain": plain_label,
            "encrypted": result.label,
            "agree": plain_label == result.label,
            "plain_s": plain_time,
            **{name + "_s": value for name, value in result.times.items()},
            "encrypted_total_s": sum(result.times.values()),
        })
        print(f"  SMS {len(rows)}/{len(positions)}: jawnie={plain_label}, "
              f"szyfrowane={result.label}, zgodność={'TAK' if plain_label == result.label else 'NIE'}")
    return pd.DataFrame(rows)


def summarize_comparison(comparison: pd.DataFrame) -> dict:
    if comparison.empty:
        raise ValueError("Brak wyników do porównania")
    time_columns = ["plain_s", "vectorize_s", "encrypt_s", "classify_s", "decrypt_s", "encrypted_total_s"]
    return {
        "sample_count": len(comparison),
        "class_counts": {label: int((comparison["actual"] == label).sum()) for label in LABELS},
        "agreement": float((comparison["plain"] == comparison["encrypted"]).mean()),
        "disagreements": int((comparison["plain"] != comparison["encrypted"]).sum()),
        "plain_accuracy": float((comparison["plain"] == comparison["actual"]).mean()),
        "encrypted_accuracy": float((comparison["encrypted"] == comparison["actual"]).mean()),
        "mean_times_s": {name: float(comparison[name].mean()) for name in time_columns},
    }
