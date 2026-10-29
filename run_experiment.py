import argparse
import time
from dataclasses import replace
from pathlib import Path

import numpy as np

from spam_he import classifier, homomorphic, reporting
from spam_he.config import Config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Projekt 9: detekcja spamu SMS na danych zaszyfrowanych szyfrem Pailliera."
    )
    parser.add_argument("--quick", action="store_true",
                        help="szybki przebieg próbny (klucz 1024 bity, 3 SMS)")
    parser.add_argument("--samples", type=int, help="liczba SMS w eksperymencie szyfrowanym")
    parser.add_argument("--key-bits", type=int, help="długość klucza Pailliera w bitach")
    parser.add_argument("--output", type=Path, help="katalog na wyniki")
    stop = parser.add_mutually_exclusive_group()
    stop.add_argument("--preprocess-only", action="store_true",
                      help="tylko preprocessing: dane, podział, TF-IDF")
    stop.add_argument("--plain-only", action="store_true",
                      help="tylko część jawna: preprocessing, trening i ocena Naive Bayes")
    stop.add_argument("--encrypt-only", action="store_true",
                      help="tylko etap II: klucze Pailliera i szyfrowanie wektorów cech klienta")
    return parser.parse_args()


def build_config(args: argparse.Namespace) -> Config:
    config = Config().quick() if args.quick else Config()
    overrides = {
        "n_crypto_samples": args.samples,
        "key_bits": args.key_bits,
        "output_dir": args.output,
    }
    return replace(config, **{name: value for name, value in overrides.items() if value is not None})


def print_plain_evaluation(trained: classifier.TrainedClassifier, metrics: dict, config: Config) -> None:
    print("Wyniki walidacji:")
    print(trained.validation.to_string(index=False))
    print(f"Wybrany alpha={trained.alpha:g}, próg log-score={trained.threshold:.6f}")
    print("Klasyfikator jawny, zbiór testowy:")
    print(f"  Accuracy:         {metrics['accuracy']:.4f}")
    print(f"  Precision (spam): {metrics['precision_spam']:.4f}")
    print(f"  Recall (spam):    {metrics['recall_spam']:.4f}")
    print(f"  F1 (spam):        {metrics['f1_spam']:.4f}")
    print(f"  Macierz pomyłek (wiersze i kolumny: ham, spam): {metrics['confusion_matrix']}")
    met = metrics["recall_spam"] >= config.target_recall
    print(f"  Recall spamu >= {config.target_recall:.0%}: {'TAK' if met else 'NIE'}")


def run_encryption_demo(trained: classifier.TrainedClassifier, split: classifier.DataSplit,
                        config: Config) -> None:
    print(f"Klient: generowanie pary kluczy Pailliera ({config.key_bits} bitów)")
    started = time.perf_counter()
    client = homomorphic.Client(trained, config)
    print(f"  czas generowania kluczy: {time.perf_counter() - started:.3f} s")

    messages = split.test_texts.iloc[:config.n_crypto_samples]
    print(f"Klient: TF-IDF i szyfrowanie {len(messages)} SMS (skala stałoprzecinkowa {config.feature_scale})")
    max_error = 0.0
    for index, message in enumerate(messages, start=1):
        started = time.perf_counter()
        features = client.vectorize(message)
        vectorize_time = time.perf_counter() - started
        started = time.perf_counter()
        request = client.encrypt(features)
        encrypt_time = time.perf_counter() - started
        error = float(np.abs(client.decrypt_features(request) - features).max())
        max_error = max(max_error, error)
        print(f"  SMS {index}: cechy={len(features)}, niezerowe={int(np.count_nonzero(features))}, "
              f"TF-IDF {vectorize_time * 1000:.1f} ms, szyfrowanie {encrypt_time:.2f} s, "
              f"rozmiar zapytania {len(request) / 1024:.0f} KiB, błąd kodowania {error:.2e}")
    limit = 0.5 / config.feature_scale
    print(f"Maksymalny błąd kodowania stałoprzecinkowego: {max_error:.2e} (dopuszczalny {limit:.2e})")
    if max_error > limit + 1e-15:
        raise RuntimeError("Błąd kodowania przekracza dopuszczalny")


def main() -> None:
    args = parse_args()
    config = build_config(args)
    if config.n_crypto_samples < 1:
        raise ValueError("Liczba SMS musi być dodatnia")

    dataset = classifier.load_dataset(classifier.obtain_dataset(config))
    print(f"Oryginał: {dataset.raw_count} SMS, po usunięciu duplikatów: {len(dataset.messages)}")
    print(dataset.messages["label"].value_counts().to_string())
    split = classifier.split_dataset(dataset, config)
    print(f"Podział: trening {len(split.train_texts)}, test {len(split.test_texts)}")
    matrix = classifier.build_vectorizer(config).fit_transform(split.train_texts)
    print(f"TF-IDF: {matrix.shape[0]} wiadomości, {matrix.shape[1]} cech")
    if args.preprocess_only:
        print("Preprocessing zakończony poprawnie.")
        return

    trained = classifier.train_classifier(split, config)
    plain_metrics = classifier.evaluate(trained, split.test_texts, split.test_labels)
    print_plain_evaluation(trained, plain_metrics, config)
    if args.plain_only:
        print("Część jawna zakończona poprawnie.")
        return
    if args.encrypt_only:
        run_encryption_demo(trained, split, config)
        print("Etap II zakończony poprawnie.")
        return

    weights, priors = homomorphic.encode_model(trained, config)
    client = homomorphic.Client(trained, config)
    server = homomorphic.Server(weights, priors)
    print(f"Etapy III-IV: klasyfikacja i porównanie do {config.n_crypto_samples} SMS "
          f"(klucz {config.key_bits} bitów)")
    comparison = homomorphic.compare_on_sample(client, server, trained, split, config)
    results = {
        "key_bits": config.key_bits,
        "feature_count": len(trained.vectorizer.get_feature_names_out()),
        "seed": config.seed,
        "plain_metrics": plain_metrics,
        "summary": homomorphic.summarize_comparison(comparison),
        "comparison": comparison,
    }
    results["checks"] = reporting.requirement_checks(results, config)
    reporting.write_outputs(results, config)
    print()
    print(reporting.render_report(results))
    print(f"Wyniki zapisano w: {config.output_dir.resolve()}")


if __name__ == "__main__":
    main()
