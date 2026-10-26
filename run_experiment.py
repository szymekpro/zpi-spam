import argparse
from dataclasses import replace
from pathlib import Path

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


def main() -> None:
    args = parse_args()
    config = build_config(args)
    config.output_dir.mkdir(parents=True, exist_ok=True)

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

    weights, priors = homomorphic.encode_model(trained, config)
    client = homomorphic.Client(trained, config)
    server = homomorphic.Server(weights, priors)
    comparison = homomorphic.compare_on_sample(client, server, trained, split, config)

    results = {
        "dataset_original_count": dataset.raw_count,
        "dataset_after_dedup": len(dataset.messages),
        "class_counts": dataset.messages["label"].value_counts().to_dict(),
        "alpha_selected": trained.alpha,
        "threshold_selected": trained.threshold,
        "plain_test_metrics": plain_metrics,
        "encrypted": homomorphic.summarize_comparison(comparison),
    }
    reporting.write_outputs(results, config)
    print(f"Wyniki zapisane w katalogu {config.output_dir}/")


if __name__ == "__main__":
    main()
