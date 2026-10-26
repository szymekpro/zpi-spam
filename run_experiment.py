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
    return parser.parse_args()


def build_config(args: argparse.Namespace) -> Config:
    config = Config().quick() if args.quick else Config()
    overrides = {
        "n_crypto_samples": args.samples,
        "key_bits": args.key_bits,
        "output_dir": args.output,
    }
    return replace(config, **{name: value for name, value in overrides.items() if value is not None})


def main() -> None:
    config = build_config(parse_args())
    config.output_dir.mkdir(parents=True, exist_ok=True)

    dataset = classifier.load_dataset(classifier.obtain_dataset(config))
    split = classifier.split_dataset(dataset, config)
    trained = classifier.train_classifier(split, config)
    plain_metrics = classifier.evaluate(trained, split.test_texts, split.test_labels)

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
