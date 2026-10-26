from dataclasses import dataclass, replace
from pathlib import Path

DATA_SOURCES = (
    "https://archive.ics.uci.edu/static/public/228/sms+spam+collection.zip",
    "https://raw.githubusercontent.com/PacktPublishing/Machine-Learning-with-Scikit-learn/"
    "master/Section%206/SMSSpamCollection",
)


@dataclass(frozen=True)
class Config:
    seed: int = 42
    data_path: Path = Path("SMSSpamCollection")
    output_dir: Path = Path("wyniki")

    test_size: float = 0.20
    validation_size: float = 0.20
    max_features: int = 200
    alpha_grid: tuple[float, ...] = (0.01, 0.05, 0.1, 0.5, 1.0)

    target_recall: float = 0.90
    target_agreement: float = 0.95

    n_crypto_samples: int = 20
    key_bits: int = 2048
    feature_scale: int = 1_000_000
    weight_scale: int = 100_000

    def quick(self) -> "Config":
        return replace(self, key_bits=1024, n_crypto_samples=3)
