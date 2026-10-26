from pathlib import Path

import pandas as pd

from spam_he.config import Config


def plot_class_distribution(class_counts: pd.Series, path: Path) -> None:
    raise NotImplementedError


def plot_confusion_matrix(matrix: list[list[int]], path: Path) -> None:
    raise NotImplementedError


def plot_times(mean_times: dict[str, float], path: Path) -> None:
    raise NotImplementedError


def requirement_checks(results: dict, config: Config) -> list[dict]:
    raise NotImplementedError


def render_report(results: dict) -> str:
    raise NotImplementedError


def markdown_to_pdf(markdown: str, path: Path) -> Path:
    raise NotImplementedError


def write_outputs(results: dict, config: Config) -> None:
    raise NotImplementedError
