from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import Image, Paragraph, SimpleDocTemplate, Spacer

from spam_he.config import Config


LABELS = ("ham", "spam")
TIME_LABELS = {
    "plain_s": "Jawny",
    "vectorize_s": "TF-IDF",
    "encrypt_s": "Szyfrowanie",
    "classify_s": "Klasyfikacja",
    "decrypt_s": "Odszyfrowanie",
    "encrypted_total_s": "Szyfrowany",
}


def plot_class_distribution(class_counts: pd.Series, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(5, 3.5))
    ax.bar(class_counts.index, class_counts.values, color=["tab:blue", "tab:red"])
    for x, value in enumerate(class_counts.values):
        ax.text(x, value, str(value), ha="center", va="bottom")
    ax.set_ylim(0, class_counts.max() * 1.12)
    ax.set_title("Rozkład klas w zbiorze")
    ax.set_ylabel("Liczba SMS")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_confusion_matrix(matrix: list[list[int]], path: Path,
                          title: str = "Macierz pomyłek (model jawny)") -> None:
    fig, ax = plt.subplots(figsize=(4, 3.5))
    ax.imshow(matrix, cmap="Blues")
    for i, row in enumerate(matrix):
        for j, value in enumerate(row):
            color = "white" if value > max(map(max, matrix)) / 2 else "black"
            ax.text(j, i, str(value), ha="center", va="center", color=color)
    ax.set_xticks([0, 1], LABELS)
    ax.set_yticks([0, 1], LABELS)
    ax.set_xlabel("Przewidziana")
    ax.set_ylabel("Rzeczywista")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_agreement(comparison: pd.DataFrame, path: Path) -> None:
    counts = [
        [int(((comparison["plain"] == plain) & (comparison["encrypted"] == encrypted)).sum())
         for encrypted in LABELS]
        for plain in LABELS
    ]
    agreed = counts[0][0] + counts[1][1]
    total = len(comparison)
    fig, ax = plt.subplots(figsize=(4.2, 3.7))
    ax.imshow(counts, cmap="Blues")
    for i, row in enumerate(counts):
        for j, value in enumerate(row):
            color = "white" if value > max(map(max, counts)) / 2 else "black"
            ax.text(j, i, str(value), ha="center", va="center", color=color)
    ax.set_xticks([0, 1], LABELS)
    ax.set_yticks([0, 1], LABELS)
    ax.set_xlabel("Decyzja modelu szyfrowanego")
    ax.set_ylabel("Decyzja modelu jawnego")
    ax.set_title(f"Zgodność decyzji: {agreed}/{total} ({agreed / total:.0%})")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_decision_matrix(comparison: pd.DataFrame, column: str, title: str, path: Path) -> None:
    matrix = [
        [int(((comparison["actual"] == actual) & (comparison[column] == predicted)).sum())
         for predicted in LABELS]
        for actual in LABELS
    ]
    correct = matrix[0][0] + matrix[1][1]
    plot_confusion_matrix(matrix, path, f"{title}: {correct}/{len(comparison)} trafnych")


def plot_times(comparison: pd.DataFrame, path: Path) -> None:
    totals = [comparison[column] * 1000 for column in ("plain_s", "encrypted_total_s")]
    means = [series.mean() for series in totals]
    fig, ax = plt.subplots(figsize=(5.5, 4.2))
    ax.bar(["Jawny", "Szyfrowany"], means, yerr=[series.std(ddof=0) for series in totals],
           color=["tab:blue", "tab:orange"], capsize=4)
    for position, (value, series) in enumerate(zip(means, totals)):
        ax.text(position, value + series.std(ddof=0), f"{value:.2f} ms" if value < 10 else f"{value:.0f} ms",
                ha="center", va="bottom")
    ax.set_ylim(0, max(means) * 1.15)
    ax.set_ylabel("Średni czas na SMS [ms]")
    ax.set_title(f"Całkowity czas klasyfikacji jednego SMS\n(szyfrowany około {means[1] / means[0]:.0f}× wolniejszy)")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_time_breakdown(comparison: pd.DataFrame, path: Path) -> None:
    stages = ["vectorize_s", "encrypt_s", "classify_s", "decrypt_s"]
    means = [comparison[column].mean() * 1000 for column in stages]
    spread = [comparison[column].std(ddof=0) * 1000 for column in stages]
    total = sum(means)
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    ax.barh([TIME_LABELS[column] for column in stages], means, xerr=spread,
            color="tab:green", capsize=3)
    for position, (value, error) in enumerate(zip(means, spread)):
        ax.text(value + error, position, f"  {value:.1f} ms ({value / total:.0%})", va="center")
    ax.invert_yaxis()
    ax.set_xlim(0, max(means) * 1.6)
    ax.set_xlabel("Średni czas [ms], słupki błędu: ±1 odch. std.")
    ax.set_title("Skład czasu wariantu szyfrowanego")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


SENSITIVITY_PANELS = (
    ("recall_default", "Recall (spam), próg domyślny 0"),
    ("recall_spam", "Recall (spam), próg z walidacji"),
    ("precision_spam", "Precision (spam)"),
    ("f1_spam", "F1 (spam)"),
    ("roc_auc", "ROC-AUC"),
)


def plot_sensitivity(trials: pd.DataFrame, x_column: str, xlabel: str, title: str,
                     path: Path, config: Config) -> None:
    values = sorted(trials[x_column].unique())
    position = trials[x_column].map({value: index for index, value in enumerate(values)})
    jitter = (trials["seed"] - trials["seed"].mean()) * 0.01
    main = trials["seed"] == config.seed
    fig, axes = plt.subplots(2, 3, figsize=(13, 7.5))
    for ax, (column, label) in zip(axes.flat, SENSITIVITY_PANELS):
        grouped = trials.groupby(x_column)[column]
        ax.scatter((position + jitter)[~main], trials[column][~main], s=14, alpha=0.4,
                   color="tab:gray", label="pojedynczy podział")
        ax.scatter((position + jitter)[main], trials[column][main], s=40, color="tab:orange",
                   zorder=3, label=f"podział główny (ziarno {config.seed})")
        ax.errorbar(range(len(values)), grouped.mean().values, yerr=grouped.std(ddof=0).values,
                    marker="o", color="tab:blue", capsize=4, linewidth=2,
                    label="średnia ± odch. std.")
        if column in ("recall_spam", "recall_default"):
            ax.axhline(config.target_recall, color="tab:red", linestyle="--",
                       label=f"cel recall = {config.target_recall:.2f}")
        ax.set_xticks(range(len(values)), [f"{value:g}" for value in values])
        ax.set_xlabel(xlabel)
        ax.set_ylabel(label)
        ax.grid(alpha=0.3)
    handles, names = axes.flat[0].get_legend_handles_labels()
    axes.flat[-1].axis("off")
    axes.flat[-1].legend(handles, names, loc="center", fontsize=9)
    fig.suptitle(f"{title} ({trials['seed'].nunique()} podziałów danych)")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def requirement_checks(results: dict, config: Config) -> list[dict]:
    summary = results["summary"]
    times = summary["mean_times_s"]
    agreement_ok = summary["agreement"] >= config.target_agreement
    slower = times["encrypted_total_s"] > times["plain_s"]
    return [
        {"name": f"Recall spamu na całym teście >= {config.target_recall:.0%}",
         "passed": results["plain_metrics"]["recall_spam"] >= config.target_recall},
        {"name": f"Zgodność decyzji na próbce >= {config.target_agreement:.0%}",
         "passed": agreement_ok},
        {"name": "Wariant szyfrowany trwa dłużej niż jawny", "passed": slower},
        {"name": "Hipoteza badawcza potwierdzona na tej próbce", "passed": agreement_ok and slower},
    ]


def render_report(results: dict) -> str:
    summary = results["summary"]
    metrics = results["plain_metrics"]
    lines = [
        "Etap IV - testy i porównanie",
        f"Klucz: {results['key_bits']} bitów, cechy TF-IDF: {results['feature_count']}, "
        f"ziarno losowania: {results['seed']}",
        f"Próbka: {summary['sample_count']} SMS "
        f"(ham: {summary['class_counts']['ham']}, spam: {summary['class_counts']['spam']})",
        "",
        "Model jawny - cały zbiór testowy:",
        f"Accuracy: {metrics['accuracy']:.2%}, Precision (spam): {metrics['precision_spam']:.2%}, "
        f"Recall (spam): {metrics['recall_spam']:.2%}, F1 (spam): {metrics['f1_spam']:.4f}",
        f"Macierz pomyłek (wiersze: rzeczywiste, kolumny: przewidziane; ham, spam): "
        f"{metrics['confusion_matrix']}",
        "",
        f"Zgodność decyzji na próbce: {summary['agreement']:.2%}, "
        f"rozbieżności: {summary['disagreements']}",
        f"Accuracy na próbce: jawnie {summary['plain_accuracy']:.2%}, "
        f"szyfrowane {summary['encrypted_accuracy']:.2%}",
        "",
        "Średni czas dla jednego SMS:",
    ]
    for label, column in [
        ("Wariant jawny łącznie", "plain_s"),
        ("TF-IDF w wariancie szyfrowanym", "vectorize_s"),
        ("Szyfrowanie", "encrypt_s"),
        ("Klasyfikacja na serwerze", "classify_s"),
        ("Odszyfrowanie i decyzja klienta", "decrypt_s"),
        ("Wariant szyfrowany łącznie", "encrypted_total_s"),
    ]:
        lines.append(f"  {label}: {summary['mean_times_s'][column] * 1000:.3f} ms")
    lines += [
        "Czasy łączne obejmują TF-IDF i decyzję; pomiar lokalny bez treningu, "
        "generowania kluczy i transmisji sieciowej.",
        "",
        "Weryfikacja:",
    ]
    for check in results["checks"]:
        lines.append(f"  {check['name']}: {'TAK' if check['passed'] else 'NIE'}")
    lines += [
        "",
        "Ograniczenia:",
        "- Mała próbka nie pozwala uogólnić wyników na wszystkie SMS-y.",
        "- Ograniczony słownik TF-IDF może pogarszać skuteczność modelu.",
        "- Zaokrąglenia stałoprzecinkowe mogą zmienić decyzję blisko progu.",
        "- Czasy zależą od sprzętu, liczby cech i długości klucza.",
        "- Paillier obsługuje dodawanie i mnożenie przez jawne wagi; decyzję podejmuje klient.",
        "- Zakładamy uczciwe wykonywanie obliczeń przez serwer; klucz prywatny ma tylko klient.",
    ]
    return "\n".join(lines) + "\n"


def markdown_to_pdf(markdown: str, path: Path, images: list[Path] = ()) -> Path:
    font = Path(matplotlib.get_data_path()) / "fonts" / "ttf" / "DejaVuSans.ttf"
    pdfmetrics.registerFont(TTFont("DejaVuSans", str(font)))
    styles = getSampleStyleSheet()
    for style in styles.byName.values():
        style.fontName = "DejaVuSans"
    story = []
    for line in markdown.splitlines():
        if line.startswith("# "):
            story.append(Paragraph(line[2:], styles["Heading1"]))
        elif line.strip():
            story.append(Paragraph(line.replace("&", "&amp;").replace("<", "&lt;"), styles["Normal"]))
        else:
            story.append(Spacer(1, 0.3 * cm))
    for image in images:
        story.append(Image(str(image), width=12 * cm, height=7 * cm, kind="proportional"))
    SimpleDocTemplate(str(path)).build(story)
    return path


def write_outputs(results: dict, config: Config) -> None:
    out = config.output_dir
    out.mkdir(parents=True, exist_ok=True)
    results["comparison"].to_csv(out / "porownanie.csv", index=False)
    report = render_report(results)
    (out / "raport.txt").write_text(report, encoding="utf-8")
    images = [out / "rozklad_klas.png", out / "macierz_pomylek.png",
              out / "macierz_probka_jawny.png", out / "macierz_probka_szyfrowany.png",
              out / "zgodnosc.png", out / "czasy.png", out / "czasy_sklad.png"]
    plot_class_distribution(pd.Series(results["class_counts"]), images[0])
    plot_confusion_matrix(results["plain_metrics"]["confusion_matrix"], images[1])
    plot_decision_matrix(results["comparison"], "plain", "Próbka, model jawny", images[2])
    plot_decision_matrix(results["comparison"], "encrypted", "Próbka, model szyfrowany", images[3])
    plot_agreement(results["comparison"], images[4])
    plot_times(results["comparison"], images[5])
    plot_time_breakdown(results["comparison"], images[6])
    markdown_to_pdf("# " + report, out / "raport.pdf", images)


def write_sensitivity_outputs(feature_trials: pd.DataFrame, alpha_trials: pd.DataFrame,
                              config: Config) -> list[Path]:
    out = config.output_dir
    out.mkdir(parents=True, exist_ok=True)
    feature_trials.to_csv(out / "wrazliwosc_cechy.csv", index=False)
    alpha_trials.to_csv(out / "wrazliwosc_alpha.csv", index=False)
    paths = [out / "wrazliwosc_cechy.png", out / "wrazliwosc_alpha.png"]
    plot_sensitivity(feature_trials, "max_features", "Liczba cech TF-IDF (słów)",
                     "Wpływ liczby słów TF-IDF na jakość klasyfikacji", paths[0], config)
    plot_sensitivity(alpha_trials, "alpha", "alpha (wygładzanie Laplace'a)",
                     f"Wpływ parametru alpha przy {config.max_features} słowach", paths[1], config)
    return paths
