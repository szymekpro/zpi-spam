import io
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

from spam_he.classifier import DataSplit, TrainedClassifier, stratified_sample
from spam_he.config import Config
from spam_he.homomorphic import Client, Server, compare_on_sample, encode_model, summarize_comparison
from spam_he.reporting import requirement_checks, write_outputs


class ComparisonTests(unittest.TestCase):
    def test_sample_is_repeatable_and_contains_both_classes(self):
        labels = pd.Series(["ham"] * 99 + ["spam"], index=range(100, 200))
        positions = stratified_sample(labels, 3, 42)
        np.testing.assert_array_equal(positions, stratified_sample(labels, 3, 42))
        self.assertEqual(len(set(positions)), 3)
        self.assertEqual(set(labels.iloc[positions]), {"ham", "spam"})

    def test_sample_sizes_and_single_class(self):
        for labels in (pd.Series(["ham", "spam", "ham"]), pd.Series(["spam"] * 3)):
            with self.subTest(labels=labels.tolist()):
                self.assertEqual(len(stratified_sample(labels, 1, 42)), 1)
                self.assertEqual(set(stratified_sample(labels, 10, 42)), {0, 1, 2})
        with self.assertRaises(ValueError):
            stratified_sample(pd.Series(["ham"]), 0, 42)
        with self.assertRaises(ValueError):
            stratified_sample(pd.Series([], dtype=str), 3, 42)

    def test_comparison_uses_the_same_messages_and_measures_all_stages(self):
        texts = pd.Series(["family dinner tonight", "meeting at home",
                           "free cash prize", "claim free cash"], index=[90, 40, 70, 10])
        labels = pd.Series(["ham", "ham", "spam", "spam"], index=texts.index)
        vectorizer = TfidfVectorizer()
        model = MultinomialNB().fit(vectorizer.fit_transform(texts), labels)
        trained = TrainedClassifier(vectorizer, model, 1.0, 0.4, pd.DataFrame())
        config = Config(key_bits=512, n_crypto_samples=4)
        client = Client(trained, config)
        server = Server(*encode_model(trained, config))
        split = DataSplit(texts, texts, labels, labels)
        with redirect_stdout(io.StringIO()):
            comparison = compare_on_sample(client, server, trained, split, config)
        positions = comparison["test_position"].to_numpy()
        self.assertEqual(set(positions), {0, 1, 2, 3})
        self.assertEqual(comparison["actual"].tolist(), labels.iloc[positions].tolist())
        self.assertEqual(comparison["plain"].tolist(), trained.predict(texts.iloc[positions]).tolist())
        self.assertTrue(comparison["agree"].all())
        stages = ["vectorize_s", "encrypt_s", "classify_s", "decrypt_s"]
        self.assertTrue((comparison[stages + ["plain_s"]] >= 0).all().all())
        np.testing.assert_allclose(comparison["encrypted_total_s"], comparison[stages].sum(axis=1))

    def test_summary_distinguishes_agreement_from_accuracy(self):
        comparison = pd.DataFrame({
            "actual": ["ham", "ham", "spam", "spam"],
            "plain": ["ham", "spam", "ham", "spam"],
            "encrypted": ["ham", "spam", "spam", "spam"],
            "plain_s": [0.001] * 4, "vectorize_s": [0.001] * 4,
            "encrypt_s": [0.02] * 4, "classify_s": [0.01] * 4,
            "decrypt_s": [0.002] * 4, "encrypted_total_s": [0.033] * 4,
        })
        summary = summarize_comparison(comparison)
        self.assertEqual(summary["agreement"], 0.75)
        self.assertEqual(summary["disagreements"], 1)
        self.assertEqual(summary["plain_accuracy"], 0.5)
        self.assertEqual(summary["encrypted_accuracy"], 0.75)
        self.assertAlmostEqual(summary["mean_times_s"]["encrypted_total_s"], 0.033)
        with tempfile.TemporaryDirectory() as directory:
            config = Config(output_dir=Path(directory) / "wyniki")
            results = {
                "key_bits": 2048, "feature_count": 200, "seed": 42,
                "plain_metrics": {"accuracy": 0.9, "precision_spam": 0.9,
                                  "recall_spam": 0.91, "f1_spam": 0.9,
                                  "confusion_matrix": [[8, 1], [1, 9]]},
                "summary": summary, "comparison": comparison,
            }
            results["checks"] = requirement_checks(results, config)
            write_outputs(results, config)
            saved = pd.read_csv(config.output_dir / "porownanie.csv")
            pd.testing.assert_frame_equal(saved, comparison)
            report = (config.output_dir / "raport.txt").read_text(encoding="utf-8")
            self.assertIn("Hipoteza badawcza potwierdzona na tej próbce: NIE", report)
            self.assertIn("Ograniczenia:", report)

    def test_hypothesis_requires_agreement_and_increased_time(self):
        for agreement, encrypted_time, expected in [(0.95, 2.0, True), (0.949, 2.0, False),
                                                    (1.0, 1.0, False)]:
            with self.subTest(agreement=agreement, encrypted_time=encrypted_time):
                results = {"plain_metrics": {"recall_spam": 0.8}, "summary": {
                    "agreement": agreement,
                    "mean_times_s": {"plain_s": 1.0, "encrypted_total_s": encrypted_time},
                }}
                checks = requirement_checks(results, Config())
                self.assertFalse(checks[0]["passed"])
                self.assertEqual(checks[-1]["passed"], expected)


if __name__ == "__main__":
    unittest.main()
