import json
import unittest
from dataclasses import replace
from unittest.mock import patch

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

from spam_he.classifier import TrainedClassifier, scores_for_sms
from spam_he.config import Config
from spam_he.homomorphic import Client, Server, classify_encrypted, encode_model


class HomomorphicClassificationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        texts = ["family dinner tonight", "meeting tonight at home",
                 "free prize cash winner", "claim free cash prize"]
        vectorizer = TfidfVectorizer()
        matrix = vectorizer.fit_transform(texts)
        model = MultinomialNB().fit(matrix, ["ham", "ham", "spam", "spam"])
        cls.trained = TrainedClassifier(vectorizer, model, 1.0, 0.4, pd.DataFrame())
        # Small keys keep tests fast; the application defaults to 2048 bits.
        cls.config = Config(key_bits=512)
        cls.client = Client(cls.trained, cls.config)
        cls.weights, cls.priors = encode_model(cls.trained, cls.config)
        cls.server = Server(cls.weights, cls.priors)

    def test_encrypted_scores_and_decisions_match_plain_model(self):
        for text in ("free cash prize", "family dinner tonight", "", "unknown vocabulary"):
            with self.subTest(text=text):
                result = classify_encrypted(self.client, self.server, text)
                features = self.client.vectorize(text)
                expected = scores_for_sms(self.trained.model, features.reshape(1, -1))[0]
                # Bound the accumulated quantization error in features, weights and priors.
                bound = (
                    np.abs(self.trained.model.feature_log_prob_).sum(axis=1)
                    / (2 * self.config.feature_scale)
                    + np.abs(features).sum() / (2 * self.config.weight_scale)
                    + len(features) / (4 * self.config.feature_scale * self.config.weight_scale)
                    + 0.5 / (self.config.feature_scale * self.config.weight_scale)
                )
                self.assertTrue(np.all(np.abs(result.scores - expected) <= bound + 1e-12))
                self.assertEqual(result.label, self.trained.predict([text])[0])
                self.assertTrue(all(value >= 0 for value in result.times.values()))

    def test_signed_weights_and_prior_scaling_are_exact(self):
        weights = np.array([[2, -3], [-5, 7]], dtype=object)
        priors = np.array([-11, 13], dtype=object)
        request = self.client.encrypt(np.array([0.25, 0.5]))
        response = Server(weights, priors).handle(request)
        _, scores = self.client.decide(response)
        scale = self.config.feature_scale * self.config.weight_scale
        np.testing.assert_allclose(scores * scale, [-1_000_011, 2_250_013], atol=1e-8)

    def test_server_needs_no_private_key(self):
        request = self.client.encrypt(self.client.vectorize("free prize"))
        with patch.object(self.client._private_key, "decrypt", side_effect=AssertionError):
            response = self.server.handle(request)
        payload = json.loads(response)
        self.assertEqual(set(payload), {"public_key", "scores"})
        self.assertEqual(len(payload["scores"]), 2)
        self.assertTrue(all(isinstance(value, str) for value in payload["scores"]))
        self.assertFalse(hasattr(self.server, "_private_key"))
        self.client.decide(response)

    def test_threshold_including_tie_is_applied_by_client(self):
        request = self.client.encrypt(self.client.vectorize("free prize"))
        response = self.server.handle(request)
        _, scores = self.client.decide(response)
        client = self.client
        original = client.threshold
        try:
            margin = scores[1] - scores[0]
            client.threshold = margin
            self.assertEqual(client.decide(response)[0], "spam")
            client.threshold = margin + 0.1
            self.assertEqual(client.decide(response)[0], "ham")
        finally:
            client.threshold = original

    def test_response_is_randomized(self):
        request = self.client.encrypt(self.client.vectorize("free prize"))
        first = self.server.handle(request)
        second = self.server.handle(request)
        self.assertNotEqual(json.loads(first)["scores"], json.loads(second)["scores"])
        np.testing.assert_array_equal(self.client.decide(first)[1], self.client.decide(second)[1])

    def test_wrong_response_key_is_rejected(self):
        other = Client(self.trained, self.config)
        response = self.server.handle(other.encrypt(other.vectorize("free prize")))
        with self.assertRaisesRegex(ValueError, "innym kluczem"):
            self.client.decide(response)

    def test_wrong_feature_count_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "Liczba cech"):
            self.server.handle(self.client.encrypt(np.array([0.5])))

    def test_wrong_score_count_is_rejected(self):
        response = json.dumps({"public_key": {"n": str(self.client.public_key.n)}, "scores": []})
        with self.assertRaisesRegex(ValueError, "dwóch wyników"):
            self.client.decide(response)

    def test_model_class_order_is_explicit(self):
        model = MultinomialNB()
        model.classes_ = self.trained.model.classes_[::-1]
        model.feature_log_prob_ = self.trained.model.feature_log_prob_[::-1]
        model.class_log_prior_ = self.trained.model.class_log_prior_[::-1]
        weights, priors = encode_model(replace(self.trained, model=model), self.config)
        np.testing.assert_array_equal(weights, self.weights)
        np.testing.assert_array_equal(priors, self.priors)


if __name__ == "__main__":
    unittest.main()
