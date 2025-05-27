# utils.py
import numpy as np

class MyMultinomialNB:
    def fit(self, X, y):
        self.classes = np.unique(y)
        n_samples, n_features = X.shape

        # Tính xác suất tiên nghiệm P(y)
        self.class_log_prior_ = np.log(np.bincount(y) / n_samples)

        # Tính xác suất có điều kiện P(xi|y) với Laplace smoothing
        self.feature_log_prob_ = []
        for c in self.classes:
            X_c = X[y == c]
            word_tfidf_sum = X_c.sum(axis=0) + 1  # Laplace smoothing
            total_tfidf = word_tfidf_sum.sum()
            log_prob = np.log(word_tfidf_sum / total_tfidf)
            self.feature_log_prob_.append(log_prob)
        self.feature_log_prob_ = np.array(self.feature_log_prob_)

    def predict(self, X):
        log_probs = self.class_log_prior_ + X @ self.feature_log_prob_.T
        return np.argmax(log_probs, axis=1)

    def predict_proba(self, X):
        log_probs = self.class_log_prior_ + X @ self.feature_log_prob_.T
        probs = np.exp(log_probs - log_probs.max(axis=1, keepdims=True))
        probs = probs / probs.sum(axis=1, keepdims=True)
        return probs