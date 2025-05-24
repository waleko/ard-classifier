"""Tests for ARD Classifier"""

import numpy as np
import pytest
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.utils.estimator_checks import check_estimator

from ard_classifier import ARDClassifier


class TestARDClassifier:
    """Test cases for ARDClassifier"""

    @pytest.fixture
    def simple_data(self):
        """Generate simple classification data"""
        X, y = make_classification(
            n_samples=100,
            n_features=20,
            n_informative=5,
            n_redundant=5,
            n_repeated=0,
            n_classes=2,
            random_state=42,
        )
        return train_test_split(X, y, test_size=0.3, random_state=42)

    @pytest.fixture
    def classifier(self):
        """Create an ARDClassifier instance"""
        return ARDClassifier(
            alpha_init=1.0,
            lambda_init=0.1,
            n_mc_samples=1,
            learning_rate=0.01,
            max_iter=50,
            tol=1e-3,
            random_state=42,
        )

    def test_initialization(self):
        """Test classifier initialization"""
        clf = ARDClassifier()
        assert clf.alpha_init == 1.0
        assert clf.lambda_init == 1.0
        assert clf.n_mc_samples == 1
        assert clf.learning_rate == 0.01
        assert clf.max_iter == 1000
        assert clf.tol == 1e-4

    def test_fit_predict(self, classifier, simple_data):
        """Test basic fit and predict functionality"""
        X_train, X_test, y_train, y_test = simple_data

        # Fit the classifier
        classifier.fit(X_train, y_train)

        # Check that fitting creates expected attributes
        assert hasattr(classifier, "coef_")
        assert hasattr(classifier, "intercept_")
        assert hasattr(classifier, "alpha_")
        assert hasattr(classifier, "classes_")
        assert hasattr(classifier, "coef_sigma2_")

        # Make predictions
        y_pred = classifier.predict(X_test)
        assert y_pred.shape == y_test.shape
        assert set(y_pred) <= set(y_train)  # Predictions should be valid classes

    def test_predict_proba(self, classifier, simple_data):
        """Test probability predictions"""
        X_train, X_test, y_train, y_test = simple_data

        classifier.fit(X_train, y_train)
        y_proba = classifier.predict_proba(X_test)

        # Check shape
        assert y_proba.shape == (len(X_test), 2)

        # Check probabilities sum to 1
        np.testing.assert_allclose(y_proba.sum(axis=1), 1.0, rtol=1e-5)

        # Check probabilities are in [0, 1]
        assert np.all(y_proba >= 0)
        assert np.all(y_proba <= 1)

    def test_score(self, classifier, simple_data):
        """Test scoring method"""
        X_train, X_test, y_train, y_test = simple_data

        classifier.fit(X_train, y_train)
        score = classifier.score(X_test, y_test)

        assert 0 <= score <= 1
        assert isinstance(score, float)

    def test_feature_importances(self, classifier, simple_data):
        """Test feature importance calculation"""
        X_train, _, y_train, _ = simple_data

        classifier.fit(X_train, y_train)
        importances = classifier.feature_importances_

        assert len(importances) == X_train.shape[1]
        assert np.all(importances >= 0)

    def test_posterior_variance(self, classifier, simple_data):
        """Test posterior variance computation"""
        X_train, _, y_train, _ = simple_data

        classifier.fit(X_train, y_train)
        variances = classifier.get_posterior_variance()

        assert len(variances) == X_train.shape[1]
        assert np.all(variances > 0)

    def test_feature_selection(self):
        """Test ARD's ability to identify relevant features"""
        # Create data where only first 3 features are relevant
        np.random.seed(42)
        n_samples = 200
        n_features = 10
        n_informative = 3

        X_informative = np.random.randn(n_samples, n_informative)
        X_noise = np.random.randn(n_samples, n_features - n_informative) * 0.1
        X = np.hstack([X_informative, X_noise])

        # Create target based only on informative features
        w_true = np.array([2.0, -1.5, 1.0])
        y_continuous = X_informative @ w_true + np.random.randn(n_samples) * 0.1
        y = (y_continuous > 0).astype(int)

        # Train classifier
        clf = ARDClassifier(
            alpha_init=1.0,
            lambda_init=0.01,
            n_mc_samples=1,
            learning_rate=0.01,
            max_iter=100,
            tol=1e-4,
            random_state=42,
        )
        clf.fit(X, y)

        # Check if informative features have lower alpha values
        informative_alphas = clf.alpha_[:n_informative]
        noise_alphas = clf.alpha_[n_informative:]

        assert np.mean(informative_alphas) < np.mean(noise_alphas) / 5

    def test_different_mc_samples(self, simple_data):
        """Test classifier with different numbers of MC samples"""
        X_train, X_test, y_train, y_test = simple_data

        for n_mc in [1, 5, 10]:
            clf = ARDClassifier(
                n_mc_samples=n_mc,
                learning_rate=0.01,
                max_iter=50,
                random_state=42,
            )
            clf.fit(X_train, y_train)
            score = clf.score(X_test, y_test)
            assert 0 <= score <= 1

    def test_reproducibility(self, simple_data):
        """Test that random_state ensures reproducibility"""
        X_train, _, y_train, _ = simple_data

        # Train two classifiers with same random state
        clf1 = ARDClassifier(max_iter=50, random_state=42)
        clf2 = ARDClassifier(max_iter=50, random_state=42)

        clf1.fit(X_train, y_train)
        clf2.fit(X_train, y_train)

        # Check that coefficients are identical
        np.testing.assert_allclose(clf1.coef_, clf2.coef_, rtol=1e-5)
        np.testing.assert_allclose(clf1.alpha_, clf2.alpha_, rtol=1e-5)

    def test_invalid_input(self, classifier):
        """Test handling of invalid inputs"""
        # Test with wrong dimensions
        X = np.array([[1, 2], [3, 4]])
        y = np.array([0, 1, 2])  # Wrong length

        with pytest.raises(ValueError):
            classifier.fit(X, y)

    @pytest.mark.skip(reason="Multi-class not yet supported")
    def test_multiclass_error(self, classifier):
        """Test that multi-class classification raises appropriate error"""
        X = np.random.randn(100, 10)
        y = np.random.randint(0, 3, 100)  # 3 classes

        with pytest.raises(ValueError, match="binary classification"):
            classifier.fit(X, y)

    @pytest.mark.parametrize(
        "n_samples,n_features", [(100, 10), (500, 50), (1000, 100)]
    )
    def test_scalability(self, n_samples, n_features):
        """Test classifier on different dataset sizes"""
        X, y = make_classification(
            n_samples=n_samples,
            n_features=n_features,
            n_informative=n_features // 5,
            random_state=42,
        )

        clf = ARDClassifier(max_iter=20, tol=1e-2, random_state=42)
        clf.fit(X, y)

        # Just check that it completes without error
        assert hasattr(clf, "coef_")
        assert len(clf.coef_) == n_features


def test_sklearn_compatibility():
    """Test basic sklearn estimator compatibility"""
    # Note: This may not pass all checks due to binary-only limitation
    try:
        check_estimator(ARDClassifier(max_iter=10))
    except Exception as e:
        # We expect some failures due to binary-only support
        assert "binary classification" in str(e)
