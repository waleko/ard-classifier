"""Basic usage example of ARD Classifier"""

import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from ard_classifier import ARDClassifier


def main():
    """Demonstrate basic usage of ARD Classifier"""
    print("ARD Classifier Basic Usage Example")
    print("=" * 50)

    # Generate synthetic data
    print("\n1. Generating synthetic data...")
    X, y = make_classification(
        n_samples=300,
        n_features=20,
        n_informative=5,
        n_redundant=5,
        n_repeated=0,
        n_classes=2,
        flip_y=0.1,  # Add 10% label noise
        random_state=42,
    )

    # Standardize features
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    print(f"   Training samples: {len(X_train)}")
    print(f"   Test samples: {len(X_test)}")
    print(f"   Features: {X.shape[1]}")

    # Create and train classifier
    print("\n2. Training ARD Classifier...")
    clf = ARDClassifier(
        alpha_init=1.0,
        lambda_init=0.1,
        n_mc_samples=1,
        learning_rate=0.02,
        max_iter=200,
        tol=1e-4,
        verbose=0,  # Set to 1 to see training progress
        random_state=42,
    )

    clf.fit(X_train, y_train)
    print("   Training complete!")

    # Evaluate
    print("\n3. Evaluation:")
    train_score = clf.score(X_train, y_train)
    test_score = clf.score(X_test, y_test)
    print(f"   Train accuracy: {train_score:.3f}")
    print(f"   Test accuracy: {test_score:.3f}")

    # Feature analysis
    print("\n4. Feature Analysis:")
    importances = clf.feature_importances_
    top_features = np.argsort(importances)[-10:][::-1]

    print("   Top 10 most important features:")
    for i, idx in enumerate(top_features):
        print(
            f"   Feature {idx:2d}: importance={importances[idx]:.4f}, "
            f"α={clf.alpha_[idx]:.2e}"
        )

    # Count effective features
    alpha_threshold = np.percentile(clf.alpha_, 80)
    n_effective = np.sum(clf.alpha_ < alpha_threshold)
    print(f"\n   Effective features: {n_effective}/{X.shape[1]}")
    print(f"   Feature sparsity: {(X.shape[1] - n_effective) / X.shape[1] * 100:.1f}%")

    # Uncertainty quantification
    print("\n5. Uncertainty Quantification:")
    variances = clf.get_posterior_variance()
    print("   Posterior std for top 5 features:")
    for i, idx in enumerate(top_features[:5]):
        print(f"   Feature {idx:2d}: σ={np.sqrt(variances[idx]):.4f}")

    # Make predictions with probabilities
    print("\n6. Sample Predictions:")
    sample_indices = [0, 1, 2]
    X_samples = X_test[sample_indices]
    y_true = y_test[sample_indices]
    y_pred = clf.predict(X_samples)
    y_proba = clf.predict_proba(X_samples)

    for i in range(len(sample_indices)):
        print(
            f"   Sample {i}: true={y_true[i]}, pred={y_pred[i]}, "
            f"prob=[{y_proba[i, 0]:.3f}, {y_proba[i, 1]:.3f}]"
        )


if __name__ == "__main__":
    main() 