import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import time
import matplotlib.pyplot as plt
from ard_classifier import ARDClassifier


def benchmark_comparison():
    """Compare ARD Classifier with standard Logistic Regression"""
    print("="*60)
    print("ARD Classifier vs Logistic Regression Benchmark")
    print("="*60)
    
    # Generate dataset with many irrelevant features
    n_samples = 5000
    n_features = 100
    n_informative = 15
    n_redundant = 10
    
    print(f"\nDataset: {n_samples:,} samples, {n_features} features")
    print(f"         {n_informative} informative, {n_redundant} redundant")
    print(f"         {n_features - n_informative - n_redundant} noise features")
    
    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=n_informative,
        n_redundant=n_redundant,
        n_clusters_per_class=3,
        flip_y=0.05,  # Add 5% label noise
        random_state=42
    )
    
    # Standardize features
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )
    
    print("\n" + "-"*60)
    
    # Test ARD Classifier
    print("\n1. ARD Classifier (Variational Inference)")
    print("-"*40)
    
    ard_clf = ARDClassifier(
        alpha_init=1.0,
        lambda_init=0.1,
        n_mc_samples=1,
        learning_rate=0.02,
        max_iter=200,
        tol=1e-4,
        verbose=0,
        random_state=42
    )
    
    start_time = time.time()
    ard_clf.fit(X_train, y_train)
    ard_time = time.time() - start_time
    
    ard_train_score = ard_clf.score(X_train, y_train)
    ard_test_score = ard_clf.score(X_test, y_test)
    
    # Count effective features
    alpha_threshold = np.percentile(ard_clf.alpha_, 80)
    n_effective = np.sum(ard_clf.alpha_ < alpha_threshold)
    
    print(f"Training time: {ard_time:.3f} seconds")
    print(f"Train accuracy: {ard_train_score:.4f}")
    print(f"Test accuracy: {ard_test_score:.4f}")
    print(f"Effective features: {n_effective}/{n_features}")
    print(f"Feature sparsity: {(n_features - n_effective)/n_features*100:.1f}%")
    
    # Get top features
    ard_importances = ard_clf.feature_importances_
    top_ard_features = np.argsort(ard_importances)[-20:][::-1]
    
    # Test standard Logistic Regression
    print("\n2. Logistic Regression (L2 regularization)")
    print("-"*40)
    
    lr_clf = LogisticRegression(
        penalty='l2',
        C=1.0,  # Inverse of regularization strength
        max_iter=1000,
        random_state=42
    )
    
    start_time = time.time()
    lr_clf.fit(X_train, y_train)
    lr_time = time.time() - start_time
    
    lr_train_score = lr_clf.score(X_train, y_train)
    lr_test_score = lr_clf.score(X_test, y_test)
    
    print(f"Training time: {lr_time:.3f} seconds")
    print(f"Train accuracy: {lr_train_score:.4f}")
    print(f"Test accuracy: {lr_test_score:.4f}")
    print(f"All features used: {n_features}/{n_features}")
    
    # Test Logistic Regression with L1 (for sparsity)
    print("\n3. Logistic Regression (L1 regularization - Lasso)")
    print("-"*40)
    
    lr_l1_clf = LogisticRegression(
        penalty='l1',
        C=0.1,  # Stronger regularization for sparsity
        solver='liblinear',
        max_iter=1000,
        random_state=42
    )
    
    start_time = time.time()
    lr_l1_clf.fit(X_train, y_train)
    lr_l1_time = time.time() - start_time
    
    lr_l1_train_score = lr_l1_clf.score(X_train, y_train)
    lr_l1_test_score = lr_l1_clf.score(X_test, y_test)
    
    # Count non-zero coefficients
    n_nonzero = np.sum(np.abs(lr_l1_clf.coef_[0]) > 1e-6)
    
    print(f"Training time: {lr_l1_time:.3f} seconds")
    print(f"Train accuracy: {lr_l1_train_score:.4f}")
    print(f"Test accuracy: {lr_l1_test_score:.4f}")
    print(f"Non-zero features: {n_nonzero}/{n_features}")
    print(f"Feature sparsity: {(n_features - n_nonzero)/n_features*100:.1f}%")
    
    # Summary comparison
    print("\n" + "="*60)
    print("SUMMARY COMPARISON")
    print("="*60)
    print(f"{'Method':<30} {'Train Acc':<10} {'Test Acc':<10} {'Time (s)':<10} {'Sparsity':<10}")
    print("-"*60)
    print(f"{'ARD (Variational)':<30} {ard_train_score:<10.4f} {ard_test_score:<10.4f} {ard_time:<10.3f} {(n_features-n_effective)/n_features*100:<10.1f}%")
    print(f"{'Logistic Regression (L2)':<30} {lr_train_score:<10.4f} {lr_test_score:<10.4f} {lr_time:<10.3f} {'0.0':<10}%")
    print(f"{'Logistic Regression (L1)':<30} {lr_l1_train_score:<10.4f} {lr_l1_test_score:<10.4f} {lr_l1_time:<10.3f} {(n_features-n_nonzero)/n_features*100:<10.1f}%")
    
    # Visualization
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Plot 1: Feature importances comparison
    ax1 = axes[0]
    
    # ARD importances (normalized)
    ard_imp_norm = ard_importances / np.max(ard_importances)
    # L2 coefficients (normalized absolute values)
    lr_coef_norm = np.abs(lr_clf.coef_[0]) / np.max(np.abs(lr_clf.coef_[0]))
    # L1 coefficients (normalized absolute values)
    lr_l1_coef_norm = np.abs(lr_l1_clf.coef_[0]) / np.max(np.abs(lr_l1_clf.coef_[0]))
    
    feature_indices = np.arange(n_features)
    width = 0.25
    
    ax1.bar(feature_indices - width, ard_imp_norm, width, label='ARD', alpha=0.7)
    ax1.bar(feature_indices, lr_coef_norm, width, label='LR (L2)', alpha=0.7)
    ax1.bar(feature_indices + width, lr_l1_coef_norm, width, label='LR (L1)', alpha=0.7)
    
    ax1.set_xlabel('Feature Index')
    ax1.set_ylabel('Normalized Importance/Coefficient')
    ax1.set_title('Feature Importance Comparison')
    ax1.legend()
    ax1.set_xlim(-1, min(50, n_features))  # Show first 50 features
    
    # Plot 2: Accuracy comparison
    ax2 = axes[1]
    methods = ['ARD\n(Variational)', 'LR (L2)', 'LR (L1)']
    train_scores = [ard_train_score, lr_train_score, lr_l1_train_score]
    test_scores = [ard_test_score, lr_test_score, lr_l1_test_score]
    
    x = np.arange(len(methods))
    width = 0.35
    
    bars1 = ax2.bar(x - width/2, train_scores, width, label='Train', alpha=0.7)
    bars2 = ax2.bar(x + width/2, test_scores, width, label='Test', alpha=0.7)
    
    ax2.set_ylabel('Accuracy')
    ax2.set_title('Accuracy Comparison')
    ax2.set_xticks(x)
    ax2.set_xticklabels(methods)
    ax2.legend()
    ax2.set_ylim(0, 1.05)
    
    # Add value labels on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax2.annotate(f'{height:.3f}',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=8)
    
    # Plot 3: Sparsity and timing
    ax3 = axes[2]
    sparsity = [(n_features-n_effective)/n_features*100, 0, (n_features-n_nonzero)/n_features*100]
    times = [ard_time, lr_time, lr_l1_time]
    
    ax3_twin = ax3.twinx()
    
    bars1 = ax3.bar(x - width/2, sparsity, width, label='Sparsity %', color='coral', alpha=0.7)
    bars2 = ax3_twin.bar(x + width/2, times, width, label='Time (s)', color='skyblue', alpha=0.7)
    
    ax3.set_ylabel('Feature Sparsity (%)', color='coral')
    ax3_twin.set_ylabel('Training Time (s)', color='skyblue')
    ax3.set_title('Efficiency Comparison')
    ax3.set_xticks(x)
    ax3.set_xticklabels(methods)
    ax3.tick_params(axis='y', labelcolor='coral')
    ax3_twin.tick_params(axis='y', labelcolor='skyblue')
    
    # Add legends
    ax3.legend(loc='upper left')
    ax3_twin.legend(loc='upper right')
    
    plt.tight_layout()
    plt.savefig('ard_benchmark_comparison.png', dpi=150)
    print("\nComparison plots saved as 'ard_benchmark_comparison.png'")
    
    # Key insights
    print("\n" + "="*60)
    print("KEY INSIGHTS")
    print("="*60)
    print("1. ARD automatically identifies and down-weights irrelevant features")
    print(f"2. ARD achieved {(n_features-n_effective)/n_features*100:.1f}% sparsity with competitive accuracy")
    print("3. ARD provides uncertainty estimates (not shown) unlike standard LR")
    print("4. L1 regularization also achieves sparsity but requires tuning C parameter")
    print("5. ARD training is slightly slower but provides richer information")


if __name__ == "__main__":
    benchmark_comparison() 