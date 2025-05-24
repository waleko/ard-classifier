"""Generate plots for README documentation"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
import time
import os

# Import from parent directory
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from ard_classifier import ARDClassifier

# Create plots directory
os.makedirs('docs/plots', exist_ok=True)


def plot_feature_importance_demo():
    """Create a feature importance visualization"""
    print("Generating feature importance plot...")
    
    # Generate data with known informative features
    X, y = make_classification(
        n_samples=500,
        n_features=20,
        n_informative=5,
        n_redundant=3,
        n_repeated=0,
        n_classes=2,
        flip_y=0.05,
        random_state=42
    )
    
    # Standardize
    X = StandardScaler().fit_transform(X)
    
    # Train ARD classifier
    ard = ARDClassifier(
        alpha_init=1.0,
        lambda_init=0.1,
        n_mc_samples=1,
        learning_rate=0.02,
        max_iter=200,
        random_state=42,
        verbose=0
    )
    ard.fit(X, y)
    
    # Create figure
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    
    # Plot 1: Feature importances
    importances = ard.feature_importances_
    indices = np.argsort(importances)[::-1]
    
    ax1.bar(range(len(importances)), importances[indices], 
            color='steelblue', alpha=0.8)
    ax1.set_xlabel('Feature Index (sorted by importance)')
    ax1.set_ylabel('Importance (1/α)')
    ax1.set_title('ARD Feature Importances - Automatic Feature Selection')
    ax1.set_xticks(range(len(importances)))
    ax1.set_xticklabels([str(i) for i in indices], rotation=45)
    
    # Add threshold line
    threshold = np.percentile(importances, 80)
    ax1.axhline(y=threshold, color='red', linestyle='--', 
                label=f'80th percentile threshold')
    ax1.legend()
    
    # Plot 2: Alpha values (log scale)
    alphas = ard.alpha_
    ax2.bar(range(len(alphas)), alphas[indices], 
            color='coral', alpha=0.8)
    ax2.set_xlabel('Feature Index (sorted by importance)')
    ax2.set_ylabel('Alpha (precision parameter)')
    ax2.set_title('ARD Precision Parameters - High α = Low Importance')
    ax2.set_yscale('log')
    ax2.set_xticks(range(len(alphas)))
    ax2.set_xticklabels([str(i) for i in indices], rotation=45)
    
    plt.tight_layout()
    plt.savefig('docs/plots/feature_importance.png', dpi=150, bbox_inches='tight')
    plt.close()


def plot_scalability_comparison():
    """Create scalability comparison plot"""
    print("Generating scalability comparison plot...")
    
    # Test different dataset sizes
    sizes = [(100, 10), (500, 50), (1000, 100), (5000, 200)]
    
    ard_times = []
    lr_times = []
    ard_scores = []
    lr_scores = []
    
    for n_samples, n_features in sizes:
        print(f"  Testing {n_samples} samples, {n_features} features...")
        
        # Generate data
        X, y = make_classification(
            n_samples=n_samples,
            n_features=n_features,
            n_informative=n_features // 5,
            random_state=42
        )
        X = StandardScaler().fit_transform(X)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Time ARD
        ard = ARDClassifier(
            learning_rate=0.05,
            max_iter=50,
            verbose=0,
            random_state=42
        )
        start = time.time()
        ard.fit(X_train, y_train)
        ard_time = time.time() - start
        ard_score = ard.score(X_test, y_test)
        
        ard_times.append(ard_time)
        ard_scores.append(ard_score)
        
        # Time standard LR
        lr = LogisticRegression(max_iter=1000, random_state=42)
        start = time.time()
        lr.fit(X_train, y_train)
        lr_time = time.time() - start
        lr_score = lr.score(X_test, y_test)
        
        lr_times.append(lr_time)
        lr_scores.append(lr_score)
    
    # Plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Dataset sizes for x-axis
    labels = [f'{n}×{d}' for n, d in sizes]
    
    # Plot 1: Training time
    x = np.arange(len(sizes))
    width = 0.35
    
    bars1 = ax1.bar(x - width/2, ard_times, width, label='ARD', 
                     color='steelblue', alpha=0.8)
    bars2 = ax1.bar(x + width/2, lr_times, width, label='Logistic Regression', 
                     color='coral', alpha=0.8)
    
    ax1.set_xlabel('Dataset Size (samples × features)')
    ax1.set_ylabel('Training Time (seconds)')
    ax1.set_title('Training Time Comparison')
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels)
    ax1.legend()
    ax1.set_yscale('log')
    
    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax1.annotate(f'{height:.2f}',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=9)
    
    # Plot 2: Accuracy
    ax2.plot(x, ard_scores, 'o-', linewidth=2, markersize=8, 
             label='ARD', color='steelblue')
    ax2.plot(x, lr_scores, 's-', linewidth=2, markersize=8, 
             label='Logistic Regression', color='coral')
    
    ax2.set_xlabel('Dataset Size (samples × features)')
    ax2.set_ylabel('Test Accuracy')
    ax2.set_title('Accuracy Comparison')
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels)
    ax2.legend()
    ax2.set_ylim(0.5, 1.05)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('docs/plots/scalability.png', dpi=150, bbox_inches='tight')
    plt.close()


def plot_sparsity_comparison():
    """Compare sparsity patterns between ARD and L1 regularization"""
    print("Generating sparsity comparison plot...")
    
    # Generate data with many features
    X, y = make_classification(
        n_samples=200,
        n_features=50,
        n_informative=10,
        n_redundant=5,
        random_state=42
    )
    X = StandardScaler().fit_transform(X)
    
    # Train ARD
    ard = ARDClassifier(
        learning_rate=0.02,
        max_iter=200,
        verbose=0,
        random_state=42
    )
    ard.fit(X, y)
    
    # Train L1 Logistic Regression
    lr_l1 = LogisticRegression(
        penalty='l1',
        C=0.1,
        solver='liblinear',
        max_iter=1000,
        random_state=42
    )
    lr_l1.fit(X, y)
    
    # Create figure
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
    
    # Plot 1: ARD coefficients with importance
    importance_threshold = np.percentile(ard.feature_importances_, 80)
    colors = ['green' if imp > importance_threshold else 'lightgray' 
              for imp in ard.feature_importances_]
    
    bars1 = ax1.bar(range(len(ard.coef_)), ard.coef_, color=colors, alpha=0.8)
    ax1.set_xlabel('Feature Index')
    ax1.set_ylabel('Coefficient Value')
    ax1.set_title('ARD Coefficients (green = important features)')
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Add importance annotation
    n_important = sum(1 for c in colors if c == 'green')
    ax1.text(0.02, 0.98, f'{n_important}/{len(ard.coef_)} features selected',
             transform=ax1.transAxes, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # Plot 2: L1 coefficients
    colors_l1 = ['orange' if abs(c) > 1e-6 else 'lightgray' 
                 for c in lr_l1.coef_[0]]
    
    bars2 = ax2.bar(range(len(lr_l1.coef_[0])), lr_l1.coef_[0], 
                     color=colors_l1, alpha=0.8)
    ax2.set_xlabel('Feature Index')
    ax2.set_ylabel('Coefficient Value')
    ax2.set_title('L1 Logistic Regression Coefficients (orange = non-zero)')
    ax2.grid(True, alpha=0.3, axis='y')
    
    # Add sparsity annotation
    n_nonzero = sum(1 for c in colors_l1 if c == 'orange')
    ax2.text(0.02, 0.98, f'{n_nonzero}/{len(lr_l1.coef_[0])} non-zero coefficients',
             transform=ax2.transAxes, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig('docs/plots/sparsity_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()


if __name__ == "__main__":
    print("Generating README plots...")
    plot_feature_importance_demo()
    plot_scalability_comparison()
    plot_sparsity_comparison()
    print("All plots generated successfully!")
    print("Plots saved in docs/plots/")

