"""Generate additional valuable plots for README documentation"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.preprocessing import StandardScaler
import os
import sys

# Import from parent directory
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from ard_classifier import ARDClassifier

# Create plots directory
os.makedirs('docs/plots', exist_ok=True)


def plot_convergence_behavior():
    """Create ELBO convergence plot"""
    print("Generating convergence behavior plot...")
    
    # Generate data
    X, y = make_classification(
        n_samples=300,
        n_features=15,
        n_informative=5,
        random_state=42
    )
    X = StandardScaler().fit_transform(X)
    
    # Create a modified ARD classifier that tracks ELBO
    class ARDWithHistory(ARDClassifier):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.elbo_history = []
            self.alpha_history = []
        
        def _elbo(self, X, y):
            """Compute the current ELBO value"""
            # Get current ELBO (this would need to be implemented in the actual class)
            # For demonstration, we'll compute a simplified version
            n_samples = X.shape[0]
            
            # Expected log likelihood (simplified)
            z = X @ self.coef_ + self.intercept_
            p = 1 / (1 + np.exp(-z))
            p = np.clip(p, 1e-10, 1-1e-10)
            log_lik = np.sum(y * np.log(p) + (1-y) * np.log(1-p))
            
            # Expected log prior
            log_prior = -0.5 * np.sum(self.alpha_ * (self.coef_**2 + self.coef_sigma2_))
            
            # Entropy
            entropy = 0.5 * np.sum(1 + np.log(2*np.pi) + np.log(self.coef_sigma2_))
            
            return log_lik + log_prior + entropy
    
    # Train with tracking
    ard = ARDWithHistory(
        learning_rate=0.02,
        max_iter=200,
        verbose=0,
        random_state=42
    )
    
    # Manually track convergence (simplified)
    elbo_values = []
    alpha_values = []
    
    for i in range(0, 200, 10):
        ard_temp = ARDClassifier(
            learning_rate=0.02,
            max_iter=i+1,
            verbose=0,
            random_state=42
        )
        ard_temp.fit(X, y)
        
        # Simplified ELBO computation
        z = X @ ard_temp.coef_ + ard_temp.intercept_
        p = 1 / (1 + np.exp(-np.clip(z, -250, 250)))
        p = np.clip(p, 1e-10, 1-1e-10)
        log_lik = np.mean(y * np.log(p) + (1-y) * np.log(1-p))
        elbo_values.append(log_lik)
        alpha_values.append(np.mean(ard_temp.alpha_))
    
    # Create plots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    
    iterations = range(0, 200, 10)
    
    # Plot 1: ELBO convergence
    ax1.plot(iterations, elbo_values, 'b-', linewidth=2, marker='o', markersize=4)
    ax1.set_xlabel('Iteration')
    ax1.set_ylabel('Log-Likelihood (proxy for ELBO)')
    ax1.set_title('Convergence Behavior - Training Progress')
    ax1.grid(True, alpha=0.3)
    ax1.axhline(y=max(elbo_values)*0.99, color='red', linestyle='--', 
                alpha=0.7, label='Near convergence')
    ax1.legend()
    
    # Plot 2: Alpha evolution
    ax2.plot(iterations, alpha_values, 'g-', linewidth=2, marker='s', markersize=4)
    ax2.set_xlabel('Iteration')
    ax2.set_ylabel('Mean Alpha Value')
    ax2.set_title('Precision Parameter Evolution')
    ax2.grid(True, alpha=0.3)
    ax2.set_yscale('log')
    
    plt.tight_layout()
    plt.savefig('docs/plots/convergence.png', dpi=150, bbox_inches='tight')
    plt.close()


def plot_uncertainty_visualization():
    """Create uncertainty quantification visualization"""
    print("Generating uncertainty visualization plot...")
    
    # Generate data with noise
    X, y = make_classification(
        n_samples=200,
        n_features=20,
        n_informative=5,
        n_redundant=2,
        flip_y=0.1,
        random_state=42
    )
    X = StandardScaler().fit_transform(X)
    
    # Train ARD classifier
    ard = ARDClassifier(
        learning_rate=0.02,
        max_iter=200,
        verbose=0,
        random_state=42
    )
    ard.fit(X, y)
    
    # Get uncertainties
    variances = ard.get_posterior_variance()
    std_devs = np.sqrt(variances)
    
    # Create figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Plot 1: Coefficient values with error bars
    indices = np.arange(len(ard.coef_))
    colors = ['steelblue' if imp > np.percentile(ard.feature_importances_, 70) 
              else 'lightcoral' for imp in ard.feature_importances_]
    
    bars = ax1.bar(indices, ard.coef_, yerr=2*std_devs, 
                   color=colors, alpha=0.8, capsize=3)
    ax1.set_xlabel('Feature Index')
    ax1.set_ylabel('Coefficient Value ± 2σ')
    ax1.set_title('Coefficient Estimates with Uncertainty')
    ax1.grid(True, alpha=0.3, axis='y')
    ax1.axhline(y=0, color='black', linestyle='-', alpha=0.5)
    
    # Add legend
    ax1.bar([], [], color='steelblue', alpha=0.8, label='Important features')
    ax1.bar([], [], color='lightcoral', alpha=0.8, label='Less important features')
    ax1.legend()
    
    # Plot 2: Uncertainty vs Importance
    ax2.scatter(ard.feature_importances_, std_devs, 
                c=colors, alpha=0.7, s=60, edgecolors='black', linewidth=0.5)
    ax2.set_xlabel('Feature Importance (1/α)')
    ax2.set_ylabel('Posterior Standard Deviation')
    ax2.set_title('Uncertainty vs Feature Importance')
    ax2.grid(True, alpha=0.3)
    
    # Add trend line
    z = np.polyfit(ard.feature_importances_, std_devs, 1)
    p = np.poly1d(z)
    x_trend = np.linspace(min(ard.feature_importances_), 
                         max(ard.feature_importances_), 100)
    ax2.plot(x_trend, p(x_trend), "r--", alpha=0.8, 
             label=f'Trend (slope={z[0]:.3f})')
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig('docs/plots/uncertainty.png', dpi=150, bbox_inches='tight')
    plt.close()


def plot_feature_selection_process():
    """Visualize the feature selection process across iterations"""
    print("Generating feature selection process plot...")
    
    # Generate data with clear signal
    X, y = make_classification(
        n_samples=400,
        n_features=25,
        n_informative=5,
        n_redundant=5,
        random_state=42
    )
    X = StandardScaler().fit_transform(X)
    
    # Track feature importance evolution
    iterations = [10, 25, 50, 100, 200]
    importance_evolution = []
    
    for max_iter in iterations:
        ard = ARDClassifier(
            learning_rate=0.02,
            max_iter=max_iter,
            verbose=0,
            random_state=42
        )
        ard.fit(X, y)
        importance_evolution.append(ard.feature_importances_.copy())
    
    # Create plot
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Plot evolution as heatmap
    importance_matrix = np.array(importance_evolution).T
    im = ax.imshow(importance_matrix, aspect='auto', cmap='viridis', 
                   interpolation='nearest')
    
    ax.set_xlabel('Training Iteration')
    ax.set_ylabel('Feature Index')
    ax.set_title('Feature Importance Evolution During Training')
    ax.set_xticks(range(len(iterations)))
    ax.set_xticklabels([f'{it} iter' for it in iterations])
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Feature Importance (1/α)', rotation=270, labelpad=20)
    
    # Highlight final important features
    final_importances = importance_evolution[-1]
    important_features = np.where(final_importances > np.percentile(final_importances, 80))[0]
    
    for feature_idx in important_features:
        ax.axhline(y=feature_idx, color='red', alpha=0.3, linewidth=2)
    
    # Add text annotation
    ax.text(0.02, 0.98, f'{len(important_features)} features selected', 
            transform=ax.transAxes, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig('docs/plots/feature_selection_evolution.png', dpi=150, bbox_inches='tight')
    plt.close()


if __name__ == "__main__":
    print("Generating additional README plots...")
    plot_convergence_behavior()
    plot_uncertainty_visualization()
    plot_feature_selection_process()
    print("Additional plots generated successfully!")
    print("Plots saved in docs/plots/") 