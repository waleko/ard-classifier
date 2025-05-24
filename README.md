# ARD Classifier - Automatic Relevance Determination for Logistic Regression

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

A Bayesian logistic regression classifier with Automatic Relevance Determination (ARD) using variational inference with the reparameterization trick. This implementation provides automatic feature selection and uncertainty quantification while maintaining compatibility with scikit-learn's API.

## 🚀 Installation

### Using pip
```bash
pip install ard-classifier
```

### Using uv (recommended)
```bash
uv add ard-classifier
```

**Note**: Currently supports binary classification only.

## 💡 Quick Start

```python
from ard_classifier import ARDClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
import numpy as np

# Generate data
X, y = make_classification(n_samples=200, n_features=20, n_informative=5,
                          n_redundant=5, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Create and train classifier
clf = ARDClassifier(
    alpha_init=1.0, 
    lambda_init=0.1, 
    n_mc_samples=1,  # Number of MC samples for likelihood
    learning_rate=0.02,
    max_iter=500, 
    verbose=1
)
clf.fit(X_train, y_train)

# Make predictions
y_pred = clf.predict(X_test)
accuracy = clf.score(X_test, y_test)
print(f"Accuracy: {accuracy:.4f}")

# Get feature importances and uncertainties
importances = clf.feature_importances_
uncertainties = clf.get_posterior_variance()
print(f"Top 5 important features: {importances.argsort()[-5:][::-1]}")
print(f"Posterior std for feature 0: {np.sqrt(uncertainties[0]):.4f}")
```

## 🎯 Key Features

- **Automatic Feature Selection**: Learns which features are relevant through ARD priors
- **Uncertainty Quantification**: Provides posterior variance estimates for all coefficients
- **Monte Carlo Sampling**: Flexible number of samples for likelihood estimation
- **Scikit-learn Compatible**: Follows sklearn's estimator API conventions
- **Bayesian Approach**: Provides principled probabilistic framework for classification
- **No Hyperparameter Tuning**: ARD automatically determines feature relevance
- **Binary Classification**: Currently supports binary classification problems only

## 🔬 How It Works

### Probabilistic Model

The ARD Classifier is based on a hierarchical Bayesian model for binary classification:

$$p(\mathbf{t} | \mathbf{X}, \boldsymbol{\theta}) = p(\mathbf{t} | \mathbf{X}, \mathbf{w})p(\mathbf{w} | \boldsymbol{\alpha})$$

Where $\boldsymbol{\theta} = \{\mathbf{w}, \boldsymbol{\alpha}\}$ represents all model parameters. The likelihood for binary classification with targets $t_n \in \{-1, +1\}$ is:

$$p(\mathbf{t} | \mathbf{X}, \mathbf{w}) = \prod_{n=1}^{N} \frac{1}{1 + \exp(-t_n \mathbf{x}_n^T \mathbf{w})}$$

The ARD prior assigns individual precision parameters to each weight:

$$p(\mathbf{w} | \boldsymbol{\alpha}) = \prod_{j=1}^{D} \mathcal{N}(w_j | 0, \alpha_j^{-1}) = \prod_{j=1}^{D} \sqrt{\frac{\alpha_j}{2\pi}} \exp\left(-\frac{\alpha_j w_j^2}{2}\right)$$

**ARD Mechanism**: Each feature $j$ has its own precision $\alpha_j$ where high $\alpha_j$ means strong regularization (less important feature) and low $\alpha_j$ means weak regularization (important feature).

### Variational Inference and ELBO

Since the posterior $p(\mathbf{w} | \mathbf{X}, \mathbf{t}, \boldsymbol{\alpha})$ is intractable due to the sigmoid nonlinearity, we use variational inference with a factorized Gaussian approximation:

$$q(\mathbf{w}|\boldsymbol{\phi}) = \prod_{j=1}^{D} \mathcal{N}(w_j | \mu_j, \sigma_j^2)$$

The Evidence Lower Bound (ELBO) can be written as:

$$\mathcal{L}(\boldsymbol{\phi}) = \int q(\mathbf{w}|\boldsymbol{\phi}) \log p(\mathbf{t}|\mathbf{X}, \mathbf{w}) d\mathbf{w} - \text{KL}[q(\mathbf{w}|\boldsymbol{\phi}) \| p(\mathbf{w}|\boldsymbol{\alpha})]$$

The challenge is that we cannot naively take the gradient $\nabla_{\boldsymbol{\phi}} \mathcal{L}(\boldsymbol{\phi})$ because the distribution $q(\mathbf{w}|\boldsymbol{\phi})$ in the integral depends on the parameters $\boldsymbol{\phi}$ we want to optimize.

### Reparameterization Trick and Stochastic Gradients

The reparameterization trick solves this by rewriting the expectation in terms of a parameter-free distribution:

$$\mathbf{w} = \boldsymbol{\mu} + \boldsymbol{\sigma} \odot \boldsymbol{\epsilon}, \quad \boldsymbol{\epsilon} \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$$

Now we can rewrite the ELBO as:

$$\mathcal{L}(\boldsymbol{\phi}) = \mathbb{E}_{\boldsymbol{\epsilon} \sim \mathcal{N}(\mathbf{0}, \mathbf{I})}[\log p(\mathbf{t}|\mathbf{X}, \boldsymbol{\mu} + \boldsymbol{\sigma} \odot \boldsymbol{\epsilon})] - \text{KL}[q(\mathbf{w}|\boldsymbol{\phi}) \| p(\mathbf{w}|\boldsymbol{\alpha})]$$

Since $\boldsymbol{\epsilon}$ doesn't depend on $\boldsymbol{\phi}$, we can now take gradients and approximate the integral with Monte Carlo sampling:

$$\nabla_{\boldsymbol{\phi}} \mathcal{L}(\boldsymbol{\phi}) \approx \frac{1}{S} \sum_{s=1}^{S} \nabla_{\boldsymbol{\phi}} \log p(\mathbf{t}|\mathbf{X}, \boldsymbol{\mu} + \boldsymbol{\sigma} \odot \boldsymbol{\epsilon}^{(s)}) - \nabla_{\boldsymbol{\phi}} \text{KL}[q(\mathbf{w}|\boldsymbol{\phi}) \| p(\mathbf{w}|\boldsymbol{\alpha})]$$

This enables **stochastic gradient optimization** where we can:
1. Sample $\boldsymbol{\epsilon}^{(s)} \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$ 
2. Compute stochastic gradients using the MC approximation
3. Update $\boldsymbol{\phi}$ using gradient ascent (e.g., Adam optimizer)

The KL divergence has a closed form:

$$\text{KL}[q(\mathbf{w}|\boldsymbol{\phi}) \| p(\mathbf{w}|\boldsymbol{\alpha})] = \frac{1}{2} \sum_{j=1}^{D} \left[\alpha_j(\mu_j^2 + \sigma_j^2) - \log \sigma_j^2 - 1\right]$$

### Hyperparameter Updates

The precision parameters are updated using empirical Bayes (Type-II ML):

$$\alpha_j = \frac{1}{\mathbb{E}_{q(\mathbf{w})}[w_j^2]} = \frac{1}{\mu_j^2 + \sigma_j^2}$$

## 📈 Performance Characteristics

### High-Dimensional Dataset Performance

ARD performance on widely-used high-dimensional datasets from ML research:

| Dataset | Samples×Features | ARD Acc. | LogReg Acc. | L1 Acc. | ARD Features | L1 Features | Overhead |
|---------|------------------|----------|-------------|---------|--------------|-------------|----------|
| MNIST Digits (even vs odd) | 1797×64 | 91.1% | 91.5% | 91.7% | 24/64 (37.5%) | 31/64 (48%) | 120× |
| 20 Newsgroups (1000 features) | 1963×1000 | 95.2% | 95.1% | 95.1% | 380/1000 (38%) | 269/1000 (27%) | 240× |
| Madelon (500 features) | 2600×500 | 57.8% | 55.6% | 57.9% | 181/500 (36%) | 313/500 (63%) | 116× |
| Arrhythmia (279 features) | 452×279 | 75.7% | 73.5% | 75.7% | 108/279 (39%) | 35/279 (13%) | 151× |
| Colon Cancer (2000 genes) | 62×2000 | 73.7% | 73.7% | 73.7% | 629/2000 (31%) | 6/2000 (0.3%) | 101× |
| Synthetic (20/300 informative) | 500×300 | 90.0% | 84.7% | 94.7% | 91/300 (30%) | 19/300 (6%) | 354× |

**Summary**: Mean accuracy difference: +1.6% (ARD vs LogReg), Mean overhead: 180×, Mean feature selection: 35.4% of features

**Key insight**: ARD provides adaptive feature selection, selecting 30-40% of features on most datasets. The selection is data-driven and varies by dataset complexity. ARD shows particular strength on challenging datasets like Madelon and Arrhythmia where it outperforms standard logistic regression.

**Datasets used**: MNIST digits, 20 Newsgroups text classification, Madelon (designed for feature selection), Arrhythmia medical diagnosis, Colon Cancer gene expression, and controlled synthetic data.

### Synthetic Dataset Validation

Controlled experiments on synthetic data with known ground truth demonstrate ARD's adaptive feature selection:

| Dataset Configuration | True Informative | ARD Selected | L1 Selected* | ARD Accuracy | L1 Accuracy |
|-------------|------------------|--------------|-------------|--------------|-------------|
| 300 samples, 20 features, 5 informative | 5/20 | 7/20 (35%) | 8/20 (40%) | 85% | 87% |
| 500 samples, 50 features, 10 informative | 10/50 | 18/50 (36%) | 15/50 (30%) | 78% | 79% |
| 1000 samples, 100 features, 15 informative | 15/100 | 23/100 (23%) | 62/100 (62%) | 76% | 77% |

*L1 results with optimally tuned C parameter

**Key insight**: ARD typically selects 23-36% of features, adapting to the signal-to-noise ratio of each dataset. Unlike the artificial 20% claim in previous versions, ARD's selection is truly data-driven and varies appropriately with dataset characteristics.

### Computational Trade-offs

ARD provides automatic feature selection at the cost of increased computation time:

| Dataset Size | ARD Time | Logistic Regression | Overhead | Benefit |
|--------------|----------|-------------------|----------|---------|
| 100×10 | 0.05s | 0.002s | 25× | Auto feature selection |
| 1,000×100 | 0.3s | 0.02s | 15× | + Uncertainty estimates |
| 5,000×100 | 0.7s | 0.004s | 175× | + No hyperparameter tuning |
| 10,000×200 | 2.1s | 0.1s | 21× | + Bayesian framework |

**Key insight**: ARD trades computational efficiency for automatic feature selection and uncertainty quantification. Overhead varies from 15× to 175× depending on dataset characteristics, making it valuable when interpretability and avoiding hyperparameter tuning are priorities over raw speed.

## 📊 Performance Overview

The ARD Classifier offers automatic feature selection with performance comparable to standard logistic regression. While it doesn't necessarily outperform traditional methods in terms of accuracy or speed, it provides additional benefits:

### Automatic Feature Selection

The ARD mechanism automatically identifies and weights important features without manual hyperparameter tuning:

![Feature Importance](docs/plots/feature_importance.png)

*ARD assigns importance scores to features automatically. Features with higher importance (lower α values) are selected as relevant without requiring manual threshold tuning.*

### Computational Performance

ARD maintains reasonable computational overhead while providing feature selection benefits:

![Scalability Comparison](docs/plots/scalability.png)

*Training time comparison shows ARD has 15-354× overhead compared to standard logistic regression, with significantly higher overhead for very high-dimensional text data. The trade-off provides automatic feature selection without manual hyperparameter optimization.*

### Sparsity vs L1 Regularization

ARD provides automatic sparsity compared to manually tuned L1 regularization:

![Sparsity Comparison](docs/plots/sparsity_comparison.png)

*Top: ARD coefficients with automatic importance detection (green = important features)*  
*Bottom: L1 logistic regression requires manual tuning of regularization strength C*

ARD automatically determines feature relevance without requiring manual hyperparameter tuning, while L1 regularization performance depends on careful selection of the regularization strength.

## 🛠️ Advanced Usage

### Uncertainty-Aware Predictions

```python
# Get posterior variance for each coefficient
variances = clf.get_posterior_variance()

# Make predictions with uncertainty
# (requires custom implementation)
def predict_with_uncertainty(clf, X, n_samples=100):
    predictions = []
    for _ in range(n_samples):
        # Sample from posterior
        w_sample = np.random.normal(clf.coef_, np.sqrt(variances))
        # Make prediction with sampled weights
        z = X @ w_sample + clf.intercept_
        predictions.append(sigmoid(z))
    
    mean_pred = np.mean(predictions, axis=0)
    std_pred = np.std(predictions, axis=0)
    return mean_pred, std_pred
```

### Custom ELBO Monitoring

```python
# Track ELBO during training
class ARDWithHistory(ARDClassifier):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.elbo_history = []
    
    # Override fit method to track ELBO
    # (implementation details in source)
```

### Generating Visualizations

Create your own plots to understand ARD behavior:

```python
# Generate the plots shown in this README
from ard_classifier.examples.generate_readme_plots import (
    plot_feature_importance_demo,
    plot_scalability_comparison, 
    plot_sparsity_comparison
)

# Run individual plot functions
plot_feature_importance_demo()
plot_scalability_comparison()
plot_sparsity_comparison()
```

## 📋 API Reference

### Parameters

- `alpha_init` (float, default=1.0): Initial value for precision parameters
- `lambda_init` (float, default=1.0): Initial regularization parameter
- `n_mc_samples` (int, default=1): Number of Monte Carlo samples for likelihood estimation
- `learning_rate` (float, default=0.01): Learning rate for Adam optimizer
- `max_iter` (int, default=1000): Maximum optimization iterations
- `tol` (float, default=1e-4): Convergence tolerance
- `verbose` (int, default=0): Verbosity level
- `random_state` (int/None, default=None): Random seed

### Methods

- `fit(X, y)`: Fit the ARD classifier
- `predict(X)`: Predict class labels
- `predict_proba(X)`: Predict class probabilities
- `score(X, y)`: Return mean accuracy
- `get_posterior_variance()`: Get posterior variance for each coefficient

### Attributes

- `coef_`: Coefficient values (mean of posterior)
- `intercept_`: Intercept value
- `alpha_`: Precision parameters for each feature
- `feature_importances_`: Feature importance scores (1/α)
- `coef_sigma2_`: Posterior variance for each coefficient

## 🧪 Testing

Run tests with pytest:
```bash
pytest tests/ -v --cov=ard_classifier
```

Current test coverage: **94%**

### Running Benchmarks

Compare ARD performance with standard methods:
```bash
python examples/benchmark_comparison.py
```

Evaluate on widely-used high-dimensional ML datasets:
```bash
python examples/real_dataset_evaluation.py
```

## 📚 References

1. Tipping, M. E. (2001). **Sparse Bayesian learning and the relevance vector machine**. Journal of Machine Learning Research.
2. Kingma, D. P., & Welling, M. (2013). **Auto-encoding variational bayes**. arXiv preprint arXiv:1312.6114.
3. Bishop, C. M. (2006). **Pattern Recognition and Machine Learning**. Springer. 