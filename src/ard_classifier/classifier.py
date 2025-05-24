import numpy as np
from scipy.special import expit
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.preprocessing import LabelBinarizer
from sklearn.utils.validation import check_array, check_is_fitted, check_X_y


class ARDClassifier(BaseEstimator, ClassifierMixin):
    """
    Automatic Relevance Determination (ARD) Classifier

    A Bayesian logistic regression classifier with automatic relevance determination
    that uses ELBO (Evidence Lower Bound) maximization for learning with variational
    inference and the reparameterization trick.

    Parameters
    ----------
    alpha_init : float, default=1.0
        Initial value for the precision parameters (inverse variance)

    lambda_init : float, default=1.0
        Initial value for the regularization parameter

    n_mc_samples : int, default=1
        Number of Monte Carlo samples for estimating the expected log-likelihood

    learning_rate : float, default=0.01
        Learning rate for Adam optimizer

    max_iter : int, default=1000
        Maximum number of iterations for optimization

    tol : float, default=1e-4
        Tolerance for convergence

    verbose : int, default=0
        Verbosity level

    random_state : int, RandomState instance or None, default=None
        Random state for reproducibility
    """

    def __init__(
        self,
        alpha_init=1.0,
        lambda_init=1.0,
        n_mc_samples=1,
        learning_rate=0.01,
        max_iter=1000,
        tol=1e-4,
        verbose=0,
        random_state=None,
    ):
        self.alpha_init = alpha_init
        self.lambda_init = lambda_init
        self.n_mc_samples = n_mc_samples
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.tol = tol
        self.verbose = verbose
        self.random_state = random_state

    def _sigmoid(self, z):
        """Numerically stable sigmoid function"""
        return expit(z)

    def _log_sigmoid(self, z):
        """Numerically stable log-sigmoid"""
        return -np.log1p(np.exp(-z))

    def _softplus(self, x):
        """Numerically stable softplus function"""
        return np.log1p(np.exp(-np.abs(x))) + np.maximum(x, 0)

    def _sample_weights(self, mu, log_sigma2, n_samples=1):
        """
        Sample weights using reparameterization trick
        w = mu + sigma * epsilon, where epsilon ~ N(0, 1)
        """
        sigma = np.exp(0.5 * log_sigma2)
        if n_samples == 1:
            epsilon = self.rng_.randn(len(mu))
            return mu + sigma * epsilon
        else:
            epsilon = self.rng_.randn(n_samples, len(mu))
            return mu[None, :] + sigma[None, :] * epsilon

    def _compute_elbo(self, X, y, mu, log_sigma2, alpha, lambda_):
        """
        Compute the Evidence Lower Bound (ELBO) using reparameterization trick

        ELBO = E_q[log p(y|X,w)] + E_q[log p(w|alpha)] - E_q[log q(w)]
        """
        n_samples, n_features = X.shape
        sigma2 = np.exp(log_sigma2)

        # Expected log-likelihood using MC sampling
        log_likelihood = 0.0
        for _ in range(self.n_mc_samples):
            w_sample = self._sample_weights(mu, log_sigma2, n_samples=1)
            z = X @ w_sample
            log_likelihood += np.sum(
                y * self._log_sigmoid(z) + (1 - y) * self._log_sigmoid(-z)
            )
        log_likelihood /= self.n_mc_samples

        # Expected log-prior: E_q[log p(w|alpha)]
        # E[w^2] = mu^2 + sigma^2
        expected_w_squared = mu**2 + sigma2
        log_prior = -0.5 * np.sum(alpha * expected_w_squared) + 0.5 * np.sum(
            np.log(alpha)
        )

        # Entropy of q(w): -E_q[log q(w)] = 0.5 * sum(1 + log(2*pi) + log(sigma^2))
        entropy = 0.5 * np.sum(1 + np.log(2 * np.pi) + log_sigma2)

        # Regularization term on mean
        reg_term = -lambda_ * np.sum(mu**2)

        elbo = log_likelihood + log_prior + entropy + reg_term
        return elbo

    def _compute_gradients(self, X, y, mu, log_sigma2, alpha, lambda_):
        """
        Compute gradients of ELBO w.r.t. mu and log_sigma2 using reparameterization trick
        """
        n_samples = X.shape[0]
        sigma2 = np.exp(log_sigma2)
        sigma = np.sqrt(sigma2)

        # Initialize gradients
        grad_mu = np.zeros_like(mu)
        grad_log_sigma2 = np.zeros_like(log_sigma2)

        # MC estimation of likelihood gradients
        for _ in range(self.n_mc_samples):
            epsilon = self.rng_.randn(len(mu))
            w_sample = mu + sigma * epsilon

            z = X @ w_sample
            p = self._sigmoid(z)

            # Gradient w.r.t. w
            grad_w = -X.T @ (p - y)  # Negative because we maximize ELBO

            # Gradients using reparameterization trick
            grad_mu += grad_w
            grad_log_sigma2 += 0.5 * grad_w * sigma * epsilon

        grad_mu /= self.n_mc_samples
        grad_log_sigma2 /= self.n_mc_samples

        # Add prior gradients
        grad_mu += -alpha * mu - 2 * lambda_ * mu
        grad_log_sigma2 += -0.5 * alpha * sigma2 + 0.5

        return grad_mu, grad_log_sigma2

    def _update_hyperparameters(self, mu, log_sigma2):
        """Update hyperparameters (alpha) using empirical Bayes"""
        sigma2 = np.exp(log_sigma2)

        # Expected value of w^2 under q(w)
        expected_w_squared = mu**2 + sigma2

        # Update alpha using the standard ARD update
        self.alpha_ = 1.0 / (expected_w_squared + 1e-10)

        # Clip alpha to reasonable values
        self.alpha_ = np.clip(self.alpha_, 1e-10, 1e10)

    def fit(self, X, y):
        """
        Fit the ARD classifier using ELBO maximization with variational inference

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data
        y : array-like of shape (n_samples,)
            Target values (binary: 0 or 1)

        Returns
        -------
        self : object
            Fitted estimator
        """
        X, y = check_X_y(X, y, accept_sparse=False)

        # Handle multi-class by checking unique labels
        self._label_binarizer = LabelBinarizer()
        Y = self._label_binarizer.fit_transform(y)

        self.classes_ = self._label_binarizer.classes_
        n_classes = len(self.classes_)

        if n_classes > 2:
            raise ValueError(
                "ARDClassifier currently only supports binary classification"
            )

        # For binary classification, sklearn's LabelBinarizer returns a column vector
        if n_classes == 2:
            y_binary = Y.ravel()
        else:
            y_binary = Y

        n_samples, n_features = X.shape

        # Set random state
        self.rng_ = np.random.RandomState(self.random_state)

        # Initialize variational parameters
        self.mu_ = self.rng_.randn(n_features + 1) * 0.01  # +1 for intercept
        self.log_sigma2_ = np.full(n_features + 1, -2.3)  # log(0.1^2) ≈ -2.3

        # Initialize hyperparameters
        self.alpha_ = np.full(n_features, self.alpha_init)
        self.lambda_ = self.lambda_init

        # Add intercept column to X
        X_with_intercept = np.column_stack([np.ones(n_samples), X])

        # Adam optimizer parameters
        m_mu = np.zeros_like(self.mu_)
        v_mu = np.zeros_like(self.mu_)
        m_log_sigma2 = np.zeros_like(self.log_sigma2_)
        v_log_sigma2 = np.zeros_like(self.log_sigma2_)
        beta1 = 0.9
        beta2 = 0.999
        epsilon = 1e-8

        # Optimization loop
        prev_elbo = -np.inf

        for iteration in range(self.max_iter):
            # Include small alpha for intercept
            alpha_full = np.concatenate([[1e-6], self.alpha_])

            # Compute gradients
            grad_mu, grad_log_sigma2 = self._compute_gradients(
                X_with_intercept,
                y_binary,
                self.mu_,
                self.log_sigma2_,
                alpha_full,
                self.lambda_,
            )

            # Adam updates
            t = iteration + 1

            # Update biased first moment estimate
            m_mu = beta1 * m_mu + (1 - beta1) * grad_mu
            m_log_sigma2 = beta1 * m_log_sigma2 + (1 - beta1) * grad_log_sigma2

            # Update biased second raw moment estimate
            v_mu = beta2 * v_mu + (1 - beta2) * grad_mu**2
            v_log_sigma2 = beta2 * v_log_sigma2 + (1 - beta2) * grad_log_sigma2**2

            # Compute bias-corrected estimates
            m_mu_hat = m_mu / (1 - beta1**t)
            m_log_sigma2_hat = m_log_sigma2 / (1 - beta1**t)
            v_mu_hat = v_mu / (1 - beta2**t)
            v_log_sigma2_hat = v_log_sigma2 / (1 - beta2**t)

            # Update parameters
            self.mu_ += self.learning_rate * m_mu_hat / (np.sqrt(v_mu_hat) + epsilon)
            self.log_sigma2_ += (
                self.learning_rate
                * m_log_sigma2_hat
                / (np.sqrt(v_log_sigma2_hat) + epsilon)
            )

            # Ensure log_sigma2 doesn't get too small
            self.log_sigma2_ = np.maximum(self.log_sigma2_, -10.0)

            # Update hyperparameters every 10 iterations
            if iteration % 10 == 0:
                self._update_hyperparameters(self.mu_[1:], self.log_sigma2_[1:])

            # Compute ELBO
            current_elbo = self._compute_elbo(
                X_with_intercept,
                y_binary,
                self.mu_,
                self.log_sigma2_,
                alpha_full,
                self.lambda_,
            )

            if self.verbose > 0 and iteration % 10 == 0:
                print(f"Iteration {iteration + 1}, ELBO: {current_elbo:.4f}")

            # Check convergence
            if abs(current_elbo - prev_elbo) < self.tol:
                if self.verbose > 0:
                    print(f"Converged after {iteration + 1} iterations")
                break

            prev_elbo = current_elbo

        # Extract final parameters
        self.intercept_ = self.mu_[0]
        self.coef_ = self.mu_[1:]
        self.intercept_sigma2_ = np.exp(self.log_sigma2_[0])
        self.coef_sigma2_ = np.exp(self.log_sigma2_[1:])

        return self

    def predict_proba(self, X):
        """
        Predict class probabilities using the mean of the variational distribution

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Samples

        Returns
        -------
        proba : array-like of shape (n_samples, n_classes)
            Class probabilities
        """
        check_is_fitted(self)
        X = check_array(X, accept_sparse=False)

        # Use mean prediction (MAP estimate)
        decision = X @ self.coef_ + self.intercept_
        proba_positive = self._sigmoid(decision)

        if len(self.classes_) == 2:
            proba = np.column_stack([1 - proba_positive, proba_positive])
        else:
            proba = proba_positive.reshape(-1, 1)

        return proba

    def predict(self, X):
        """
        Predict class labels

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Samples

        Returns
        -------
        y : array-like of shape (n_samples,)
            Predicted class labels
        """
        proba = self.predict_proba(X)
        indices = proba.argmax(axis=1)
        return self.classes_[indices]

    def score(self, X, y, sample_weight=None):
        """
        Return the mean accuracy on the given test data and labels

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Test samples
        y : array-like of shape (n_samples,)
            True labels for X
        sample_weight : array-like of shape (n_samples,), default=None
            Sample weights

        Returns
        -------
        score : float
            Mean accuracy
        """
        from sklearn.metrics import accuracy_score

        return accuracy_score(y, self.predict(X), sample_weight=sample_weight)

    @property
    def feature_importances_(self):
        """
        Get feature importances based on inverse alpha values

        Higher alpha means lower importance (more regularization)
        """
        check_is_fitted(self)
        return 1.0 / (self.alpha_ + 1e-10)

    def get_posterior_variance(self):
        """
        Get the posterior variance for each coefficient

        Returns
        -------
        variance : array-like of shape (n_features,)
            Posterior variance for each feature
        """
        check_is_fitted(self)
        return self.coef_sigma2_
