"""
Isolating the Irreducible Error Floor in Function Approximation
=================================================================

Extension of Unit 1, Module 1 (Function Approximation): Y = f(X) + eps.

The module explains that prediction error splits into a REDUCIBLE piece
(how far your model f_hat is from the true f) and an IRREDUCIBLE piece
(the noise variance, sigma^2, which no model can ever fit away). In real
data you can never actually see this split, because you never know the
true f.

This script sidesteps that limitation: it defines its own ground-truth
function f(x), generates noisy data from it, fits polynomial models of
increasing flexibility, and evaluates each fit two different ways:

  1. Against the NOISY test targets  -> total error   (reducible + irreducible)
  2. Against the TRUE noiseless f(x) -> reducible error only

The gap between those two curves should hover right around sigma^2,
regardless of model flexibility, while the reducible-error curve should
dip (better fit) and then rise again (overfitting) as flexibility grows.
The experiment is repeated at two noise levels to show the floor moves
with sigma^2, but the shape of the reducible-error curve does not.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.pipeline import make_pipeline

RNG = np.random.default_rng(seed=42)


def true_function(x):
    """Ground-truth f(x): a nonlinear signal with a trend and a wiggle."""
    return 3 * np.sin(1.3 * x) + 0.5 * x


def make_dataset(n, sigma, rng):
    """Draw n noisy (x, y) samples from y = f(x) + eps, eps ~ N(0, sigma^2)."""
    x = rng.uniform(0, 10, size=n)
    noise = rng.normal(0, sigma, size=n)
    y = true_function(x) + noise
    return x, y


def fit_polynomial(x_train, y_train, degree):
    """
    Fit an OLS polynomial regression of the given degree.

    Raw powers of x (x^15, x^20, ...) get numerically huge on this scale
    and make (X^T X) ill-conditioned, which produces error blowups that
    reflect numerical instability rather than genuine overfitting. Scaling
    the polynomial features first keeps the comparison fair across degrees
    so the resulting curve reflects statistical overfitting, not arithmetic.
    """
    model = make_pipeline(
        PolynomialFeatures(degree),
        StandardScaler(),
        LinearRegression(),
    )
    model.fit(x_train.reshape(-1, 1), y_train)
    return model


def run_experiment(sigma, degrees, n_train=150, n_test=5000, n_trials=30, rng=RNG):
    """
    For each polynomial degree, repeat n_trials times:
      - draw a fresh noisy training set,
      - fit a polynomial of that degree,
      - evaluate on a large fresh test set both against noisy y and true f(x).
    Average the two MSEs over trials to smooth out sampling noise.
    """
    total_error = np.zeros(len(degrees))
    reducible_error = np.zeros(len(degrees))

    # One large, fixed test grid reused across trials for a stable comparison
    x_test = np.linspace(0, 10, n_test)
    f_true_test = true_function(x_test)

    for trial in range(n_trials):
        x_train, y_train = make_dataset(n_train, sigma, rng)
        # fresh noisy test targets each trial, so "total error" reflects
        # genuinely new noise draws, not the same noise reused
        y_test_noisy = f_true_test + rng.normal(0, sigma, size=n_test)

        for i, d in enumerate(degrees):
            model = fit_polynomial(x_train, y_train, d)
            preds = model.predict(x_test.reshape(-1, 1))

            total_error[i] += np.mean((y_test_noisy - preds) ** 2)
            reducible_error[i] += np.mean((f_true_test - preds) ** 2)

    return total_error / n_trials, reducible_error / n_trials


def main():
    degrees = [1, 2, 3, 5, 7, 9, 11, 13, 15, 17, 19]
    sigma_levels = [0.5, 1.5]

    fig, axes = plt.subplots(1, len(sigma_levels), figsize=(13, 5), sharey=False)

    for ax, sigma in zip(axes, sigma_levels):
        total_err, reducible_err = run_experiment(sigma, degrees)

        ax.plot(degrees, total_err, "o-", color="crimson", label="Total error (vs. noisy y)")
        ax.plot(degrees, reducible_err, "o-", color="steelblue", label="Reducible error (vs. true f)")
        ax.axhline(sigma ** 2, color="gray", linestyle="--", label=f"sigma^2 = {sigma**2:.2f}")

        ax.set_title(f"Noise level sigma = {sigma}")
        ax.set_xlabel("Polynomial degree (model flexibility)")
        ax.set_ylabel("Mean squared error")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)

        # Print the numeric gap as evidence, not just a visual claim
        gap = total_err - reducible_err
        print(f"\n--- sigma = {sigma} (sigma^2 = {sigma**2:.3f}) ---")
        for d, t, r, g in zip(degrees, total_err, reducible_err, gap):
            print(f"degree={d:>2}  total={t:6.3f}  reducible={r:6.3f}  gap={g:6.3f}")

    plt.suptitle("Total error = Reducible error + Irreducible error (sigma^2), across flexibility")
    plt.tight_layout()
    plt.savefig("irreducible_error_floor.png", dpi=150)
    print("\nSaved plot to irreducible_error_floor.png")


if __name__ == "__main__":
    main()
