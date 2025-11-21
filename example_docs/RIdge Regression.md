---
title: Ridge Regression
aliases: [L2 Regularization, Tikhonov regularization, Shrinkage Regression, Ridge]
tags: [machine-learning, regression, regularization, statistics, overfitting, bias-variance-tradeoff]
summary: A regularization technique that adds an L2 penalty to linear regression to shrink coefficients, prevent overfitting, and improve model generalization.
---
Ridge regression adds a penalty term to the standard linear regression cost function. This penalty is proportional to the square of the coefficients.

**Intuition behind the "ridge"**:

*   **Shrinkage**: The penalty term "shrinks" the regression coefficients towards zero. This is like adding a "ridge" or a constraint that prevents coefficients from becoming too large.
*   **Regularization**: It's a form of regularization that helps prevent overfitting, especially when dealing with multicollinearity (highly correlated predictors) or when the number of features is large relative to the number of observations.
*   **[[Bias Variance Tradeoff]]**: By shrinking coefficients, we introduce a small amount of bias but significantly reduce the variance of the model.
*   **Low-Degree Polynomial Analogy**: For polynomials, large coefficients can lead to wiggly, high-degree curves that fit the training data perfectly but generalize poorly.

**Cost Function**:

The standard Ordinary Least Squares (OLS) cost function is:
$$
J(\beta) = \sum_{i=1}^{n} (y_i - \hat{y}_i)^2
$$
where $\hat{y}_i = \beta_0 + \sum_{j=1}^{p} \beta_j x_{ij}$.

Ridge regression modifies this to:
$$
J_{ridge}(\beta) = \sum_{i=1}^{n} (y_i - \hat{y}_i)^2 + \lambda \sum_{j=1}^{p} \beta_j^2
$$
Here, $\lambda$ (lambda) is the tuning parameter that controls the strength of the penalty.

**💡 Key Takeaways**:

*   **Penalty**: Ridge regression adds an L2 penalty ($\lambda \sum \beta_j^2$) to the cost function.
*   **Shrinkage**: This penalty shrinks coefficients towards zero, reducing their magnitude.
*   **Overfitting**: It combats overfitting by simplifying the model.
*   **Bias-Variance**: It balances bias and variance, favoring lower variance.
*   **Smoothness**: Encourages simpler, smoother model fits, analogous to lower-degree polynomials.