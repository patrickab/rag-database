---
title: Parametric vs. Non-Parametric Models
aliases:
  - Parametric Models
  - Non-Parametric Models
  - Parametric vs Non-Parametric
  - Model Assumptions
  - Model Flexibility
tags:
  - machine-learning
  - statistical-learning
  - statistical-modeling
summary: Parametric models (fixed parameters, strong assumptions, interpretable) vs. non-parametric models (flexible, data-driven, robust, less interpretable).
---
**Parametric vs. Non-Parametric Models**

# **Parametric Models**

Assume a fixed, finite number of parameters to define the functional form of the relationship between variables. The model structure is predetermined.

**Examples**
-   **Linear Regression**: $y = \beta_0 + \beta_1 x_1 + \dots + \beta_p x_p + \epsilon$, where $\beta_i$ are the parameters.
-   **Logistic Regression**: $P(Y=1|X) = \frac{1}{1 + e^{-(\beta_0 + \beta_1 x_1 + \dots + \beta_p x_p)}}$, with $\beta_i$ as parameters.

**Pros**
-   **Interpretability**: Parameters often have clear statistical meaning.
-   **Efficiency**: Require less data to estimate parameters accurately if assumptions hold.
-   **Computational Speed**: Generally faster to train.

**Cons**
-   **Strong Assumptions**: Performance heavily relies on the correctness of the assumed functional form and data distribution.
-   **Bias**: Can suffer from high bias if the true relationship deviates from the assumed form.

---

# **Non-Parametric Models**

Do not assume a fixed functional form or a finite number of parameters. The model structure adapts to the data, often growing in complexity with more data.

**Examples**
-   **k-Nearest Neighbors (k-NN)**: Classifies/regresses based on the majority class/average of its $k$ nearest data points.
-   **Decision Trees/Random Forests**: Partition the feature space into regions, with predictions based on the data within each region.

**Pros**
-   **Flexibility**: Can model complex, non-linear relationships without strong prior assumptions.
-   **Robustness**: Less prone to misspecification bias.

**Cons**
-   **Data Hungry**: Typically require large datasets to achieve good performance.
-   **Interpretability**: Often harder to interpret the underlying relationships.
-   **Computational Cost**: Can be more computationally intensive, especially during prediction (e.g., k-NN).

---

**💡 Key Takeaways**
-   **Parametric**: Fixed parameters, assumed distribution, high interpretability, efficient with small data, but sensitive to assumptions.
-   **Non-Parametric**: Flexible structure, data-driven, robust to unknown relationships, but data-intensive and less interpretable.