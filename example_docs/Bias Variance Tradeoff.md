---
title: Bias Variance Tradeoff
aliases:
tags:
  - machine-learning
  - model-evaluation
  - statistics
  - overfitting
  - underfitting
summary: Describes the fundamental relationship between bias and variance errors in predictive models, where reducing one often increases the other, aiming for optimal generalization.
---
The bias-variance tradeoff is a fundamental concept in machine learning that describes the relationship between two sources of error in predictive models: bias and variance.


![[Pasted image 20251105133736.png]]


**💡 Bias**
- Bias is the error from model assumptions.
- High bias leads to underfitting.
**💡 Variance**
- Variance measures a model's sensitivity to training data fluctuations.
-  High variance can cause the model to learn unwanted patterns from the training data that don't generalize to new data (overfitting).
- Low variance leads to more stable predictions across datasets.

The tradeoff arises because reducing one type of error often increases the other. A simple model might have high bias but low variance, while a complex model might have low bias but high variance. The goal is to find a model that balances both to achieve the best possible generalization performance.

**Underfitting (High Bias)**
- Linear model or non-linear data
- Consistent but inaccurate predictions
- Model is too simple

 **Overfitting (High Variance)**
 - High-degree polynomial
 - Fits training data perfectly
 - Poor generalization


**💡 Key Takeaways**
*   **Bias**: Model's simplifying assumptions; high bias leads to underfitting.
*   **Variance**: Model's sensitivity to training data; high variance leads to overfitting.
*   **Tradeoff**: Decreasing bias often increases variance, and vice-versa.
*   **Goal**: Find an optimal balance for best generalization.