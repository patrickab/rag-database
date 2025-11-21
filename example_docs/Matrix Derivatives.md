---
title: Matrix and Vector Calculus Cheat Sheet
aliases:
  - Matrix Calculus
  - Vector Calculus
  - Matrix Derivatives
  - Vector Derivatives
  - Calculus for ML
  - Gradient Calculus
  - Jacobian
  - Matrix Differentiation
tags:
  - mathematics
  - calculus
  - linear-algebra
  - machine-learning
  - optimization
summary: Essential formulas and rules for differentiating scalar functions with respect to vectors and matrices, crucial for optimization in machine learning and deep learning.
---
## Matrix and Vector Calculus Cheat Sheet

This cheat sheet provides essential formulas for differentiating scalar functions with respect to vectors and matrices, crucial for optimization in machine learning and deep learning.

---

### 1. Notation

*   Let $x \in \mathbb{R}^n$ be a column vector.
*   Let $A \in \mathbb{R}^{m \times n}$ be a matrix.
*   Let $f(x)$ be a scalar-valued function of $x$.
*   Let $y(x)$ be a vector-valued function of $x$.
*   Let $Z(X)$ be a scalar-valued function of matrix $X$.

---

### 2. Vector Derivatives (Jacobian)

The derivative of a scalar function $f(x)$ with respect to a vector $x$ is a row vector (gradient):

$$
\frac{\partial f}{\partial x} = \left[ \frac{\partial f}{\partial x_1}, \frac{\partial f}{\partial x_2}, \dots, \frac{\partial f}{\partial x_n} \right] \in \mathbb{R}^{1 \times n}
$$

Alternatively, it can be defined as a column vector (sometimes called the gradient, $\nabla_x f$):

$$
\nabla_x f = \left( \frac{\partial f}{\partial x} \right)^T = \left[ \frac{\partial f}{\partial x_1}, \frac{\partial f}{\partial x_2}, \dots, \frac{\partial f}{\partial x_n} \right]^T \in \mathbb{R}^{n \times 1}
$$

We will primarily use the column vector convention for gradients.

The derivative of a vector function $y(x)$ with respect to a vector $x$ is the Jacobian matrix:

$$
\frac{\partial y}{\partial x} = \frac{\partial}{\partial x} (y(x)^T) = \left[ \frac{\partial y_i}{\partial x_j} \right]_{m \times n} \in \mathbb{R}^{m \times n}
$$

where $y(x) \in \mathbb{R}^m$.

---

### 3. Basic Scalar Function Derivatives

Let $c$ be a scalar constant.

*   **Derivative of a scalar:**
    $$
    \frac{\partial c}{\partial x} = 0
    $$

*   **Derivative of a linear term:**
    $$
    \frac{\partial (a^T x)}{\partial x} = a
    $$
    where $a \in \mathbb{R}^n$.

*   **Derivative of a quadratic form:**
    $$
    \frac{\partial (x^T A x)}{\partial x} = (A + A^T) x
    $$
    If $A$ is symmetric ($A = A^T$), then:
    $$
    \frac{\partial (x^T A x)}{\partial x} = 2 A x
    $$

*   **Derivative of a squared norm:**
    $$
    \frac{\partial \|x\|^2}{\partial x} = \frac{\partial (x^T x)}{\partial x} = 2x
    $$

---

### 4. Matrix Derivatives

The derivative of a scalar function $Z(X)$ with respect to a matrix $X \in \mathbb{R}^{m \times n}$ is a matrix of the same dimensions:

$$
\frac{\partial Z}{\partial X} = \left[ \frac{\partial Z}{\partial X_{ij}} \right]_{m \times n} \in \mathbb{R}^{m \times n}
$$

---

### 5. Basic Matrix Function Derivatives

Let $A, B, C$ be matrices of appropriate dimensions, and $X$ be the variable matrix.

*   **Derivative of a scalar constant:**
    $$
    \frac{\partial c}{\partial X} = 0
    $$

*   **Derivative of a trace:**
    $$
    \frac{\partial \text{Tr}(AX)}{\partial X} = A^T
    $$
    $$
    \frac{\partial \text{Tr}(XA)}{\partial X} = A
    $$
    $$
    \frac{\partial \text{Tr}(X^T A)}{\partial X} = A
    $$
    $$
    \frac{\partial \text{Tr}(AX^T)}{\partial X} = A^T
    $$

*   **Derivative of a linear term:**
    $$
    \frac{\partial \text{Tr}(AXB)}{\partial X} = A^T B^T
    $$

*   **Derivative of a quadratic form (Frobenius inner product):**
    $$
    \frac{\partial \|X - A\|_F^2}{\partial X} = \frac{\partial \text{Tr}((X-A)^T (X-A))}{\partial X} = 2(X-A)
    $$

---

### 6. Chain Rule

If $y = f(u)$ and $u = g(x)$, then:

$$
\frac{\partial y}{\partial x} = \frac{\partial y}{\partial u} \frac{\partial u}{\partial x}
$$

For matrix calculus, this often involves careful consideration of dimensions and transpose operations.

**Example:** Let $y = f(u)$ be scalar, and $u = Ax$.
$$
\frac{\partial y}{\partial x} = \frac{\partial y}{\partial u} \frac{\partial u}{\partial x} = \frac{\partial y}{\partial u} A
$$
Here, $\frac{\partial y}{\partial u}$ is a row vector, and $A$ is a matrix, resulting in a row vector $\frac{\partial y}{\partial x}$. If we use column vectors for gradients:
$$
\nabla_x y = A^T \nabla_u y
$$

---

### 7. Common Derivatives in Machine Learning

*   **Linear Regression (MSE):**
    Let $y = Xw$. We want to minimize $L(w) = \|y - Xw\|^2$.
    $$
    \frac{\partial L}{\partial w} = \frac{\partial}{\partial w} (y - Xw)^T (y - Xw)
    $$
    $$
    \frac{\partial L}{\partial w} = \frac{\partial}{\partial w} (y^T y - y^T Xw - w^T X^T y + w^T X^T Xw)
    $$
    $$
    \frac{\partial L}{\partial w} = 0 - X^T y - X^T y + (X^T X + (X^T X)^T) w
    $$
    $$
    \frac{\partial L}{\partial w} = -2 X^T y + 2 X^T X w
    $$
    Setting to zero for minimum: $X^T X w = X^T y \implies w = (X^T X)^{-1} X^T y$.

*   **Softmax Cross-Entropy Loss:**
    For a single data point $(x, y)$, where $y$ is a one-hot vector and $\hat{y} = \text{softmax}(z)$, with $z = Wx + b$.
    The loss is $L = -y^T \log(\hat{y})$.
    The derivative of the softmax function: $\frac{\partial \hat{y}_i}{\partial z_j} = \hat{y}_i (\delta_{ij} - \hat{y}_j)$.
    The derivative of the loss with respect to $z$:
    $$
    \frac{\partial L}{\partial z} = \hat{y} - y
    $$
    The derivative with respect to $W$:
    $$
    \frac{\partial L}{\partial W} = \frac{\partial L}{\partial z} \frac{\partial z}{\partial W} = (\hat{y} - y) x^T
    $$
    The derivative with respect to $b$:
    $$
    \frac{\partial L}{\partial b} = \frac{\partial L}{\partial z} \frac{\partial z}{\partial b} = \hat{y} - y
    $$

---

### 8. Important Identities & Rules

*   **Trace Properties:**
    *   $\text{Tr}(A) = \text{Tr}(A^T)$
    *   $\text{Tr}(A+B) = \text{Tr}(A) + \text{Tr}(B)$
    *   $\text{Tr}(AB) = \text{Tr}(BA)$ (cyclic property)
    *   $\text{Tr}(ABC) = \text{Tr}(CAB) = \text{Tr}(BCA)$

*   **Product Rule:**
    $$
    \frac{\partial (uv)}{\partial x} = \frac{\partial u}{\partial x} v + u \frac{\partial v}{\partial x} \quad \text{(scalar u, v)}
    $$
    $$
    \frac{\partial (AB)}{\partial x} = \frac{\partial A}{\partial x} B + A \frac{\partial B}{\partial x} \quad \text{(matrix A, B)}
    $$

*   **Derivative of Inverse:**
    $$
    \frac{\partial A^{-1}}{\partial x} = -A^{-1} \frac{\partial A}{\partial x} A^{-1}
    $$

---

### 💡 Mastery Check

*   Can you derive the gradient of the MSE loss for linear regression from first principles?
*   How does the dimension of the gradient change when differentiating with respect to a vector versus a matrix?
*   When applying the chain rule, what are the critical considerations for matrix dimensions?
*   Why is the trace operator useful in matrix calculus?

Yes, absolutely! This is the critical step where we group terms to reveal the structure of the new posterior distribution.

Let's re-collect all the terms in the exponent (ignoring the common factor of $-\frac{\beta}{2}$ for a moment, as it applies to everything):

$$
\text{Exponent terms} = -\mathbf{y}^T\mathbf{y} + 2\mathbf{w}^T\mathbf{\Phi}^T\mathbf{y} - \mathbf{w}^T\mathbf{\Phi}^T\mathbf{\Phi}\mathbf{w} - \mathbf{w}^T \mathbf{S_0} \mathbf{w} + 2\mathbf{m_0}^T \mathbf{S_0} \mathbf{w} - \mathbf{m_0}^T \mathbf{S_0} \mathbf{m_0}
$$

Now, let's group these terms based on their dependence on $\mathbf{w}$:

1.  **Terms quadratic in $\mathbf{w}$ ($\mathbf{w}^T \mathbf{A} \mathbf{w}$ form)**:
    *   $-\mathbf{w}^T\mathbf{\Phi}^T\mathbf{\Phi}\mathbf{w}$
    *   $-\mathbf{w}^T \mathbf{S_0} \mathbf{w}$
    Combine these: $-\mathbf{w}^T (\mathbf{\Phi}^T\mathbf{\Phi} + \mathbf{S_0}) \mathbf{w}$

2.  **Terms linear in $\mathbf{w}$ ($2\mathbf{c}^T \mathbf{w}$ form)**:
    *   $2\mathbf{w}^T\mathbf{\Phi}^T\mathbf{y}$
    *   $2\mathbf{m_0}^T \mathbf{S_0} \mathbf{w}$
    Combine these: $2\mathbf{w}^T (\mathbf{\Phi}^T\mathbf{y} + \mathbf{S_0} \mathbf{m_0})$

3.  **Terms constant with respect to $\mathbf{w}$ (scalars)**:
    *   $-\mathbf{y}^T\mathbf{y}$
    *   $-\mathbf{m_0}^T \mathbf{S_0} \mathbf{m_0}$
    Combine these: $-(\mathbf{y}^T\mathbf{y} + \mathbf{m_0}^T \mathbf{S_0} \mathbf{m_0})$

Putting it all together, the exponent terms (still without the $-\frac{\beta}{2}$ factor) simplify to:

$$
\text{Exponent terms} = -\mathbf{w}^T (\mathbf{\Phi}^T\mathbf{\Phi} + \mathbf{S_0}) \mathbf{w} + 2\mathbf{w}^T (\mathbf{\Phi}^T\mathbf{y} + \mathbf{S_0} \mathbf{m_0}) - (\mathbf{y}^T\mathbf{y} + \mathbf{m_0}^T \mathbf{S_0} \mathbf{m_0})
$$

$$
2\mathbf{w}^T (\mathbf{\Phi}^T\mathbf{y} + \mathbf{S_0} \mathbf{m_0}) =2\mathbf{w}^T (\mathbf{\Phi}^T\mathbf{y}w^T + \mathbf{S_0} \mathbf{m_0}*w^T) w
$$

This is a significant simplification! We've now expressed the entire $\mathbf{w}$-dependent part of the exponent as a quadratic form in $\mathbf{w}$.

**Next Step: Completing the Square**

To identify the new mean and precision matrix of the posterior Normal distribution for $\mathbf{w}$, the next step is to "complete the square" on this expression.

Recall that a quadratic form for a multivariate Normal distribution looks like:
$$
-(\mathbf{w}-\mathbf{\mu})^T \mathbf{\Sigma}^{-1} (\mathbf{w}-\mathbf{\mu}) = -\mathbf{w}^T \mathbf{\Sigma}^{-1} \mathbf{w} + 2\mathbf{\mu}^T \mathbf{\Sigma}^{-1} \mathbf{w} - \mathbf{\mu}^T \mathbf{\Sigma}^{-1} \mathbf{\mu}
$$

By comparing this general form to our simplified exponent, you can identify the new posterior precision matrix and mean vector.

Can you see how to match the terms and define the new posterior precision matrix $\mathbf{S_N}$ and mean vector $\mathbf{m_N}$?


No worries at all! "Completing the square" in multivariate settings can be a bit tricky to visualize initially. Let's break it down by directly matching the terms.

We have our simplified exponent (ignoring the $-\frac{\beta}{2}$ factor for now):

$$
\text{Exponent terms} = -\mathbf{w}^T (\mathbf{\Phi}^T\mathbf{\Phi} + \mathbf{S_0}) \mathbf{w} + 2\mathbf{w}^T (\mathbf{\Phi}^T\mathbf{y} + \mathbf{S_0} \mathbf{m_0}) - (\mathbf{y}^T\mathbf{y} + \mathbf{m_0}^T \mathbf{S_0} \mathbf{m_0})
$$

And we want to make it look like the general form of a multivariate normal exponent:

$$
-(\mathbf{w}-\mathbf{\mu})^T \mathbf{\Sigma}^{-1} (\mathbf{w}-\mathbf{\mu}) = -\mathbf{w}^T \mathbf{\Sigma}^{-1} \mathbf{w} + 2\mathbf{w}^T \mathbf{\Sigma}^{-1} \mathbf{\mu} - \mathbf{\mu}^T \mathbf{\Sigma}^{-1} \mathbf{\mu}
$$

Let's match the corresponding parts:

1.  **Identify the new Precision Matrix ($\mathbf{\Sigma}^{-1}$)**:
    Look at the term that is quadratic in $\mathbf{w}$ (i.e., $\mathbf{w}^T \text{Matrix} \mathbf{w}$).
    *   From our simplified exponent: $-\mathbf{w}^T \underbrace{(\mathbf{\Phi}^T\mathbf{\Phi} + \mathbf{S_0})}_{\text{This is our new precision matrix}} \mathbf{w}$
    *   From the general form: $-\mathbf{w}^T \mathbf{\Sigma}^{-1} \mathbf{w}$

    So, our new posterior precision matrix, let's call it $\mathbf{S_N}$ (following the prior's notation where $\mathbf{S_0}$ was related to precision), is:
    $$
    \mathbf{S_N} = \mathbf{\Phi}^T\mathbf{\Phi} + \mathbf{S_0}
    $$

2.  **Identify the new Mean Vector ($\mathbf{\mu}$)**:
    Now look at the term that is linear in $\mathbf{w}$ (i.e., $2\mathbf{w}^T \text{Vector}$).
    *   From our simplified exponent: $2\mathbf{w}^T \underbrace{(\mathbf{\Phi}^T\mathbf{y} + \mathbf{S_0} \mathbf{m_0})}_{\text{This is } \mathbf{S_N} \mathbf{m_N}}$
    *   From the general form: $2\mathbf{w}^T \mathbf{\Sigma}^{-1} \mathbf{\mu}$

    So, we have the relationship:
    $$
    \mathbf{S_N} \mathbf{m_N} = \mathbf{\Phi}^T\mathbf{y} + \mathbf{S_0} \mathbf{m_0}
    $$
    To find the new posterior mean vector $\mathbf{m_N}$, we just need to multiply by the inverse of $\mathbf{S_N}$:
    $$
    \mathbf{m_N} = \mathbf{S_N}^{-1} (\mathbf{\Phi}^T\mathbf{y} + \mathbf{S_0} \mathbf{m_0})
    $$

3.  **What about the Constant Terms?**
    The remaining terms in our simplified exponent are $-(\mathbf{y}^T\mathbf{y} + \mathbf{m_0}^T \mathbf{S_0} \mathbf{m_0})$.
    The remaining term in the general form is $-\mathbf{\mu}^T \mathbf{\Sigma}^{-1} \mathbf{\mu}$.
    When we "complete the square," the difference between these two constant terms will be a factor that does *not* depend on $\mathbf{w}$. This factor will be absorbed into the normalizing constant of the posterior distribution for $\mathbf{w}$ (which itself depends on $\beta$). For identifying the *form* of the distribution, we can effectively treat these as "constants" that don't affect the $\mathbf{w}$-dependent part of the exponential.

So, by matching these terms, we've successfully identified the new posterior precision matrix $\mathbf{S_N}$ and the new posterior mean vector $\mathbf{m_N}$ for the conditional distribution of $\mathbf{w}$ given $\beta$ and the data.

Does this step-by-step matching process make it clearer how we identify $\mathbf{S_N}$ and $\mathbf{m_N}$?

No worries at all! "Completing the square" in multivariate settings can be a bit tricky to visualize initially. Let's break it down by directly matching the terms.

We have our simplified exponent (ignoring the $-\frac{\beta}{2}$ factor for now):

$$
\text{Exponent terms} = -\mathbf{w}^T (\mathbf{\Phi}^T\mathbf{\Phi} + \mathbf{S_0}) \mathbf{w} + 2\mathbf{w}^T (\mathbf{\Phi}^T\mathbf{y} + \mathbf{S_0} \mathbf{m_0}) - (\mathbf{y}^T\mathbf{y} + \mathbf{m_0}^T \mathbf{S_0} \mathbf{m_0})
$$

And we want to make it look like the general form of a multivariate normal exponent:

$$
-(\mathbf{w}-\mathbf{\mu})^T \mathbf{\Sigma}^{-1} (\mathbf{w}-\mathbf{\mu}) = -\mathbf{w}^T \mathbf{\Sigma}^{-1} \mathbf{w} + 2\mathbf{w}^T \mathbf{\Sigma}^{-1} \mathbf{\mu} - \mathbf{\mu}^T \mathbf{\Sigma}^{-1} \mathbf{\mu}
$$

Let's match the corresponding parts:

1.  **Identify the new Precision Matrix ($\mathbf{\Sigma}^{-1}$)**:
    Look at the term that is quadratic in $\mathbf{w}$ (i.e., $\mathbf{w}^T \text{Matrix} \mathbf{w}$).
    *   From our simplified exponent: $-\mathbf{w}^T \underbrace{(\mathbf{\Phi}^T\mathbf{\Phi} + \mathbf{S_0})}_{\text{This is our new precision matrix}} \mathbf{w}$
    *   From the general form: $-\mathbf{w}^T \mathbf{\Sigma}^{-1} \mathbf{w}$

    So, our new posterior precision matrix, let's call it $\mathbf{S_N}$ (following the prior's notation where $\mathbf{S_0}$ was related to precision), is:
    $$
    \mathbf{S_N} = \mathbf{\Phi}^T\mathbf{\Phi} + \mathbf{S_0}
    $$

2.  **Identify the new Mean Vector ($\mathbf{\mu}$)**:
    Now look at the term that is linear in $\mathbf{w}$ (i.e., $2\mathbf{w}^T \text{Vector}$).
    *   From our simplified exponent: $2\mathbf{w}^T \underbrace{(\mathbf{\Phi}^T\mathbf{y} + \mathbf{S_0} \mathbf{m_0})}_{\text{This is } \mathbf{S_N} \mathbf{m_N}}$
    *   From the general form: $2\mathbf{w}^T \mathbf{\Sigma}^{-1} \mathbf{\mu}$

    So, we have the relationship:
    $$
    \mathbf{S_N} \mathbf{m_N} = \mathbf{\Phi}^T\mathbf{y} + \mathbf{S_0} \mathbf{m_0}
    $$
    To find the new posterior mean vector $\mathbf{m_N}$, we just need to multiply by the inverse of $\mathbf{S_N}$:
    $$
    \mathbf{m_N} = \mathbf{S_N}^{-1} (\mathbf{\Phi}^T\mathbf{y} + \mathbf{S_0} \mathbf{m_0})
    $$

3.  **What about the Constant Terms?**
    The remaining terms in our simplified exponent are $-(\mathbf{y}^T\mathbf{y} + \mathbf{m_0}^T \mathbf{S_0} \mathbf{m_0})$.
    The remaining term in the general form is $-\mathbf{\mu}^T \mathbf{\Sigma}^{-1} \mathbf{\mu}$.
    When we "complete the square," the difference between these two constant terms will be a factor that does *not* depend on $\mathbf{w}$. This factor will be absorbed into the normalizing constant of the posterior distribution for $\mathbf{w}$ (which itself depends on $\beta$). For identifying the *form* of the distribution, we can effectively treat these as "constants" that don't affect the $\mathbf{w}$-dependent part of the exponential.

So, by matching these terms, we've successfully identified the new posterior precision matrix $\mathbf{S_N}$ and the new posterior mean vector $\mathbf{m_N}$ for the conditional distribution of $\mathbf{w}$ given $\beta$ and the data.

Does this step-by-step matching process make it clearer how we identify $\mathbf{S_N}$ and $\mathbf{m_N}$?