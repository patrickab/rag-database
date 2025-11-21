---
title: Multidimensional Differential Calculus
aliases:
  - matrix calculus
  - vector calculus
  - Jacobian
  - gradient
  - chain rule
tags:
  - calculus
  - linear-algebra
  - machine-learning
  - optimization
  - mathematics
  - differential-equations
summary: An introduction to multidimensional differential calculus, covering gradients, Jacobians, and matrix derivatives, with applications in machine learning like linear regression.
---
We often encounter functions in machine learning and computer science that don't just map a single number to another. Instead, they operate on entire vectors or matrices. Think of a loss function, which takes a vector of model parameters and outputs a single scalar loss. Or a neural network layer, which transforms an input vector into an output vector.

To optimize or analyze these functions, we need a way to describe how they change with respect to their high-dimensional inputs. This is precisely the role of matrix and vector derivatives.

But how do we generalize the familiar concept of a derivative, $f'(x)$, to a function $f(\mathbf{x})$ where $\mathbf{x}$ is a vector? What would such a "derivative" even look like?

---

### 1. From Scalar Derivative to the Gradient

Let's start with the simplest case: a function that maps a vector to a scalar, $f: \mathbb{R}^n \to \mathbb{R}$.

In single-variable calculus, the derivative $f'(x_0)$ gives us the slope of the tangent line at $x_0$. It's the best linear approximation of the function's change near that point:
$$
f(x_0 + \Delta x) \approx f(x_0) + f'(x_0) \Delta x
$$

How can we extend this to $f(\mathbf{x})$? The input is now a vector $\mathbf{x} \in \mathbb{R}^n$. A small change is no longer a scalar $\Delta x$, but a vector $\Delta \mathbf{x}$. What object, when combined with $\Delta \mathbf{x}$, gives us the scalar change in $f$?

The natural candidate is the dot product. This leads us to the **gradient vector**:
$$
\nabla_{\mathbf{x}} f(\mathbf{x}) = \begin{bmatrix} \frac{\partial f}{\partial x_1} \\ \frac{\partial f}{\partial x_2} \\ \vdots \\ \frac{\partial f}{\partial x_n} \end{bmatrix} \in \mathbb{R}^n
$$

The gradient allows us to form the same linear approximation, now in higher dimensions:
$$
f(\mathbf{x}_0 + \Delta \mathbf{x}) \approx f(\mathbf{x}_0) + (\nabla_{\mathbf{x}} f(\mathbf{x}_0))^T \Delta \mathbf{x}
$$

💡 **Geometric Interpretation**: The gradient $\nabla_{\mathbf{x}} f(\mathbf{x})$ is a vector that points in the direction of the steepest ascent of the function $f$ at point $\mathbf{x}$. Its magnitude indicates the rate of that increase. This is why moving in the *opposite* direction, $-\nabla f$, is the basis of gradient descent.

#### Example: Derivative of a Linear Form

Let's consider a simple function $f(\mathbf{x}) = \mathbf{a}^T \mathbf{x}$ for a constant vector $\mathbf{a} \in \mathbb{R}^n$.
$$
f(\mathbf{x}) = \sum_{i=1}^n a_i x_i = a_1 x_1 + a_2 x_2 + \dots + a_n x_n
$$
To find the gradient, we compute the partial derivative with respect to each component $x_k$:
$$
\frac{\partial f}{\partial x_k} = \frac{\partial}{\partial x_k} \left( \sum_{i=1}^n a_i x_i \right) = a_k
$$
Assembling these partials into a vector, what do we get?

$$
\nabla_{\mathbf{x}} (\mathbf{a}^T \mathbf{x}) = \begin{bmatrix} a_1 \\ a_2 \\ \vdots \\ a_n \end{bmatrix} = \mathbf{a}
$$
This is beautifully analogous to the scalar case $\frac{d}{dx}(ax) = a$.

#### Example: Derivative of a Quadratic Form

Now for a slightly more complex case, the squared L2-norm: $f(\mathbf{x}) = \mathbf{x}^T \mathbf{x}$.
$$
f(\mathbf{x}) = \sum_{i=1}^n x_i^2 = x_1^2 + x_2^2 + \dots + x_n^2
$$
The partial derivative with respect to a specific component $x_k$ is:
$$
\frac{\partial f}{\partial x_k} = \frac{\partial}{\partial x_k} \left( \sum_{i=1}^n x_i^2 \right) = 2x_k
$$
Therefore, the gradient is:
$$
\nabla_{\mathbf{x}} (\mathbf{x}^T \mathbf{x}) = \begin{bmatrix} 2x_1 \\ 2x_2 \\ \vdots \\ 2x_n \end{bmatrix} = 2\mathbf{x}
$$
Again, this mirrors the scalar case $\frac{d}{dx}(x^2) = 2x$.

> **Reflections on the Gradient**
> - The gradient $\nabla_{\mathbf{x}} f$ is the generalization of the derivative for functions $f: \mathbb{R}^n \to \mathbb{R}$.
> - It's a vector of the same dimension as the input $\mathbf{x}$.
> - It provides the best linear approximation of the function's change via the dot product: $\Delta f \approx (\nabla f)^T \Delta \mathbf{x}$.

---

### 2. The Jacobian: Generalizing to Vector Functions

What if the function's *output* is also a vector? Consider a function $\mathbf{f}: \mathbb{R}^n \to \mathbb{R}^m$.
$$
\mathbf{f}(\mathbf{x}) = \begin{bmatrix} f_1(x_1, \dots, x_n) \\ f_2(x_1, \dots, x_n) \\ \vdots \\ f_m(x_1, \dots, x_n) \end{bmatrix}
$$
Each component function $f_i$ is a scalar-valued function of the vector $\mathbf{x}$. We already know how to find the derivative of each $f_i$—it's the gradient $\nabla_{\mathbf{x}} f_i$.

How should we arrange these gradients to represent the derivative of the entire function $\mathbf{f}$? The most natural structure is a matrix. This is the **Jacobian matrix**.

The Jacobian $J_{\mathbf{f}}(\mathbf{x})$ is an $m \times n$ matrix where each entry $(i, j)$ is the partial derivative of the $i$-th output component with respect to the $j$-th input component:
$$
J_{ij} = \frac{\partial f_i}{\partial x_j}
$$
$$
J_{\mathbf{f}}(\mathbf{x}) = \frac{\partial \mathbf{f}}{\partial \mathbf{x}^T} = \begin{bmatrix}
\frac{\partial f_1}{\partial x_1} & \frac{\partial f_1}{\partial x_2} & \cdots & \frac{\partial f_1}{\partial x_n} \\
\frac{\partial f_2}{\partial x_1} & \frac{\partial f_2}{\partial x_2} & \cdots & \frac{\partial f_2}{\partial x_n} \\
\vdots & \vdots & \ddots & \vdots \\
\frac{\partial f_m}{\partial x_1} & \frac{\partial f_m}{\partial x_2} & \cdots & \frac{\partial f_m}{\partial x_n}
\end{bmatrix} = \begin{bmatrix}
— (\nabla_{\mathbf{x}} f_1)^T — \\
— (\nabla_{\mathbf{x}} f_2)^T — \\
\vdots \\
— (\nabla_{\mathbf{x}} f_m)^T —
\end{bmatrix}
$$

🎯 **The Core Idea**: The Jacobian matrix represents the best linear transformation that approximates the change in the function $\mathbf{f}$ at a point $\mathbf{x}_0$.
$$
\mathbf{f}(\mathbf{x}_0 + \Delta \mathbf{x}) \approx \mathbf{f}(\mathbf{x}_0) + J_{\mathbf{f}}(\mathbf{x}_0) \Delta \mathbf{x}
$$
Notice the dimensions: $J_{\mathbf{f}}$ is $m \times n$, and $\Delta \mathbf{x}$ is $n \times 1$. The product is an $m \times 1$ vector, which correctly represents the change in the output.

#### Example: An Affine Transformation

Consider the function $\mathbf{f}(\mathbf{x}) = A\mathbf{x} + \mathbf{b}$, where $A \in \mathbb{R}^{m \times n}$, $\mathbf{x} \in \mathbb{R}^n$, and $\mathbf{b} \in \mathbb{R}^m$. This is a fundamental operation in any neural network layer.

Let's write out the $i$-th component of the output:
$$
f_i(\mathbf{x}) = (A\mathbf{x})_i + b_i = \sum_{j=1}^n A_{ij} x_j + b_i
$$
Now, let's find the partial derivative of $f_i$ with respect to an input component $x_k$:
$$
\frac{\partial f_i}{\partial x_k} = \frac{\partial}{\partial x_k} \left( \sum_{j=1}^n A_{ij} x_j + b_i \right) = A_{ik}
$$
This is the entry $(i, k)$ of our Jacobian matrix. Since this holds for all $i$ and $k$, what is the Jacobian matrix itself?

$$
J_{\mathbf{f}}(\mathbf{x}) = A
$$
This is a profound result. The derivative of the linear transformation $\mathbf{x} \mapsto A\mathbf{x}$ is the matrix $A$ itself. This is perfectly analogous to $\frac{d}{dx}(ax) = a$. The Jacobian *is* the linear map.

> **Reflections on the Jacobian**
> - The Jacobian matrix is the generalization of the derivative for functions $\mathbf{f}: \mathbb{R}^n \to \mathbb{R}^m$.
> - It's an $m \times n$ matrix of all possible partial derivatives.
> - It represents the best *linear transformation* approximating the function's local behavior.
> - The gradient is a special case of the Jacobian where $m=1$ (the output is a scalar). The Jacobian would be a $1 \times n$ row vector, which is the transpose of our gradient definition.

---

### 3. Scalar-by-Matrix Derivatives

We often encounter functions that map a matrix to a scalar, $f: \mathbb{R}^{m \times n} \to \mathbb{R}$. A prime example is the loss function in linear regression, which depends on a weight matrix $W$.

The derivative $\frac{\partial f}{\partial X}$ will be a matrix of the same shape as $X$, where each element $(i, j)$ is the partial derivative of $f$ with respect to the element $X_{ij}$.
$$
\left( \frac{\partial f}{\partial X} \right)_{ij} = \frac{\partial f}{\partial X_{ij}}
$$

#### Example: The Trace Function

Let $f(X) = \text{Tr}(X)$ for $X \in \mathbb{R}^{n \times n}$. The trace is the sum of the diagonal elements:
$$
f(X) = \sum_{k=1}^n X_{kk}
$$
Let's find the partial derivative with respect to an arbitrary element $X_{ij}$:
$$
\frac{\partial f}{\partial X_{ij}} = \frac{\partial}{\partial X_{ij}} \left( \sum_{k=1}^n X_{kk} \right) =
\begin{cases}
1 & \text{if } i=j \\
0 & \text{if } i \neq j
\end{cases}
$$
This is precisely the definition of the identity matrix. Therefore:
$$
\frac{\partial}{\partial X} \text{Tr}(X) = I
$$

---

### 4. The Chain Rule in Action: Linear Regression

Let's see how these concepts are indispensable in a real-world scenario. Consider a simple linear regression model without bias: $\hat{\mathbf{y}} = XW$, where $X \in \mathbb{R}^{N \times D}$ is the data matrix, $W \in \mathbb{R}^{D \times 1}$ is the weight vector, and $\hat{\mathbf{y}} \in \mathbb{R}^{N \times 1}$ are the predictions.

Our goal is to minimize the Mean Squared Error (MSE) loss function:
$$
L(W) = \frac{1}{N} ||XW - \mathbf{y}||_2^2 = \frac{1}{N} (\mathbf{\hat{y}} - \mathbf{y})^T (\mathbf{\hat{y}} - \mathbf{y})
$$
To optimize this with gradient descent, we need to compute $\nabla_W L$. We can think of this as a chain of functions:
1.  $W \xrightarrow{g} \mathbf{\hat{y}} = XW$
2.  $\mathbf{\hat{y}} \xrightarrow{h} L = \frac{1}{N} ||\mathbf{\hat{y}} - \mathbf{y}||_2^2$

Let's define an error vector $\mathbf{e} = \mathbf{\hat{y}} - \mathbf{y}$. Then $L = \frac{1}{N} \mathbf{e}^T \mathbf{e}$.

**Step 1: Compute $\nabla_{\mathbf{\hat{y}}} L$**
This is the derivative of the loss with respect to the model's predictions.
$$
\nabla_{\mathbf{\hat{y}}} L = \frac{\partial}{\partial \mathbf{\hat{y}}} \left( \frac{1}{N} (\mathbf{\hat{y}} - \mathbf{y})^T (\mathbf{\hat{y}} - \mathbf{y}) \right)
$$
Using the identity $\nabla_{\mathbf{x}}(\mathbf{x}^T\mathbf{x}) = 2\mathbf{x}$ and the chain rule, we get:
$$
\nabla_{\mathbf{\hat{y}}} L = \frac{2}{N} (\mathbf{\hat{y}} - \mathbf{y}) = \frac{2}{N} (XW - \mathbf{y})
$$

**Step 2: Compute the Jacobian of $\mathbf{\hat{y}}$ with respect to $W$**
Our function is $\mathbf{\hat{y}}(W) = XW$. This is an affine transformation mapping $\mathbb{R}^D \to \mathbb{R}^N$. From our earlier result, what is the Jacobian $\frac{\partial \mathbf{\hat{y}}}{\partial W^T}$?
$$
\frac{\partial \mathbf{\hat{y}}}{\partial W^T} = X
$$
The Jacobian is an $N \times D$ matrix.

**Step 3: Apply the Chain Rule**
The multivariate chain rule states:
$$
\frac{\partial L}{\partial W^T} = \frac{\partial L}{\partial \mathbf{\hat{y}}^T} \frac{\partial \mathbf{\hat{y}}}{\partial W^T}
$$
Let's check the dimensions. $\frac{\partial L}{\partial W^T}$ should be $1 \times D$.
- $\frac{\partial L}{\partial \mathbf{\hat{y}}^T}$ is the transpose of the gradient $\nabla_{\mathbf{\hat{y}}} L$, so it's a $1 \times N$ row vector.
- $\frac{\partial \mathbf{\hat{y}}}{\partial W^T}$ is the Jacobian, which is $N \times D$.

The product of a $(1 \times N)$ and an $(N \times D)$ matrix is indeed $(1 \times D)$. The dimensions match perfectly.

$$
\frac{\partial L}{\partial W^T} = \left( \frac{2}{N} (XW - \mathbf{y})^T \right) X
$$
This gives us the derivative as a row vector. To get the gradient (a column vector), we simply transpose the result:
$$
\nabla_W L = \left( \frac{\partial L}{\partial W^T} \right)^T = X^T \left( \frac{2}{N} (XW - \mathbf{y}) \right) = \frac{2}{N} X^T(XW - \mathbf{y})
$$
This is the famous normal equation gradient, the workhorse of linear regression. We derived it systematically by composing derivatives of vector functions.

---

### 📝 Matrix Calculus Cheat Sheet

Here is a summary of the core concepts and common differentiation rules.

⚠️ **A Note on Layouts**: There are two common conventions for organizing derivatives: the **numerator layout** and the **denominator layout**. This can affect the shape of the resulting derivative (e.g., whether the gradient is a column or row vector). The explanation above and the table below use the **denominator layout**, which is prevalent in machine learning literature.

**Definitions & Shapes**

| Type | Function $y=f(x)$ | Derivative $\frac{dy}{dx}$ | Shape of Derivative |
| :--- | :--- | :--- | :--- |
| Scalar-by-Scalar | $y \in \mathbb{R}$, $x \in \mathbb{R}$ | $\frac{df}{dx}$ | $1 \times 1$ (Scalar) |
| Scalar-by-Vector | $y \in \mathbb{R}$, $\mathbf{x} \in \mathbb{R}^n$ | $\nabla_{\mathbf{x}} y$ (Gradient) | $n \times 1$ (Vector) |
| Vector-by-Scalar | $\mathbf{y} \in \mathbb{R}^m$, $x \in \mathbb{R}$ | $\frac{d\mathbf{y}}{dx}$ | $m \times 1$ (Vector) |
| Vector-by-Vector | $\mathbf{y} \in \mathbb{R}^m$, $\mathbf{x} \in \mathbb{R}^n$ | $\frac{\partial \mathbf{y}}{\partial \mathbf{x}^T}$ (Jacobian) | $m \times n$ (Matrix) |
| Scalar-by-Matrix | $y \in \mathbb{R}$, $X \in \mathbb{R}^{m \times n}$ | $\frac{\partial y}{\partial X}$ | $m \times n$ (Matrix) |

**Common Identities & Rules**

| Operation | Function | Derivative | Notes |
| :--- | :--- | :--- | :--- |
| **Vector Derivatives** | | | |
| Linear Form | $\mathbf{a}^T \mathbf{x}$ | $\nabla_{\mathbf{x}}(\mathbf{a}^T \mathbf{x}) = \mathbf{a}$ | $\mathbf{a}, \mathbf{x} \in \mathbb{R}^n$ |
| Quadratic Form | $\mathbf{x}^T A \mathbf{x}$ | $\nabla_{\mathbf{x}}(\mathbf{x}^T A \mathbf{x}) = (A + A^T)\mathbf{x}$ | For symmetric $A$, this is $2A\mathbf{x}$ |
| L2 Norm Squared | $\mathbf{x}^T \mathbf{x}$ | $\nabla_{\mathbf{x}}(\mathbf{x}^T \mathbf{x}) = 2\mathbf{x}$ | Special case of quadratic form with $A=I$ |
| Affine Transform | $A\mathbf{x} + \mathbf{b}$ | $\frac{\partial (A\mathbf{x} + \mathbf{b})}{\partial \mathbf{x}^T} = A$ | The Jacobian is the matrix $A$ |
| **Matrix Derivatives** | | | |
| Linear Form | $\text{Tr}(AX)$ | $\frac{\partial}{\partial X} \text{Tr}(AX) = A^T$ | |
| Bilinear Form | $\mathbf{a}^T X \mathbf{b}$ | $\frac{\partial}{\partial X} (\mathbf{a}^T X \mathbf{b}) = \mathbf{a}\mathbf{b}^T$ | Result is an outer product |
| Quadratic Form | $\text{Tr}(X^T A X)$ | $\frac{\partial}{\partial X} \text{Tr}(X^T A X) = (A+A^T)X$ | |
| Determinant | $\det(X)$ | $\frac{\partial}{\partial X} \det(X) = \det(X) (X^{-1})^T$ | For invertible $X$ |
| **Chain Rule** | | | |
| Vector Chain Rule | $L = f(g(\mathbf{x}))$ | $\frac{\partial L}{\partial \mathbf{x}^T} = \frac{\partial L}{\partial \mathbf{y}^T} \frac{\partial \mathbf{y}}{\partial \mathbf{x}^T}$ | where $\mathbf{y} = g(\mathbf{x})$ |

Mastering these rules provides the foundation for understanding and developing optimization algorithms for nearly all modern machine learning models.