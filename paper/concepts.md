# Concepts

Here’s a **markdown section** that describes cubic spline interpolation using 4 points, starting from the cubic equations, listing derivatives, and showing the linear system you need to solve. Equations are in LaTeX as requested.

---

## Cubic Spline Interpolation with Four Points

Suppose you have four data points: \$(x\_0, y\_0), (x\_1, y\_1), (x\_2, y\_2), (x\_3, y\_3)\$. We seek a smooth piecewise cubic function \$f(x)\$ that passes through all points and has continuous first and second derivatives.

### Cubic Polynomial Segments

We define three cubic segments, one for each interval:

$$
S_i(x) = a_i x^3 + b_i x^2 + c_i x + d_i \qquad \text{for}~i=0,1,2
$$

where \$S\_0(x)\$ is valid on $\[x\_0, x\_1]\$, \$S\_1(x)\$ on $\[x\_1, x\_2]\$, and \$S\_2(x)\$ on $\[x\_2, x\_3]\$.

There are **12 unknown coefficients**: \$a\_0, b\_0, c\_0, d\_0, a\_1, \dots, d\_2\$.

---

### Continuity and Boundary Conditions

To determine the coefficients, we impose the following **12 conditions**:

#### 1. Interpolation at Endpoints (6 equations)

$$
\begin{align}
S_0(x_0) &= y_0 \\
S_0(x_1) &= y_1 \\
S_1(x_1) &= y_1 \\
S_1(x_2) &= y_2 \\
S_2(x_2) &= y_2 \\
S_2(x_3) &= y_3
\end{align}
$$

#### 2. First Derivative Continuity at Interior Points (2 equations)

$$
\begin{align}
S_0'(x_1) &= S_1'(x_1) \\
S_1'(x_2) &= S_2'(x_2)
\end{align}
$$

#### 3. Second Derivative Continuity at Interior Points (2 equations)

$$
\begin{align}
S_0''(x_1) &= S_1''(x_1) \\
S_1''(x_2) &= S_2''(x_2)
\end{align}
$$

#### 4. Natural Spline Boundary Conditions (2 equations)

$$
\begin{align}
S_0''(x_0) &= 0 \\
S_2''(x_3) &= 0
\end{align}
$$

---

### System of Linear Equations

Each condition is a linear equation for the coefficients. For example, the value condition \$S\_0(x\_0) = y\_0\$ expands to:

$$
a_0 x_0^3 + b_0 x_0^2 + c_0 x_0 + d_0 = y_0
$$

The complete system can be written in **matrix form** as:

$$
A \cdot \mathbf{c} = \mathbf{y}
$$

where \$A\$ is a \$12 \times 12\$ matrix (see below), \$\mathbf{c}\$ is the vector of unknowns, and \$\mathbf{y}\$ is the right-hand side.

**Example matrix \$A\$ and vector \$\mathbf{y}\$:**

$$
\texttt{matrixA} = 
\begin{bmatrix}
x_0^3 & x_0^2 & x_0 & 1 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 \\
x_1^3 & x_1^2 & x_1 & 1 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 \\
0 & 0 & 0 & 0 & x_1^3 & x_1^2 & x_1 & 1 & 0 & 0 & 0 & 0 \\
0 & 0 & 0 & 0 & x_2^3 & x_2^2 & x_2 & 1 & 0 & 0 & 0 & 0 \\
0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & x_2^3 & x_2^2 & x_2 & 1 \\
0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & x_3^3 & x_3^2 & x_3 & 1 \\
3x_1^2 & 2x_1 & 1 & 0 & -3x_1^2 & -2x_1 & -1 & 0 & 0 & 0 & 0 & 0 \\
0 & 0 & 0 & 0 & 3x_2^2 & 2x_2 & 1 & 0 & -3x_2^2 & -2x_2 & -1 & 0 \\
6x_1 & 2 & 0 & 0 & -6x_1 & -2 & 0 & 0 & 0 & 0 & 0 & 0 \\
0 & 0 & 0 & 0 & 6x_2 & 2 & 0 & 0 & -6x_2 & -2 & 0 & 0 \\
6x_0 & 2 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 \\
0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 6x_3 & 2 & 0 & 0 \\
\end{bmatrix}
$$

$$
\texttt{matrixC} = 
\begin{bmatrix}
y_0 \\ y_1 \\ y_1 \\ y_2 \\ y_2 \\ y_3 \\
0 \\ 0 \\ 0 \\ 0 \\ 0 \\ 0
\end{bmatrix}
$$

Solve for the coefficients:

$$
\mathbf{c} = A^{-1} \mathbf{y}
$$

---

### Summary

This system guarantees that:

* The spline passes through all data points
* The first and second derivatives are continuous at interior knots
* The boundary conditions are satisfied (natural spline: zero curvature at the ends)

You can use any linear algebra solver to solve for \$\mathbf{c}\$ given \$A\$ and \$\mathbf{y}\$.
