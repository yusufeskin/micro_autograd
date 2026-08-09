# Autograd: A Scalar-Valued Automatic Differentiation Engine

## Abstract
This repository implements a dynamic, reverse-mode automatic differentiation (AD) engine from scratch. Operating over a custom Directed Acyclic Graph (DAG) architecture, the engine evaluates mathematical expressions in a forward pass and computes exact gradients via backpropagation. It is designed to serve as the foundational differential calculus engine for constructing parameter-space topologies, such as Multi-Layer Perceptrons.

## Mathematical Framework

### 1. The Computational Graph
Every scalar operation constructs a node within a Directed Acyclic Graph. Let $\mathcal{G} = (\mathcal{V}, \mathcal{E})$ be the computational graph where $\mathcal{V}$ represents the scalar values (nodes) and $\mathcal{E}$ represents the directed edges defining the functional dependencies. A node $v_k \in \mathcal{V}$ is defined by a transformation $f$:
$$ v_k = f(v_1, v_2, \dots, v_n) $$
where $v_1, \dots, v_n$ are the topological parents of $v_k$ in the backward context.

### 2. Multivariate Chain Rule and Gradient Accumulation
The core mechanism of the reverse-mode AD relies on the multivariate chain rule to propagate the gradient of a target objective function $L$ with respect to any intermediate node $v_i$. If $v_i$ acts as an operand for multiple subsequent operations (i.e., its out-degree $> 1$ in the forward graph), the local gradients must be accumulated:
$$ \frac{\partial L}{\partial v_i} = \sum_{j \in \text{children}(v_i)} \frac{\partial L}{\partial v_j} \frac{\partial v_j}{\partial v_i} $$
In the implementation, this mathematical necessity dictates the use of the `+=` operator during the `_backward()` phase. This strictly preserves gradient integrity across independent computational branches.

### 3. Topological Sort via Depth-First Search
To guarantee causality during gradient propagation, the backward pass must be strictly ordered. The gradient of a node $v_i$ can only be resolved once the gradients of all $v_j \in \text{children}(v_i)$ are fully computed. 
This constraint is satisfied by executing a recursive Depth-First Search to extract a post-order traversal of the DAG. The nodes are then processed in reverse topological order, seamlessly mapping the continuous chain rule expansion onto discrete algorithmic steps.

## Architecture Architecture
* **`Value` Class:** The primitive scalar data structure. Encapsulates the scalar `data`, the accumulated `grad`, the local derivative closure `_backward`, and the structural topology (`_prev`, `_op`).
* **Operator Overloading:** Native Python magic methods (`__add__`, `__mul__`, etc.) are overloaded to dynamically build the DAG during standard arithmetic operations.

## Roadmap
* **Vectorized Operations:** Extending the scalar topology into $\mathbb{R}^n$ vector spaces.
* **Neural Architecture:** Implementing hierarchical functional compositions to define affine transformations $z = \mathbf{w}^T \mathbf{x} + b$ mapped through non-linear activation functions (e.g., $\tanh$).
* **Optimization:** Implementing a Gradient Descent routine over the isolated parameter space.

## Acknowledgments
This implementation is heavily inspired by and structurally follows the foundational concepts taught in **Andrej Karpathy's** excellent [Neural Networks: Zero to Hero](https://karpathy.ai/zero-to-hero.html) series (specifically the *micrograd* lecture). The series served as the primary pedagogical guide for understanding the algorithmic translation of calculus and graph theory into a working automatic differentiation engine.