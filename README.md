# Autograd: A Scalar-Valued Automatic Differentiation Engine

## Abstract
This repository implements a dynamic, reverse-mode automatic differentiation (AD) engine from scratch. Operating over a custom Directed Acyclic Graph (DAG) architecture, the engine evaluates mathematical expressions in a forward pass and computes exact gradients via backpropagation. It includes a built-in Neural Network API, serving as a foundational differential calculus engine for constructing and optimizing parameter-space topologies, such as Multi-Layer Perceptrons.

## Mathematical Framework

### 1. The Computational Graph
Every scalar operation constructs a node within a Directed Acyclic Graph. Let $\mathcal{G} = (\mathcal{V}, \mathcal{E})$ be the computational graph where $\mathcal{V}$ represents the scalar values (nodes) and $\mathcal{E}$ represents the directed edges defining the functional dependencies. A node $v_k \in \mathcal{V}$ is defined by a transformation $f$:

$$v_k = f(v_1, v_2, \dots, v_n)$$

where $v_1, \dots, v_n$ are the topological parents of $v_k$ in the backward context.

### 2. Multivariate Chain Rule and Gradient Accumulation
The core mechanism of the reverse-mode AD relies on the multivariate chain rule to propagate the gradient of a target objective function $L$ with respect to any intermediate node $v_i$. If $v_i$ acts as an operand for multiple subsequent operations (i.e., its out-degree $> 1$ in the forward graph), the local gradients must be accumulated:

$$\frac{\partial L}{\partial v_i} = \sum_{j \in \text{children}(v_i)} \frac{\partial L}{\partial v_j} \frac{\partial v_j}{\partial v_i}$$

In the implementation, this mathematical necessity dictates the use of the `+=` operator during the `_backward()` phase. This strictly preserves gradient integrity across independent computational branches.

### 3. Topological Sort via Depth-First Search
To guarantee causality during gradient propagation, the backward pass must be strictly ordered. The gradient of a node $v_i$ can only be resolved once the gradients of all $v_j \in \text{children}(v_i)$ are fully computed. 
This constraint is satisfied by executing a recursive Depth-First Search to extract a post-order traversal of the DAG. The nodes are then processed in reverse topological order, seamlessly mapping the continuous chain rule expansion onto discrete algorithmic steps.

## Project Structure & Architecture 

* **`engine.py`**: The core autodiff engine. Contains the `Value` class, which encapsulates the scalar data, accumulated gradients, and overloads native Python magic methods (`__add__`, `__mul__`, `__pow__`, etc.) to dynamically build the DAG.
* **`nn.py`**: The Neural Network API. Implements hierarchical functional compositions (`Neuron`, `Layer`, `MLP`) to define affine transformations $z = \mathbf{w}^T \mathbf{x} + b$ mapped through non-linear activation functions (e.g., $\tanh$).
* **`visualize.py`**: A graph visualization utility using `graphviz`. It traces the topological connections and renders the entire computational graph, displaying forward data and backward gradients at every node.
* **`train.py`**: An implementation script demonstrating the forward pass, Mean Squared Error (MSE) loss computation, and backpropagation over a small dataset.

## Quick Start Example

```python
from nn import MLP
from engine import Value
from visualize import draw_dot

# 1. Define dataset
xs = [[2.0, 3.0, -1.0], [3.0, -1.0, 0.5], [0.5, 1.0, 1.0], [1.0, 1.0, -1.0]]
ys = [1.0, -1.0, -1.0, 1.0]

# 2. Initialize a Multi-Layer Perceptron (3 inputs, two hidden layers of 4, 1 output)
n = MLP(3, [4, 4, 1])

# 3. Forward pass and Loss computation (MSE)
ypred = [n(x) for x in xs]
loss = sum(((yout[0] - ygt) ** 2 for ygt, yout in zip(ys, ypred)))

# 4. Backward pass (computes gradients for all weights and biases)
loss.backward()
print(f"Total Loss: {loss.data}")

# 5. One step gradient descent (looped version can be found in train.py)
for p in n.parameters():
    p.data += -0.01 * p.grad


# 6. Visualize the Computational Graph
dot = draw_dot(loss)
dot.render('nn_graph', view=True, cleanup=True)
```

## Roadmap

- [x] **Computational Graph & Automatic Differentiation:** Scalar-valued `Value` engine with topological sort and backpropagation.
- [x] **Neural Network API:** Implementation of `Neuron`, `Layer`, and `MLP` architectures.
- [x] **Visualization Tooling:** DAG rendering pipeline using Graphviz.
- [x] **Training Loop (Gradient Descent):** Implementing a full optimization routine to update the isolated parameter space iteratively ($w \leftarrow w - \alpha \cdot \nabla w$) over multiple epochs.
- [ ] **Vectorized Operations (Tensor Conversion):** Extending the scalar topology into $\mathbb{R}^n$ vector spaces using NumPy to replace `for` loops with optimized matrix multiplications (Dot Products) for significant performance gains.

## Acknowledgments
This implementation is heavily inspired by and structurally follows the foundational concepts taught in **Andrej Karpathy's** excellent [Neural Networks: Zero to Hero](https://karpathy.ai/zero-to-hero.html) series (specifically the *micrograd* lecture). The series served as the primary pedagogical guide for understanding the algorithmic translation of calculus and graph theory into a working automatic differentiation engine.