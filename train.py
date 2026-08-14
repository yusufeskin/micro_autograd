from nn import MLP
from engine import Value
from visualize import draw_dot

xs = [
    [2.0, 3.0, -1.0],
    [3.0, -1.0, 0.5],
    [0.5, 1.0, 1.0],
    [1.0, 1.0, -1.0]
]
ys = [1.0, -1.0, -1.0, 1.0]

n = MLP(3, [4, 4, 1])

EPOCHS = 50
LEARNING_RATE = 0.01

for step in range(EPOCHS):
    # forward pass 
    ypred = [n(x) for x in xs]
    loss = sum((yout[0] - ygt) ** 2 for ygt, yout in zip(ys, ypred))
    
    # backward pass
    for p in n.parameters():
        p.grad = 0.0
        
    loss.backward()
    
    # gradient descent
    for p in n.parameters():
        p.data += -LEARNING_RATE * p.grad
        
    print(f"Step {step:02d} | Loss: {loss.data:.4f}")

dot = draw_dot(loss)
dot.render('nn_graph', view=True, cleanup=True)