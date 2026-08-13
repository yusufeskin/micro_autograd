from nn import MLP
from engine import Value
from visualize import draw_dot


xs = [
  [2.0, 3.0, -1.0],
  [2.0, 3.0, -1.0],
  [2.0, 3.0, -1.0],
  [3.0, -1.0, 0.5],
  [0.5, 1.0, 1.0],
  [1.0, 1.0, -1.0]
]
ys = [1.0, 1.0, 1.0, -1.0, -1.0, 1.0]

n = MLP(3, [4, 4, 1])
ypred = [n(x) for x in xs]

loss = sum(((yout[0] - ygt) ** 2 for ygt, yout in zip(ys, ypred)))

print(loss.data)
loss.backward()

dot = draw_dot(loss)
dot.render('nn', view=True, cleanup=True)