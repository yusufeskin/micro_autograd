from nn import MLP
from engine import Value
from visualize import draw_dot

x = [2.0, 3.0, -1.0]
n = MLP(3, [2, 2, 1])
a = n(x)
m = n.parameters()

dot = draw_dot(a[0])
dot.render('nn', view=True)