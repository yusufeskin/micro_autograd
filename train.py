from nn import MLP
import engine


x = [2.0, 3.0, -1.0]
n = MLP(3, [4, 4, 1])
a = n(x)
m = n.parameters()
