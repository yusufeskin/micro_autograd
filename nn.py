import random 
from engine import Value

class Neuron:
    def __init__(self, nin):
        self.w = []
        for _ in range(nin):
            random_number = random.uniform(-1,1)
            self.w.append(Value(random_number))

        self.b = Value(random.uniform(-1,1))

    def __call__(self, x):
        if len(self.w) != len(x):
            raise ValueError('Dimension mismatch')
        # w*x + b
        act = []
        for wi, xi in zip(self.w, x):
            mult_value = wi*xi
            act.append(mult_value)

        sum_act = sum(act, self.b)
        out = sum_act.tanh()
        return out
    
    def parameters(self):
        return self.w + [self.b]
    
class Layer:
    def __init__(self, nin, nout):
        self.neurons = []
        for _ in range(nout):
            self.neurons.append(Neuron(nin))

    def __call__(self, x):
        outs = []
        for n in self.neurons:
            outs.append(n(x))
        return outs  # now it always returns a list
    
    def parameters(self):
        params = []
        for neuron in self.neurons:
            ps = neuron.parameters()
            params.extend(ps)
        return params

class MLP:
    def __init__(self, nin, nouts):
        sz = [nin] + nouts
        self.layers = []
        for i in range(len(sz)- 1):
            layer = Layer(sz[i], sz[i+1])
            self.layers.append(layer)
            
    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
    
    def parameters(self):
        params = []
        for layer in self.layers:
            ps = layer.parameters()
            params.extend(ps)
        return params