import math

class Value:
    def __init__(self, data, _children=(), _op='', label=''):
        self.data = data
        self._prev = set(_children)
        self._op = _op
        self.label = label
        self.grad = 0.0
        self._backward = lambda : None

    def __repr__(self):
        return f"Value(data={self.data}, op={self._op}, label={self.label})"

    def _checktype(self, obj):
        if isinstance(obj, Value):
            return obj
        elif isinstance(obj, (int, float)):
            obj = Value(obj)
            return obj
        else:
            return NotImplemented

    def __add__(self, other):
        other = self._checktype(other)
        if other is NotImplemented:
            return NotImplemented
        out = Value(self.data + other.data, (self, other), '+')
        def _backward():
            self.grad += 1.0 * out.grad
            other.grad += 1.0 * out.grad
        out._backward = _backward
        return out
    
    def __sub__(self, other):
        other = self._checktype(other)
        if other is NotImplemented:
            return NotImplemented
        out = Value(self.data - other.data, (self, other), '-')
        def _backward():
            self.grad += 1.0 * out.grad
            other.grad += -1.0 * out.grad
        out._backward = _backward
        return out

    def __mul__(self, other):
        other = self._checktype(other)
        if other is NotImplemented:
            return NotImplemented
        out = Value(self.data * other.data, (self, other), '*')
        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
        out._backward = _backward
        return out
    
    def __truediv__(self, other):
        other = self._checktype(other)
        if other is NotImplemented:
            return NotImplemented
        if other.data == 0:
            raise ZeroDivisionError("zero div error!")

        out = Value(self.data / other.data, (self, other), '/')
        def _backward():
            self.grad += (1.0 / other.data) * out.grad
            other.grad += (-self.data / (other.data ** 2)) * out.grad
        out._backward = _backward
        return out
           
    
    def exp(self):
        x = self.data
        out = Value(math.exp(x), (self, ), 'exp')
        def _backward():
            self.grad += out.data * out.grad
        out._backward = _backward
        return out
    
    def tanh(self):
        x = self.data
        out = Value((math.exp(x) - math.exp(-x)) / (math.exp(x) + math.exp(-x)), (self, ), 'tanh')
        def _backward():
            self.grad += (1 - out.data ** 2) * out.grad
        out._backward = _backward
        return out

    def __pow__(self, other):
        out = Value(self.data ** other, (self,), f'**{other}')
        def _backward():
            self.grad += (other * (self.data ** (other - 1))) * out.grad
        out._backward = _backward
        return out
    
    # commutavity and right operants
    def __rmul__(self,other):
        return self * other
    
    def __radd__(self,other):
        return self + other
    
    def __rsub__(self,other):
         return self._checktype(other) - self
    
    # __rtruediv__ will be added later (it needs more advance techniques due to the derivative calculations)

    # topological sort
    def backward(self):
        topo = []
        visited = set()
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)
        
        build_topo(self)
        self.grad = 1.0
        for node in reversed(topo):
            node._backward()
