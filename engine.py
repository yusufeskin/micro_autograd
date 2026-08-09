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
            self.grad = 1.0 * out.grad
            other.grad = 1.0 * out.grad
        out._backward = _backward
        return out
    
    def __sub__(self, other):
        other = self._checktype(other)
        if other is NotImplemented:
            return NotImplemented
        out = Value(self.data - other.data, (self, other), '-')
        def _backward():
            self.grad = 1.0 * out.grad
            other.grad = -1.0 * out.grad
        out._backward = _backward
        return out

    def __mul__(self, other):
        other = self._checktype(other)
        if other is NotImplemented:
            return NotImplemented
        out = Value(self.data * other.data, (self, other), '*')
        def _backward():
            self.grad = other.data * out.grad
            other.grad = self.data * out.grad
        out._backward = _backward
        return out
    
    def __truediv__(self, other):
        other = self._checktype(other)
        if other is NotImplemented:
            return NotImplemented
        if other.data == 0:
            raise ZeroDivisionError("Sıfıra bölme hatası!")

        out = Value(self.data / other.data, (self, other), '/')
        def _backward():
            self.grad = (1.0 / other.data) * out.grad
            other.grad = (-self.data / (other.data ** 2)) * out.grad
        out._backward = _backward
        return out
           
           
    
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

a = Value(3.0, label='a')
b = Value(2.0, label='b')
e = Value(-4.0, label='e')

c = a * b; c.label='c'
d = c + e; d.label='d'
print(d)
print(c)
d.grad = 1.0
d.backward()
print(f"a.grad: {a.grad}")
print(f"b.grad: {b.grad}")