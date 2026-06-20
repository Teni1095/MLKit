from node import Node
from opsenum import Ops
from link import Link

class Graph:
    def __init__(self, data, collapsible=False):
        self.node = Node(data)
        self._left = None
        self._right = None
        self.parent = []
        self.ops = None
        self.collapsible = collapsible
        self.grad = None

    @property
    def left(self):
        return self._left

    @left.setter
    def left(self, value):
        self._left = value

    @property
    def right(self):
        return self._right

    @right.setter
    def right(self, value):
        self._right = value

    def _createGraph(self, other, ops):
        g = Graph(None)

        left_link = Link(self, g)
        g.left = left_link
        self.parent.append(left_link)

        if other is not None:
            right_link = Link(other, g)
            g.right = right_link
            other.parent.append(right_link)

        g.ops = ops
        return g

    def clone(self):
        g = Graph(self.node.data)
        g._left = self._left
        g._right = self._right
        g.parent = self.parent
        g.ops = self.ops
        return g

    def clearParents(self):
        self.parent = []
        return self

    @property
    def T(self):
        return self._createGraph(None, Ops.TRANSPOSE)

    def __add__(self, other):
        if not isinstance(other, Graph):
            other = Graph(other)
        return self._createGraph(other, Ops.ADD)

    def __radd__(self, other):
        return Graph(other) + self

    def __sub__(self, other):
        if not isinstance(other, Graph):
            other = Graph(other)
        return self._createGraph(other, Ops.SUB)

    def __rsub__(self, other):
        return Graph(other) - self

    def __mul__(self, other):
        if not isinstance(other, Graph):
            other = Graph(other)
        return self._createGraph(other, Ops.MUL)

    def __rmul__(self, other):
        return Graph(other) * self

    def __truediv__(self, other):
        if not isinstance(other, Graph):
            other = Graph(other)
        return self._createGraph(other, Ops.DIV)

    def __rtruediv__(self, other):
        return Graph(other) / self

    def __matmul__(self, other):
        if not isinstance(other, Graph):
            other = Graph(other)
        return self._createGraph(other, Ops.MATMUL)

    def exp(self):
        return self._createGraph(None, Ops.EXP)

    def log(self):
        return self._createGraph(None, Ops.LOG)

    def log10(self):
        return self._createGraph(None, Ops.LOG10)

    def sin(self):
        return self._createGraph(None, Ops.SIN)

    def cos(self):
        return self._createGraph(None, Ops.COS)

    def tan(self):
        return self._createGraph(None, Ops.TAN)

    def sqrt(self):
        return self._createGraph(None, Ops.SQRT)

    def abs(self):
        return self._createGraph(None, Ops.ABS)

    def neg(self):
        return self._createGraph(None, Ops.NEG)

    def pow(self, n):
        if not isinstance(n, Graph):
            n = Graph(n)
        return self._createGraph(n, Ops.POW)

    def max(self, other=None):
        if other is None:
            return self._createGraph(None, Ops.MAX_REDUCE)
        if not isinstance(other, Graph):
            other = Graph(other)
        return self._createGraph(other, Ops.MAX)

    def sum(self):
        return self._createGraph(None, Ops.SUM)
