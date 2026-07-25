from node import Node
from opsenum import Ops
from link import Link
from transforms import Transforms

class Graph:
    def __init__(self, data, collapsible=False):
        self.node = Node(data)
        self._left = None
        self._right = None
        self.parent = []
        self.ops = None
        self.collapsible = collapsible
        self.transforms = []

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

    def transform(self, transform, **params):
        self.transforms.append({
            "transform": transform,
            "params": params,
            "original_shape": self.node.data.shape,
        })
        return self

    def _transferTransforms(self, link):
        if not self.transforms:
            return

        link.transforms.extend(self.transforms)
        self.transforms = []

    def _createGraph(self, other, ops):
        g = Graph(None)

        left_link = Link(self, g)
        g.left = left_link
        self.parent.append(left_link)

        if other is not None:
            right_link = Link(other, g)
            g.right = right_link
            other.parent.append(right_link)

        self._transferTransforms(left_link)
        if other is not None:
            other._transferTransforms(right_link)

        g.ops = ops
        return g

    def clone(self):
        g = Graph(self.node.data)
        g._left = self._left
        g._right = self._right
        g.parent = list(self.parent)
        g.ops = self.ops
        g.transforms = list(self.transforms)
        return g

    def set(self, other):
        self.node = other.node
        self._left = other._left
        self._right = other._right
        self.parent = list(other.parent)
        self.ops = other.ops
        self.transforms = list(other.transforms)
        return self

    def clearParents(self):
        self.parent = []
        return self

    @property
    def shape(self):
        return self.node.data.shape

    def transpose(self, axes=None):
        return self.transform(Transforms.TRANSPOSE, axes=axes)

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

    def ge(self, other):
        if not isinstance(other, Graph):
            other = Graph(other)
        return self._createGraph(other, Ops.GE)

    def gt(self, other):
        if not isinstance(other, Graph):
            other = Graph(other)
        return self._createGraph(other, Ops.GT)

    def eq(self, other):
        if not isinstance(other, Graph):
            other = Graph(other)
        return self._createGraph(other, Ops.EQ)

    def sign(self):
        return self._createGraph(None, Ops.SIGN)

    def round(self):
        return self._createGraph(None, Ops.ROUND)

    def argmax(self):
        return self._createGraph(None, Ops.ARGMAX)
