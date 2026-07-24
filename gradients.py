import numpy as np
from opsenum import Ops
from graph import Graph


class Gradients:

    @staticmethod
    def compute(ops, isLeft, left, right):
        if ops == Ops.ADD:
            return Gradients.add(isLeft)
        elif ops == Ops.SUB:
            return Gradients.sub(isLeft)
        elif ops == Ops.MUL:
            return Gradients.mul(isLeft, left, right)
        elif ops == Ops.DIV:
            return Gradients.div(isLeft, left, right)
        elif ops == Ops.MATMUL:
            return Gradients.matmul(isLeft, left, right)
        elif ops == Ops.EXP:
            return Gradients.exp(left)
        elif ops == Ops.LOG:
            return Gradients.log(left)
        elif ops == Ops.LOG10:
            return Gradients.log10(left)
        elif ops == Ops.SIN:
            return Gradients.sin(left)
        elif ops == Ops.COS:
            return Gradients.cos(left)
        elif ops == Ops.TAN:
            return Gradients.tan(left)
        elif ops == Ops.SQRT:
            return Gradients.sqrt(left)
        elif ops == Ops.ABS:
            return Gradients.abs(left)
        elif ops == Ops.NEG:
            return Gradients.neg()
        elif ops == Ops.POW:
            return Gradients.pow(isLeft, left, right)
        elif ops == Ops.MAX:
            return Gradients.max(isLeft, left, right)
        elif ops == Ops.MAX_REDUCE:
            return Gradients.max_reduce(left)
        elif ops == Ops.SUM:
            return Gradients.sum(left)
        elif ops == Ops.GE:
            return Gradients.ge(isLeft, left, right)
        elif ops == Ops.GT:
            return Gradients.gt(isLeft, left, right)
        elif ops == Ops.EQ:
            return Gradients.eq(isLeft, left, right)
        elif ops == Ops.SIGN:
            return Gradients.sign(left)
        elif ops == Ops.ROUND:
            return Gradients.round(left)
        elif ops == Ops.ARGMAX:
            return Gradients.argmax(left)

    @staticmethod
    def combine(ops, upstream, local, isLeft):
        if upstream is None:
            return local
        if ops == Ops.MATMUL:
            return upstream @ local if isLeft else local @ upstream
        return upstream * local

    # --- pure-operator ops: already type-unaware via Python's operator
    # dunders, work identically whether operands are raw data or Graph ---

    @staticmethod
    def add(isLeft):
        return Graph(1)

    @staticmethod
    def sub(isLeft):
        return Graph(1) if isLeft else Graph(-1)

    @staticmethod
    def mul(isLeft, left, right):
        return Graph(right) if isLeft else Graph(left)

    @staticmethod
    def div(isLeft, left, right):
        if isLeft:
            return Graph(1) / Graph(right)
        return Graph(-1) * Graph(left) / (Graph(right) * Graph(right))

    @staticmethod
    def matmul(isLeft, left, right):
        return Graph(right).transpose() if isLeft else Graph(left).transpose()

    @staticmethod
    def neg():
        return Graph(-1)

    # --- method-based ops: build symbolic Graph expressions via Graph's
    # own methods, wrapping raw left/right into fresh disconnected leaves ---

    @staticmethod
    def exp(left):
        return Graph(left).exp()

    @staticmethod
    def log(left):
        return Graph(1) / Graph(left)

    @staticmethod
    def log10(left):
        return Graph(1) / (Graph(left) * Graph(np.log(10)))

    @staticmethod
    def sin(left):
        return Graph(left).cos()

    @staticmethod
    def cos(left):
        return Graph(-1) * Graph(left).sin()

    @staticmethod
    def tan(left):
        return Graph(1) / (Graph(left).cos().pow(Graph(2)))

    @staticmethod
    def sqrt(left):
        return Graph(1) / (Graph(2) * Graph(left).sqrt())

    @staticmethod
    def abs(left):
        return Graph(left) / Graph(left).abs()

    @staticmethod
    def pow(isLeft, left, right):
        if isLeft:
            return Graph(right) * Graph(left).pow(Graph(right) - Graph(1))
        return Graph(left).pow(Graph(right)) * Graph(left).log()

    @staticmethod
    def max(isLeft, left, right):
        if isLeft:
            return Graph(left).ge(Graph(right))
        return Graph(right).gt(Graph(left))

    @staticmethod
    def max_reduce(left):
        return Graph(left).eq(Graph(left).max())

    @staticmethod
    def sum(left):
        return Graph(np.ones(np.shape(left)))

    @staticmethod
    def ge(isLeft, left, right):
        return Graph(np.zeros(np.shape(left) if isLeft else np.shape(right)))

    @staticmethod
    def gt(isLeft, left, right):
        return Graph(np.zeros(np.shape(left) if isLeft else np.shape(right)))

    @staticmethod
    def eq(isLeft, left, right):
        return Graph(np.zeros(np.shape(left) if isLeft else np.shape(right)))

    @staticmethod
    def sign(left):
        return Graph(np.zeros(np.shape(left)))

    @staticmethod
    def round(left):
        return Graph(np.zeros(np.shape(left)))

    @staticmethod
    def argmax(left):
        return Graph(np.zeros(np.shape(left)))
