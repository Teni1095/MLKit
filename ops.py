import numpy as np
from opsenum import Ops
from graph import Graph


class Operations:

    @staticmethod
    def compute(ops, left, right):
        if ops == Ops.ADD:
            return Operations.add(left, right)
        elif ops == Ops.SUB:
            return Operations.sub(left, right)
        elif ops == Ops.MUL:
            return Operations.mul(left, right)
        elif ops == Ops.DIV:
            return Operations.div(left, right)
        elif ops == Ops.MATMUL:
            return Operations.matmul(left, right)
        elif ops == Ops.TRANSPOSE:
            return Operations.transpose(left)
        elif ops == Ops.EXP:
            return Operations.exp(left)
        elif ops == Ops.LOG:
            return Operations.log(left)
        elif ops == Ops.LOG10:
            return Operations.log10(left)
        elif ops == Ops.SIN:
            return Operations.sin(left)
        elif ops == Ops.COS:
            return Operations.cos(left)
        elif ops == Ops.TAN:
            return Operations.tan(left)
        elif ops == Ops.SQRT:
            return Operations.sqrt(left)
        elif ops == Ops.ABS:
            return Operations.abs(left)
        elif ops == Ops.NEG:
            return Operations.neg(left)
        elif ops == Ops.POW:
            return Operations.pow(left, right)
        elif ops == Ops.MAX:
            return Operations.max(left, right)
        elif ops == Ops.MAX_REDUCE:
            return Operations.max_reduce(left)
        elif ops == Ops.SUM:
            return Operations.sum(left)
        elif ops == Ops.GE:
            return Operations.ge(left, right)
        elif ops == Ops.GT:
            return Operations.gt(left, right)
        elif ops == Ops.EQ:
            return Operations.eq(left, right)
        elif ops == Ops.SIGN:
            return Operations.sign(left)
        elif ops == Ops.ROUND:
            return Operations.round(left)
        elif ops == Ops.ARGMAX:
            return Operations.argmax(left)

    @staticmethod
    def add(left, right):
        return left + right

    @staticmethod
    def sub(left, right):
        return left - right

    @staticmethod
    def mul(left, right):
        return left * right

    @staticmethod
    def div(left, right):
        return left / right

    @staticmethod
    def matmul(left, right):
        return left @ right

    @staticmethod
    def transpose(left):
        return left.T

    @staticmethod
    def exp(left):
        return np.exp(left)

    @staticmethod
    def log(left):
        return np.log(left)

    @staticmethod
    def log10(left):
        return np.log10(left)

    @staticmethod
    def sin(left):
        return np.sin(left)

    @staticmethod
    def cos(left):
        return np.cos(left)

    @staticmethod
    def tan(left):
        return np.tan(left)

    @staticmethod
    def sqrt(left):
        return np.sqrt(left)

    @staticmethod
    def abs(left):
        return np.abs(left)

    @staticmethod
    def neg(left):
        return -left

    @staticmethod
    def pow(left, right):
        return left ** right

    @staticmethod
    def max(left, right):
        return np.maximum(left, right)

    @staticmethod
    def max_reduce(left):
        return np.max(left)

    @staticmethod
    def sum(left):
        return np.sum(left)

    @staticmethod
    def ge(left, right):
        return np.where(left >= right, 1.0, 0.0)

    @staticmethod
    def gt(left, right):
        return np.where(left > right, 1.0, 0.0)

    @staticmethod
    def eq(left, right):
        return np.where(left == right, 1.0, 0.0)

    @staticmethod
    def sign(left):
        return np.sign(left)

    @staticmethod
    def round(left):
        return np.round(left)

    @staticmethod
    def argmax(left):
        return np.argmax(left)


class Gradients:

    @staticmethod
    def compute(ops, upstream, isLeft, left, right):
        if ops == Ops.ADD:
            return Gradients.add(upstream, isLeft)
        elif ops == Ops.SUB:
            return Gradients.sub(upstream, isLeft)
        elif ops == Ops.MUL:
            return Gradients.mul(upstream, isLeft, left, right)
        elif ops == Ops.DIV:
            return Gradients.div(upstream, isLeft, left, right)
        elif ops == Ops.MATMUL:
            return Gradients.matmul(upstream, isLeft, left, right)
        elif ops == Ops.TRANSPOSE:
            return Gradients.transpose(upstream, left)
        elif ops == Ops.EXP:
            return Gradients.exp(upstream, left)
        elif ops == Ops.LOG:
            return Gradients.log(upstream, left)
        elif ops == Ops.LOG10:
            return Gradients.log10(upstream, left)
        elif ops == Ops.SIN:
            return Gradients.sin(upstream, left)
        elif ops == Ops.COS:
            return Gradients.cos(upstream, left)
        elif ops == Ops.TAN:
            return Gradients.tan(upstream, left)
        elif ops == Ops.SQRT:
            return Gradients.sqrt(upstream, left)
        elif ops == Ops.ABS:
            return Gradients.abs(upstream, left)
        elif ops == Ops.NEG:
            return Gradients.neg(upstream)
        elif ops == Ops.POW:
            return Gradients.pow(upstream, isLeft, left, right)
        elif ops == Ops.MAX:
            return Gradients.max(upstream, isLeft, left, right)
        elif ops == Ops.MAX_REDUCE:
            return Gradients.max_reduce(upstream, left)
        elif ops == Ops.SUM:
            return Gradients.sum(upstream, left)
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
    def add(upstream, isLeft):
        grad = Graph(1)
        if upstream is None:
            return Graph(1) * grad
        return upstream * grad

    @staticmethod
    def sub(upstream, isLeft):
        grad = Graph(1) if isLeft else Graph(-1)
        if upstream is None:
            return Graph(1) * grad
        return upstream * grad

    @staticmethod
    def mul(upstream, isLeft, left, right):
        grad = right if isLeft else left
        if upstream is None:
            return Graph(1) * grad
        return upstream * grad

    @staticmethod
    def div(upstream, isLeft, left, right):
        if isLeft:
            grad = Graph(1) / right
        else:
            grad = Graph(-1) * left / (right * right)
        if upstream is None:
            return Graph(1) * grad
        return upstream * grad

    @staticmethod
    def matmul(upstream, isLeft, left, right):
        if upstream is None:
            if isLeft:
                return right.T
            else:
                return left.T
        if isLeft:
            return upstream @ right.T
        else:
            return left.T @ upstream

    @staticmethod
    def transpose(upstream, left):
        if upstream is None:
            return left
        return upstream.T

    @staticmethod
    def exp(upstream, left):
        grad = left.exp()
        if upstream is None:
            return Graph(1) * grad
        return upstream * grad

    @staticmethod
    def log(upstream, left):
        grad = Graph(1) / left
        if upstream is None:
            return Graph(1) * grad
        return upstream * grad

    @staticmethod
    def log10(upstream, left):
        grad = Graph(1) / (left * Graph(np.log(10)))
        if upstream is None:
            return Graph(1) * grad
        return upstream * grad

    @staticmethod
    def sin(upstream, left):
        grad = left.cos()
        if upstream is None:
            return Graph(1) * grad
        return upstream * grad

    @staticmethod
    def cos(upstream, left):
        grad = Graph(-1) * left.sin()
        if upstream is None:
            return Graph(1) * grad
        return upstream * grad

    @staticmethod
    def tan(upstream, left):
        grad = Graph(1) / (left.cos().pow(Graph(2)))
        if upstream is None:
            return Graph(1) * grad
        return upstream * grad

    @staticmethod
    def sqrt(upstream, left):
        grad = Graph(1) / (Graph(2) * left.sqrt())
        if upstream is None:
            return Graph(1) * grad
        return upstream * grad

    @staticmethod
    def abs(upstream, left):
        grad = left / left.abs()
        if upstream is None:
            return Graph(1) * grad
        return upstream * grad

    @staticmethod
    def neg(upstream):
        grad = Graph(-1)
        if upstream is None:
            return Graph(1) * grad
        return upstream * grad

    @staticmethod
    def pow(upstream, isLeft, left, right):
        if isLeft:
            grad = right * left.pow(right - Graph(1))
        else:
            grad = left.pow(right) * left.log()
        if upstream is None:
            return Graph(1) * grad
        return upstream * grad

    @staticmethod
    def max(upstream, isLeft, left, right):
        if isLeft:
            grad = left.ge(right)
        else:
            grad = right.gt(left)
        if upstream is None:
            return Graph(1) * grad
        return upstream * grad

    @staticmethod
    def max_reduce(upstream, left):
        grad = left.eq(left.max())
        if upstream is None:
            return Graph(1) * grad
        return upstream * grad

    @staticmethod
    def sum(upstream, left):
        grad = Graph(np.ones(left.shape))
        if upstream is None:
            return Graph(1) * grad
        return upstream * grad

    @staticmethod
    def ge(isLeft, left, right):
        return Graph(np.zeros(left.shape if isLeft else right.shape))

    @staticmethod
    def gt(isLeft, left, right):
        return Graph(np.zeros(left.shape if isLeft else right.shape))

    @staticmethod
    def eq(isLeft, left, right):
        return Graph(np.zeros(left.shape if isLeft else right.shape))

    @staticmethod
    def sign(left):
        return Graph(np.zeros(left.shape))

    @staticmethod
    def round(left):
        return Graph(np.zeros(left.shape))

    @staticmethod
    def argmax(left):
        return Graph(np.zeros(left.shape))
