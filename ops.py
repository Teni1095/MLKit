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


class Gradients:

    @staticmethod
    def compute(ops, upstream, isLeft, parent):
        if ops == Ops.ADD:
            return Gradients.add(upstream, isLeft)
        elif ops == Ops.SUB:
            return Gradients.sub(upstream, isLeft)
        elif ops == Ops.MUL:
            return Gradients.mul(upstream, isLeft, parent)
        elif ops == Ops.DIV:
            return Gradients.div(upstream, isLeft, parent)
        elif ops == Ops.MATMUL:
            return Gradients.matmul(upstream, isLeft, parent)
        elif ops == Ops.TRANSPOSE:
            return Gradients.transpose(upstream, parent)
        elif ops == Ops.EXP:
            return Gradients.exp(upstream, parent)
        elif ops == Ops.LOG:
            return Gradients.log(upstream, parent)
        elif ops == Ops.LOG10:
            return Gradients.log10(upstream, parent)
        elif ops == Ops.SIN:
            return Gradients.sin(upstream, parent)
        elif ops == Ops.COS:
            return Gradients.cos(upstream, parent)
        elif ops == Ops.TAN:
            return Gradients.tan(upstream, parent)
        elif ops == Ops.SQRT:
            return Gradients.sqrt(upstream, parent)
        elif ops == Ops.ABS:
            return Gradients.abs(upstream, parent)
        elif ops == Ops.NEG:
            return Gradients.neg(upstream)
        elif ops == Ops.POW:
            return Gradients.pow(upstream, isLeft, parent)
        elif ops == Ops.MAX:
            return Gradients.max(upstream, isLeft, parent)
        elif ops == Ops.MAX_REDUCE:
            return Gradients.max_reduce(upstream, parent)
        elif ops == Ops.SUM:
            return Gradients.sum(upstream, parent)

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
    def mul(upstream, isLeft, parent):
        grad = parent.right.child.clone().clearParents() if isLeft else parent.left.child.clone().clearParents()
        if upstream is None:
            return Graph(1) * grad
        return upstream * grad

    @staticmethod
    def div(upstream, isLeft, parent):
        left_clone = parent.left.child.clone().clearParents()
        right_clone = parent.right.child.clone().clearParents()
        if isLeft:
            grad = Graph(1) / right_clone
        else:
            grad = Graph(-1) * left_clone / (right_clone * right_clone)
        if upstream is None:
            return Graph(1) * grad
        return upstream * grad

    @staticmethod
    def matmul(upstream, isLeft, parent):
        if upstream is None:
            if isLeft:
                return parent.right.child.clone().clearParents().T
            else:
                return parent.left.child.clone().clearParents().T
        if isLeft:
            return upstream @ parent.right.child.clone().clearParents().T
        else:
            return parent.left.child.clone().clearParents().T @ upstream

    @staticmethod
    def transpose(upstream, parent):
        if upstream is None:
            return parent.left.child.clone().clearParents()
        return upstream.T

    @staticmethod
    def exp(upstream, parent):
        grad = parent.left.child.clone().clearParents().exp()
        if upstream is None:
            return Graph(1) * grad
        return upstream * grad

    @staticmethod
    def log(upstream, parent):
        grad = Graph(1) / parent.left.child.clone().clearParents()
        if upstream is None:
            return Graph(1) * grad
        return upstream * grad

    @staticmethod
    def log10(upstream, parent):
        grad = Graph(1) / (parent.left.child.clone().clearParents() * Graph(np.log(10)))
        if upstream is None:
            return Graph(1) * grad
        return upstream * grad

    @staticmethod
    def sin(upstream, parent):
        grad = parent.left.child.clone().clearParents().cos()
        if upstream is None:
            return Graph(1) * grad
        return upstream * grad

    @staticmethod
    def cos(upstream, parent):
        grad = Graph(-1) * parent.left.child.clone().clearParents().sin()
        if upstream is None:
            return Graph(1) * grad
        return upstream * grad

    @staticmethod
    def tan(upstream, parent):
        grad = Graph(1) / (parent.left.child.clone().clearParents().cos().pow(Graph(2)))
        if upstream is None:
            return Graph(1) * grad
        return upstream * grad

    @staticmethod
    def sqrt(upstream, parent):
        grad = Graph(1) / (Graph(2) * parent.left.child.clone().clearParents().sqrt())
        if upstream is None:
            return Graph(1) * grad
        return upstream * grad

    @staticmethod
    def abs(upstream, parent):
        grad = parent.left.child.clone().clearParents() / parent.left.child.clone().clearParents().abs()
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
    def pow(upstream, isLeft, parent):
        left_clone = parent.left.child.clone().clearParents()
        right_clone = parent.right.child.clone().clearParents()
        if isLeft:
            grad = right_clone * left_clone.pow(right_clone - Graph(1))
        else:
            grad = parent.clone().clearParents() * left_clone.log()
        if upstream is None:
            return Graph(1) * grad
        return upstream * grad

    @staticmethod
    def max(upstream, isLeft, parent):
        left_clone = parent.left.child.clone().clearParents()
        right_clone = parent.right.child.clone().clearParents()
        if isLeft:
            grad = Graph(np.where(left_clone.node.data >= right_clone.node.data, 1.0, 0.0))
        else:
            grad = Graph(np.where(right_clone.node.data > left_clone.node.data, 1.0, 0.0))
        if upstream is None:
            return Graph(1) * grad
        return upstream * grad

    @staticmethod
    def max_reduce(upstream, parent):
        left_clone = parent.left.child.clone().clearParents()
        grad = Graph(np.where(left_clone.node.data == parent.node.data, 1.0, 0.0))
        if upstream is None:
            return Graph(1) * grad
        return upstream * grad

    @staticmethod
    def sum(upstream, parent):
        grad = Graph(np.ones_like(parent.left.child.node.data))
        if upstream is None:
            return Graph(1) * grad
        return upstream * grad
