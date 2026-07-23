import numpy as np
from opsenum import Ops


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
