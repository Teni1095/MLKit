import numpy as np
from collections import deque
from graph import Graph
from opsenum import Ops
from ops import Operations
from gradients import Gradients
from transforms import TransformFunctions

class Engine:
    def __init__(self, graph):
        self.root = graph
        self.round = 0

    def _applyTransforms(self, data, link, inverse=False):
        if data is None or link is None:
            return data

        transforms = reversed(link.transforms) if inverse else link.transforms
        for transform in transforms:
            params = transform.get("params", {})
            data = TransformFunctions.apply(
                data,
                transform["transform"],
                inverse=inverse,
                **params,
            )
        return data

    def _compute(self, graph):
        left = self._applyTransforms(graph.left.child.node.data if graph.left is not None else None, graph.left)
        right = self._applyTransforms(graph.right.child.node.data if graph.right is not None else None, graph.right)
        graph.node.data = Operations.compute(graph.ops, left, right)
        graph.node.computedBy = self
        graph.node.computedRound = self.round

    def _isStale(self, graph):
        if graph.left is None and graph.right is None:
            return False
        if graph.node.computedBy is None:
            return True
        return graph.node.computedBy is self and graph.node.computedRound != self.round

    def _computeGraph(self, graph):
        stack = [graph]
        while stack:
            current = stack[-1]
            if current is None:
                stack.pop()
                continue
            left = current.left.child if current.left is not None else None
            right = current.right.child if current.right is not None else None

            if left is None and right is None:
                stack.pop()
                continue

            left_ready = left is None or not self._isStale(left)
            right_ready = right is None or not self._isStale(right)

            if left_ready and right_ready:
                stack.pop()
                self._compute(current)
            else:
                if not left_ready:
                    stack.append(left)
                if not right_ready:
                    stack.append(right)

    def forwardPass(self):
        self.round += 1
        self._computeGraph(self.root)

    def _computeGrad(self, graph, upstream, link):
        return None

    def backwardPass(self):
        return None
