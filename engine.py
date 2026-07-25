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
        self.gradients = {}

    def _updateGradient(self, graph, upstream):
        if graph in self.gradients and self.gradients[graph] is not None:
            old = self.gradients[graph].clone()
            combined = old + upstream
            self.gradients[graph].set(combined)
        else:
            self.gradients[graph] = upstream

    def _applyTransforms(self, data, link, inverse=False):
        if data is None or link is None:
            return data

        transforms = reversed(link.transforms) if inverse else link.transforms
        for transform in transforms:
            params = transform.get("params", {})
            original_shape = transform.get("original_shape", {})
            data = TransformFunctions.apply(
                data,
                transform["transform"],
                inverse=inverse,
                original_shape=original_shape,
                **params,
            )
        return data

    def _compute(self, graph, inverse=False):
        left = self._applyTransforms(graph.left.child.node.data if graph.left is not None else None, graph.left, inverse=inverse)
        right = self._applyTransforms(graph.right.child.node.data if graph.right is not None else None, graph.right, inverse=inverse)
        graph.node.data = Operations.compute(graph.ops, left, right)
        graph.node.computedBy = self
        graph.node.computedRound = self.round

    def _isStale(self, graph):
        if graph.left is None and graph.right is None:
            return False
        if graph.node.computedBy is None:
            return True
        return graph.node.computedBy is self and graph.node.computedRound != self.round

    def _computeGraph(self, graph, inverse=False):
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
                self._compute(current, inverse=inverse)
            else:
                if not left_ready:
                    stack.append(left)
                if not right_ready:
                    stack.append(right)

    def forwardPass(self):
        self.round += 1
        self._computeGraph(self.root)

    def getGradValue(self, graph):
        grad_graph = self.gradients[graph];
        self._computeGraph(grad_graph, inverse=True)
        return grad_graph.node.data

    def _computeGrad(self, graph, upstream, link):
        return None

    def backwardPass(self):
        stack = [(self.root, None)]
        while stack:
            current_tuple = stack[-1]
            if current_tuple is None:
                stack.pop()
                continue
            current, upstream = current_tuple
            stack.pop()
            left = current.left.child if current.left is not None else None
            right = current.right.child if current.right is not None else None
            if current not in self.gradients or self.gradients[current] is None:
                if left is not None:
                    left_graph = Gradients.compute(current.ops, True, left.node.data, right.node.data if right is not None else None)
                    if current.left.transforms and upstream:
                        upstream.transforms = list(current.left.transforms)
                    left_grad = Gradients.combine(current.ops, upstream, left_graph, True)
                    left_tuple = (left, left_grad)
                    stack.append(left_tuple)
                if right is not None:
                    right_graph = Gradients.compute(current.ops, False, left.node.data if left is not None else None, right.node.data)
                    if current.right.transforms and upstream:
                        upstream.transforms = list(current.right.transforms)
                    right_grad = Gradients.combine(current.ops, upstream, right_graph, False)
                    right_tuple = (right, right_grad)
                    stack.append(right_tuple)
            self._updateGradient(current, upstream)
        return None
