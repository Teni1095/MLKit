import numpy as np
from collections import deque
from graph import Graph
from opsenum import Ops
from ops import Operations
from gradients import Gradients
from transforms import TransformFunctions
from transforms import Transforms

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

    def _applyTransforms(self, data, link, inverse=False, opposite_shape=None):
        if data is None or link is None:
            return data

        ops = link.parent.ops if link.parent is not None else None

        transforms = reversed(link.transforms) if inverse else link.transforms
        for transform in transforms:
            params = transform.get("params", {})
            original_shape = transform.get("original_shape", {})
            data = TransformFunctions.apply(
                data,
                transform["transform"],
                inverse=inverse,
                original_shape=original_shape,
                ops=ops,
                opposite_shape=opposite_shape,
                **params,
            )
        return data

    def _compute(self, graph, inverse=False):
        left_data = graph.left.child.node.data if graph.left is not None else None
        right_data = graph.right.child.node.data if graph.right is not None else None
        left_shape = graph.left.child.node.shape if graph.left is not None else None
        right_shape = graph.right.child.node.shape if graph.right is not None else None

        left = self._applyTransforms(left_data, graph.left, inverse=inverse, opposite_shape=right_shape)
        right = self._applyTransforms(right_data, graph.right, inverse=inverse, opposite_shape=left_shape)
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

    def _isBackpropable(self, graph):
        if graph.left is None and graph.right is None:
            return True
        return graph.node.computedBy is self and graph.node.computedRound == self.round

    def _propagateTo(self, current, link, upstream):
        if link is None:
            return None
        child = link.child
        if not self._isBackpropable(child):
            return None
        isLeft = link is current.left
        left_data = current.left.child.node.data if current.left is not None else None
        right_data = current.right.child.node.data if current.right is not None else None
        local = Gradients.compute(current.ops, isLeft, left_data, right_data)
        if link.transforms and upstream:
            upstream.transforms = list(link.transforms)
        return (child, Gradients.combine(current.ops, upstream, local, isLeft))

    def backwardPass(self):
        stack = [(self.root, None)]
        while stack:
            current, upstream = stack.pop()
            if current not in self.gradients or self.gradients[current] is None:
                for link in (current.left, current.right):
                    result = self._propagateTo(current, link, upstream)
                    if result is not None:
                        stack.append(result)
            self._updateGradient(current, upstream)
        return None
