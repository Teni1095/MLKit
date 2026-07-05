import numpy as np
from graph import Graph
from opsenum import Ops
from ops import Operations, Gradients
from transforms import TransformFunctions

class Engine:
    def __init__(self):
        pass

    def _applyTransforms(self, data, link, inverse=False):
        if data is None or link is None:
            return data

        for transform in link.transforms:
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

            left_ready = left is None or left.node.data is not None
            right_ready = right is None or right.node.data is not None

            if left_ready and right_ready:
                stack.pop()
                self._compute(current)
            else:
                if not left_ready:
                    stack.append(left)
                if not right_ready:
                    stack.append(right)

    def forwardPass(self, graph):
        self._computeGraph(graph)

    def _getGrad(self, graph, upstream, parent):
        return None

    def backwardPass(self, graph):
        return None

    def getGradStruct(self, graph, leaves):
        return []
