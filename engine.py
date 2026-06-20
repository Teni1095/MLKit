import numpy as np
from graph import Graph
from opsenum import Ops
from ops import Operations, Gradients

class Engine:
    def __init__(self):
        pass

    def _compute(self, graph):
        left = graph.left.child.node.data
        right = graph.right.child.node.data if graph.right is not None else None
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
        isLeft = parent.left.child is graph
        return Gradients.compute(parent.ops, upstream, isLeft, parent)

    def backwardPass(self, graph):
        stack = [(graph, None)]
        while stack:
            current, upstream = stack.pop()
            if current.left is None and current.right is None and upstream is not None:
                self._computeGraph(upstream)
                grad = upstream.node.data
                if current.collapsible:
                    grad = np.sum(grad, axis=0, keepdims=True)
                if current.grad is None:
                    current.grad = grad
                else:
                    current.grad = current.grad + grad
            else:
                if current.left is not None:
                    left_child = current.left.child
                    left_grad = self._getGrad(left_child, upstream, current)
                    stack.append((left_child, left_grad))
                if current.right is not None:
                    right_child = current.right.child
                    right_grad = self._getGrad(right_child, upstream, current)
                    stack.append((right_child, right_grad))

    def getGradStruct(self, graph, leaves):
        target_set = set(id(leaf) for leaf in leaves)
        results = []
        stack = [(graph, None)]
        while stack:
            current, upstream = stack.pop()
            if id(current) in target_set:
                results.append((current, upstream))
            if current.left is not None:
                left_child = current.left.child
                left_grad = self._getGrad(left_child, upstream, current)
                stack.append((left_child, left_grad))
            if current.right is not None:
                right_child = current.right.child
                right_grad = self._getGrad(right_child, upstream, current)
                stack.append((right_child, right_grad))
        return results
