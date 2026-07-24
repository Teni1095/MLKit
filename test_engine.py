import unittest
import numpy as np
from graph import Graph
from engine import Engine
from transforms import Transforms


class TestEngineForwardPass(unittest.TestCase):
    def test_forward_pass_computes_root_value(self):
        x = Graph(2.0)
        y = Graph(3.0)
        root = x * y + Graph(4.0)

        engine = Engine(root)
        engine.forwardPass()

        self.assertEqual(root.node.data, 10.0)
        self.assertEqual(x.node.data, 2.0)
        self.assertEqual(y.node.data, 3.0)

    def test_forward_pass_broadcast_transform(self):
        x = Graph(np.array([1.0, 2.0]))
        b = Graph(np.array([[1.0], [1.0]]))

        x.transform(Transforms.BROADCAST_TO, shape=(2, 2))
        root = x + b

        engine = Engine(root)
        engine.forwardPass()

        expected = np.array([[2.0, 3.0], [2.0, 3.0]])
        np.testing.assert_array_equal(root.node.data, expected)

    def test_backward_pass_broadcast_transform(self):
        x = Graph(np.array([1.0, 2.0]))
        b = Graph(np.array([[1.0], [1.0]]))

        x.transform(Transforms.BROADCAST_TO, shape=(2, 2))
        out = x + b
        root = out.sum()

        engine = Engine(root)
        engine.forwardPass()
        engine.backwardPass()

        self.assertIn(x, engine.gradients)
        self.assertIn(b, engine.gradients)

        grad_x = engine.gradients[x]
        grad_b = engine.gradients[b]

        self.assertIsInstance(grad_x, Graph)
        self.assertIsInstance(grad_b, Graph)

        np.testing.assert_array_equal(engine.getGradValue(x), np.array([2.0, 2.0]))
        np.testing.assert_array_equal(engine.getGradValue(b), np.ones((2, 2)))

    def test_forward_pass_recomputes_on_new_round(self):
        x = Graph(2.0)
        y = Graph(3.0)
        root = x * y + Graph(4.0)

        engine = Engine(root)
        engine.forwardPass()
        self.assertEqual(root.node.data, 10.0)

        x.node.data = 4.0
        engine.forwardPass()
        self.assertEqual(root.node.data, 16.0)

    def test_backward_pass_builds_gradient_graphs(self):
        x = Graph(2.0)
        y = Graph(3.0)
        root = x * y + Graph(4.0)

        engine = Engine(root)
        engine.forwardPass()
        engine.backwardPass()

        self.assertIn(x, engine.gradients)
        self.assertIn(y, engine.gradients)
        self.assertIn(root, engine.gradients)

        grad_x = engine.gradients[x]
        grad_y = engine.gradients[y]

        self.assertIsInstance(grad_x, Graph)
        self.assertIsInstance(grad_y, Graph)

        Engine(grad_x).forwardPass()
        Engine(grad_y).forwardPass()

        self.assertEqual(grad_x.node.data, 3.0)
        self.assertEqual(grad_y.node.data, 2.0)

    def test_backward_pass_with_matrices(self):
        X = Graph(np.array([[1.0, 2.0], [3.0, 4.0]]))
        W = Graph(np.array([[2.0, 0.0], [0.0, 2.0]]))
        b = Graph(np.array([[1.0, 1.0], [1.0, 1.0]]))

        out = X @ W + b
        root = out.sum()

        engine = Engine(root)
        engine.forwardPass()
        engine.backwardPass()

        self.assertIn(X, engine.gradients)
        self.assertIn(W, engine.gradients)
        self.assertIn(b, engine.gradients)

        grad_X = engine.gradients[X]
        grad_W = engine.gradients[W]
        grad_b = engine.gradients[b]

        self.assertIsInstance(grad_X, Graph)
        self.assertIsInstance(grad_W, Graph)
        self.assertIsInstance(grad_b, Graph)

        Engine(grad_X).forwardPass()
        Engine(grad_W).forwardPass()
        Engine(grad_b).forwardPass()

        np.testing.assert_array_equal(engine.getGradValue(X), np.array([[2.0, 2.0], [2.0, 2.0]]))
        np.testing.assert_array_equal(engine.getGradValue(W), np.array([[4.0, 4.0], [6.0, 6.0]]))
        np.testing.assert_array_equal(engine.getGradValue(b), np.ones((2, 2)))


if __name__ == "__main__":
    unittest.main()
