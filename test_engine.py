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

        np.testing.assert_array_equal(engine.getGradValue(x), np.array([[2.0, 2.0]]))
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
        W = Graph(np.array([[1.0, 2.0],[3.0, 4.0]]))
        b = Graph(np.array([[1.0, 1.0], [1.0, 1.0]]))

        out = X @ W + b
        root = out.sum()

        engine = Engine(root)
        engine.forwardPass()
        engine.backwardPass()

        self.assertIn(X, engine.gradients)
        self.assertIn(W, engine.gradients)
        self.assertIn(b, engine.gradients)




        np.testing.assert_array_equal(engine.getGradValue(X), np.array([[3.0, 7.0], [3.0, 7.0]]))
        np.testing.assert_array_equal(engine.getGradValue(W), np.array([[4.0, 4.0], [6.0, 6.0]]))
        np.testing.assert_array_equal(engine.getGradValue(b), np.ones((2, 2)))

    def test_backward_gradient_accumulation_multiple_children(self):
        X = Graph(np.array([[1.0, 2.0],
                            [3.0, 4.0]]))

        W = Graph(np.array([[1.0, 0.0],
                            [0.0, 1.0]]))

        # z is used twice downstream
        z = X @ W

        out = z + z

        root = out.sum()

        engine = Engine(root)
        engine.forwardPass()
        engine.backwardPass()

        self.assertIn(X, engine.gradients)
        self.assertIn(W, engine.gradients)
        self.assertIn(z, engine.gradients)

        # out = z + z
        # d(out)/d(z) = 1 + 1 = 2
        np.testing.assert_array_equal(
            engine.getGradValue(z),
            np.array([[2.0, 2.0],
                    [2.0, 2.0]])
        )

        # z = X @ W
        # dz/dX = W.T
        # gradient arriving at z is all 2s
        #
        # dX = dz @ W.T
        # W is identity, so dX = dz
        np.testing.assert_array_equal(
            engine.getGradValue(X),
            np.array([[2.0, 2.0],
                    [2.0, 2.0]])
        )

        # dW = X.T @ dz
        # X.T = [[1,3],[2,4]]
        # dz  = [[2,2],[2,2]]
        #
        # result:
        # [[8,8],
        #  [12,12]]
        np.testing.assert_array_equal(
            engine.getGradValue(W),
            np.array([[8.0, 8.0],
                    [12.0, 12.0]])
        )

    def test_backward_complex_shared_subgraph(self):
        X = Graph(np.array([[1.0, 2.0],
                            [3.0, 4.0]]))

        W1 = Graph(np.array([[1.0, 2.0],
                            [3.0, 4.0]]))

        W2 = Graph(np.array([[2.0, 1.0],
                            [1.0, 2.0]]))

        h = X @ W1

        a = h @ W2
        b = h + h

        out = a + b + h

        root = out.sum()

        engine = Engine(root)
        engine.forwardPass()
        engine.backwardPass()

        self.assertIn(X, engine.gradients)
        self.assertIn(W1, engine.gradients)
        self.assertIn(W2, engine.gradients)

        # Main thing: no crash, no cycles, all gradients exist
        self.assertIsNotNone(engine.getGradValue(X))
        self.assertIsNotNone(engine.getGradValue(W1))
        self.assertIsNotNone(engine.getGradValue(W2))

    def test_backward_deep_chain(self):
        X = Graph(np.array([[2.0, 3.0]]))

        W1 = Graph(np.array([[2.0],
                            [4.0]]))

        W2 = Graph(np.array([[5.0]]))

        z1 = X @ W1
        z2 = z1 @ W2
        z3 = z2 + z2

        root = z3.sum()

        engine = Engine(root)
        engine.forwardPass()
        engine.backwardPass()

        np.testing.assert_array_equal(
            engine.getGradValue(z3),
            np.array([[1.0]])
        )

        np.testing.assert_array_equal(
            engine.getGradValue(z2),
            np.array([[2.0]])
        )

        np.testing.assert_array_equal(
            engine.getGradValue(z1),
            np.array([[10.0]])
        )
    def test_backward_with_reshape_and_manual_broadcast(self):
        X = Graph(np.array([[1.0, 2.0],
                            [3.0, 4.0]]))

        W = Graph(np.array([[1.0, 2.0],
                            [3.0, 4.0]]))

        b = Graph(np.array([[1.0, 2.0]]))

        # (2,2)
        z = X @ W

        # Manually create broadcast view of b
        # b: (1,2) -> (2,2)
        b.transforms.append({
            "transform": Transforms.BROADCAST_TO,
            "original_shape": (1, 2),
            "params": {
                "shape": (2, 2)
            }
        })

        z = z + b

        # Reshape view
        # (2,2) -> (4,1)
        z.transforms.append({
            "transform": Transforms.RESHAPE,
            "original_shape": (2,2),
            "params": {
                "shape": (4,1)
            }
        })

        # Operation after reshape
        out = z.sum()

        engine = Engine(out)
        engine.forwardPass()
        engine.backwardPass()

        self.assertIn(X, engine.gradients)
        self.assertIn(W, engine.gradients)
        self.assertIn(b, engine.gradients)

        # Reshape backward:
        # (4,1) gradient -> (2,2)
        np.testing.assert_array_equal(
            engine.getGradValue(z),
            np.ones((2,2))
        )

        # Broadcast backward:
        # collapse rows
        np.testing.assert_array_equal(
            engine.getGradValue(b),
            np.array([[2.0, 2.0]])
        )

    def test_backward_heavy_graph_with_multiple_transforms(self):
        X = Graph(np.array([[1.0, 2.0],
                            [3.0, 4.0]]))

        W1 = Graph(np.array([[1.0, 2.0],
                            [3.0, 4.0]]))

        W2 = Graph(np.array([[2.0, 1.0],
                            [4.0, 3.0]]))

        b = Graph(np.array([[1.0, 2.0]]))

        # First layer
        h = X @ W1

        # Manual broadcast:
        # b (1,2) -> (2,2)
        # b_view = b.view()
        b.transforms.append({
            "transform": Transforms.BROADCAST_TO,
            "original_shape": (1, 2),
            "params": {
                "shape": (2, 2)
            }
        })

        h = h + b

        # Branch 1:
        # reshape then matmul
        h.transforms.append({
            "transform": Transforms.RESHAPE,
            "original_shape": (2,2),
            "params": {
                "shape": (4,1)
            }
        })

        # Bring it back
        h.transforms.append({
            "transform": Transforms.RESHAPE,
            "original_shape": (4,1),
            "params": {
                "shape": (2,2)
            }
        })

        a = h @ W2

        # Branch 2:
        # direct use of h
        c = h * h

        # Merge everything
        out = a + c + h

        root = out.sum()

        engine = Engine(root)
        engine.forwardPass()
        engine.backwardPass()

        self.assertIn(X, engine.gradients)
        self.assertIn(W1, engine.gradients)
        self.assertIn(W2, engine.gradients)
        self.assertIn(b, engine.gradients)

        # Ensure all gradients can be evaluated
        dx = engine.getGradValue(X)
        dw1 = engine.getGradValue(W1)
        dw2 = engine.getGradValue(W2)
        db = engine.getGradValue(b)

        self.assertEqual(dx.shape, X.node.data.shape)
        self.assertEqual(dw1.shape, W1.node.data.shape)
        self.assertEqual(dw2.shape, W2.node.data.shape)
        self.assertEqual(db.shape, b.node.data.shape)

if __name__ == "__main__":
    unittest.main()
