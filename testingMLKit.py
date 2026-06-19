import unittest
import numpy as np
from node import Node
from graph import Graph
from engine import Engine

class TestEngineArithmetic(unittest.TestCase):
    def setUp(self):
        self.engine = Engine()

    def test_addition(self):
        # Setup: x = 5.0, y = 3.0, z = x + y
        x = Graph(5.0)
        y = Graph(3.0)
        z = x + y

        # 1. Test Forward Pass
        self.engine.forwardPass(z)
        self.assertEqual(z.node.data, 8.0)

        # 2. Test Backward Pass (dz/dx = 1, dz/dy = 1)
        grad_x = self.engine.backwardPass(x)
        grad_y = self.engine.backwardPass(y)
        
        self.assertEqual(grad_x.node.data, 1.0)
        self.assertEqual(grad_y.node.data, 1.0)

    def test_subtraction(self):
        # Setup: x = 10.0, y = 4.0, z = x - y
        x = Graph(10.0)
        y = Graph(4.0)
        z = x - y

        # 1. Test Forward Pass
        self.engine.forwardPass(z)
        self.assertEqual(z.node.data, 6.0)

        # 2. Test Backward Pass (dz/dx = 1, dz/dy = -1)
        grad_x = self.engine.backwardPass(x)
        grad_y = self.engine.backwardPass(y)
        
        self.assertEqual(grad_x.node.data, 1.0)
        self.assertEqual(grad_y.node.data, -1.0)

    def test_multiplication(self):
        # Setup: x = 4.0, y = 3.0, z = x * y
        x = Graph(4.0)
        y = Graph(3.0)
        z = x * y

        # 1. Test Forward Pass
        self.engine.forwardPass(z)
        self.assertEqual(z.node.data, 12.0)

        # 2. Test Backward Pass (dz/dx = y = 3, dz/dy = x = 4)
        grad_x = self.engine.backwardPass(x)
        grad_y = self.engine.backwardPass(y)
        
        self.assertEqual(grad_x.node.data, 3.0)
        self.assertEqual(grad_y.node.data, 4.0)

    def test_division(self):
        # Setup: x = 6.0, y = 2.0, z = x / y
        x = Graph(6.0)
        y = Graph(2.0)
        z = x / y

        # 1. Test Forward Pass
        self.engine.forwardPass(z)
        self.assertEqual(z.node.data, 3.0)

        # 2. Test Backward Pass 
        # dz/dx = 1 / y = 1 / 2 = 0.5
        # dz/dy = -x / y^2 = -6 / 4 = -1.5
        grad_x = self.engine.backwardPass(x)
        grad_y = self.engine.backwardPass(y)
        
        self.assertEqual(grad_x.node.data, 0.5)
        self.assertEqual(grad_y.node.data, -1.5)

    def test_chained_operations(self):
        # Setup: complex expression z = (x * y) + (x / y)
        # For x = 4.0, y = 2.0 -> z = (4 * 2) + (4 / 2) = 8 + 2 = 10
        x = Graph(4.0)
        y = Graph(2.0)
        z = (x * y) + (x / y)

        self.engine.forwardPass(z)
        self.assertEqual(z.node.data, 10.0)

        # Gradients analytical check:
        # dz/dx = y + (1/y) = 2 + 0.5 = 2.5
        # dz/dy = x - (x / y^2) = 4 - (4 / 4) = 3.0
        grad_x = self.engine.backwardPass(x)
        grad_y = self.engine.backwardPass(y)

        self.assertEqual(grad_x.node.data, 2.5)
        self.assertEqual(grad_y.node.data, 3.0)
# Ensure you have 'import numpy as np' at the top of your test file

    def test_matrix_operations(self):
        # 1. Setup raw NumPy data
        # X: Shape (2, 3) -> Matrix of features
        # W: Shape (3, 2) -> Weights matrix
        x_data = np.array([[1.0, 2.0, 3.0], 
                           [4.0, 5.0, 6.0]])
        w_data = np.array([[0.1, 0.2], 
                           [0.3, 0.4], 
                           [0.5, 0.6]])
        
        # Wrap them in your Graph nodes
        X = Graph(x_data)
        W = Graph(w_data)
        
        # 2. Forward Equation: Z = X @ W (Resulting shape: 2, 2)
        Z = X @ W
        
        self.engine.forwardPass(Z)
        
        # Check forward pass against NumPy's native matmul
        expected_Z = x_data @ w_data
        np.testing.assert_array_almost_equal(Z.node.data, expected_Z)

        # 3. Backward Pass Configuration
        # Upstream gradient acting as dLoss/dZ (Shape: 2, 2)
        # We will simulate an upstream incoming gradient of all ones
        upstream_grad_data = np.ones((2, 2))
        upstream_grad = Graph(upstream_grad_data)
        
        # 4. Evaluate your internal gradient engine step
        # Using your upward traversal layout, we extract the gradient of Z with respect to X and W
        # Note: In your current loop setup, ensure 'upstream' starts as Graph(upstream_grad_data)
        
        # Analytical expected math:
        # dLoss/dX = upstream_grad @ W^T  (Shape: 2, 2 @ 2, 3 -> 2, 3)
        # dLoss/dW = X^T @ upstream_grad  (Shape: 3, 2 @ 2, 2 -> 3, 2)
        expected_grad_X = upstream_grad_data @ w_data.T
        expected_grad_W = x_data.T @ upstream_grad_data

        # Execute your engine's upward backward passes
        # (Assuming you temporarily feed the upstream matrix node or configure the stack base)
        grad_X = self.engine.backwardPass(X)
        grad_W = self.engine.backwardPass(W)
        
        # Run forward passes on the resulting gradient structures to populate data
        self.engine.forwardPass(grad_X)
        self.engine.forwardPass(grad_W)

        # Validate structural array accuracy
        np.testing.assert_array_almost_equal(grad_X.node.data, expected_grad_X)
        np.testing.assert_array_almost_equal(grad_W.node.data, expected_grad_W)

if __name__ == '__main__':
    unittest.main()
