import numpy as np
from graph import Graph
from engine import Engine
from activations import Activations

engine = Engine()

# Basic tests
B, C = Graph(3.0), Graph(4.0)
A = B * C
engine.forwardPass(A)
engine.backwardPass(A)
print(f'Multiplication: A={A.node.data}, dA/dB={B.grad}, dA/dC={C.grad}')

B, C = Graph(3.0), Graph(4.0)
A = B + C
engine.forwardPass(A)
engine.backwardPass(A)
print(f'Addition: A={A.node.data}, dA/dB={B.grad}, dA/dC={C.grad}')

# XOR batch test
np.random.seed(0)
X = np.array([[0,0],[0,1],[1,0],[1,1]], dtype=float)
y = np.array([[0],[1],[1],[0]], dtype=float)

W1 = Graph(np.random.randn(2, 4) * 0.5)
b1 = Graph(np.zeros((1, 4)), collapsible=True)
W2 = Graph(np.random.randn(4, 1) * 0.5)
b2 = Graph(np.zeros((1, 1)), collapsible=True)

lr = 0.5
for epoch in range(10000):
    W1.grad = None
    b1.grad = None
    W2.grad = None
    b2.grad = None

    xi = Graph(X)
    yi = Graph(y)
    z1 = xi @ W1 + b1
    a1 = Activations.sigmoid(z1)
    z2 = a1 @ W2 + b2
    a2 = Activations.sigmoid(z2)
    diff = a2 - yi
    loss = diff * diff

    engine.forwardPass(loss)
    engine.backwardPass(loss)

    W1.node.data -= lr * W1.grad
    b1.node.data -= lr * b1.grad
    W2.node.data -= lr * W2.grad
    b2.node.data -= lr * b2.grad

xi = Graph(X)
z1 = xi @ W1 + b1
a1 = Activations.sigmoid(z1)
z2 = a1 @ W2 + b2
a2 = Activations.sigmoid(z2)
engine.forwardPass(a2)
print('XOR predictions:')
for i in range(4):
    print(f'Input: {X[i]} -> Output: {a2.node.data[i][0]:.4f} (expected {y[i][0]})')
"
