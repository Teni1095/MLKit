from graph import Graph

class Activations:

    @staticmethod
    def relu(graph):
        return graph.max(Graph(0))

    @staticmethod
    def sigmoid(graph):
        return Graph(1) / (Graph(1) + graph.neg().exp())

    @staticmethod
    def tanh(graph):
        exp_pos = graph.exp()
        exp_neg = graph.neg().exp()
        return (exp_pos - exp_neg) / (exp_pos + exp_neg)

    @staticmethod
    def softmax(graph):
        exp_x = graph.exp()
        return exp_x / exp_x.sum()
