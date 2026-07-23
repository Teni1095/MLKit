class Link:
    def __init__(self, child, parent):
        self.child = child
        self.parent = parent
        self.transforms = []
        self.grad = None
