class Node:
    def __init__(self, data):
        self._data = data
        self.computedBy = None
        self.computedRound = None

    @property
    def data(self):
        return self._data

    @data.setter
    def data(self, value):
        self._data = value
