class Graph:

    def __init__(self, matrix: list[list[int]] = None):
        self._edges = None
        self._vertices = None
        self._matrix = None
        self._edges_dict = None
        self.set_values(matrix)

    def set_values(self, matrix):
        self._vertices = []
        self._edges = []
        self._edges_dict = {}
        self._matrix = matrix

        for v, values in enumerate(matrix):
            if v not in self._vertices:
                self._vertices.append(v)
            for u, value in enumerate(values):
                if value == 1 and [u, v] not in self._edges:
                    self._edges.append([v, u])

        self._edges_dict = dict(zip([i for i in range(len(self.edges))], self.edges))

    def recreate(self, matrix):
        self.set_values(matrix)

    def check_independence(self, probable):
        if ([u, v] in self.edges
                or [v, u] in self.edges):
            return True
        return False

    def count_independent_edges(self, edges):
        count = 0
        tmp = set()
        for edge in edges:
            if self.is_edge(*edge) and set(edge) not in tmp:
                tmp.add(*edge)
                count += 1
        return count

    def to_gf(self):
        pass

    @property
    def vertices(self):
        return self._vertices

    @property
    def edges(self):
        return self._edges

    @property
    def edges_dict(self):
        return self._edges_dict

    @property
    def matrix(self):
        return self._matrix

    @property
    def number_of_edges(self):
        return len(self._edges)

    @property
    def number_of_vertices(self):
        return len(self._vertices)


if __name__ == '__main__':
    mp = [
        [0, 1, 1, 1, 0],
        [1, 0, 1, 0, 1],
        [1, 1, 0, 1, 0],
        [1, 0, 1, 0, 1],
        [0, 1, 0, 1, 0],
    ]

    mp2 = [
        [0, 1, 1, 1, 0],
        [1, 0, 1, 0, 1],
        [1, 1, 0, 0, 0],
        [1, 0, 0, 0, 0],
        [0, 1, 0, 0, 0],
    ]

    g = Graph(matrix=mp)

    print(g.vertices)
    print(g.edges)
    print(g.edges_dict)

    g.recreate(mp2)

    print(g.vertices)
    print(g.edges)
