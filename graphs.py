from random import randint


def check_adj(edge_1, edge_2):
    if (edge_1[0] == edge_2[0] or edge_1[1] == edge_2[0]
            or edge_1[0] == edge_2[1] or edge_1[1] == edge_2[1]):
        return True
    return False


class Graph:

    def __init__(self, matrix: list[list[int]] = None, n: int = None):
        self._edges = None
        self._vertices = None
        self._vertex_adj_matrix = None
        self._edge_adj_matrix = None
        self._edges_dict = None
        if matrix is None:
            self.create_random_graph(n)
        else:
            self.set_values(matrix)

    def set_values(self, vertex_adj_matrix):
        if vertex_adj_matrix is None:
            return None
        self._vertices = []
        self._edges = []
        self._edges_dict = {}
        self._vertex_adj_matrix = vertex_adj_matrix

        for v, values in enumerate(vertex_adj_matrix):
            if v not in self._vertices:
                self._vertices.append(v)
            for u, value in enumerate(values):
                if value == 1 and (u, v) not in self._edges:
                    self._edges.append((v, u))

        self._edges_dict = dict(zip([i for i in range(len(self.edges))], self.edges))
        self._edge_adj_matrix = self.create_edge_adj_matrix(self.number_of_edges)

    def recreate(self, matrix):
        self.set_values(matrix)

    def create_edge_adj_matrix(self, number_of_edges):
        adj = [[0] * number_of_edges for _ in range(number_of_edges)]

        for i in range(number_of_edges):
            for j in range(i, number_of_edges):
                if check_adj(self.edges_dict[i], self.edges_dict[j]) and i != j:
                    adj[i][j] = adj[j][i] = 1
        return adj

    def to_edges(self, dna):
        edges = []
        for index, gen in enumerate(dna.genes):
            if gen == 1:
                edges.append(tuple(self._edges_dict[index]))
        return edges

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
    def vertex_adj_matrix(self):
        return self._vertex_adj_matrix

    @property
    def edge_adj_matrix(self):
        return self._edge_adj_matrix

    @property
    def number_of_edges(self):
        return len(self._edges)

    @property
    def number_of_vertices(self):
        return len(self._vertices)

    def create_random_graph(self, n: int):
        graph = []
        for i in range(n):
            vertex = []
            for j in range(n):
                vertex.append(randint(0, 1))
            graph.append(vertex)
        for i in range(n):
            for j in range(0, i):
                graph[i][j] = graph[j][i]
        for i in range(n):
            graph[i][i] = 0

        self.set_values(graph)