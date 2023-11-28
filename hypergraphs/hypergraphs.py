import random
import random as rd


def check_adj(edge_1, edge_2):
    if (edge_1[0] == edge_2[0] or edge_1[1] == edge_2[0]
            or edge_1[0] == edge_2[1] or edge_1[1] == edge_2[1]):
        return True
    return False


class HyperGraph:
    def __init__(self, n: int):
        self._max_number_edges = 0
        self._edges = []
        self._vertices = []
        self._edges_dict = {}
        self.create_random_graph(n)

    def recreate(self, n):
        self.create_random_graph(n)

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
    def max_number_edges(self):
        return self._max_number_edges

    @max_number_edges.setter
    def max_number_edges(self, value):
        self._max_number_edges = value

    @vertices.setter
    def vertices(self, value):
        self._vertices = value

    @property
    def edges(self):
        return self._edges

    @property
    def edges_dict(self):
        return self._edges_dict

    @property
    def number_of_edges(self):
        return len(self._edges)

    @property
    def number_of_vertices(self):
        return len(self._vertices)

    def create_random_graph(self, n: int):
        v = [i for i in range(n)]
        random.shuffle(v)
        for _ in range(3):

        self.vertices = [v[x:x + 3] for x in range(4)]
        self.max_number_edges = (n / 3) ** 3
        number_of_edges = rd.randint(2, self.max_number_edges)

        for _ in range(number_of_edges):
            edge = []
            for part in self.vertices:
                edge.append(rd.choice(part))
            self.edges.append(edge)


if __name__ == '__main__':
    hg = HyperGraph(9)
