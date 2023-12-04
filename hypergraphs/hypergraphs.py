import random
import random as rd
import numpy as np
import itertools as it


class HyperGraph:
    """
    Класс Гиперграфа
    """
    MAX_VERTEX_SET_COUNT = 3  # 3-дольный
    MAX_EDGE_SET_COUNT = 3  # 3-однородный

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

    @edges.setter
    def edges(self, value):
        self._edges = value

    @property
    def edges_dict(self):
        return self._edges_dict

    @edges_dict.setter
    def edges_dict(self, value):
        self._edges_dict = value

    @property
    def number_of_edges(self):
        return len(self._edges)

    @property
    def number_of_vertices(self):
        return len(self.vertices_set)

    def __init__(self, n: int, vertices=None, edges=None):
        """
        Инициализация рандомного 3-дольного и 3-однородного гиперграфа
        :param n:
        """
        self.vertices_set = None
        if edges is None:
            edges = []
        if vertices is None:
            vertices = []

        self.max_number_edges = 0
        self.edges = edges
        self.vertices = vertices
        self.edges_dict = {}
        self.create_random_graph(n)

    def recreate(self, n) -> None:
        """
        Метод пересоздания гиперграфа
        :param n: количество вершин
        :return: создает новый граф (объект остается тем же)
        """
        self._max_number_edges = 0
        self._edges = []
        self._vertices = []
        self._edges_dict = {}
        self.create_random_graph(n)

    def to_edges(self, dna):
        edges = []
        for index, gen in enumerate(dna.genes):
            if gen == 1:
                edges.append(tuple(self.edges[index]))
        return edges

    def create_random_graph(self, n: int) -> None:
        """
        Метод для создания рандомного 3-дольного и 3-однородного гипреграфа
        :param n: количество вершин кратное 3
        :return: None
        """
        v = [i for i in range(n)]
        # random.shuffle(v) # не нужно скорее всего

        self.vertices = list(map(list, np.array_split(v, self.MAX_VERTEX_SET_COUNT)))
        self.vertices_set = set(v)

        self.max_number_edges = int((n // self.MAX_VERTEX_SET_COUNT) ** self.MAX_VERTEX_SET_COUNT)
        number_of_edges = rd.randint(3, self.max_number_edges)

        # ручной выбор ребер
        # current_number_of_edge = 0
        # while current_number_of_edge != number_of_edges:
        #     edge = []
        #     for part in self.vertices:
        #         edge.append(rd.choice(part))
        #     if edge not in self.edges:
        #         self.edges.append(edge)
        #         current_number_of_edge += 1

        edges = list(map(list, it.product(*self.vertices)))
        random.shuffle(edges)
        self.edges = edges[:number_of_edges]
        self.edges_dict = dict(zip([i for i in range(len(self.edges))], self.edges))

    def check_intersection(self, index_edge_1: int, index_edge_2: int) -> bool:
        """
        Метод для проверки пересечений ребер.
        :param index_edge_1: Номер ребра №1.
        :param index_edge_2: Номер ребра №2.
        :return: Возвращает - True, если ребра смежны, False - в противном случае.
        """
        if self.intersection(index_edge_1, index_edge_2):
            return True
        return False

    def intersection(self, index_edge_1: int, index_edge_2: int) -> list:
        """
        Метод для выявления вершин, которые есть в обоих ребрах.
        :param index_edge_1: Ребро №1.
        :param index_edge_2: Ребро №2.
        :return: Возвращает список вершин, которые есть в обоих ребрах.
        """
        return [value for value in self.edges[index_edge_1] if value in set(self.edges[index_edge_2])]

    def check_all_nodes(self, genes):
        edges = set()
        for index, gen in enumerate(genes):
            if gen == 1:
                edges |= set(self.edges[index])

        if len(self.vertices_set - edges) == 0:
            return True
        return False


if __name__ == '__main__':
    hg = HyperGraph(12)
    print(hg.number_of_edges)
    print(hg.edges)
    print(hg.vertices)
    print(hg.edges_dict)
