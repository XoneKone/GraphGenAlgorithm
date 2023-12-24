import random
import random as rd
import numpy as np
import itertools as it
import copy
from math import factorial
from matplotlib import pyplot as plt
import hypernetx as hnx


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
    def edges(self) -> list[list[int]]:
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

    def __init__(self, n: int, k: int = None, vertices=None, edges=None):
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
        self.create_random_graph(n, k)

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

    def create_random_graph(self, n: int, k: int = None) -> None:
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

        if k is None:
            number_of_edges = rd.randint(3, self.max_number_edges)
        else:
            number_of_edges = k

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

    def find_all_intersections(self) -> list[list[int]]:
        """
        Метод для выявления всех пересечений
        :return: список уникальных пересечений
        """
        intersection_list = []
        for i in range(len(self.edges) - 1):
            for j in range(i + 1, len(self.edges)):
                possible_intersection = self.intersection(i, j)
                if possible_intersection and possible_intersection not in intersection_list:
                    intersection_list.append(possible_intersection)
        return intersection_list

    def remove_vertices(self, possible_articulation_point: list[int]) -> (list[list[int]], list[int]):
        """
        Метод, который удаляет возможную точку сочленения
        :param possible_articulation_point: возможная точка сочленения
        :return: список ребер и список вершин без точек сочленения
        """
        edges = copy.deepcopy(self.edges)
        vertices = list(self.vertices_set)
        for vertex in possible_articulation_point:
            for edge in edges:
                if vertex in edge:
                    edge.remove(vertex)
            if vertex in vertices:
                vertices.remove(vertex)

        return edges, vertices

    def find_articulation_points(self) -> list[list[int]]:
        """
        Метод, который ищет все точки сочленеия
        :return: список множеств точек сочленения
        """

        intersection_list = self.find_all_intersections()
        articulation_points = []
        if not self.check_articulation_point([]):
            for possible_articulation_point in intersection_list:
                if self.check_articulation_point(possible_articulation_point):
                    articulation_points.append(possible_articulation_point)
            return articulation_points

    def check_articulation_point(self, possible_articulation_point: list[int]) -> bool:
        """
        Метод, проверяющий возможную точку сочленения на то, что это действительно точка сочленения
        :param possible_articulation_point: возможная точка сочленения (список вершин)
        :return: True либо False
        """
        edges_without_possible_artic_point, vertices_without_artic_points = self.remove_vertices(
            possible_articulation_point)

        visited = {v: False for v in vertices_without_artic_points}

        edge_dict = dict(
            zip([i for i in range(len(edges_without_possible_artic_point))], edges_without_possible_artic_point))
        adj_matrix = self.create_adj_matrix(vertices_without_artic_points, edge_dict)
        for v in visited.keys():
            visited = {v: False for v in vertices_without_artic_points}

            self.dfs(v, visited, v, adj_matrix, edge_dict)
            if not all(visited.values()):
                return True

        return False

    def dfs(self, vertex: int, visited: dict, parent: int, adj_matrix: dict, edge_dict: dict) -> None:
        """
        Метод поиска в глубину, отмечает возможные пути из заданной вершины
        :param vertex: заданная вершина, с которой начинается поиск
        :param visited: словарь вершин, которые уже посетили
        :param parent: вершина-родитель
        :param adj_matrix: матрица инцидентности
        :param edge_dict: словарь ребер
        :return:
        """
        visited[vertex] = True

        for adj_edge in adj_matrix[vertex]:
            for neighbor in edge_dict[adj_edge]:
                if not visited[neighbor] and neighbor != parent:
                    self.dfs(neighbor, visited, vertex, adj_matrix, edge_dict)

    def create_adj_matrix(self, vertices: list, edges_dict: dict) -> dict:
        """
        Метод создающий из списка вершин и словаря ребер матрицу инцидентности, где строки - это вершины, а столбцы - это ребра
        :param vertices: список вершин
        :param edges_dict: словарь ребер
        :return: словарь, где ключ - это вершина, а значение - это список инцидентных ребер
        """
        adj_matrix = {i: [] for i in vertices}
        for i in vertices:
            for j in edges_dict.keys():
                if i in edges_dict[j]:
                    adj_matrix[i].append(j)
        return adj_matrix

    def check_all_nodes(self, genes: list[int]) -> bool:
        """
        Метод проверяющий, что сочетание - совершенное
        :param genes: бинарная последовательность
        :return: True или False
        """
        edges = set()
        for index, gen in enumerate(genes):
            if gen == 1:
                edges |= set(self.edges[index])

        if len(self.vertices_set - edges) == 0:
            return True
        return False

    def to_edges_dict(self, best_edges: list[tuple[int]]):
        matching_dict = []
        for matching in best_edges:
            tmp = []
            for edge in matching:
                tmp.append([key for key in self.edges_dict if tuple(self.edges_dict[key]) == edge].pop())
            matching_dict.append([tuple(tmp)])
        return matching_dict

    def bound(self):
        return int(factorial(self.number_of_vertices // self.MAX_VERTEX_SET_COUNT) ** (self.MAX_VERTEX_SET_COUNT - 1))

    def save_art_points_graph(self, title: str, art_point: list, path: str):
        """
        Функция рисующая граф и точки сочленения
        :param path: путь до файла
        :param art_point: Ребра паросочетания
        :param title: Название
        :return:
        """
        edges = self.edges_dict
        plt.clf()
        H = hnx.Hypergraph(edges)
        colors = [
            'red' if int(vertex) in art_point else 'black' for vertex in list(H.nodes)
        ]
        hnx.draw(H,
                 with_edge_labels=False,
                 edges_kwargs={
                     'linewidths': 2
                 },
                 nodes_kwargs={
                     'facecolors': colors
                 },
                 node_labels_kwargs={
                     'fontsize': 24
                 },
                 layout_kwargs={
                     'seed': 39
                 })
        plt.title(title)
        plt.savefig(path)

    def show_hypergraph(self, title: str):
        """
        Функция рисующая граф и точки сочленения
        :param path: путь до файла
        :param art_point: Ребра паросочетания
        :param title: Название
        :return:
        """
        edges = self.edges_dict
        plt.clf()
        H = hnx.Hypergraph(edges)
        hnx.draw(H,
                 with_edge_labels=False,
                 edges_kwargs={
                     'linewidths': 2
                 },
                 node_labels_kwargs={
                     'fontsize': 24
                 },
                 layout_kwargs={
                     'seed': 39
                 })
        plt.title(title)
        plt.show()


if __name__ == '__main__':
    hg = HyperGraph(9, 5)
    print(f"Количество ребер: {hg.number_of_edges}")
    print(f"Ребра: {hg.edges}")
    print(f"Вершины (разбитые на доли): {hg.vertices}")
    print(f"Словарь ребер: {hg.edges_dict}")
    print(f"Все возможные пересечения ребер: {hg.find_all_intersections()}")
    art_points = hg.find_articulation_points()
    print(f"Точки сочленения: {art_points}")

    path_to_graphs = "../resources/graphs/"
    hg.show_hypergraph("Гиперграф")
    if art_points:
        for index, art_point in enumerate(art_points):
            ptgp = path_to_graphs + f"hypergraph/vertex_" + str(
                hg.number_of_vertices) + f"_atr_points_{index}_" + ".jpg"
            hg.save_art_points_graph(title="Точки сочленения",
                                     art_point=art_point,
                                     path=ptgp)
    else:
        print("Гиперграф несвязный")

