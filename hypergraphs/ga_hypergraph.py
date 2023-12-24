import random
import time
from random import randrange, uniform, randint
import numpy as np
import networkx as nx
from matplotlib import pyplot as plt
from tqdm import tqdm
import hypernetx as hnx
from hypergraphs import HyperGraph
from records import Record


class DNA:
    """
        Класс, представляющий решение задачи нахождения совершенных паросочетания.
        Решение представляется двоичной последовательностью длины n - количество ребер в исходной графе.
    """
    MAX_VALUE = 2147483647

    def __init__(self, length: int = None, dna=None):
        self._length = None
        self._genes = None
        self._fitness = 0
        self.length = length
        self.genes = dna

    @property
    def length(self):
        return self._length

    @length.setter
    def length(self, value):
        if value is not None:
            self._length = value

    @property
    def genes(self):
        return self._genes

    @genes.setter
    def genes(self, dna):
        if dna is not None:
            self._genes = dna.genes
        else:
            if self.length < 10:
                self._genes = [randint(0, 1) for _ in range(self.length)]
            else:
                self._genes = [0 for _ in range(self.length)]
                positions = [randrange(0, self.length) for _ in range(randint(1, 3))]
                for position in positions:
                    self._genes[position] = 1

    @property
    def fitness(self):
        return self._fitness

    @fitness.setter
    def fitness(self, value):
        if value is not None:
            self._fitness = value

    def evaluate_fitness(self, hypergraph: HyperGraph):
        fs = 0
        if not hypergraph.check_all_nodes(self.genes):
            self.fitness = self.MAX_VALUE
            return
        for i_index in range(self.length):
            if self.genes[i_index] == 1:
                for j_index in range(i_index + 1, self.length):
                    if self.genes[j_index] == 1 and hypergraph.check_intersection(i_index, j_index):
                        self.fitness = self.MAX_VALUE
                        return
            else:
                fs += 1
        self.fitness = fs

    def __repr__(self):
        return "-".join(map(str, self.genes))

    def __eq__(self, other):
        if isinstance(other, DNA):
            return self.genes == other.genes
        return False

    def __hash__(self):
        return hash(tuple(self.genes))


class Launcher:
    """
        Класс для запуска и настройки генетического и иммунного алгоритма
    """
    UPPER_BOUND = 10_000

    def __init__(self, hypergraph: HyperGraph):
        self._generation_count = None
        self._hypergraph = None
        self._population = []

        self._avg_fitness = None
        self._best_fitness = None
        self._fittest = None

        self.hypergraph = hypergraph
        self.records = []

        self.population_size = 200  # randrange(200, 500)
        self.cloning_rate = 0.2
        self.crossing_over_rate = 0.9  # uniform(0.8, 0.95)
        self.mutation_rate = 0.4  # uniform(0.5, 1.0)

    @property
    def population(self):
        return self._population

    @population.setter
    def population(self, value):
        if value is not None:
            if len(self._population) != 0:
                self._population.clear()
            self._population = value

    @property
    def hypergraph(self):
        return self._hypergraph

    @hypergraph.setter
    def hypergraph(self, value):
        if value is not None:
            self._hypergraph = value

    @property
    def best_fitness(self):
        return self._best_fitness

    @best_fitness.setter
    def best_fitness(self, value):
        if value is not None:
            self._best_fitness = value

    @property
    def avg_fitness(self):
        return self._avg_fitness

    @avg_fitness.setter
    def avg_fitness(self, value):
        if value is not None:
            self._avg_fitness = value

    @property
    def fittest(self):
        return self._fittest

    @fittest.setter
    def fittest(self, value):
        if value is not None:
            self._fittest = value

    @property
    def generation_count(self):
        return self._generation_count

    @generation_count.setter
    def generation_count(self, value):
        if value is not None:
            self._generation_count = value

    def record_statistic(self):
        """
        Записывает информацию о поколении в массив records
        :return: None
        """
        if self.best_fitness != DNA.MAX_VALUE:
            flag = False
            for i in self.records:
                a = [self.hypergraph.to_edges(dna) for dna in self.fittest]
                if i.best_edges == a:
                    flag = True
            if not flag:
                self.records.append(
                    Record(
                        number_of_generations=self.generation_count,
                        fittest=self.fittest,
                        best_fitness=self.best_fitness,
                        avg_fitness=self.avg_fitness,
                        best_edges=[self.hypergraph.to_edges(dna) for dna in self.fittest],
                        hypergraph=self.hypergraph
                    )
                )

    def initialize_population(self):
        """
        Инициализация поколения. Популяция заполняется случайными решениями.
        :return: None
        """
        self._population.clear()
        self.records.clear()
        for _ in range(self.population_size):
            self._population.append(DNA(self.hypergraph.number_of_edges))
        for dna in self._population:
            dna.evaluate_fitness(self.hypergraph)
        self.generation_count = 1
        self.fittest, self.best_fitness = self.get_fittests()
        self.avg_fitness = self.calculate_avg_fitness()
        self.record_statistic()

    def crossover(self, parent1: DNA, parent2: DNA):
        """
        Операция одноточечного кроссовера. Вероятность кроссовера определяется параметром crossing_over_rate.
        :param parent1: Первый родитель
        :param parent2: Второй родитель
        :return: child1 child2 || parent1 parent2
        """
        check = uniform(0, 1)
        if check <= self.crossing_over_rate:
            n = parent1.length
            child1, child2 = DNA(n), DNA(n)
            position = randint(2, n - 2)

            for i in range(position + 1):
                child1.genes[i] = parent1.genes[i]
                child2.genes[i] = parent2.genes[i]

            for i in range(position + 1, n):
                child1.genes[i] = parent2.genes[i]
                child2.genes[i] = parent1.genes[i]

            return child1, child2
        else:
            return parent1, parent2

    def mutation(self, dna):
        """
        Операция мутации. С вероятностью mutation_rate мутируют все гены.
        :param dna: Решение
        :return: mutated_dna
        """
        check = uniform(0, 1)
        position = randrange(1, dna.length)
        if check <= self.mutation_rate:
            if dna.genes[position] == 0:
                dna.genes[position] = 1
            else:
                dna.genes[position] = 0
        return dna

    def mutation2(self, dna):
        """
        Операция многоточечной мутации. С вероятностью mutation_rate мутируют от 1 до n генов (n-длина решения).
        :param dna: Решение
        :return: mutated_dna
        """
        check = uniform(0, 1)
        number_of_mutations = randint(1, dna.length)
        positions = [randrange(0, dna.length) for _ in range(number_of_mutations)]
        if check <= self.mutation_rate:
            for position in positions:
                if dna.genes[position] == 0:
                    dna.genes[position] = 1
                else:
                    dna.genes[position] = 0
        return dna

    def tournament_selection(self):
        """
        Турнирная выборка (селекция). В новом поколении только сильнейшие.
        :return:
        """
        new_population = []
        for j in range(2):
            random.shuffle(self.population)
            for i in range(0, self.population_size - 1, 2):
                if self.population[i].fitness < self.population[i + 1].fitness:
                    new_population.append(self.population[i])
                else:
                    new_population.append(self.population[i + 1])
        return new_population

    def roulette_wheel_selection(self):
        """
        Выборка методом рулетки. Выбор определяется вероятностью.
        :return:
        """
        total_fitness = 0
        for dna in self.population:
            total_fitness += 1 / (1 + dna.fitness)
        cumulative_fitness = []
        cumulative_fitness_sum = 0
        for i in range(len(self.population)):
            cumulative_fitness_sum += 1 / (1 + self.population[i].fitness) / total_fitness
            cumulative_fitness.append(cumulative_fitness_sum)

        new_population = []
        for i in range(len(self.population)):
            roulette = uniform(0, 1)
            for j in range(len(self.population)):
                if roulette <= cumulative_fitness[j]:
                    new_population.append(self.population[j])
                    break
        return new_population

    def next_generation_darvin(self):
        """
        Операция создания следующего поколения в генетическом алгоритме по модели Дарвина.
        :return: None
        """
        population_without_gen_operators = self.tournament_selection()
        new_population = []
        # random.shuffle(population_without_gen_operators)

        for i in range(0, self.population_size - 1, 2):
            child1, child2 = self.crossover(population_without_gen_operators[i],
                                            population_without_gen_operators[i + 1])
            new_population.append(child1)
            new_population.append(child2)
        if len(new_population) < self.population_size:
            new_population.append(population_without_gen_operators[-1])

        self._population.clear()

        for dna in new_population:
            dna.evaluate_fitness(self.hypergraph)
            mutated = self.mutation2(dna)
            mutated.evaluate_fitness(self.hypergraph)
            if dna.fitness < mutated.fitness:
                self._population.append(dna)
            else:
                self._population.append(mutated)

        self.generation_count += 1
        self.avg_fitness = self.calculate_avg_fitness()
        self.fittest, self.best_fitness = self.get_fittests()
        self.record_statistic()

    def calculate_avg_fitness(self):
        """
        Функция считает среднее значение функции приспособленности популяции. Считаются только решения задачи.
        :return: Среднее значение функции приспособленности популяции
        """
        return sum(
            [dna.fitness for dna in self.population if dna.fitness != DNA.MAX_VALUE]) / self.population_size

    def get_fittests(self):
        """
        Находит и возвращает лучшие решения в поколении и наилучшее значение функции приспособленности.
        :return: [Лучшие решения в поколении], Наилучшее значение функции приспособленности
        """
        min_fitness = min([ind.fitness for ind in self.population])
        fittests = set([ind for ind in self.population if ind.fitness == min_fitness])
        return list(fittests), min_fitness

    def get_best_info(self):
        """
        Выводит наилучшие решения за всю работу алгоритма.
        :return:
        """
        if self.records:
            min_value = min([record.best_fitness for record in self.records])
            bests_decisions = [record for record in self.records if record.best_fitness == min_value]
            return bests_decisions
        else:
            return None

    def to_file(self, path):
        """
        Запись в файл информации о всех поколениях и их лучших решениях.
        :param path: Название или путь к файлу
        :return:
        """
        with open(path, 'w', encoding="UTF-8") as file:
            for record in self.records:
                file.write("*" * 100 + '\n' + str(record))
            file.write("*" * 50 + "\n")

    def start_darvin(self):
        """
        Функция старта генетического алгоритма по модели Дарвина
        :return: None
        """
        self.initialize_population()
        count = 0
        bound = self.hypergraph.bound()
        while self.generation_count != self.UPPER_BOUND:
            if self.best_fitness == (self.hypergraph.number_of_edges - (self.hypergraph.number_of_vertices // self.hypergraph.MAX_VERTEX_SET_COUNT)):
                count += 1
            if count == bound:
                break
            self.next_generation_darvin()

    def save_graphic(self, title: str, path: str):
        """
        Функция показывает Зависимость максимальной и средней приспособленности от поколения
        :param path: Путь до файла
        :param title: Название графика
        :return: None
        """
        max_fitness = [record.best_fitness for record in self.records]
        avg_fitness = [record.avg_fitness for record in self.records]
        plt.clf()
        plt.plot(max_fitness, color="red", label="Max")
        plt.plot(avg_fitness, color="green", label="Avg")
        plt.xlabel("Поколение")
        plt.ylabel("Макс/средняя приспособленность")
        plt.title(f'{title}\nЗависимость максимальной и средней приспособленности от поколения')
        plt.legend(loc='lower left')
        plt.savefig(path)

    def save_matching_graph(self, title: str, matching: list, path: str):
        """
        Функция рисующая граф и совершенные паросочетания
        :param matching: Ребра паросочетания
        :param title: Название
        :return:
        """
        edges = self.hypergraph.edges_dict
        plt.clf()
        H = hnx.Hypergraph(edges)
        colors = [
            'red' if tuple(H.incidence_dict[edge]) in matching else 'black' for edge in list(H.edges)
        ]
        hnx.draw(H,
                 with_edge_labels=False,
                 edges_kwargs={
                     'edgecolors': colors,
                     'linewidths': 2
                 },
                 node_labels_kwargs={
                     'fontsize': 24
                 },
                 layout_kwargs={
                     'seed': 39
                 })
        plt.title(title)
        plt.savefig(path)


def main(number_of_vertex, number_of_edges):
    path_to_graphics = "../resources/graphics/"
    path_to_graphs = "../resources/graphs/"
    ga_title = "Генетический алгоритм"
    ga_time = []

    hypergraph = HyperGraph(n=number_of_vertex, k=number_of_edges)
    launcher = Launcher(hypergraph=hypergraph)

    # измерение времени работы генетического алгоритма
    start = time.perf_counter()
    launcher.start_darvin()
    end = time.perf_counter()
    ga_time.append(end - start)

    # отображение графика и сохранение
    ptg = path_to_graphics + "ga_hypergraph/max_avg_graphic_" + "vertex_" + str(number_of_vertex) + ".jpg"
    launcher.save_graphic(title=ga_title,
                          path=ptg)

    # Находим все лучшие решения и отображаем и сохраняем
    bests = launcher.get_best_info()

    if bests:
        for best in tqdm(bests):
            print(best)
            for index, pair_edges in enumerate(best.best_edges):
                ptgp = path_to_graphs + f"hypergraph/vertex_" + str(
                    number_of_vertex) + f"_matching_{index}_" + ".jpg"
                launcher.save_matching_graph(title=ga_title,
                                             matching=pair_edges,
                                             path=ptgp)
    else:
        print("NOT FOUND GEN " + str(number_of_vertex))

    launcher.to_file("../resources/ga_hypergraph.txt")

    print(f"Vertices: {launcher.hypergraph.vertices}")
    print(f"Edges: {launcher.hypergraph.edges_dict}")

    plt.clf()
    plt.title("График зависимости скорости работы от количества вершин")
    plt.plot(ga_time, color="red", label="GA")
    plt.xlabel("Количество вершин")
    plt.ylabel("Время решения (с)")
    plt.legend(loc='lower left')
    plt.savefig(path_to_graphics + "time_graphic.jpg")


if __name__ == '__main__':
    main(15, 15)
