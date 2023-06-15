import random
import time
from random import randrange, uniform, randint
import numpy as np
import networkx as nx
from matplotlib import pyplot as plt
from tqdm import tqdm

from graphs import Graph


class DNA:
    """
        Класс, представляющий решение задачи нахождения наибольшего паросочетания.
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

    def evaluate_fitness(self, graph):
        fitness = 0
        for index in range(len(self.genes)):
            if self.genes[index] == 1:
                for j in range(index + 1, len(self.genes)):
                    if self.genes[j] == 1 and graph.edge_adj_matrix[index][j] == 1:
                        self.fitness = self.MAX_VALUE
                        return
            else:
                fitness += 1
        self.fitness = fitness

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

    def __repr__(self):
        return "-".join(map(str, self.genes))

    def __eq__(self, other):
        if isinstance(other, DNA):
            return self.genes == other.genes
        return False

    def __hash__(self):
        return hash(tuple(self.genes))


class Record:
    """
        Класс, представляющий собой информацию о поколении
    """

    def __init__(self,
                 number_of_generations: int,
                 fittest: list[DNA],
                 best_fitness: int,
                 avg_fitness: float,
                 best_edges: list):
        self.number_of_generations = number_of_generations
        self.fittest = fittest
        self.best_fitness = best_fitness
        self.avg_fitness = avg_fitness
        self.best_edges = best_edges

    def __str__(self):
        return str(f"Номер поколения: {self.number_of_generations},\n"
                   + f"Наилучшие особи: {self.fittest},\n"
                   + f"Наилучшая приспособленность: {self.best_fitness} \n"
                   + f"Средняя приспособленность поколения: {self.avg_fitness}\n"
                   + f"Наибольшее паросочетание: {self.best_edges}\n")


class Launcher:
    """
        Класс для запуска и настройки генетического и иммунного алгоритма
    """
    UPPER_BOUND = 1000

    def __init__(self, graph: Graph):
        self._generation_count = None
        self._graph = None
        self._population = []

        self._avg_fitness = None
        self._best_fitness = None
        self._fittest = None

        self.graph = graph
        self.records = []

        self.population_size = 1000  # randrange(200, 500)
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
    def graph(self):
        return self._graph

    @graph.setter
    def graph(self, value):
        if value is not None:
            self._graph = value

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
            self.records.append(
                Record(
                    self.generation_count,
                    self.fittest,
                    self.best_fitness,
                    self.avg_fitness,
                    [self.graph.to_edges(dna) for dna in self.fittest]
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
            self._population.append(DNA(self.graph.number_of_edges))
        for dna in self._population:
            dna.evaluate_fitness(self.graph)
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
        if check <= self.mutation_rate:
            for position in range(len(dna.genes)):
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
        population_without_gen_operators = self.roulette_wheel_selection()
        new_population = []
        #random.shuffle(population_without_gen_operators)

        for i in range(0, self.population_size - 1, 2):
            child1, child2 = self.crossover(population_without_gen_operators[i],
                                            population_without_gen_operators[i + 1])
            new_population.append(child1)
            new_population.append(child2)
        if len(new_population) < self.population_size:
            new_population.append(population_without_gen_operators[-1])

        self._population = new_population

        for dna in self._population:
            dna = self.mutation2(dna)
            dna.evaluate_fitness(self.graph)

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
            min_avg_value = min([decision.avg_fitness for decision in bests_decisions])
            bests_decisions = [decision for decision in bests_decisions if decision.avg_fitness == min_avg_value]
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
                file.write(
                    f"Номер поколения: {record.number_of_generationsgeneration_count},\n"
                    f"Наилучшие особи: {record.fittest},\n"
                    f"Наибольшее паросочетание: {[record.graph.to_edges(variant) for variant in self.fittest]}\n"
                    f"Наилучшая приспособленность: {record.best_fitness} \n"
                    f"Средняя приспособленность поколения: {record.avg_fitness}\n")
            file.write("*" * 50 + "\n")

    def start_darvin(self):
        """
        Функция старта генетического алгоритма по модели Дарвина
        :return: None
        """
        self.initialize_population()

        while self.best_fitness != (self.graph.number_of_edges - self.graph.number_of_vertices / 2) \
                and self.generation_count != self.UPPER_BOUND:
            self.next_generation_darvin()

    def start_immune(self):
        """
         Функция старта иммунного алгоритма
         :return: None
         """
        self.initialize_population()

        while self.best_fitness != (self.graph.number_of_edges - self.graph.number_of_vertices / 2) \
                and self.generation_count != self.UPPER_BOUND:
            self.next_generation_immune()

    def next_generation_immune(self):
        """
        Функция создания следующего поколения для иммунного алгоритма.
        :return: None
        """
        numb = round(self.cloning_rate * self.population_size)
        random.shuffle(self.population)

        clones = self.population[:numb]
        for clone in clones:
            clone = self.mutation2(clone)
            clone.evaluate_fitness(self.graph)

        new_population = []
        for clone, parent in zip(clones, self.population[:numb]):
            if clone.fitness <= parent.fitness:
                new_population.append(clone)
            else:
                new_population.append(parent)

        for i in self.population[numb:]:
            new_population.append(i)

        self._population = new_population

        self.generation_count += 1
        self.avg_fitness = self.calculate_avg_fitness()
        self.fittest, self.best_fitness = self.get_fittests()
        self.record_statistic()

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
        Функция рисующая граф и наибольшее паросочетание
        :param matching: Ребра паросочетания
        :param title: Название
        :return:
        """
        G = nx.Graph()
        G.add_edges_from(self.graph.edges)
        plt.clf()
        edge_color_list = ["grey"] * len(G.edges)
        for i, edge in enumerate(G.edges()):
            if edge in matching or (edge[1], edge[0]) in matching:
                edge_color_list[i] = 'red'
        nx.draw_circular(G, with_labels=True, edge_color=edge_color_list)
        plt.title(title)
        plt.savefig(path)


def save(path: str):
    pass


def main():
    path_to_graphics = "resources/graphics/"
    path_to_graphs = "resources/graphs/"
    ga_title = "Генетический алгоритм"
    immune_title = "Иммунный алгоритм"
    ga_time = []
    immune_time = []
    for number_of_vertex in tqdm(range(20, 21)):
        graph = Graph(n=number_of_vertex)
        launcher = Launcher(graph=graph)

        # измерение времени работы генетического алгоритма
        start = time.perf_counter()
        launcher.start_darvin()
        end = time.perf_counter()
        ga_time.append(end - start)
        # отображение графика и сохранение
        launcher.save_graphic(ga_title,
                              path_to_graphics + "ga/max_avg_graphic_" + "vertex_" + str(number_of_vertex) + ".jpg")

        # Находим все лучшие решения и отображаем и сохраняем
        bests = launcher.get_best_info()
        if bests:
            for best in bests:
                for index, pair_edges in enumerate(best.best_edges):
                    launcher.save_matching_graph(ga_title, pair_edges,
                                                 path_to_graphs + f"ga/vertex_" + str(
                                                     number_of_vertex) + f"_matching_{index}_" + ".jpg")
        else:
            print("NOT FOUND GEN " + str(number_of_vertex))

        # измерение времени работы иммунного алгоритма
        start = time.perf_counter()
        launcher.start_immune()
        end = time.perf_counter()
        immune_time.append(end - start)

        launcher.save_graphic(immune_title,
                              path_to_graphics + "immune/max_avg_graphic_" + "vertex_" + str(number_of_vertex) + ".jpg")

        # Находим все лучшие решения и отображаем и сохраняем
        bests = launcher.get_best_info()
        if bests:
            for best in bests:
                for index, pair_edges in enumerate(best.best_edges):
                    launcher.save_matching_graph(immune_title, pair_edges,
                                                 path_to_graphs + f"immune/vertex_" + str(
                                                     number_of_vertex) + f"_matching_{index}_" + ".jpg")
        else:
            print("NOT FOUND IMMUNE " + str(number_of_vertex))

    plt.clf()
    plt.title("График зависимости скорости работы от количества вершин")
    plt.plot(ga_time, color="red", label="GA")
    plt.plot(immune_time, color="blue", label="Immune")
    plt.xlabel("Количество вершин")
    plt.ylabel("Время решения (с)")
    plt.legend(loc='lower left')
    plt.savefig(path_to_graphics + "time_graphic.jpg")


if __name__ == '__main__':
    main()
