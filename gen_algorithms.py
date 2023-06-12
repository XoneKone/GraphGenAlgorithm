import random
from random import randrange, uniform, randint
import numpy as np
import networkx as nx
from matplotlib import animation
from matplotlib import pyplot as plt

from graphs import Graph


class DNA:
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
            self._genes = [randint(0, 1) for _ in range(self.length)]

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


class RecordGA:
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


class GA:

    def __init__(self, graph: Graph):
        self._generation_count = None
        self._graph = None
        self._population = []

        self._avg_fitness = None
        self._best_fitness = None
        self._fittest = None

        self.graph = graph
        self.records = []

        self.population_size = 800 # randrange(200, 500)
        self.crossing_over_rate = 0.9  # uniform(0.8, 0.95)
        self.mutation_rate = 0.9  # uniform(0.5, 1.0)

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
        if self.best_fitness != DNA.MAX_VALUE:
            self.records.append(
                RecordGA(
                    self.generation_count,
                    self.fittest,
                    self.best_fitness,
                    self.avg_fitness,
                    [self.graph.to_edges(dna) for dna in self.fittest]
                )
            )

    def initialize_population(self):
        for _ in range(self.population_size):
            self._population.append(DNA(self.graph.number_of_edges))
        for dna in self._population:
            dna.evaluate_fitness(self.graph)
        self.generation_count = 1
        self.fittest, self.best_fitness = self.get_fittests()
        self.avg_fitness = self.calculate_avg_fitness()
        self.record_statistic()

    def crossover(self, parent1: DNA, parent2: DNA):
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
        check = uniform(0, 1)
        if check <= self.mutation_rate:
            for position in range(len(dna.genes)):
                if dna.genes[position] == 0:
                    dna.genes[position] = 1
                else:
                    dna.genes[position] = 0
        return dna

    def mutation2(self, dna):
        check = uniform(0, 1)
        position = randrange(0, dna.length)
        if check <= self.mutation_rate:
            if dna.genes[position] == 0:
                dna.genes[position] = 1
            else:
                dna.genes[position] = 0
        return dna

    def roulette_wheel_selection(self):
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
        population_without_gen_operators = self.roulette_wheel_selection()
        new_population = []
        random.shuffle(population_without_gen_operators)

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
        return sum(
            [dna.fitness for dna in self.population if dna.fitness != DNA.MAX_VALUE]) / self.population_size

    def get_fittests(self):
        min_fitness = min([ind.fitness for ind in self.population])
        fittests = set([ind for ind in self.population if ind.fitness == min_fitness])
        return list(fittests), min_fitness

    def get_best_info(self):
        if self.records:
            min_value = min([record.best_fitness for record in self.records])
            bests_decisions = [record for record in self.records if record.best_fitness == min_value]
            min_avg_value = min([decision.avg_fitness for decision in bests_decisions])
            bests_decisions = [decision for decision in bests_decisions if decision.avg_fitness == min_avg_value]
            return bests_decisions
        else:
            return None

    def to_file(self, path):
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
        self.initialize_population()

        while self.best_fitness != 0 and self.generation_count != 1001:
            self.next_generation_darvin()

    def show_graphics(self):
        max_fitness = [record.best_fitness for record in self.records]
        avg_fitness = [record.avg_fitness for record in self.records]
        plt.plot(max_fitness, color="red")
        plt.plot(avg_fitness, color="green")
        plt.xlabel("Поколение")
        plt.ylabel("Макс/средняя приспособленность")
        plt.title('Зависимость максимальной и средней приспособленности от поколения')
        plt.show()

    def show_graph(self):
        bests = self.get_best_info()
        if bests is None:
            return "Решение не найдено!"
        G = nx.Graph()
        G.add_edges_from(self.graph.edges)
        pair_edges = bests[0].best_edges[0]
        edge_color_list = ["grey"] * len(G.edges)
        for i, edge in enumerate(G.edges()):
            if edge in pair_edges or (edge[1], edge[0]) in pair_edges:
                edge_color_list[i] = 'red'
        nx.draw(G, with_labels=True, edge_color=edge_color_list)
        plt.show()

    def simple_update(self, num, layout, G, ax, bests):
        ax.clear()
        # Draw the graph with random node colors
        edge_color_list = ["grey"] * len(G.edges)
        for best in bests:
            pair_edges = best.best_edges
            for i, edge in enumerate(G.edges()):
                if edge in pair_edges or (edge[1], edge[0]) in pair_edges:
                    edge_color_list[i] = 'red'
            nx.draw(G, pos=layout, with_labels=True, edge_color=edge_color_list, ax=ax)

        # Set the title
        ax.set_title("Frame {}".format(num))

    def simple_animation(self):
        # Build plot
        fig, ax = plt.subplots(figsize=(6, 4))

        # Create a graph and layout
        G = nx.Graph()
        G.add_edges_from(self.graph.edges)
        bests = self.get_best_info()
        if bests is None:
            return "Решение не найдено!"
        layout = nx.spring_layout(G)

        ani = animation.FuncAnimation(fig, self.simple_update, frames=20,
                                      fargs=(layout, G, ax, bests))
        ani.save('animation_2.gif', writer='pillow')

        plt.show()


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
    mp3 = [
        [0, 0, 0, 0, 1, 1, 0, 1, 1],
        [0, 0, 0, 0, 1, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 1, 1, 1, 0],
        [0, 0, 0, 0, 0, 1, 0, 1, 0],
        [1, 1, 0, 0, 0, 0, 0, 0, 0],
        [1, 0, 1, 1, 0, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 0, 0, 0, 0],
        [1, 0, 1, 1, 0, 0, 0, 0, 0],
        [1, 0, 0, 0, 0, 0, 0, 0, 0],
    ]
    # {0: [0, 1], 1: [0, 2], 2: [0, 3], 3: [1, 2], 4: [1, 4], 5: [2, 3], 6: [3, 4]}
    g = Graph(matrix=mp3)
    g2 = Graph(n=13)
    ga = GA(graph=g2)
    ga.start_darvin()
    ga.show_graphics()
    bests = ga.get_best_info()
    for i in bests:
        print(i)
    ga.show_graph()
