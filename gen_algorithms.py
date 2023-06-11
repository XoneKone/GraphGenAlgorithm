import random
from random import randrange, uniform, randint

from graphs import Graph


class DNA:

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


class GA:
    MAX_VALUE = 2147483647

    def __init__(self, graph: Graph = None):
        self._generation_count = None
        self._graph = None
        self._population = []

        self._avg_fitness = None
        self._best_fitness = None
        self._fittest = None

        self.graph = graph
        self.population_size = randrange(20, 30)
        self.crossing_over_rate = uniform(0.8, 0.95)
        self.mutation_rate = uniform(0.5, 1.0)

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

    def step(self):
        pass

    def initialize_population(self):
        for _ in range(self.population_size):
            self._population.append(DNA(self.graph.number_of_edges))
        for dna in self._population:
            dna.fitness = self.evaluate_fitness(dna)
        self.generation_count = 1
        self.fittest = self.get_fittest()
        self.best_fitness = self.fittest[0].fitness

    def evaluate_fitness(self, dna):
        fitness = 0
        for i in range(len(dna.genes)):
            if dna.genes[i] == 1:
                for j in range(i + 1, len(dna.genes)):
                    if dna.genes[j] == 1 and self.graph.edge_adj_matrix[i][j] == 1:
                        return self.MAX_VALUE
            else:
                fitness += 1
        return fitness

    def crossover(self, parent1: DNA, parent2: DNA):
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

    def mutation(self, dna):
        mutated_dna = DNA(dna.length, dna)
        check = uniform(0, 1)
        if check <= self.mutation_rate:
            position = randint(0, len(mutated_dna.genes) - 1)
            if mutated_dna.genes[position] == 0:
                mutated_dna.genes[position] = 1
            else:
                mutated_dna.genes[position] = 0
        return mutated_dna

    def mutation2(self, dna):
        mutated_dna = DNA(dna.length, dna)
        check = uniform(0, 1)
        position = randrange(0, mutated_dna.length)
        if check <= self.mutation_rate:
            if mutated_dna.genes[position] == 0:
                mutated_dna.genes[position] = 1
            else:
                mutated_dna.genes[position] = 0
        return mutated_dna

    def choose_parents(self):
        parents = self.roulette_wheel_selection()[0:2]
        return parents

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

    def next_generation(self):
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

        self.population.clear()

        for dna in new_population:
            dna = self.mutation2(dna)
            dna.fitness = self.evaluate_fitness(dna)
            self.population.append(dna)

        self.generation_count += 1
        self.avg_fitness = self.calculate_avg_fitness()
        self.fittest = self.get_fittest()
        self.best_fitness = self.fittest[0].fitness

    def calculate_avg_fitness(self):
        return sum([dna.fitness for dna in self.population if dna.fitness != self.MAX_VALUE]) / self.population_size

    def get_fittest(self):
        min_fitness = min([i.fitness for i in self.population])
        fittests = set([i for i in self.population if i.fitness == min_fitness])
        return list(fittests)

    def get_population_info(self):
        return self.generation_count, self.fittest

    def start(self):
        self.initialize_population()
        while self.best_fitness != 0 and self.generation_count != 1001:
            if self.generation_count % 10 == 0:
                print(
                    f"Номер поколения: {self.generation_count},\n"
                    f"Наилучшие особи: {self.fittest},\n"
                    f"Наибольшее паросочетание: {[self.graph.to_edges(variant) for variant in self.fittest]}\n"
                    f"Наилучшая приспособленность: {self.best_fitness} \n"
                    f"Средняя приспособленность поколения: {self.avg_fitness}")
                print("*" * 50 + "\n")
            self.next_generation()
        else:
            print(
                f"Номер поколения: {self.generation_count},\n"
                f"Наилучшие особи: {self.fittest},\n"
                f"Наибольшее паросочетание: {[self.graph.to_edges(variant) for variant in self.fittest]}\n"
                f"Наилучшая приспособленность: {self.best_fitness} \n"
                f"Средняя приспособленность поколения: {self.avg_fitness}")
            print("*" * 50 + "\n")


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
    ga = GA(graph=g)
    ga.start()
