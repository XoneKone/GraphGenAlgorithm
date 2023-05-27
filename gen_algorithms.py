from random import randrange, uniform, randint

from graphs import Graph


class DNA:

    def __init__(self, length: int, dna=None):
        self._length = None
        self._genes = None
        self._fitness = 0
        self.genes = dna
        self.length = length

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


class PopulationInfo:
    def __init__(self):
        self.generation_count = 0
        self.avg_fitness = None
        self.best_fitness = None
        self.fittest = None


class GA:
    MIN_VALUE = -2147483647

    def __init__(self, graph: Graph = None):
        self._graph = None
        self._population = []

        self.graph = graph
        self.population_size = randrange(20, 30)
        self.crossing_over_rate = uniform(0.8, 0.95)
        self.mutation_rate = uniform(0.5, 1.0)
        self.population_info = PopulationInfo()

    @property
    def population(self):
        return self._population

    @population.setter
    def population(self, value):
        if value is not None:
            self._population = value

    @property
    def graph(self):
        return self._graph

    @graph.setter
    def graph(self, value):
        if value is not None:
            self._graph = value

    def step(self):
        pass

    def initialize_population(self):
        for _ in range(self.population_size):
            self._population.append(DNA(self.graph.number_of_edges))
            for dna in self._population:
                dna.fitness = self.evaluate_fitness(dna)

    def evaluate_fitness(self, dna):
        fitness = 0
        for i in range(len(dna.genes)):
            for j in range(i + 1, len(dna.genes)):
                if (((dna.genes[i] != 0 and dna.genes[j] != 0) and self.graph.edge_adj_matrix[i][j] == 1)
                        or (dna.genes[i] == 0)):
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

        if self.evaluate_fitness(child1) < self.evaluate_fitness(child2):
            return child1

        return child2

    def mutation(self, dna):
        mutated_dna = DNA(dna)
        check = uniform(0, 1)
        if check <= self.mutation_rate:
            position = randint(0, len(mutated_dna.genes) - 1)
            if mutated_dna.genes[position] == 0:
                mutated_dna.genes[position] = 1
            else:
                mutated_dna.genes[position] = 0
        return mutated_dna

    def mutation2(self, dna):
        mutated_dna = DNA(dna)
        for position in range(mutated_dna.length):
            check = uniform(0, 1)
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
        for _ in population_without_gen_operators:
            parents = self.choose_parents()
            child = self.crossover(parents[0], parents[1])
            child = self.mutation2(child)
            child.fitness = self.evaluate_fitness(child)
            new_population.append(child)
        self.population = new_population
        self.population_info.generation_count += 1
        self.population_info.avg_fitness =

    def calculate_avg_fitness(self):
        avg_fitness = None
        #todo: Доделать)

    def get_population_info(self):

        return

    def start(self, graph: list[list[int]]):
        self.initialize_population()
        while self.population_info.best_fitness != 0 and self.population_info.generation_count != 1000:
            self.population_info.generation_count += 1
