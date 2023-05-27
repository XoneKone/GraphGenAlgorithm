from random import randrange, uniform

from graphs import Graph


class DNA:

    def __init__(self, length, dna=None):
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
            self._genes = [randrange(0, self.length) for _ in range(self.length)]

    @property
    def fitness(self):
        return self._fitness

    @fitness.setter
    def fitness(self, value):
        if value is not None:
            self._fitness = value


class GA:
    MIN_VALUE = -2147483647

    def __init__(self, graph: Graph = None):
        self._graph = None
        self._population = []

        self.graph = graph
        self.population_size = randrange(20, 30)
        self.crossing_over_rate = uniform(0.8, 0.95)
        self.mutation_rate = uniform(0.5, 1.0)

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
        #todo Переделать надо, значения должны
        count = 0
        for index, value in enumerate(dna.genes):
            if [index, value] in self.graph.edges:
                count += 1
            else:
                return self.MIN_VALUE
        return count

    def crossing_over(self):
        pass

    def mutation(self):
        pass

    def choose_parents(self):
        pass

    def solve(self, graph: list[list[int]]):
        pass
