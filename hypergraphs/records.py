from hypergraphs import HyperGraph


class Record:
    """
        Класс, представляющий собой информацию о поколении
    """

    def __init__(self,
                 number_of_generations: int,
                 fittest: list,
                 best_fitness: int,
                 avg_fitness: float,
                 best_edges: list,
                 hypergraph: HyperGraph):
        self.number_of_generations = number_of_generations
        self.fittest = fittest
        self.best_fitness = best_fitness
        self.avg_fitness = avg_fitness
        self.best_edges = best_edges
        self.hypergraph = hypergraph

    def __str__(self):
        return str(f"Номер поколения: {self.number_of_generations},\n"
                   + f"Наилучшие особи: {self.fittest},\n"
                   + f"Наилучшая приспособленность: {self.best_fitness} \n"
                   + f"Средняя приспособленность поколения: {self.avg_fitness}\n"
                   + f"Количество ребер в сочетании: {max(list(map(len, self.best_edges)))}\n"
                   + f"Совершенные сочетания: {self.best_edges}\n")
