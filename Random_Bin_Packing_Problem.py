import numpy as np
from time import time
np.random.seed(0)
import gc

class Item:
    def __init__(self, label: int = None, size: int = None):
        self._label = label
        self._size = size


class Bin:
    def __init__(self, items: list = None, capacity: int = 0):
        if items is None:
            items = []
        self._items = items
        self._capacity = capacity

    def add_item(self, item: Item):
        self._items.append(item)
        self._capacity += item._size

    def get_labels(self):
        return [item._label for item in self._items]


class Chromosome:
    def __init__(self, bins: list = None, max_capacity: int = 0):
        if bins is None:
            bins = []
        self._bins = bins
        self._max_capacity = max_capacity
        self._fitness = None

    def calculate_fitness(self):
        num_bins = len(self._bins)
        bins_capacity = np.asarray([_bin._capacity for _bin in self._bins], dtype=np.int)
        self._fitness = np.sum(np.power(bins_capacity / self._max_capacity, 2)) / num_bins
        return self._fitness

    def count_items(self):
        return len([item for _bin in self._bins for item in _bin._items])

    def count_indexes(self):
        items = [item for _bin in self._bins for item in _bin._items]
        indexes = [item._label for item in items]
        return len(set(indexes))


class Population:

    @staticmethod
    def population_initialization(D: int = None, N: int = None, B: int = None, d_list: list = None,
                                  population_size: int = 250):
        chromos = []
        indexes = np.arange(N)
        start = time()
        current_generation_fitness_best = -1
        current_solution_best = -1
        
        bins = []
        current_bin = Bin()
        for i in range(population_size):
            bins.clear()
            current_bin._items.clear()
            current_bin._capacity = 0
            
            np.random.shuffle(indexes)
            for idx in indexes:
                if current_bin._capacity + d_list[idx] <= D:
                    current_bin.add_item(Item(label=idx, size=d_list[idx]))
                else:
                    bins.append(current_bin)
                    current_bin = Bin()
                    current_bin.add_item(Item(label=idx, size=d_list[idx]))
            bins.append(current_bin)
            chromos.append(Chromosome(bins, max_capacity=D))
            a = chromos[-1]
            current_generation_fitness = a.calculate_fitness()
            
            if i % 1000 == 0:
                print('\r Random number: {}'.format(i), end='')
            if current_generation_fitness_best < current_generation_fitness:
                current_generation_fitness_best =  current_generation_fitness
                current_solution_best = len(a._bins)
                print('fitness_best:{}'.format(current_generation_fitness_best))
                print('solution_best:{}'.format(current_solution_best))
            #print('\r\n solution_best:{}'.format(current_solution_best), end='')
            #print('\r\n fitness_best:{}'.format(current_generation_fitness_best), end='')
            #print('Initial fitness: {}'.format(current_generation_fitness))
            #print('Initial solution: {} bins'.format(current_solution))

            # if (time() - start) % 60 == 0:
            #    gc.collect()
            if time() - start > 750:
                print('fitness_best:{}'.format(current_generation_fitness_best))
                print('solution_best:{}'.format(current_solution_best))
                # current_generation_fitness = [np.max(np.asarray([chromo.calculate_fitness() for chromo in chromos]))]
                #current_solution = [np.min(np.asarray([len(chromo._bins) for chromo in chromos]))]
                #print('Initial fitness: {}'.format(current_generation_fitness))
                #print('Initial solution: {} bins'.format(current_solution))
                break;

def load_data(data_path):
    f = open(data_path, 'r')
    lines = f.readlines()
    num_set = int(lines[0])
    length = int((len(lines) - 1) / num_set)
    sets = [lines[i * length + 1: (i + 1) * length + 1] for i in range(num_set)]
    test_sets = []
    for _set in sets:
        D, N, B = [int(num) for num in _set[1].split()]
        d_list = [int(num) for num in _set[2:]]
        test_sets.append({'D': D, 'N': N, 'B': B, 'd_list': d_list})
    return test_sets


if __name__ == '__main__':
    generate_config = {'population_size': 100000000000, 'offspring_number': 50, 'chromosomes_replace': 50,
                       'crossover_probability': 1.0, 'mutation_probability': 0.66, 'mutation_size': 2,
                       'generations_number': 500, 'stop_criterion_depth': 50}
    generate_config['offspring_number'] = int(generate_config['population_size'] / 2)
    generate_config['chromosomes_replace'] = int(generate_config['population_size'] / 2)
    path = 'binpack3.txt'
    test_sets = load_data(path)
    for _set in test_sets:
        population = Population.population_initialization(D=_set['D'], N=_set['N'], B=_set['B'],
                                                          d_list=_set['d_list'],
                                                          population_size=generate_config['population_size'], )
        break
