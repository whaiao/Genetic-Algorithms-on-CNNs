from typing import List
import random
import pickle

import torch
import torch.nn as nn
import torch.nn.functional as F

from data import get_train_valid_loader, plot_images
from trainer import train
from model import ModelFromDecoding
from mutation import MUTATION_OPERATIONS


class GA():
    def __init__(self,
                 n_individuals: int = 10,
                 cnn_depth: int = 10,
                 stopping_criteria: int = 5):
        self.n_individuals = n_individuals
        self.cnn_depth = cnn_depth
        self.generations = 0
        self.stopping = stopping_criteria
        self.individuals = []
        self.offsprings = []
        self._generate_new_population()
        self.fitness = {s: None for s in self.individuals}
        self.device = torch.device(
            'cuda') if torch.cuda.is_available() else torch.device('cpu')

    def _generate_new_population(self):
        # choices = [32, 64, 128, 256, 512]
        for _ in range(self.n_individuals):
            enc = ''
            for _ in range(self.cnn_depth):
                spec = random.uniform(0., 1.)
                if spec < 0.5:
                    conv_1 = str(random.randint(1, 512))
                    conv_2 = str(random.randint(1, 512))
                    # conv_1 = str(random.choice(choices))
                    # conv_2 = str(random.choice(choices))
                    enc += f'{conv_1}-{conv_2}-'
                else:
                    oracle = str(round(random.uniform(0., 1.), 1))
                    enc += f'{oracle}-'
            self.individuals.append(enc[:-1])

    def calc_fitness(self):
        trainset, _ = get_train_valid_loader('./data/', 128, False, 42)
        criterion = nn.CrossEntropyLoss()
        for i in self.individuals:
            if self.fitness[i] is not None: continue
            model = ModelFromDecoding(i, self.device)
            self.fitness[i] = train(model,
                                    criterion,
                                    trainset,
                                    epochs=50,
                                    device=self.device)

    def generate_offsprings(self, p_crossover: float, p_mutation: float,
                            mutation_operations: list):
        offsprings = []
        while len(offsprings) < len(self.individuals):
            p1, p2 = random.choices(self.individuals, k=2)
            # if self.fitness[p1] < self.fitness[p2]:
            # p1, p2 = p2, p1
            while p1 == p2:
                p1, p2 = random.choices(self.individuals, k=2)
            r = random.uniform(0., 1.)
            if r < p_crossover:
                p1_part = p1.split('-')
                p2_part = p2.split('-')
                crossover_point = random.randint(1, len(p1_part) - 1)
                o1 = '-'.join(p1_part[:crossover_point] +
                              p2_part[crossover_point:])
                o2 = '-'.join(p2_part[:crossover_point] +
                              p1_part[crossover_point:])
                offsprings.append(o1)
                offsprings.append(o2)
        for offspring in self.offsprings:
            r = random.uniform(0., 1.)
            if r < p_mutation:
                m = random.randint(0, len(mutation_operations) - 1)
                pi = random.randint(0, len(offspring.split('-')) - 1)
                offspring[pi] = mutation_operations[m](offspring[pi])
        self.offsprings.extend(offsprings)

    def selection(self,
                  add_n_to_population: int = 2,
                  get_fittest: bool = False):
        def remove_individual(least_fittest: str):
            self.population.pop(least_fittest)
            del self.fitness[least_fittest]

        next_generation = []

        fittest = dict(
            sorted(self.fitness.items(), key=lambda i: i[1], reverse=True))
        rank = list(fittest.keys())

        least_fittest = fittest[rank[-1]]
        fittest = fittest[rank[0]]
        remove_individual(least_fittest)
        include = random.choices(self.offsprings, k=add_n_to_population)
        next_generation.extend(self.population)

        for i in include:
            next_generation.append(i)

        self.population = next_generation
        self.offsprings.clear()

        if get_fittest:
            return fittest

    def run(self):
        while self.generations < self.stopping:

            print(f'Generation: {self.generations+1}')
            self.calc_fitness()
            print('Current Fitness: ', self.fitness)
            self.generate_offsprings(random.uniform(0., 1.),
                                     random.uniform(0., 1.),
                                     MUTATION_OPERATIONS)
            self.selection()
            self.generations += 1


if __name__ == '__main__':
    p = GA()
    p.run()
