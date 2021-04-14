from typing import List
from pprint import pprint
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
                 n_individuals: int = 8,
                 cnn_depth: int = 10,
                 stopping_criteria: int = 100):
        self.n_individuals = n_individuals
        self.cnn_depth = cnn_depth
        self.generations = 0
        self.stopping = stopping_criteria
        self.individuals = []
        self.offsprings = []
        self._generate_new_population()
        self.fitness = {s: None for s in self.individuals}
        self.history = {}
        self.device = torch.device(
            'cuda') if torch.cuda.is_available() else torch.device('cpu')

    def _generate_new_population(self):
        choices = [64, 128, 256, 512]
        for _ in range(self.n_individuals):
            enc = ''
            for _ in range(self.cnn_depth):
                spec = random.uniform(0., 1.)
                if spec < 0.5:
                    #conv_1 = str(random.randint(1, 512))
                    #conv_2 = str(random.randint(1, 512))
                    conv_1 = str(random.choice(choices))
                    conv_2 = str(random.choice(choices))
                    enc += f'{conv_1}-{conv_2}-'
                else:
                    oracle = str(round(random.uniform(0., 1.), 1))
                    enc += f'{oracle}-'
            self.individuals.append(enc[:-1])

    def calc_fitness(self):
        trainset, _ = get_train_valid_loader('./data/', 64, True, 42, num_workers=1, pin_memory=True)
        # don't apply softmax
        criterion = nn.CrossEntropyLoss()
        for i in self.individuals:
            if i not in self.fitness.keys(): 
                self.fitness[i] = None
            if self.fitness[i] is not None: continue
            model = ModelFromDecoding(i, self.device).to(self.device)
            epochs = 2 if self.generations < (self.stopping // 2) else 10  # let model evolve faster in the start and train longer in the end
            self.fitness[i] = train(model,
                                    criterion,
                                    trainset,
                                    epochs=epochs,
                                    device=self.device)
        self.history[self.generations] = self.fitness

    def generate_offsprings(self, p_crossover: float, p_mutation: float,
                            mutation_operations: list):
        offsprings = []
        while len(offsprings) < len(self.individuals):
            p1, p2 = random.choices(self.individuals, k=2)
            if self.fitness[p1] < self.fitness[p2]:
                p1, p2 = p2, p1
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
                  add_n_to_population: int = 5,
                  remove_n_weakest: int = 4,
                  get_fittest: bool = False):
        def remove_individual(least_fittest: str):
            if least_fittest in self.individuals:
                self.individuals.remove(least_fittest)
                del self.fitness[least_fittest]
            else:
                return 

        next_generation = []

        fittest = dict(
            sorted(self.fitness.items(), key=lambda i: i[1], reverse=True))
        rank = list(fittest.keys())

        for w in range(1, remove_n_weakest):
            rmv = rank[-w]
            remove_individual(rmv)

        for _ in range(add_n_to_population):
            a, b = random.choices(self.offsprings, k=2)
            while a == b:
                a, b = random.choices(self.offsprings, k=2)
            next_generation.append(a)
            next_generation.append(b)
        
        individuals_2be = rank[:len(rank)//2]  # select the upper best performing for next generation
        next_generation.extend(individuals_2be)


        self.individuals = next_generation
        self.offsprings.clear()

        if get_fittest:
            return fittest

    def run(self):
        while self.generations < self.stopping:
            print(f'Generation: {self.generations+1}')
            self.calc_fitness()
            print('Current Fitness:')
            pprint(self.fitness)
            self.generate_offsprings(random.uniform(0., 1.),
                                     random.uniform(0., 1.),
                                     MUTATION_OPERATIONS)
            self.selection()
            self.generations += 1
            # print('Next gen: ')
            # pprint(self.individuals)


if __name__ == '__main__':
    p = GA()
    p.run()
    with open('results-pref100gen.pickle', 'wb') as f:
        pickle.dump(p, f)
