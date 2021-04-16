from pprint import pprint
import random

import torch
import torch.nn as nn
import torch.nn.functional as F

from data import get_train_valid_loader, plot_images
from trainer import train
from model import ModelFromDecoding
from mutation import MUTATION_OPERATIONS
"""
Main module which represents the GA training
"""


class GA():
    def __init__(self,
                 expert_mode: bool = True,
                 initial_epochs: int = 3,
                 final_epochs: int = 10,
                 batch_size: int = 64,
                 n_individuals: int = 8,
                 cnn_depth: int = 10,
                 stopping_criteria: int = 100):
        """
        Generative Algorithm class which contains the lifecycle of generative algorithms.

        Args:
            expert_mode (bool, optional): Sample values based from human designed networks. Defaults to True.
            initial_epochs (int, optional): Initial epochs to train individual. Defaults to 3.
            final_epochs (int, optional): Final epochs that an individual is being trained. Defaults to 10.
            batch_size (int, optional): Batch size. Defaults to 64.
            n_individuals (int, optional): Number of individuals generated at initialization. Defaults to 8.
            cnn_depth (int, optional): Initial depth of architecture. Defaults to 10.
            stopping_criteria (int, optional): Number of generations where the algorithm should stop. Defaults to 100.
        """

        self.expert_mode = expert_mode
        self.batch_size = batch_size
        self.initial_epochs = initial_epochs
        self.final_epochs = final_epochs
        self.n_individuals = n_individuals
        self.cnn_depth = cnn_depth
        self.generations = 0
        self.stopping = stopping_criteria
        self.individuals = []
        self.offsprings = []
        self._generate_new_population(
        )  # initializes first pool at initialization
        self.fitness = {s: None for s in self.individuals}
        self.history = {}
        self.device = torch.device(
            'cuda') if torch.cuda.is_available() else torch.device('cpu')

    def _generate_new_population(self):
        """
        Generates initial population of string sequences representing a NN architecture
        """
        if self.expert_mode:
            choices = [
                64, 128, 256, 512
            ]  # known values that have worked well in current networks
        for _ in range(self.n_individuals):
            enc = ''
            for _ in range(self.cnn_depth):
                spec = random.uniform(0., 1.)
                if spec < 0.5:
                    conv_1 = str(random.randint(1, 512))
                    conv_2 = str(random.randint(1, 512))
                    if self.expert_mode:
                        conv_1 = str(random.choice(choices))
                        conv_2 = str(random.choice(choices))
                    enc += f'{conv_1}-{conv_2}-'
                else:
                    pool = str(round(random.uniform(0., 1.), 1))
                    enc += f'{pool}-'
            self.individuals.append(enc[:-1])

    def calc_fitness(self):
        """
        Calculates fitness (training set accuracy) for models that have not been trained
        """
        if torch.cuda.is_available():
            trainset, _ = get_train_valid_loader('./data/',
                                                 self.batch_size,
                                                 True,
                                                 42,
                                                 num_workers=1,
                                                 pin_memory=True)
        else:
            trainset, _ = get_train_valid_loader('./data/', self.batch_size,
                                                 True, 42)

        criterion = nn.CrossEntropyLoss()
        for i in self.individuals:
            if i not in self.fitness.keys():
                self.fitness[i] = None
            if self.fitness[i] is not None: continue
            model = ModelFromDecoding(i, self.device).to(self.device)
            epochs = self.initial_epochs if self.generations < (
                self.stopping // 2
            ) else self.final_epochs  # let model evolve faster in the start and train longer in the end
            self.fitness[i] = train(model,
                                    criterion,
                                    trainset,
                                    epochs=epochs,
                                    device=self.device)
        self.history[self.generations] = self.fitness

    def generate_offsprings(self, p_crossover: float, p_mutation: float,
                            mutation_operations: list):
        """
        Generates offsprings from sampled parents in current population.
        Offsprings have a certain chance to mutate.

        Args:
            p_crossover (float): Crossover probability
            p_mutation (float): Mutation probability
            mutation_operations (list): List of mutation operations which are defined in `mutation.py`
        """
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
                  remove_n_weakest: int = 4):
        """
        Selects best and drops worst performing individuals.

        Args:
            add_n_to_population (int, optional): Number of offsprings being added to the next generation. Defaults to 5.
            remove_n_weakest (int, optional): Number of individuals being dropped. Defaults to 4.
        """
        def remove_individual(least_fittest: str):
            """
            Helper function to remove individuals

            Args:
                least_fittest (str): Representation of individual
            """
            if least_fittest in self.individuals:
                self.individuals.remove(least_fittest)
                del self.fitness[least_fittest]
            else:
                return

        next_generation = []

        # sort individuals based on their fitness
        fittest = dict(
            sorted(self.fitness.items(), key=lambda i: i[1], reverse=True))
        rank = list(fittest.keys())

        for w in range(1, remove_n_weakest):
            rmv = rank[-w]
            remove_individual(rmv)

        # sample from pool of offsprings
        for _ in range(add_n_to_population):
            a, b = random.choices(self.offsprings, k=2)
            while a == b:
                a, b = random.choices(self.offsprings, k=2)
            next_generation.append(a)
            next_generation.append(b)

        individuals_2be = rank[:len(
            rank) // 2]  # select the upper best performing for next generation
        next_generation.extend(individuals_2be)

        self.individuals = next_generation
        self.offsprings.clear()

    def run(self):
        """
        Training loop.
        """
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
