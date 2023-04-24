"""
This module provides an implementation of a Linear Genetic Algorithm (LGA), which evolves a population of
programs through multiple generations.

The main class, LGA, is responsible for handling the population, fitness
evaluation, and the evolution process. The LGA class supports spawning, mutation, crossover, and elimination
methods for evolving the programs.
"""

from pprint import pformat
from numbers import Number
from os import makedirs
from typing import Tuple, Iterable, Union, List

import torch
import numpy as np
from tqdm import tqdm

from utils.losses import success_rate, FITNESS_FUNCTIONS
from utils.datasets import Dataset
from LP.program import Program
from LP.operations import UNARY, BINARY, AREA

torch.set_printoptions(profile="full")

class PopulationNotEvaluatedError(Exception):
    """Raised when the population was not evaluated and acessed based upon fitness values"""
    pass


class LGA:
    def __init__(
        self,
        dataset: Dataset,
        program: Program,
        population: int = 42,
        elite: int = 3,
        equal_elite: bool = False,
        generations: int = 60,
        min_instructions: int = 1,
        max_instructions: int = 20,
        fitness: str = "ac",
        hidden_register_shape: Tuple[int] = (42,),
        area_instruction_p: int = 10,
        program_grow_p: int = 25,
        program_shrink_p: int = 25,
        mutation_p: int = 25,
        crossover_p: int = 25,
        mutate_instructions: int = 1,
        mutate_registers: int = 1,
        model_dir: str = "models",
        logging_dir: str = "logging",
        binary: Union[None, List[str]] = None,
        unary: Union[None, List[str]] = None,
        area: Union[None, List[str]] = None,
    ) -> None:
        """
        Initialize LGA algorithm object.

        Args:

            dataset (Dataset): Dataset object.
            program (Program): Program object.
            population (int): Population size. Defaults to 42.
            elite (int): Elite size. Defaults to 3.
            equal_elite (bool): Sample elite uniformly, instead of sampling with probability equal to their fitness. Defaults to False.
            generations (int): Number of generations to evolve. Defaults to 60.
            min_instructions (int): Minimum number of instructions. Defaults to 1.
            max_instructions (int): Maximum number of instructions. Defaults to 20.
            fitness (str): Fitness function identifier. Defaults to "ac".
            hidden_register_shape (Tuple[int]): Shape of the hidden register. Defaults to (42,).
            area_instruction_p (int): Probability for area instructions. Defaults to 10.
            program_grow_p (int): Probability for a program to grow (an instruction). Defaults to 25.
            program_shrink_p (int): Probability for a program to shrink (an instruction). Defaults to 25.
            mutation_p (int): Probability for a program to mutate. Defaults to 25.
            crossover_p (int): Probability for a program to crossover. Defaults to 25.
            mutate_instructions (int): Number of instructions to mutate WHEN mutation occurs. Defaults to 1.
            mutate_registers (int): Registers to mutate WHEN mutation occurs. Defaults to 1.
            model_dir (str): Path to the directory for saving models. Defaults to "models".
            logging_dir (str): Path to the directory for logging. Defaults to "logging".
            binary (Union[None, List[str]]): List of chosen binary functions, None if all. Defaults to None.
            unary (Union[None, List[str]]): List of chosen unary functions, None if all. Defaults to None.
            area (Union[None, List[str]]): List of chosen area functions, None if all. Defaults to None.
        """
        self.proto_program = program
        self.torch_device = dataset.torch_device
        self.object_shape = dataset.object_shape
        self.dataset_name = dataset.dataset_name
        self.dataset_edge_size = dataset.edge_size
        self.num_of_classes = len(dataset.classes)
        self.current_generation = 0
        self.fitness_fn = FITNESS_FUNCTIONS[fitness]

        self.evaluated = False
        self.population = np.array([])
        self.evaluated_fitness = np.array([])
        self.mutation_operation_probabilities = \
            np.array([self.crossover_p, self.mutation_p, self.program_grow_p, self.program_shrink_p])

        self._set_hyperparameters(population, generations, min_instructions, max_instructions, hidden_register_shape,
                                  mutation_p, crossover_p, area_instruction_p, program_grow_p, program_shrink_p,
                                  mutate_registers, mutate_instructions, elite, equal_elite)
        self._set_directories(model_dir, logging_dir)
        self._set_operations(unary, binary, area)


    def _set_hyperparameters(self, population, generations, min_instructions, max_instructions, hidden_register_shape,
                            mutation_p, crossover_p, area_instruction_p, program_grow_p, program_shrink_p, mutate_registers,
                            mutate_instructions, elite_size, equal_elite):
        """Set LGA hyperparameters"""
        self.population_bound = population
        self.generations = generations
        self.min_instructions = min_instructions
        self.max_instructions = max_instructions
        self.hidden_register_shape = hidden_register_shape
        self.mutation_p = mutation_p / 100.0
        self.crossover_p = crossover_p / 100.0
        self.area_instruction_p = area_instruction_p / 100.0
        self.program_grow_p = program_grow_p / 100.0
        self.shrink_p = program_shrink_p / 100.0
        self.mutate_registers = mutate_registers
        self.mutate_instructions = mutate_instructions
        self.elite_size = elite_size
        self.equal_elite = equal_elite

    def _set_directories(self, model_dir, logging_dir):
        """Set and ensure model and loggin directories exist"""
        self.model_dir = model_dir
        self.logging_dir = logging_dir
        makedirs(model_dir, exist_ok=True)
        makedirs(logging_dir, exist_ok=True)

    def _set_operations(self, unary, binary, area):
        """Set operations of linear program"""
        global UNARY, BINARY, AREA
        if unary:
            UNARY = [U for U in UNARY if U.__name__ in unary]
        if binary:
            BINARY = [B for B in BINARY if B.__name__ in binary]
        if area:
            AREA = [A for A in AREA if A.__name__ in area]

    def fill_population(self) -> None:
        """Generate Program instances to meet the population criteria"""
        population_missing = self.population_bound - len(self.population)
        # Create new individuals
        self.population = np.append(self.population, [self.new_individual() for _ in range(population_missing)])
        # Invalidate population evaluation, since new individuals were added
        self.evaluated = False

    def new_individual(self) -> Program:
        """
        Create a new Program individual based on LGA object properties

        Returns:
            Program: The new Program instance
        """
        if not self.evaluated:
            return Program.create_random(self)

        elite_population, elite_fitness = self.elite_plus_fitness

        if self.equal_elite:
            elite_distribution_probability = np.ones_like(elite_fitness) / len(elite_population)
        else:
            elite_distribution_probability = np.array(elite_fitness) / sum(elite_fitness)

        # Determine which mutation operations to perform based on their probabilities
        crossover, mutate, grow, shrink = self.mutation_operation_probabilities >= np.random.random(4)

        # When neither of following actions was chosen, create a random program for exploration
        if not crossover and not mutate and not grow and not shrink:
            return Program.create_random(self)

        # Initiate the new life
        if crossover:
            father, mother = np.random.choice(elite_population, size=2, replace=False, p=elite_distribution_probability)
            offspring = Program.crossover(father, mother)
        else:
            parent = np.random.choice(elite_population, p=elite_distribution_probability)
            offspring = Program.transcription(parent)

        if mutate:
            offspring.mutate(self.mutate_registers, self.mutate_instructions)

        if grow:
            offspring.grow()

        if shrink:
            offspring.shrink()

        # long live the newborn
        return offspring


    def evaluate_population(self, in_data: torch.Tensor, gt_labels: torch.Tensor, use_percent: bool = False) -> None:
        """
        Evaluate the whole population of programs and sort it from best to worse.

        Args:
            in_data (torch.Tensor): Input data tensor.
            gt_labels (torch.Tensor): Ground truth labels tensor.
            use_percent (bool, optional): Evaluate success in percents of correctly predicted objects. Defaults to False.
        """
        # Rank each program with it's fitness
        self.evaluated_fitness = np.array(
            [
                individual.eval(in_data, gt_labels, success_rate if use_percent else self.fitness_fn)
                for individual in self.population
            ]
        ).reshape(-1)
        # Rank each individual with instruction lenght
        instruction_lengths = np.array([len(individual.instructions) for individual in self.population])

        # Sort population and fitness according to instructions length (secondary key) and fitness (primary key)
        sorted_indices = np.argsort(instruction_lengths)
        self.population = self.population[sorted_indices]
        self.evaluated_fitness = self.evaluated_fitness[sorted_indices]

        sorted_indices = np.argsort(self.evaluated_fitness, kind="stable")[::-1]
        self.population = self.population[sorted_indices]
        self.evaluated_fitness = self.evaluated_fitness[sorted_indices]
        self.evaluated = True

    def eliminate(self) -> None:
        """Eliminate inferior Programs which did not make it into the elite"""
        # Delete non-elite programs and retain only the elite
        delete_programs = self.population[self.elite_size :]
        self.population = self.population[: self.elite_size]
        for delete_program in delete_programs:
            del delete_program

    def train(self, dataset: Dataset, runs: int) -> None:
        """
        Execute the evolution in the name of The God itself

        Args:
            dataset (Dataset): Dataset providing test and eval data&labels
            runs (int): How many times to run the LGA
        """
        self.best_programs = []
        for run in range(1, runs+1):
            # Initialize the population and other configuration
            self.population = np.array([self.proto_program] if self.proto_program else [])
            self.current_generation = 0
            self.evaluated = False
            # Train the population
            with tqdm(range(self.generations), desc="Evolving ...") as pbar:
                for _ in pbar:
                    # Update generation iteration
                    self.current_generation += 1
                    # Fill the whole population with individuals
                    self.fill_population()
                    # Evaluate the population
                    self.evaluate_population(dataset.test_X, dataset.test_y)
                    # Eliminate all individuals but the elite
                    self.eliminate()
                    # Update progress bar description
                    pbar.set_description(f"Evolving ... ({int(self.evaluated_fitness[0])} %)")
            # Log the run best program and its fitness
            self.best_programs.append((self.best_program, self.evaluated_fitness[0]))

            # TODO: Logging some data for further analysis
            # TODO: Pickle the best program out there, evaluate all the best programs beforehand on eval dataset


    @property
    def best_program(self) -> Program:
        """Return the program from the population with the highest accuracy."""
        if not self.evaluated:
            raise PopulationNotEvaluatedError("Tried to get the best program without evaluation of fitness!")
        return self.population[0]

    @property
    def elite_plus_fitness(self) -> Tuple[Union[Iterable[Program], Iterable[Number]]]:
        """Return the array of elite individuals and their fitness."""
        return self.population[: self.elite_size], self.evaluated_fitness[: self.elite_size]

    def to_dict(self):
        '''Create JSON-dumpable dictionary with data needed for further analysis'''
        return {
            "dataset-name": self.dataset_edge_size,
            "dataset-edge-size": self.dataset_edge_size,
            "max-population": self.population_bound,
            "elite-size": self.elite_size,
            "elite-equal": self.equal_elite,
            "generations to evolve": self.generations,
            "max-i": self.max_instructions,
            "min-i": self.min_instructions,
            "mutation-p": self.mutation_p,
            "crossover-p": self.crossover_p,
            "grow-p": self.program_grow_p,
            "area-p": self.area_instruction_p,
            "fitness-fn": self.fitness_fn.__name__,
            "hidden-reg-shape": self.hidden_register_shape,
            "mutate-registers": self.mutate_registers,
            "mutate-instructions": self.mutate_instructions,
            "UNARY": [U.__name__ for U in UNARY],
            "BINARY": [B.__name__ for B in BINARY],
            "AREA": [A.__name__ for A in AREA]
        }


    def __repr__(self) -> str:
        """String representation of program and it's instructions (human friendly)"""
        return pformat(self.to_dict(), width=120)
