"""
This module provides an implementation of a Linear Genetic Algorithm (LGA), which evolves a population of
programs through multiple generations.

The main class, LGA, is responsible for handling the population, fitness
evaluation, and the evolution process. The LGA class supports spawning, mutation, crossover, and elimination
methods for evolving the programs.
"""

from pickle import dump
from numbers import Number
from os import makedirs, path
from datetime import datetime
from pprint import pformat, pprint
from typing import Tuple, Iterable, Union, List

import numpy as np
import torch
from torch.nn.functional import softmax
from tqdm import tqdm

from LGA.mutations import Mutations
from LGA.operations import UNARY, BINARY, AREA
from LGA.program import Program, TENSOR_FACOTRY
from utils.datasets import Dataset
from utils.other import PopulationNotEvaluatedError
from utils.losses import accuracy_score, FITNESS_FUNCTIONS


# pylint: disable=too-many-locals, too-many-arguments
class LGA:
    """Implementation of linear genetic algorithm."""

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
            area_instruction_p (int): Probability (in %) for area instructions. Defaults to 10.
            program_grow_p (int): Probability (in %) for a program to grow (an instruction). Defaults to 25.
            program_shrink_p (int): Probability (in %) for a program to shrink (an instruction). Defaults to 25.
            mutation_p (int): Probability (in %) for a program to mutate. Defaults to 25.
            crossover_p (int): Probability (in %) for a program to crossover. Defaults to 25.
            mutate_instructions (int): Number of instructions to mutate WHEN mutation occurs. Defaults to 1.
            mutate_registers (int): Registers to mutate WHEN mutation occurs. Defaults to 1.
            model_dir (str): Path to the directory for saving models. Defaults to "models".
            logging_dir (str): Path to the directory for logging. Defaults to "logging".
            binary (Union[None, List[str]]): List of chosen binary functions, None if all. Defaults to None.
            unary (Union[None, List[str]]): List of chosen unary functions, None if all. Defaults to None.
            area (Union[None, List[str]]): List of chosen area functions, None if all. Defaults to None.
        """
        # Program loaded prior to LGA initialization, serves as a seed for single individual elite at 1st generation
        self.proto_program = program
        self.torch_device = dataset.torch_device
        self.object_shape = dataset.object_shape
        self.num_of_classes = len(dataset.classes)
        self.current_generation = 0
        self.fitness_fn = FITNESS_FUNCTIONS[fitness]

        # Array of Program objects
        self.population = np.array([])
        self.evaluated = False
        self.evaluated_fitness = None
        self.best_program_candidates = None
        self.run_fitness_history = None

        self.elite_size_tensor = TENSOR_FACOTRY.IntTensor(1).fill_(elite)
        self.elite_distribution_probability = TENSOR_FACOTRY.FloatTensor(elite).fill_(1)

        # set LGA hyperparameters
        self._set_hyperparameters(
            population,
            generations,
            min_instructions,
            max_instructions,
            hidden_register_shape,
            mutation_p,
            crossover_p,
            area_instruction_p,
            program_grow_p,
            program_shrink_p,
            mutate_registers,
            mutate_instructions,
            elite,
            equal_elite,
        )
        self._set_directories(model_dir, logging_dir)
        self._set_operations(unary, binary, area)
        # Set self as lga for Program class
        Program.lga = self

    def _set_hyperparameters(
        self,
        population: int,
        generations: int,
        min_instructions: int,
        max_instructions: int,
        hidden_register_shape: Tuple[int],
        mutation_p: int,
        crossover_p: int,
        area_instruction_p: int,
        program_grow_p: int,
        program_shrink_p: int,
        mutate_registers: int,
        mutate_instructions: int,
        elite_size: int,
        equal_elite: int,
    ) -> None:
        """
        Configure LGA algorithm with hyperparameters
        """
        self.population_bound = population
        self.generations = generations
        self.min_instructions = min_instructions
        self.max_instructions = max_instructions
        self.hidden_register_shape = hidden_register_shape
        self.mutate_registers = mutate_registers
        self.mutate_instructions = mutate_instructions
        self.elite_size = elite_size
        self.equal_elite = equal_elite
        self.mutation_p = mutation_p / 100.0
        self.crossover_p = crossover_p / 100.0
        self.area_instruction_p = area_instruction_p / 100.0
        self.program_grow_p = program_grow_p / 100.0
        self.shrink_p = program_shrink_p / 100.0
        self.mutation_operation_probabilities = np.array(
            [self.crossover_p, self.mutation_p, self.program_grow_p, self.shrink_p]
        )

    def _set_directories(self, model_dir: str, logging_dir: str) -> None:
        """Set and create model and loggin directories if not exist"""
        self.model_dir = model_dir
        self.logging_dir = logging_dir
        makedirs(model_dir, exist_ok=True)
        makedirs(logging_dir, exist_ok=True)

    def _set_operations(self, unary: List[str], binary: List[str], area: List[str]) -> None:
        """If specified - select algorithm operations"""
        # pylint: disable=global-statement
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
        self.population = np.append(self.population, [self.new_individual() for _ in range(population_missing)])
        # Sort population according to its' population. Later sorted with fitness (stable sort).
        self.population = self.population[
            np.argsort(np.array([len(individual.instructions) for individual in self.population]))
        ]
        self.evaluated = False

    # pylint: disable=attribute-defined-outside-init
    def new_individual(self) -> Program:
        """
        Create a new Program individual based on LGA object properties

        Returns:
            Program: The new Program instance
        """
        if not self.evaluated:
            return Program.create_random_program(self)

        # Determine which mutation operations to perform based on their probabilities (could be 4, could be 0)
        crossover, mutate, grow, shrink = self.mutation_operation_probabilities >= np.random.random(4)

        # When neither of following actions was chosen, create a random program for exploration
        if not crossover and not mutate and not grow and not shrink:
            return Program.create_random_program(self)

        elite_population = self.population[: self.elite_size]
        # If elite is equal, this vector is set in constructor
        if not self.equal_elite:
            self.elite_distribution_probability = softmax(self.evaluated_fitness[: self.elite_size_tensor], dim=0)

        # Initiate the new life
        if crossover:
            father_id, mother_id = self.elite_distribution_probability.multinomial(2, replacement=False)
            father, mother = elite_population[father_id], elite_population[mother_id]
            offspring = Program.crossover(mother, father)
        else:
            parent_id = self.elite_distribution_probability.multinomial(1)
            parent = elite_population[parent_id]
            offspring = Program.transcription(parent)

        if mutate:
            Mutations.mutate_program(offspring, self.mutate_registers, self.mutate_instructions)
        if grow:
            offspring.grow()
        if shrink:
            offspring.shrink()

        # long live the newborn
        return offspring

    def argsort_population_and_fitness(self, key: Iterable, **kwargs) -> None:
        """Sort population and fitness tensor according to provided kay array"""
        sorted_indices = torch.argsort(key, **kwargs)
        self.evaluated_fitness = self.evaluated_fitness[sorted_indices]
        self.population = self.population[sorted_indices.cpu().numpy()]

    def evaluate_population(self, in_data: torch.Tensor, gt_labels: torch.Tensor, use_percent: bool = False) -> None:
        """
        Evaluate the whole population of programs and sort it from best to worse.

        Args:
            in_data (torch.Tensor): Input data tensor.
            gt_labels (torch.Tensor): Ground truth labels tensor.
            use_percent (bool, optional): Evaluate success in percents of correctly predicted objects. Defaults to False.
        """
        # Rank each program with it's fitness
        self.evaluated_fitness = torch.stack(
            [
                individual.evaluate(in_data, gt_labels, accuracy_score if use_percent else self.fitness_fn)
                for individual in self.population
            ]
        )
        self.argsort_population_and_fitness(self.evaluated_fitness, stable=True, descending=True)
        self.evaluated = True

    def eliminate(self) -> None:
        """Eliminate inferior Programs which did not make it into the elite"""
        # Delete non-elite programs and retain only the elite
        self.population = np.delete(self.population, range(self.elite_size, self.population_bound))
        # delete_programs = self.population[self.elite_size :]
        # self.population = self.population[: self.elite_size]
        # for delete_program in delete_programs:
        #     del delete_program

    def train(self, dataset: Dataset, runs: int) -> None:
        """
        Execute the evolution in the name of The God itself

        Args:
            dataset (Dataset): Dataset providing test and eval data&labels
            runs (int): How many times to run the LGA
        """
        # Save loss data and best programs from each run for further analysis
        self.best_program_candidates = []
        self.run_fitness_history = []

        for run in range(runs):
            # Initialize the population and other configuration
            self.population = np.array([self.proto_program] if self.proto_program else [])
            self.current_generation = 0
            self.evaluated = False
            self.run_fitness_history.append([])

            with tqdm(range(self.generations), desc="Evolving ...") as pbar:
                for _ in pbar:
                    self.current_generation += 1
                    self.fill_population()
                    self.evaluate_population(dataset.test_X, dataset.test_y)
                    self.eliminate()
                    self.run_fitness_history[run].append(self.evaluated_fitness[0])

                    pbar.set_description("Evolving ... ({:.5f})".format(self.evaluated_fitness[0]))

            self.best_program_candidates.append(self.population)

        self.final_evaluation(dataset)

    def final_evaluation(self, dataset: Dataset) -> Program:
        """After training, evaluate best programs from each run and return the best performing one"""
        # Evaluate best population candidates on evaluation dataset, pickle the best program
        self.population = np.array(self.best_program_candidates).flatten()
        self.evaluate_population(dataset.eval_X, dataset.eval_y, use_percent=True)
        model_filename = (
            f"{dataset.dataset_name}_{dataset.edge_size}_{self.evaluated_fitness[0]}_{datetime.now().isoformat()}.p"
        )
        model_path = path.join(self.model_dir, model_filename)

        with open(model_path, "wb") as file:
            dump(self.best_program, file)

        # Log fitness for all runs
        fitness_log_filename = (
            f"{dataset.dataset_name}_{dataset.edge_size}_runs{len(self.run_fitness_history)}_{datetime.now().isoformat()}.p"
        )
        fitness_log_path = path.join(self.logging_dir, fitness_log_filename)

        with open(fitness_log_path, "wb") as file:
            dump(self.run_fitness_history, file)

        print("Results of evaluated best program candidates:")
        pprint(self.evaluated_fitness)
        print("The best program is:")
        pprint(self.best_program)

    @property
    def best_program(self) -> Program:
        """Return the program from the population with the highest accuracy."""
        if not self.evaluated:
            raise PopulationNotEvaluatedError("Tried to get the best program without evaluation of fitness!")
        return self.population[0]

    def to_dict(self):
        """Create JSON-dumpable dictionary with data needed for further analysis"""
        return {
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
            "AREA": [A.__name__ for A in AREA],
        }

    def __repr__(self) -> str:
        """String representation of program and it's instructions (human friendly)"""
        return pformat(self.to_dict(), width=120)
