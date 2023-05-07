"""
This module provides an implementation of a Linear Genetic Algorithm (LGA), which evolves a population of
programs through multiple generations.

The main class, LGA, is responsible for handling the population, fitness
evaluation, and the evolution process. The LGA class supports spawning, mutation, crossover, and elimination
methods for evolving the programs.
"""

from math import prod
from pickle import dump
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
            mutate_instructions (int): Max number of instructions to mutate WHEN mutation occurs. Defaults to 1.
            mutate_registers (int): Max negisters to mutate WHEN mutation occurs. Defaults to 1.
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
        self._set_turboboost_properties()

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

    def _set_operations(self, unary: Union[List[str], None], binary: Union[List[str], None], area: Union[List[str], None]) -> None:
        """If specified - select algorithm operations"""
        # pylint: disable=global-statement
        global UNARY, BINARY, AREA
        if unary is not None:
            UNARY = [U for U in UNARY if U.__name__ in unary]
        if binary is not None:
            BINARY = [B for B in BINARY if B.__name__ in binary]
        if area is not None:
            AREA = [A for A in AREA if A.__name__ in area]

    def _set_turboboost_properties(self) -> None:
        """
        Initialize properties for Program and Instruction to read

        Not passing arguments here and there, also not creating millions of instances of some attributes,
        self (LGA instance) will be acessable from each Instruction and Program instance
        """
        Program.lga = self
        self.register_shapes_dict = {
            "input_registers": self.object_shape,
            "hidden_registers": self.hidden_register_shape,
            "result_registers": (self.num_of_classes,),
        }
        self.hidden_register_count = prod(self.hidden_register_shape)
        self.result_register_count = self.num_of_classes

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
            self.run_fitness_history.append(
                (
                    {
                        "fitness": [],
                        "instr-len": [],
                    }
                )
            )

            with tqdm(range(self.generations), desc="Evolving ...", ncols=100) as pbar:
                for _ in pbar:
                    self.current_generation += 1
                    # Create new programs until population criterion met
                    self.fill_population()
                    # Evaluate population on test dataset
                    self.evaluate_population(dataset.test_X, dataset.test_y)
                    # Eliminate programs which could not make it into the elite
                    self.eliminate()
                    # Log fitness of best programs throughout generations
                    self.run_fitness_history[run]["fitness"].append(self.evaluated_fitness[0])
                    # Log instruction length of best programs throughout generations
                    self.run_fitness_history[run]["instr-len"].append(len(self.best_program.instructions))
                    pbar.set_description(
                        f"Evolving ... ({self.evaluated_fitness[0]:.5f}) ({len(self.best_program.instructions)})"
                    )

                # fitness is list of single-element tensors, instruction lengths is list of ints
                # make both a numpy array
                self.run_fitness_history[run]["fitness"] = torch.stack(self.run_fitness_history[run]["fitness"]).cpu().numpy()
                self.run_fitness_history[run]["instr-len"] = np.array(self.run_fitness_history[run]["instr-len"])

            self.best_program_candidates.append(self.population)

        self.final_evaluation(dataset)

    def final_evaluation(self, dataset: Dataset) -> Program:
        """After training, evaluate best programs from each run and return the best performing one"""
        # Evaluate best population candidates on evaluation dataset, pickle the best program
        self.population = np.array(self.best_program_candidates).flatten()
        self.evaluate_population(dataset.eval_X, dataset.eval_y, use_percent=True)
        timestamp = datetime.now().isoformat()
        filename = f"{dataset.dataset_name}_{dataset.edge_size}_{self.evaluated_fitness[0]}_{timestamp}.p"
        model_path = path.join(self.model_dir, filename)

        with open(model_path, "wb") as file:
            self.best_program.pickle(file)

        fitness_log_path = path.join(self.logging_dir, filename)

        with open(fitness_log_path, "wb") as file:
            dump(self.run_fitness_history, file)

        self.print_results(model_path, fitness_log_path)

    def print_results(self, model_path: str, logging_path: str) -> None:
        """Print the final message after algorithm evaluation"""
        print("Results of evaluated best program candidates:")
        pprint(self.evaluated_fitness)
        print("The best program is:")
        program_instructions = [repr(instr) for instr in self.best_program.instructions]
        print('\n'.join(program_instructions))
        torch.set_printoptions(profile="full")
        print("Hidden register initialization values:")
        pprint(self.best_program.hidden_register_initial_values)
        print("Result register initialization values:")
        pprint(self.best_program.result_register_initial_values)
        torch.set_printoptions(profile="default")
        print(f"The best program was pickled into file: {model_path}")
        print(f"The fitness and etc. history was pickled into file: {logging_path}")

    @property
    def best_program(self) -> Program:
        """Return the program from the population with the highest accuracy."""
        if not self.evaluated:
            raise PopulationNotEvaluatedError("Tried to get the best program without evaluation of fitness!")
        return self.population[0]

    def register_shapes(self, register: str):
        """Return register shape given register name"""
        return self.register_shapes_dict[register]

    def to_dict(self):
        """Create JSON-dumpable dictionary with data needed for further analysis"""
        return {
            "UNARY": [U.__name__ for U in UNARY],
            "BINARY": [B.__name__ for B in BINARY],
            "AREA": [A.__name__ for A in AREA],
            "area-p": self.area_instruction_p,
            "crossover-p": self.crossover_p,
            "mutation-p": self.mutation_p,
            "grow-p": self.program_grow_p,
            "shrink-p": self.shrink_p,
            "elite-size": self.elite_size,
            "elite-equal": self.equal_elite,
            "fitness-fn": self.fitness_fn.__name__,
            "hidden-reg-shape": self.hidden_register_shape,
            "generations": self.generations,
            "max-i": self.max_instructions,
            "min-i": self.min_instructions,
            "mutate-registers": self.mutate_registers,
            "mutate-instructions": self.mutate_instructions,
            "max-population": self.population_bound,
        }

    def __repr__(self) -> str:
        """String representation of program and it's instructions (human friendly)"""
        return pformat(self.to_dict(), width=120)
