"""
This module provides an implementation of a Linear Genetic Programming (LGP) algorithm, which evolves a population of
programs through multiple generations.

The main class, LGP, is responsible for handling the population, fitness
evaluation, and the evolution process. The LGP class supports spawning, mutation, crossover, and elimination
methods for evolving the programs. It also contains methods for logging the current state of the population and
pickling the best program in each generation.
"""

from pickle import dump
from pprint import pformat
from numbers import Number
from os.path import join, isfile
from typing import Tuple, Iterable, Union, Dict

import torch
from torch.nn import CrossEntropyLoss
import numpy as np
from tqdm import tqdm

# That's the best solution i got for importing modules correctly from wherever (not really)
# I'm not satisfied though but this covers all my use cases even with interactive python
try:
    from src.program import Program
    from src.datasets import Dataset
except:
    from program import Program
    from datasets import Dataset

# This setting makes torch print full content of tensors
torch.set_printoptions(profile="full")

# Exceptions
class PopulationNotEvaluatedError(Exception):
    pass

def success_rate(result_registers: torch.Tensor, gt_labels: torch.Tensor) -> float:
    """
    Compute the percentage of correctly predicted classes.

    Args:
        result_registers (torch.Tensor): Tensor containing the model's output.
        gt_labels (torch.Tensor): Ground truth labels.

    Returns:
        float: The percentage of correct predictions.
    """
    # Collapse the classification probabilities and calculate accuracy
    predicted_labels = result_registers.argmax(dim=1)
    predictions = predicted_labels == gt_labels
    # Return the percentage of correct predictions
    return ((predictions.sum() / len(predictions)) * 100).item()

def cross_entropy(result_registers: torch.Tensor, gt_labels: torch.Tensor) -> float:
    """
    Compute the percentage of correctly predicted classes.

    Args:
        result_registers (torch.Tensor): Tensor containing the model's output.
        gt_labels (torch.Tensor): Ground truth labels.

    Returns:
        float: Cross-entropy result.
    """
    # Normalize result registers with softmax
    return CrossEntropyLoss()(result_registers.type(torch.double), gt_labels.type(torch.long)).item()


# Mapping os tring to funtions
FITNESS_FUNCTIONS = {
    "p": success_rate,
    "ce": cross_entropy
}


class LGP:
    """
    Linear Genetic Programming controll algorithm/object

    Attributes:
        dataset (Dataset): The dataset used for training.
        program (Program): The initial program in the population.
        population_bound (int): The maximum size of the population.
        generations (int): The number of generations to evolve.
        current_generation (int): The current generation number.
        grow_p (float): Probability for a program to grow an instruction.
        min_instructions (int): The minimum number of program instructions.
        max_instructions (int): The maximum number of program instructions.
        hidden_register_shape (Tuple[int]): Shape of hidden register field.
        fitness_fn (function): The fitness function used to evaluate the programs.
        fitness (List[float]): Fitness values for each individual.
        object_shape (Tuple[int]): Shape of the data object.
        classes (int): The number of classes to discriminate.
        torch_device (torch.device): The torch device to be used.
        mutation_p (float): The mutation probability.
        crossover_p (float): The crossover probability.
        area_instruction_p (float): Probability of area-processing instruction.
        elite_size (int): Size of the elite to keep for the next generation.
        equal_elite (bool): If True, distribute elite equally (not proportionally to fitness) when repopulating.
        mutate_registers (int): Registers to mutate when mutation occurs.
        mutate_instructions (int): Instructions to mutate when mutation occurs.
        model_dir (str): Path to directory where models will be saved.
        logging_dir (str): Path to directory where training progress will be documented.
        evaluated (bool): Indicator whether the population was evaluated.
    """

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
        fitness: str = "p",
        hidden_register_shape: Tuple[int] = (42,),
        area_instruction_p: int = 10,
        grow_p: int = 25,
        mutation_p: int = 25,
        crossover_p: int = 25,
        mutate_instructions: int = 1,
        mutate_registers: int = 1,
        model_dir: str = "models",
        logging_dir: str = "logging",
    ) -> None:
        """
        Initialize LGP algorithm object.

        Args:
            dataset (Dataset): Dataset object.
            program (Program): Program object.
            population (int, optional): Population size. Defaults to 42.
            elite (int, optional): Elite size. Defaults to 3.
            equal_elite (bool, optional): Whether to distribute elite equally. Defaults to False.
            generations (int, optional): Number of generations to evolve. Defaults to 60.
            min_instructions (int, optional): Minimal number of program instructions. Defaults to 1.
            max_instructions (int, optional): Maximal number of program instructions. Defaults to 20.
            fitness (str, optional): Fitness function identifier. Defaults to "p".
            hidden_register_shape (Tuple[int], optional): Shape of hidden register field. Defaults to (42,).
            area_instruction_p (int, optional): Probability of area-processing instruction. Defaults to 10.
            grow_p (int, optional): Probability for a program to grow (an instruction). Defaults to 25.
            mutation_p (int, optional): Probability for a program to mutate. Defaults to 25.
            crossover_p (int, optional): Probability for a program to crossover. Defaults to 25.
            mutate_instructions (int, optional): Instructions to mutate WHEN mutation occurs. Defaults to 1.
            mutate_registers (int, optional): Registers to mutate WHEN mutation occurs. Defaults to 1.
            model_dir (str, optional): Path to the directory for saving models. Defaults to "models".
            logging_dir (str, optional): Path to the directory for logging. Defaults to "logging".
        """
        # Save the configuration into properties
        self.population_bound = population  # Max population
        # Initialize either empty population or single individual "population"
        self.population = np.array([program] if program else [])

        self.generations = generations  # How many generation to evolve
        self.current_generation = 0  # Generation evolving right now

        self.grow_p = grow_p / 100.0  # Probability (in %) for a program to grow_p (an instruction)
        self.min_instructions = min_instructions  # Minimal number of Program instructions
        self.max_instructions = max_instructions  # Maximal number of Program instructions
        self.hidden_register_shape = hidden_register_shape  # Shape of hidden register field

        self.fitness_fn = FITNESS_FUNCTIONS[fitness]  # Fitness function
        self.evaluated_fitness = []  # Fitness values for each individual

        self.object_shape = dataset.object_shape  # Shape of data object
        self.num_of_classes = len(dataset.classes)  # How many classes to discriminate
        self.torch_device = dataset.torch_device  # Tensor torch_device

        self.mutation_p = mutation_p / 100.0  # Main GA parameters, mutation and
        self.crossover_p = crossover_p / 100.0  # crossover probabilities
        self.area_instruction_p = area_instruction_p / 100.0  # Probability of area-processing instruction
        self.elite_size = elite  # Size of elite to keep to another gen
        self.equal_elite = equal_elite  # Distribute elite equally (not proportionally to fitness) when repopulating
        self.mutate_registers = mutate_registers  # Registers to mutate WHEN mutation occurs
        self.mutate_instructions = mutate_instructions  # Instructions to mutate WHEN mutation occurs

        # Logging and other utility configuration
        self.dataset_name = dataset.dataset_name
        self.dataset_edge_size = dataset.edge_size
        self.model_dir = model_dir
        self.logging_dir = logging_dir
        self.evaluated = False  # Indicator whether the population was evaluated

    def spawn_individuals(self) -> None:
        """Generate Program instances to meet the population criteria"""
        # How many individuals are missing from population
        population_missing = self.population_bound - len(self.population)
        # Create new individuals
        self.population = np.append(self.population, [self.new_individual() for _ in range(population_missing)])
        # Invalidate population evaluation, since new individuals were added
        # fn new_individual requires fitness, therefore invalidate it at the end
        self.evaluated = False

    def new_individual(self) -> Program:
        """
        Create a new Program individual based on LGP object properties

        Returns:
            Program: The new Program instance
        """
        # If population was not evaluated, we do not have elite, generate individuals randomly
        # Mutation not even considered because it's randomly generated at the first place
        if not self.evaluated:
            return Program.creation(self)

        # If population was evaluated, let's take the elite and repopulate
        elite_population, elite_fitness = self.elite
        # Use fitness values as probability distribution of individuals or
        # use uniform distribution among elite if self.equal_elite
        if self.equal_elite:
            elite_distribution_probability = np.ones_like(elite_fitness) / len(elite_population)
        else:
            elite_distribution_probability = np.array(elite_fitness) / sum(elite_fitness)

        # Decide which actions to perform. If neither, program will be shrunk
        # (it's already in the new population, so the new one gotta change a bit)
        crossover, mutate, grow = self.repopulation_probs >= np.random.random(3)

        ## Actual process of generating the new offspring begins here
        # Create new offspring either by crossover of 2 parents or by simple selection of elite individual
        if crossover:
            father, mother = np.random.choice(elite_population, size=2, replace=False, p=elite_distribution_probability)
            offspring = Program.crossover(father, mother)
        else:
            parent = np.random.choice(elite_population, p=elite_distribution_probability)
            offspring = Program.transcription(parent)

        # Mutate if randomly chosen to
        if mutate:
            offspring.mutate(self.mutate_registers, self.mutate_instructions)

        # Grow_p if randomly chosen to
        if grow:
            offspring.grow()

        # When neither of following actions was chosen, let's delete individual's
        # random instruction to decrease the probability of the same programs in population
        if not crossover and not mutate and not grow:
            offspring.shrink()

        # Finally return the newborn
        return offspring

    @property
    def repopulation_probs(self) -> Iterable[Number]:
        """Return numpy array of crossover, mutation and grow_pth probabilities"""
        return np.array([self.crossover_p, self.mutation_p, self.grow_p])

    def evaluate_population(self, in_data: torch.Tensor, gt_labels: torch.Tensor, use_percent: bool = False) -> None:
        """
        Evaluate the whole population of programs and sort it from best to worse

        This method is used for evolving and evaluating, therefore data are passed as arguments

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
        # Sort population according to fitness (Best first)
        self.population = self.population[np.argsort(self.evaluated_fitness)[::-1]]
        # Once we sorted the population, sort fitness values to weight individual
        self.evaluated_fitness.sort()
        self.evaluated_fitness = self.evaluated_fitness[::-1]
        self.evaluated = True

    def eliminate(self) -> None:
        """Eliminate inferior Programs"""
        # Delete non-elite programs and retain only the elite
        delete_programs = self.population[self.elite_size :]
        for delete_program in delete_programs:
            del delete_program
        self.population = self.population[: self.elite_size]

    def fit(self, in_data: torch.Tensor, gt_labels: torch.Tensor, model_dir: str, dump_dir: str) -> None:
        """
        Execute the evolution in the name of The God itself

        Args:
            in_data (torch.Tensor): Training dataset (data).
            gt_labels (torch.Tensor): Training dataset (labels).
            model_dir (str): Directory to save the best model.
            dump_dir (str): Directory to save the evolution log.
        """
        with tqdm(range(self.generations), desc="Evolving ...") as pbar:
            for _ in pbar:
                # Update generation iteration
                self.current_generation += 1
                # Fill the whole population with individuals
                self.spawn_individuals()
                # Evaluate the population
                self.evaluate_population(in_data, gt_labels)
                # Save best individual
                self.pickle_best_program()
                # Log actual generation
                self.log_generation()
                # Eliminate all individuals but the elite
                self.eliminate()
                # Update progress bar description
                pbar.set_description(f"Evolving ... ({self.evaluated_fitness[0]} %)")

    def generate_unique_filename(self, suffix: str = "p") -> str:
        """
        Generate unique filename with provided suffix

        Args:
            suffix (str): Desired file suffix.
        Returns:
            str: Unique path to a file with suffix suffix :)
        """
        # Create pseudo-uniqe name for current LGP config
        unique_name = f"{self.dataset_name}_{self.dataset_edge_size}_{self.current_generation}_{self.evaluated_fitness[0]}"

        # If the file already exists, generate unique path with appending a number
        if isfile(f"{unique_name}.{suffix}"):
            pre_suffix_number = 1
            while isfile():
                pre_suffix_number += 1
            return f"{unique_name}_{pre_suffix_number}.{suffix}"
        else:
            return f"{unique_name}.{suffix}"

    def pickle_best_program(self):
        """Pickle best program into unique file"""
        # Get unique path for current best model
        filename = self.generate_unique_filename()
        pickle_path = join(self.model_dir, filename)
        with open(pickle_path, "wb") as file:
            dump(self.best_program, file)

    def log_generation(self):
        """Log population of actual generation into unique log file"""
        # Get unique path for current generation logging
        filename = self.generate_unique_filename(suffix="log")
        logging_path = join(self.logging_dir, filename)
        with open(logging_path, "w") as file:
            file.write(self._generation_log)

    @property
    def _generation_log(self) -> str:
        """Create log of current generation and LGP state"""
        return f"{self}{[program for program in self.population]}"

    @property
    def best_program(self) -> Program:
        """Return program from population with highest accuracy"""
        if not self.evaluated:
            raise PopulationNotEvaluatedError("Tried to get the best program without evaluation of fitness!")
        return self.population[0]

    @property
    def elite(self) -> Tuple[Union[Iterable[Program], Iterable[Number]]]:
        """Return the array of elite individuals and theirs fitness"""
        return self.population[: self.elite_size], self.evaluated_fitness[: self.elite_size]

    @property
    def _info_dict(self) -> Dict[str, Dict]:
        """Obtain information about Program instance in a dictionary"""
        return {
            "Population": {
                "size": len(self.population),
                "max": self.population_bound,
                "elite size": self.elite_size,
                "elite equal (sample uniformly)": self.equal_elite,
                "Generations to evolve": self.generations,
                "Generations evolved": self.current_generation,
            },
            "Programs": {
                "Max instructions": self.max_instructions,
                "Min instructions": self.min_instructions,
                "Probability of program mutation": self.mutation_p,
                "Probability of programs crossover": self.crossover_p,
                "Probability of program growing": self.grow_p,
                "Probability of areal data processing": self.area_instruction_p,
            },
            "Fitness": {
                "function": self.fitness_fn.__name__,
                "values": self.evaluated_fitness,
                "fitness valid": self.evaluated,
            },
        }

    def __repr__(self) -> str:
        """String representation of program and it's instructions (developer friendly)"""
        return pformat(self._info_dict, width=120)
