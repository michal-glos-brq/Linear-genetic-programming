#######################
# @$%&           &%$@ #
#!    Michal Glos    !#
#!     EVO - 2023    !#
#!        __         !#
#!      <(o )___     !#
#!       ( ._> /     !#
# @$%&     `---' &%$@ #
#######################


from pprint import pformat
from numbers import Number
from typing import Tuple, Iterable, Union, Dict

import torch
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


################################################################################
#####                       Fitness functions                              #####
################################################################################
#####      Fitness functions take 2 args: predicted labels, gt labels      #####


def success_rate(y_pred: torch.Tensor, y_gt: torch.Tensor) -> Number:
    """Compute the percentage of correctly predicted classes"""
    predictions = y_pred == y_gt
    # Return the percentage of correct predictions
    return ((predictions.sum() / len(predictions)) * 100).item()


# TODO: Implement cross-entropy with fuzzy results

# Mapping os tring to funtions
FITNESS = {"p": success_rate}

################################################################################
#####                      LGP algorithm class                             #####
################################################################################


class LGP:
    """
    Linear Genetic Programming controll algorithm/object

    important properties: (see __init__ for comments next to property declaration/definition)
        population:         NumPy array of Program instances (the whole population)
        population_bound:   Maximum number of individuals in population
        elite_size:         Number of individuals consisting elite
        hidden_reg_shape:   Shape of hidden registers
        fitness_fn:         Function computing fitness
        fitness:            Fitness ranking of individuals
        evaluated:          Current population was evaluated nad fitness values are current
    """

    def __init__(
        self,
        dataset: Dataset,
        program: Program,
        population: int = 42,
        elite: int = 3,
        equal_elite: bool = False,
        generations: int = 60,
        min_inst: int = 1,
        max_inst: int = 20,
        fitness: str = "p",
        hidden_reg_shape: Tuple[int] = (42,),
        area_p: int = 10,
        grow_p: int = 25,
        mutation_p: int = 25,
        crossover_p: int = 25,
        mutate_inst: int = 1,
        mutate_reg: int = 1,
    ) -> None:
        """
        Initialize LGP algorithm object

        @args:
            population:         Upper population bound
            elite:              Number of best individuals considered elite
            equal_elite:        Do not weight elite individuals chances to procreate by fitness
            generations:        For how many generations to evolve
            program:            Program instance or None (Proto Program ... :D )
            min_inst:           Minimum required lenght of program in instructions
            max_inst:           Maximum required lenght of program in instructions
            fitness:            Fitness function (choose from 'ce' and 'p'), see Program class
            dataset:            Dataset instance
            grow_p:             Incrementaly increase the lenght of program (in %)
            mutation_p:         Mutation probability (in %)
            crossover_p:        Crossover probability (in %)
            hidden_reg_shape:   Shape of hidden register field
            mutate_inst:        How many instructions to mutate
            mutate_reg:         How many registers to mutate
        """
        # Save the configuration into properties
        self.population_bound = population  # Max population
        # Initialize either empty population or single individual "population"
        self.population = np.array([program] if program else [])

        self.generations = generations  # How many generation to evolve
        self.actual_generation = 0  # Generation evolving right now

        self.grow_p = grow_p / 100.0  # Probability (in %) for a program to grow_p (an instruction)
        self.min_inst = min_inst  # Minimal number of Program instructions
        self.max_inst = max_inst  # Maximal number of Program instructions
        self.hidden_reg_shape = hidden_reg_shape  # Shape of hidden register field

        self.fitness_fn = FITNESS[fitness]  # Fitness function
        self.fitness = []  # Fitness values for each individual

        self.obj_shape = dataset.obj_shape  # Shape of data object
        self.classes = len(dataset.classes)  # How many classes to discriminate
        self.device = dataset.device  # Tensor device

        self.mutation_p = mutation_p / 100.0  # Main GA parameters, mutation and
        self.crossover_p = crossover_p / 100.0  # crossover probabilities
        self.area_p = area_p / 100.0  # Probability of area-processing instruction
        self.elite_size = elite  # Size of elite to keep to another gen
        self.equal_elite = equal_elite  # Distribute elite equally (not proportionally to fitness) when repopulating
        self.mutate_reg = mutate_reg  # Registers to mutate WHEN mutation occurs
        self.mutate_inst = mutate_inst  # Instructions to mutate WHEN mutation occurs

        # Other utility configuration
        self.evaluated = False  # Indicator whether the population was evaluated

    ################################################################################
    #####                     Adding on population                             #####
    ################################################################################

    def spawn_individuals(self) -> None:
        """Generate Program instances to meet the population criteria"""
        # How many individuals are missing from population
        pop_missing = self.population_bound - len(self.population)
        # Create new individuals
        self.population = np.append(self.population, [self.new_individual() for _ in range(pop_missing)])
        # Invalidate population evaluation, since new individuals were added
        # fn new_individual requires fitness, therefore invalidate it at the end
        self.evaluated = False

    def new_individual(self) -> Program:
        """Create a new Program individual based on LGP object properties"""
        # If population was not evaluated, we do not have elite, generate individuals randomly
        # Mutation not even considered because it's randomly generated at the first place
        if not self.evaluated:
            return Program.creation(self)

        # If population was evaluated, let's take the elite and repopulate
        elite_pop, elite_fitness = self.elite
        # Use fitness values as probability distribution of individuals or
        # use uniform distribution among elite if self.equal_elite
        if self.equal_elite:
            elite_probs = np.ones_like(elite_fitness) / len(elite_pop)
        else:
            elite_probs = np.array(elite_fitness) / sum(elite_fitness)

        # Decide which actions to perform. If neither, program will be shrunk (it's already in the new population, so the new one gotta change a bit)
        crossover, mutate, grow_p = self.repopulation_probs >= np.random.random(3)

        ## Actual process of generating the new offspring begins here
        # Create new offspring either by crossover of 2 parents or by simple selection of elite individual
        if crossover:
            father, mother = np.random.choice(elite_pop, size=2, replace=False, p=elite_probs)
            offspring = Program.crossover(father, mother)
        else:
            parent = np.random.choice(elite_pop, p=elite_probs)
            offspring = Program.transcription(parent)

        # Mutate if randomly chosen to
        if mutate:
            offspring.mutate(self.mutate_reg, self.mutate_inst)

        # Grow_p if randomly chosen to
        if grow_p:
            offspring.grow()

        # When neither of following actions was chosen, let's delete individual's
        # random instruction to decrease the probability of the same programs in population
        if not crossover and not mutate and not grow_p:
            offspring.shrink()

        # Finally return the newborn
        return offspring

    @property
    def repopulation_probs(self) -> Iterable[Number]:
        """Return numpy array of crossover, mutation and grow_pth probabilities"""
        return np.array([self.crossover_p, self.mutation_p, self.grow_p])

    ################################################################################
    #####             Evaluating and eliminating population                    #####
    ################################################################################

    def evaluate_population(self, X: torch.Tensor, y_gt: torch.Tensor) -> None:
        """
        Evaluate the whole population of programs and sort it from best to worse

        This method is used for evolving and evaluating, therefore data are passed as arguments

        @args:
            X:      Data Tensor
            y_gt:   Labels Tensor
        """
        # Rank each program with it's fitness
        self.fitness = np.array([individual.eval(X, y_gt, self.fitness_fn) for individual in self.population]).reshape(
            -1
        )
        # Sort population according to fitness (Best first)
        self.population = self.population[np.argsort(self.fitness)[::-1]]
        # Once we sorted the population, sort fitness values to weight individual
        self.fitness.sort()
        self.fitness = self.fitness[::-1]
        self.evaluated = True

    def eliminate(self) -> None:
        """Eliminate inferior Programs"""
        # Delete non-elite programs and retain only the elite
        delete_programs = self.population[self.elite_size :]
        for delete_program in delete_programs:
            del delete_program
        self.population = self.population[: self.elite_size]

    ################################################################################
    #####                     LGP algorithm controll                           #####
    ################################################################################

    def fit(self, X: torch.Tensor, y_gt: torch.Tensor) -> None:
        """
        Execute the evolution in the name of The God itself

        @args:
            X:      Training dataset (data)
            y_gt:   Training dataset (labels)
        """
        with tqdm(range(self.generations), desc="Evolving ...") as pbar:
            for _ in pbar:
                # Update generation iteration
                self.actual_generation += 1
                # Fill the whole population with individuals
                self.spawn_individuals()
                # Rank them with fitness function
                self.evaluate_population(X, y_gt)
                # Keep only the elite
                self.eliminate()
                # Repeat till the last generation, update the progress bar
                pbar.set_description(f"Evolving ... ({self.fitness[0]} %)")

    ################################################################################
    #####                                Utils                                 #####
    ################################################################################

    @property
    def elite(self) -> Tuple[Union[Iterable[Program], Iterable[Number]]]:
        """Return the array of elite individuals and theirs fitness"""
        return self.population[: self.elite_size], self.fitness[: self.elite_size]

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
                "Generations evolved": self.actual_generation,
            },
            "Programs": {
                "Max instructions": self.max_inst,
                "Min instructions": self.min_inst,
                "Probability of program mutation": self.mutation_p,
                "Probability of programs crossover": self.crossover_p,
                "Probability of program growing": self.grow_p,
                "Probability of areal data processing": self.area_p,
            },
            "Fitness": {"function": self.fitness_fn.__name__, "values": self.fitness, "fitness valid": self.evaluated},
        }

    def __repr__(self) -> str:
        """String representation of program and it's instructions (developer friendly)"""
        return pformat(self._info_dict, width=120)
