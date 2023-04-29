"""
This module provides implementation of linear program - solution to Linear Genetic Programming.

This class implements all necessary genetic operations: crossover, mutation and transcription.
It also implements program evaluation.
"""

import pickle
from math import prod
from numbers import Number
from pprint import pformat
from random import random, randint
from typing import Iterable, Union, Callable, Dict, Tuple

import torch

from LGA.instruction import Instruction
from utils.other import InstructionIndexError, ProgramRegisterShapeError

TENSOR_FACOTRY = (torch.cuda if torch.cuda.is_available() else torch)


# pylint: disable=too-many-arguments
class Program:
    """
    Representation of linear program - a list of Instruction instances
    
    Attributes:
        instructions (List[Instruction]): List of linear instructions.
        input_register_shape (Iterable[int]): Shape of input register field.
        hidden_register_shape (Iterable[int]): Shape of hidden (working) memory.
        hidden_register_initial_values (torch.Tensor): Initialization values of hidden (working) memory.
        result_register_shape (Iterable[int]): Shape of result vector memory.
        result_register_initial_values (torch.Tensor): Initialization values of result vector memory.
        register_shapes (Dict[str, Tuple[int]]): Mapping of register names to their shapes.
    """
    # Linear Genetic Algorithm
    lga = None

    def __init__(
        self,
        min_instructions: int,
        max_instructions: int,
        object_shape: Iterable[int],
        result_register_shape: Iterable[int],
        hidden_register_shape: Iterable[int],
        area_instruction_p: Number,
        torch_device: Number,
    ) -> None:
        """
        Initialize the Program class, create register initial states and instructions

        Args:
            max_instructions (int): Maximal length of program (number of instructions).
            min_instructions (int): Minimal length of program (number of instructions).
            object_shape (Iterable[int]): Shape of objects to be classified.
            result_register_shape (Iterable[int]): Output register shape / number of classes.
            hidden_register_shape (Iterable[int]): Shape of hidden register field.
            area_instruction_p (Number): Probability of instruction to process a slice instead of a single value.
            torch_device (torch.device): Pytorch device to store tensors.
        """
        self.min_instructions = min_instructions
        self.max_instructions = max_instructions

        self.input_register_shape = object_shape
        self.hidden_register_shape = hidden_register_shape
        self.result_register_shape = result_register_shape

        self.hidden_register_initial_values = None
        self.result_register_initial_values = None
        self.input_registers = None
        self.result_registers = None
        self.hidden_registers = None

        self.instructions = []

        self.torch_device = torch_device
        self.area_instruction_p = area_instruction_p

        self.register_shapes = {
            "input_registers": self.input_register_shape,
            "hidden_registers": self.hidden_register_shape,
            "result_registers": self.result_register_shape,
        }

        self.hidden_register_count = prod(hidden_register_shape)
        self.result_register_count = prod(result_register_shape)

    @staticmethod
    def load_program(path: str, reg_shape: Tuple[int]) -> 'Program':
        """
        Load program from pickled file and assert configuration \
        complience with current dataset configuration
        Args:
            path:   Path to pickled Program instance
        """
        with open(path, "rb") as file:
            program = pickle.load(file)

        if not program.hidden_regfiled.shape == reg_shape:
            raise ProgramRegisterShapeError(
                f"Loaded programs hidden register fields ({', '.join(str(dim) for dim in program.hidden_regfiled.shape)})"
                f" has to be equal to CLI option provided ({', '.join(str(dim) for dim in reg_shape)})!"
            )

        return program

    @staticmethod
    def create_random_program(lga) -> "Program":
        """
        Create random individual with no ancestors

        Args:
            lga (Program): LGP object instance
        Returns:
            Program: Program instance with random genome
        """
        new_individual = Program(
            lga.max_instructions,
            lga.min_instructions,
            lga.object_shape,
            (lga.num_of_classes,),
            lga.hidden_register_shape,
            lga.area_instruction_p,
            lga.torch_device,
        )

        new_individual.hidden_register_initial_values = torch.normal(
            0, 1, new_individual.hidden_register_shape, device=lga.torch_device
        )
        new_individual.result_register_initial_values = torch.normal(
            0, 1, new_individual.result_register_shape, device=lga.torch_device
        )

        new_individual.instructions = [Instruction.random(new_individual) for _ in range(new_individual.max_instructions)]

        return new_individual

    @staticmethod
    def crossover(mother: "Program", father: "Program") -> "Program":
        """
        Perform crossover of 2 parents, do not mutate yet

        Args:
            mother (Program): First parent
            father (Program): Second parent
        Returns:
            offspring (Program): New program made from mother and father programs
        """
        new_individual = Program(
            mother.min_instructions,
            mother.max_instructions,
            mother.input_register_shape,
            mother.result_register_shape,
            mother.hidden_register_shape,
            mother.area_instruction_p,
            mother.torch_device,
        )

        # Crossover of registers
        result_register_mask = TENSOR_FACOTRY.FloatTensor(*new_individual.result_register_shape).uniform_() < 0.5
        hidden_register_mask = TENSOR_FACOTRY.FloatTensor(*new_individual.hidden_register_shape).uniform_() < 0.5

        new_individual.result_register_initial_values = (result_register_mask * mother.result_register_initial_values) + (
            ~result_register_mask * father.result_register_initial_values
        )
        new_individual.hidden_register_initial_values = (hidden_register_mask * mother.hidden_register_initial_values) + (
            ~hidden_register_mask * father.hidden_register_initial_values
        )

        # Crossover of instructions
        instruction_ratio = random()
        mother_crossover_index = int(instruction_ratio * len(mother.instructions))
        father_crossover_index = int(instruction_ratio * len(father.instructions))

        new_instruction_list = mother.instructions[:mother_crossover_index] + father.instructions[father_crossover_index:]
        new_individual.instructions = [instr.copy(new_individual) for instr in new_instruction_list]

        return new_individual

    @staticmethod
    def transcription(parent: "Program") -> "Program":
        """
        Copy parents genome to it's offspring, no mutations yet

        Args:
            parent (Program): Program instance
        Return:
            offspring (Program): New program instance equivalent to it's parent
        """
        new_individual = Program(
            parent.min_instructions,
            parent.max_instructions,
            parent.input_register_shape,
            parent.result_register_shape,
            parent.hidden_register_shape,
            parent.area_instruction_p,
            parent.torch_device,
        )

        new_individual.result_register_initial_values = parent.result_register_initial_values.clone()
        new_individual.hidden_register_initial_values = parent.hidden_register_initial_values.clone()

        new_individual.instructions = [instr.copy(new_individual) for instr in parent.instructions]

        return new_individual


    def delete_instruction(self, idx: Union[None, int] = None) -> None:
        """Delete instruction on index idx, delete random instruction if idx is None"""
        if idx is None:
            idx = randint(0, len(self.instructions) - 1)
        elif not 0 <= idx < len(self.instructions):
            raise InstructionIndexError(
                f"Tried to delete instruction on index {idx} from list of {len(self.instructions)} instructions!"
            )

        del self.instructions[idx]

    def grow(self) -> None:
        """Add random instruction. Also remove another when the max instruction limit would be exceeded"""
        if len(self.instructions) >= self.min_instructions:
            self.delete_instruction()
        self.instructions.append(Instruction.random(self))

    def shrink(self) -> None:
        """Delete random instruction. Also add another when the min instruction limit would be exceeded"""
        self.delete_instruction()
        if len(self.instructions) < self.max_instructions:
            self.instructions.append(Instruction.random(self))

    # pylint: disable=invalid-name
    def evaluate(
        self, X: torch.Tensor, y_gt: torch.Tensor, fitness_fn: Callable[[torch.Tensor, torch.Tensor], Number]
    ) -> torch.Tensor:
        """
        Evaluate the program on the given dataset

        Args:
            X (torch.Tensor): Tensor of objects to classify.
            y_gt (torch.Tensor): Tensor of ground truth labels.
            fitness_fn (Callable[[torch.Tensor, torch.Tensor], Number]): Fitness function.
        Returns:
            torch.Tensor: Fitness score
        """
        self.input_registers = X    # Programs can't write into input register, therfore no cloning

        self.hidden_registers = self.hidden_register_initial_values.repeat(
            X.size(0), *(len(self.hidden_register_initial_values.shape) * [1])
        )
        self.result_registers = self.result_register_initial_values.repeat(X.size(0), 1)

        for instruction in self.instructions:
            instruction.execute()

        result = fitness_fn(self.result_registers, y_gt)

        del self.hidden_registers
        del self.result_registers

        return result


    @property
    def _info_dict(self) -> Dict[str, Dict]:
        """Obtain information about Program instance in a dictionary"""
        return {
            "Hidden register field": {
                "shape": self.hidden_register_shape,
                "init_values": self.hidden_register_initial_values,
            },
            "Result register field": {
                "shape": self.result_register_shape,
                "init_values": self.result_register_initial_values,
            },
            "Instructions": [f"{i}: {instr}" for i, instr in enumerate(self.instructions, 1)],
        }

    def __repr__(self) -> str:
        """String representation of program and it's instructions (developer friendly)"""
        return pformat(self._info_dict, width=116)
