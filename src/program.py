"""
This module provides implementation of linear program - solution to Linear Genetic Programming.

This class implements all necessary genetic operations: crossover, mutation and transcription.
It also implements program evaluation.
"""


import torch
import numpy as np

from pprint import pformat
from typing import Iterable, Union, Callable, Dict
from numbers import Number

# That's the best solution i got for importing modules correctly from wherever (not really)
# I'm not satisfied though but this covers all my use cases even with interactive python
try:
    from src.instruction import Instruction
except:
    from instruction import Instruction


class InstructionIndexError(Exception):
    pass


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
    """

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
        Initialize the Program class, initialize register init values,
        initialize instructions

        Args:
            max_instructions (int): Maximal length of program (number of instructions).
            min_instructions (int): Minimal length of program (number of instructions).
            object_shape (Iterable[int]): Shape of objects to be classified.
            result_register_shape (Iterable[int]): Output register shape / number of classes.
            hidden_register_shape (Iterable[int]): Shape of hidden register field.
            area_instruction_p (Number): Probability of instruction to process a slice instead of a single value.
            torch_device (torch.device): Pytorch device to store tensors.
        """
        # Define max and min length of program
        self.min_instructions = min_instructions
        self.max_instructions = max_instructions

        # Obtain dimensions of register fields (Used in Instruction.choose_operands)
        self.input_register_shape = object_shape
        self.hidden_register_shape = hidden_register_shape
        self.result_register_shape = result_register_shape

        # Declare register init values tensors, would be set in one of classmethods
        self.hidden_register_initial_values = None
        self.result_register_initial_values = None

        # Declare the instruction field
        self.instructions = []

        self.torch_device = torch_device
        self.area_instruction_p = area_instruction_p

    @staticmethod
    def creation(lgp) -> "Program":
        """
        Create random individual with no ancestors

        Args
            lgp (Program): LGP object instance
        """
        # Create the new individual
        new_individual = Program(
            lgp.max_instructions,
            lgp.min_instructions,
            lgp.object_shape,
            (lgp.classes,),
            lgp.hidden_register_shape,
            lgp.area_instruction_p,
            lgp.torch_device,
        )

        # Generate initial values of register fields
        new_individual.hidden_register_initial_values = torch.normal(0, 1, new_individual.hidden_register_shape)
        new_individual.result_register_initial_values = torch.normal(0, 1, new_individual.result_register_shape)
        # Move to used torch_device
        new_individual.hidden_register_initial_values = new_individual.hidden_register_initial_values.to(
            new_individual.torch_device
        )
        new_individual.result_register_initial_values = new_individual.result_register_initial_values.to(
            new_individual.torch_device
        )

        # Generate the shortest possible program randomly
        for _ in range(new_individual.max_instructions):
            new_individual.instructions.append(Instruction.random(new_individual))

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
        # Create the new individual
        new_individual = Program(
            mother.min_instructions,
            mother.max_instructions,
            mother.input_register_shape,
            mother.result_register_shape,
            mother.hidden_register_shape,
            mother.area_instruction_p,
            mother.torch_device,
        )

        # Crossover of registers (random access crossover)
        # Create masks for result and hidden registers (apprx. 50% is False, 50% True)
        result_register_mask = (torch.randint(0, 2, new_individual.result_register_shape) == 1).to(
            new_individual.torch_device
        )
        hidden_register_mask = (torch.randint(0, 2, new_individual.hidden_register_shape) == 1).to(
            new_individual.torch_device
        )
        # Compose the offspring register initialization tensors
        new_individual.result_register_initial_values = (result_register_mask * mother.result_register_initial_values) + (
            ~result_register_mask * father.result_register_initial_values
        )
        new_individual.hidden_register_initial_values = (hidden_register_mask * mother.hidden_register_initial_values) + (
            ~hidden_register_mask * father.hidden_register_initial_values
        )

        # Now perform the instruction crossover
        # Ratio - how many (in fraction) instruction would be taken from mother and (1-ratio) from father
        instruction_ratio = np.random.random()
        # Index of crossover into mothers instructions
        mother_crossover_index = int(instruction_ratio * len(mother.instructions))
        # Index of crossover into fathers instructions
        father_crossover_index = int(instruction_ratio * len(father.instructions))
        # Now just slice parents instruction arrays and concatenate them into their offspring
        new_instruction_list = mother.instructions[:mother_crossover_index] + father.instructions[father_crossover_index:]
        # Copy the new genome into the new individual (instructions are bound to parents)
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
        # Create the new individual
        new_individual = Program(
            parent.min_instructions,
            parent.max_instructions,
            parent.input_register_shape,
            parent.result_register_shape,
            parent.hidden_register_shape,
            parent.area_instruction_p,
            parent.torch_device,
        )

        # Copy the register field init values
        new_individual.result_register_initial_values = parent.result_register_initial_values.clone()
        new_individual.hidden_register_initial_values = parent.hidden_register_initial_values.clone()

        # Copy the instructions from parent
        new_individual.instructions = [instr.copy(new_individual) for instr in parent.instructions]

        return new_individual

    def mutate(self, registers_to_mutate: int, instructions_to_mutate: int) -> None:
        """
        Perform program mutation

        Mutation is performed in both: register init values and instructions
            - registers get values added sampled from normal distribution (0,1)
            - instructions are instruction related, mutation command is passed upon instruction

        Args:
            registers_to_mutate (int):  How many registers to mutate?
            instructions_to_mutate (int):   How many instructions to mutate?
        """
        # Calculate how many mutations would be performed on hidden registers
        register_sizes = torch.FloatTensor([self._hid_regs, self._res_regs])
        hid_reg_mutations = register_sizes.multinomial(registers_to_mutate, replacement=True).sum()
        res_reg_mutations = registers_to_mutate - hid_reg_mutations

        # Perform register mutations (This gets hacky ... makes perfect sense though)
        ## [torch.randperm(self._hid_regs)[:hid_reg_mutations]] - This indexes hid_reg_mutations
        ## random elements of the Tensor without repeating, adds values sampled from normal dist.
        self.hidden_register_initial_values.view(-1)[torch.randperm(self._hid_regs)[:hid_reg_mutations]] += torch.normal(
            0, 1, (hid_reg_mutations,)
        ).to(self.torch_device)
        self.result_register_initial_values.view(-1)[torch.randperm(self._res_regs)[:res_reg_mutations]] += torch.normal(
            0, 1, (res_reg_mutations,)
        ).to(self.torch_device)

        # Choose random distinct instructions and perform mutations
        for instr in np.random.choice(self.instructions, size=instructions_to_mutate, replace=False):
            instr.mutate()

    def delete_instruction(self, idx: Union[None, int] = None) -> None:
        """Delete instruction on index idx, random if idx is None"""
        # Obtain valid index into instruction list
        if idx is None:
            idx = np.random.randint(0, len(self.instructions))
        elif 0 <= idx < len(self.instructions):
            raise InstructionIndexError(
                f"Tried to delete instruction on index {idx} from list of {len(self.instructions)} instructions!"
            )
        # Delete the chosen instruction
        del self.instructions[idx]

    def grow(self) -> None:
        """Add random instruction, remove another when max limit exceeded to compensate"""
        # Remove instruction first to not delete new instruction when over limit
        if len(self.instructions) >= self.min_instructions:
            self.delete_instruction()
        # Append new and random instruction at the end of instruction list
        self.instructions.append(Instruction.random(self))

    def shrink(self) -> None:
        """Delete one random instruction, add one when lower limit exceeded to stay within limits"""
        # Delete random instruction
        self.delete_instruction()
        # Add instruction if lower limit was exceeded
        if len(self.instructions) < self.max_instructions:
            self.instructions.append(Instruction.random(self))

    def eval(
        self, X: torch.Tensor, y_gt: torch.Tensor, fitness_fn: Callable[[torch.Tensor, torch.Tensor], Number]
    ) -> Number:
        """
        Evaluate the program on the given dataset

        Returns the fitness value calculated by the fitness function.

        Args:
            X (torch.Tensor): Tensor of objects to classify.
            y_gt (torch.Tensor): Tensor of ground truth labels.
            fitness_fn (Callable[[torch.Tensor, torch.Tensor], Number]): Fitness function.
        Returns:
            Number: Fitness score
        """
        # Start with register initialization
        self.input_registers = X.clone().to(self.torch_device)
        self.hidden_registers = self.hidden_register_initial_values.repeat(
            X.shape[0], *(len(self.hidden_register_shape) * [1])
        ).to(self.torch_device)
        self.result_registers = self.result_register_initial_values.repeat(X.shape[0], 1).to(self.torch_device)

        # Now execute all instructions on data
        for instruction in self.instructions:
            instruction.execute()

        # Return the value for chosen fitness functions
        return fitness_fn(self.result_registers, y_gt)

    @property
    def _hid_regs(self) -> int:
        """Number of hidden registers"""
        return self.hidden_register_initial_values.shape.numel()

    @property
    def _res_regs(self) -> int:
        """Number of result registers"""
        return self.result_register_initial_values.shape.numel()

    @property
    def _info_dict(self) -> Dict[str, Dict]:
        """Obtain information about Program instance in a dictionary"""
        return {
            "1 - Hidden register field": {
                "shape": self.hidden_register_shape,
                "init values": self.hidden_register_initial_values,
            },
            "2 - Result register field": {
                "shape": self.result_register_shape,
                "init_values": self.result_register_initial_values,
            },
            # This should be sorted by pformat, if not, make it a list of pair tuples
            "3 - Instructions": [f"{i}: {instr}" for i, instr in enumerate(self.instructions, 1)],
        }

    def __repr__(self) -> str:
        """String representation of program and it's instructions (developer friendly)"""
        return pformat(self._info_dict, width=120)
