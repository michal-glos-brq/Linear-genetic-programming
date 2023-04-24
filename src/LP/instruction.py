"""
This python module implements Instruction class - a single instruction in linear program
Instruction class provides it's initialization, its' random initialization and its' transcription
"""

import torch
import numpy as np
from typing import Dict, Iterable

from utils.other import true_with_prob_p
from LP.operations import UNARY, BINARY, AREA, identity, INPUT_REGISTERS, OUTPUT_REGISTERS

class Instruction:
    """
    Single instruction of Program

    Attributes:
        parent (Program): The parent program containing this Instruction instance.
        operation_parity (int): The instruction parity (1, 2, etc.).
        operation_function (Callable): The instruction function.
        input_registers (List[str]): The input registers to be passed into the operation_function.
        output_register (str): The output register where the output of operation_function is stored.
        input_register_indices (List[Union[int, Tuple[int]]]): The indices into the input registers.
        output_register_indices (Union[int, Tuple[int]]): The index into the output register.
        is_area_operation (bool): Whether the Instruction instance works with tensor slices.
        area_operation_function (Callable): The area operation, None if area is False.
    """

    def __init__(self, parent) -> None:
        """
        Initialize the instruction as object

        Args:
            parent (Program):     Program class instance (parent program)
        """
        self.parent = parent
        self.operation_parity = None
        self.operation_function = None
        self.input_registers = None
        self.input_register_indices = None
        self.output_register_indices = None
        self.output_register = None
        self.is_area_operation = None
        self.area_operation_function = None
        self.register_shapes = {
            "input_registers": self.parent.input_register_shape,
            "hidden_registers": self.parent.hidden_register_shape,
            "result_registers": self.parent.result_register_shape,
        }

    def execute(self) -> None:
        """Execute vectorized instruction on the whole dataset"""
        # Obtain input register fields
        input_registers = [getattr(self.parent, _input_registers) for _input_registers in self.input_registers]
        output_register = getattr(self.parent, self.output_register)
        
        # Obtain instruction operands
        operands = []
        for _idx, _reg in zip(self.input_register_indices, input_registers):
            # When slicing with tuples, 1st dimension (object array) have to be
            # moved to the last position (slicing from 1st dim.) and back after slicing 
            operands.append(torch.moveaxis(torch.moveaxis(_reg, 0, -1)[_idx], -1, 0))

        result = self.operation_function(*operands)

        if self.is_area_operation:
            result = self.area_operation_function(result)

        # Save result into register (indexing with tuples again)
        torch.moveaxis(output_register, 0, -1)[self.output_register_indices] = result 

    def copy(self, parent) -> "Instruction":
        """
        Create new instruction belonging to a different parent

        Args:
            parent (Program): The new parent program
        Return:
            Instruction:    self (Instruction) copy with updated parent program
        """
        # Create the object instance
        new_instruction = Instruction(parent)

        # Proceed with copying self properties into the new instruction
        new_instruction.operation_parity = self.operation_parity
        new_instruction.operation_function = self.operation_function
        new_instruction.output_register = self.output_register
        new_instruction.output_register_indices = self.output_register_indices
        new_instruction.is_area_operation = self.is_area_operation
        new_instruction.area_operation_function = self.area_operation_function
        # List shall be copied
        new_instruction.input_registers = self.input_registers.copy()
        new_instruction.input_register_indices = self.input_register_indices.copy()

        return new_instruction

    @staticmethod
    def random(parent) -> "Instruction":
        """Generate a random instruction object instance"""
        # Create the object instance
        new_instruction = Instruction(parent)

        # Decide on the matter of area processing or single value processing
        new_instruction.is_area_operation = true_with_prob_p(parent.area_instruction_p)

        # Decide on parity (randomly, proportionally to number of operations)
        unary = (np.random.randint(0, (len(UNARY) + len(BINARY))) < len(UNARY)) or new_instruction.is_area_operation
        new_instruction.operation_parity = 1 if unary else 2
        # Choose random function of chosen parity (If this is an area instruction, add possibility od _id unary fn)
        new_instruction.operation_function = (
            np.random.choice(UNARY + ([identity] if new_instruction.is_area_operation else []))
            if unary
            else np.random.choice(BINARY)
        )

        # Now choose input and output registers (just strings, getattr will be used on parent)
        new_instruction.input_registers = [
            np.random.choice(INPUT_REGISTERS) for _ in range(new_instruction.operation_parity)
        ]
        new_instruction.output_register = np.random.choice(OUTPUT_REGISTERS)

        # If area instruction, generate slice ids
        if new_instruction.is_area_operation:
            # For each register, for each dimenstion, get 2 unique indices in bounds of tensor shape
            # We can slice how many dimensions we want with tuple of par-tuples
            new_instruction.input_register_indices = []
            for dim in new_instruction.register_shapes[new_instruction.input_registers[0]]:
                indices = tuple([num.item() for num in torch.randperm(dim)[:2].sort().values])
                new_instruction.input_register_indices.append(indices)
            new_instruction.input_register_indices = [tuple(new_instruction.input_register_indices)]

            # And finally, randomly obtain the area operation (Add identity - only area operation would be applied)
            new_instruction.area_operation_function = np.random.choice(AREA)
        # Otherwise, generate only normal ids
        else:
            # Randomly generate indices into registers (Nasty!)
            new_instruction.input_register_indices = [
                tuple(map(lambda x: np.random.randint(0, x), new_instruction.register_shapes[register]))
                for register in new_instruction.input_registers
            ]

        # Generate result register id
        new_instruction.output_register_indices = tuple(
            [np.random.randint(0, x) for x in new_instruction.register_shapes[new_instruction.output_register]]
        )

        return new_instruction


    def __repr__(self) -> str:
        """String representation of the istruction (single line of code)-ish"""
        # Create the instruction destination
        destination = f'{self.output_register}[{", ".join(map(str, self.output_register_indices))}]'

        # Make the line below readable ...
        # operands = [f'{_reg}[{",".join(map(str, _idx) if not isinstance(_idx[0], int) else map(lambda x: f"{x[0]}:{x[1]}"))}]' for _reg, _idx in zip(self.input_registers, self.input_register_indices)]
        operands = []
        for register, indices in zip(self.input_registers, self.input_register_indices):
            operands.append(
                # start with register name, add indices in [] - single indices [a,b,c], slices [a1:a2, b1:b2]
                f'{register}[{",".join(map(str, indices) if isinstance(indices[0], int) else map(lambda x: f"{x[0]}:{x[1]}", indices))}]'
            )

        # Add opearnds as strings into "function call" string
        instruction = f'{destination} = {self.operation_function.__name__}({", ".join(operands)})'

        # Add area function into the string
        if self.is_area_operation:
            instruction = f"{instruction}.{self.area_operation_function.__name__}()"

        return instruction
