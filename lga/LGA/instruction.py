"""
This python module implements Instruction class - a single instruction in linear program
Instruction class provides it's initialization, its' random initialization and its' transcription
"""
from random import random, choice, sample, randint
from typing import List, Union, Tuple

import torch

from utils.other import true_with_probability
from LGA.operations import UNARY, BINARY, AREA, identity, INPUT_REGISTERS, OUTPUT_REGISTERS, UNARY_OP_RATIO


class Instruction:
    """
    Single instruction of Program

    Attributes:
        parent (Program): The parent program containing this Instruction instance.
        operation_parity (int): The instruction parity {1, 2}.
        operation_function (Callable): The instruction function.
        input_registers (List[str]): The input registers to be passed into the operation_function.
        output_register (str): The output register where the output of operation_function is stored.
        input_register_indices (List[Union[int, Tuple[int]]]): The indices into the input registers.
        output_register_indices (Union[int, Tuple[int]]): The index into the output register.
        is_area_operation (bool): Whether the Instruction instance works with tensor slices.
        area_operation_function (Callable): The area operation, None if area is False.

        Note: Indices (not slices) into register Tensors always start with ellipsis (...)
    """

    def __init__(self, parent) -> None:
        """
        Initialize the instruction as object.

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

    def obtain_operands(self) -> List[torch.Tensor]:
        """Obtain instruction operands (value Tensors)"""
        if self.is_area_operation:
            # Operands are subtensors
            return [
                Instruction.multi_slice(getattr(self.parent, register), indices)
                for indices, register in zip(self.input_register_indices, self.input_registers)
            ]
        # Otherwise, operands are single values
        return [
            getattr(self.parent, register)[indices]
            for indices, register in zip(self.input_register_indices, self.input_registers)
        ]

    def compute(self, operands: List[torch.Tensor]) -> torch.Tensor:
        """Perform instruction calculations"""
        result = self.operation_function(*operands)

        if self.is_area_operation:
            result = self.area_operation_function(result.view((result.size(0), -1)), dim=1)

        return result

    def save_result(self, result: torch.Tensor) -> None:
        """Save result into output register on output register indices"""
        getattr(self.parent, self.output_register)[self.output_register_indices] = result

    def execute(self) -> None:
        """Execute vectorized instruction on the whole (Program) input register field"""
        operands = self.obtain_operands()
        result = self.compute(operands)
        self.save_result(result)

    def copy(self, parent) -> "Instruction":
        """
        Copy Instruction class instance with different parent

        Args:
            parent (Program): The new parent program
        Return:
            Instruction:    self (Instruction) copy with updated parent program
        """
        new_instr = Instruction(parent)

        new_instr.operation_parity = self.operation_parity
        new_instr.operation_function = self.operation_function
        new_instr.output_register = self.output_register
        new_instr.output_register_indices = self.output_register_indices
        new_instr.is_area_operation = self.is_area_operation
        new_instr.area_operation_function = self.area_operation_function

        new_instr.input_registers = self.input_registers.copy()
        new_instr.input_register_indices = self.input_register_indices.copy()

        return new_instr

    @staticmethod
    def random(parent) -> "Instruction":
        """Generate a random Instruction object instance"""
        new_instr = Instruction(parent)
        new_instr.is_area_operation = true_with_probability(parent.lga.area_instruction_p)

        is_unary_operation = random() < UNARY_OP_RATIO or new_instr.is_area_operation
        new_instr.operation_parity = 1 if is_unary_operation else 2

        if is_unary_operation:
            new_instr.operation_function = choice(UNARY + ([identity] if new_instr.is_area_operation else []))
        else:
            new_instr.operation_function = choice(BINARY)

        new_instr.input_registers = sample(INPUT_REGISTERS, k=new_instr.operation_parity)
        new_instr.output_register = choice(OUTPUT_REGISTERS)

        if new_instr.is_area_operation:
            # For each register and dimenstion obtain 2 ascending unique indices in bounds of tensor shape
            new_instr.input_register_indices = [
                Instruction.get_random_slice(
                    new_instr.parent.lga.register_shapes_dict[new_instr.input_registers[0]],
                    new_instr.parent.lga.torch_device,
                )
            ]
            new_instr.area_operation_function = choice(AREA)
        else:
            new_instr.input_register_indices = [
                Instruction.get_random_index(new_instr.parent.lga.register_shapes_dict[register])
                for register in new_instr.input_registers
            ]

        new_instr.output_register_indices = Instruction.get_random_index(
            new_instr.parent.lga.register_shapes_dict[new_instr.output_register]
        )
        return new_instr

    @staticmethod
    def multi_slice(tensor: torch.Tensor, indices: Tuple[int], dim=1):
        """Slice tensor in multiple dimensions"""
        for _dim, _dim_value in enumerate(indices, dim):
            tensor = torch.index_select(tensor, _dim, _dim_value)
        return tensor

    @staticmethod
    def get_random_slice(register_shape: Union[torch.Size, Tuple], torch_device: torch.device) -> Tuple[int]:
        """Obtain random slice indices into a tensor with provided shape"""
        return [torch.randint(0, dim, (randint(1, dim >> 1),), device=torch_device) for dim in register_shape]

    # pylint: disable=consider-using-generator
    @staticmethod
    def get_random_index(register_shape: Union[torch.Size, Tuple]) -> Tuple[int]:
        """Obtain random indices into a tensor with provided shape (indices to a single value)"""
        return (...,) + tuple(randint(0, dim - 1) for dim in register_shape)

    def __repr__(self) -> str:
        """String representation of the istruction (single line of code)-ish"""
        destination = f'{self.output_register}[{", ".join(map(str, self.output_register_indices[1:]))}]'

        operands = []
        for register, indices in zip(self.input_registers, self.input_register_indices):
            if isinstance(indices[0], type(...)):
                operands.append(f'{register}[{",".join(map(str, indices[1:]))}]')
            else:
                _indices = []
                for _id in indices:
                    _indices.append(f'({",".join(_id.cpu().numpy().astype(str))})')
                operands.append(f'{register}[{",".join(_indices)}')

        instruction = f'{destination} = {self.operation_function.__name__}({", ".join(operands)})'

        if self.is_area_operation:
            instruction = f"{instruction}.{self.area_operation_function.__name__}()"

        return instruction
