from typing import Any, Dict, Iterable, Union, List
import torch
import numpy as np


def true_with_prob_p(p: float) -> bool:
    """
    Returns True with a probability of 'p' and False with a probability of '1-p'.

    Args:
        p (float): Probability value between 0 and 1.

    Returns:
        bool: True with probability 'p', False otherwise.
    """
    return np.random.random < p


def identity(input_value: Any) -> Any:
    """
    Identity function that returns its only argument.

    This function is used as an unary operation, it's purpose is to apply
    area operations without input-modifying unary operation preceding it.

    Args:
        input_value (Any): The input value to return.

    Returns:
        Any: The input value.
    """
    return input_value


def safe_div(tensor1: torch.Tensor, tensor2: torch.Tensor) -> torch.Tensor:
    """
    Divide tensors safely. If attempted to divide by 0, set the result to 0.

    Args:
        tensor1 (torch.Tensor): The first tensor.
        tensor2 (torch.Tensor): The second tensor.

    Returns:
        torch.Tensor: The resulting tensor after safe division.
    """
    result = torch.div(tensor1, tensor2)
    result[result != result] = 0
    return result


# Binary operations
BINARY = [torch.add, torch.sub, torch.mul, torch.pow, safe_div]

# Unary operations
UNARY = [torch.exp, torch.log, torch.sin, torch.cos, torch.tan, torch.tanh, torch.sqrt]

# Area operations (Reducing area to single value)
AREA = [torch.mean, torch.median, torch.sum, torch.max, torch.min, torch.prod]

INPUT_REGISTERS = ["input_registers", "hidden_registers"]
OUTPUT_REGISTERS = ["hidden_registers", "result_registers"]


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
        is_area_opearation (bool): Whether the Instruction instance works with tensor slices.
        area_operation_function (Callable): The area operation, None if area is False.
    """

    def __init__(self, parent) -> None:
        """
        Initialize the instruction as object

        Args:
            parent (Program):     Program class instance (parent program)
        """
        # Keep pointing to the parent program
        self.parent = parent
        # Initialize other properties
        self.operation_parity = None
        self.operation_function = None
        self.input_registers = None
        self.input_register_indices = None
        self.output_register_indices = None
        self.output_register = None
        self.is_area_opearation = None
        self.area_operation_function = None

    def execute(self) -> None:
        """Execute vectorized instruction on the whole dataset"""
        # Obtain the register fields (Tensors)
        input_registers = [getattr(self.parent, _input_registers) for _input_registers in self.input_registers]
        output_register = getattr(self.parent, self.output_register)
        # Obtain the operands (move first axis to the end for easy indexing
        # and slicing single register value for the whole dataset)
        operands = []
        for _idx, _reg in zip(self.input_register_indices, input_registers):
            # This is a little hacky, I want to choose slice for each dataset entry
            # with indexing tuples, therefore the first dimension (number of entries in dataset)
            # has to be moved to the end, so tuple indexing would index in registerfield for each entry
            # The result is swapped back - area functions would obtain different operands in shape
            operands.append(torch.moveaxis(torch.moveaxis(_reg, 0, -1)[_idx], -1, 0))

        # Calculate the result
        result = self.operation_function(*operands)

        # If area instruction, reduce tensor with area operation to scalar
        if self.is_area_opearation:
            result = self.area_operation_function(result)

        # Save the result into the result register
        # The same process as with obtaining the operands has to be followed
        torch.moveaxis(output_register, 0, -1)[self.output_register_indices] = result

    def _mutate_operation(self) -> None:
        """Mutate the operation used in this Instruction instance"""
        # If mutating area operation, it's 50/50 which operation would be changed
        if self.is_area_opearation and true_with_prob_p(0.5):
            self.area_operation_function = np.random.choice(AREA)
        # Otherwise, start with choosing parity of operation randomly with p proportional to their count
        else:
            unary = (np.random.randint(0, (len(UNARY) + len(BINARY))) < len(UNARY)) or self.is_area_opearation
            self.operation_function = np.random.choice(UNARY) if unary else np.random.choice(BINARY)

            # Changed from binary to unary, just get rid of the second operand
            if unary and self.operation_parity == 2:
                self.operation_parity = 1
                self.input_registers = self.input_registers[:1]
                self.input_register_indices = self.input_register_indices[:1]
            # Changed from unary to binary, we have to add one random input register with index
            elif not unary and self.operation_parity == 1:
                self.operation_parity = 2
                self.input_registers.append(np.random.choice(INPUT_REGISTERS))
                self.input_register_indices.append(
                    tuple(map(lambda x: np.random.randint(0, x), self.input_registers[1].shape))
                )

    def _mutate_input_registers(self) -> None:
        """Mutate one of inputs (change index or register and index)"""
        # Choose register on which the mutation is performed
        input_registers_indices = np.random.randint(0, self.operation_parity)
        self.input_registers[input_registers_indices] = np.random.choice(INPUT_REGISTERS)

        # Randomly choose new index
        if self.is_area_opearation:
            new_indices = []
            for dim in self.reg_shapes[self.input_registers[input_registers_indices]]:
                # Choose random 2 numbers for each dimension with correct values
                indices, _ = torch.randperm(dim)[:2].sort()
                new_indices.append(tuple(indices))
            self.input_register_indices[input_registers_indices] = tuple(new_indices)
        else:
            self.input_register_indices[input_registers_indices] = tuple(
                map(lambda x: np.random.randint(0, x), self.reg_shapes[self.input_registers[input_registers_indices]])
            )

    def _mutate_output_register(self) -> None:
        """Change the destination of Instruction result"""
        # Randomly choose new register
        self.output_register = np.random.choice(OUTPUT_REGISTERS)
        # Randomly generate index into that new register
        self.output_register_indices = tuple(map(lambda x: np.random.randint(0, x), self.reg_shapes[self.output_register]))

    def _mutate_area(self) -> None:
        """Change index into output register or output register and its' index"""
        # From area operation to scalar operation, unary parity is implicit here, could index with 0
        if self.is_area_opearation:
            self.input_register_indices[0] = tuple([np.random.randint(lb, hb) for lb, hb in self.input_register_indices[0]])
        else:
            # This instruction has to be unary in order to work now with area tensor slices
            if self.operation_parity == 2:
                self.operation_parity = 1
                self.operation_function = np.random.choice(UNARY)
                self.input_registers = self.input_registers[:1]
                self.input_register_indices = self.input_register_indices[:1]
            # Now we have unary operation, let's add random area operation and generate slice indices
            self.area_operation_function = np.random.choice(AREA)
            new_indices = []
            for dim in self.reg_shapes[self.input_registers[0]]:
                indices, _ = torch.randperm(dim)[:2].sort()
                new_indices.append(tuple(indices))
            self.input_register_indices[0] = tuple(new_indices)
        # Flag the change 'officialy'
        self.is_area_opearation = not self.is_area_opearation

    def mutate(self) -> None:
        """
        Perform instruction mutation

        According to LGA convention, let's just really change only one thing here

        Could be one of following things:
            Operation
            Input index/register
            Output index/register
            Area operation added or removed
        Area instructions could mutate also on:
            Area of operands
        """
        # Area mutation has fixed probability
        # (1 - area prob.) will be equally distributed alongside other mutation types
        not_area_p = (1 - self.parent.area_p) / 3.0
        # Randomly choose one of mutation functions and execute it
        mutation_function = np.random.choice(
            [self._mutate_area, self._mutate_input_registers, self._mutate_output_register, self._mutate_area],
            p=[not_area_p, not_area_p, not_area_p, self.parent.area_p],
        )
        # Execute the mutation function
        mutation_function()

    def copy(self, parent) -> "Instruction":
        """
        Create new instruction under provided parent
        
        Args:
            parent (Program): Parent program
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
        new_instruction.is_area_opearation = self.is_area_opearation
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
        new_instruction.is_area_opearation = true_with_prob_p(parent.area_p)

        # Decide on parity (randomly, proportionally to number of operations)
        unary = (np.random.randint(0, (len(UNARY) + len(BINARY))) < len(UNARY)) or new_instruction.is_area_opearation
        new_instruction.operation_parity = 1 if unary else 2
        # Choose random function of chosen parity (If this is an area instruction, add possibility od _id unary fn)
        new_instruction.operation_function = (
            np.random.choice(UNARY + ([identity] if new_instruction.is_area_opearation else []))
            if unary
            else np.random.choice(BINARY)
        )

        # Now choose input and output registers (just strings, getattr will be used on parent)
        new_instruction.input_registers = [
            np.random.choice(INPUT_REGISTERS) for _ in range(new_instruction.operation_parity)
        ]
        new_instruction.output_register = np.random.choice(OUTPUT_REGISTERS)

        # If area instruction, generate slice ids
        if new_instruction.is_area_opearation:
            # For each register, for each dimenstion, get 2 unique indices in bounds of tensor shape
            # We can slice how many dimensions we want with tuple of par-tuples
            new_instruction.input_register_indices = []
            for dim in new_instruction.reg_shapes[new_instruction.input_registers[0]]:
                indices, _ = tuple(torch.randperm(dim)[:2].sort())
                new_instruction.input_register_indices.append(indices)
            new_instruction.input_register_indices = [tuple(new_instruction.input_register_indices)]

            # And finally, randomly obtain the area operation (Add identity - only area operation would be applied)
            new_instruction.area_operation_function = np.random.choice(AREA)
        # Otherwise, generate only normal ids
        else:
            # Randomly generate indices into registers (Nasty!)
            new_instruction.input_register_indices = [
                tuple(map(lambda x: np.random.randint(0, x), new_instruction.reg_shapes[register]))
                for register in new_instruction.input_registers
            ]

        # Generate result register id
        new_instruction.output_register_indices = tuple(
            [np.random.randint(0, x) for x in new_instruction.reg_shapes[new_instruction.output_register]]
        )

        return new_instruction

    @property
    def reg_shapes(self) -> Dict[str, Iterable[int]]:
        """Shapes of registers given their names"""
        return {
            "input_registers": self.parent.input_register_shape,
            "hidden_registers": self.parent.hidden_register_shape,
            "result_registers": self.parent.result_register_shape,
        }

    @property
    def _info_dict(self) -> Dict[str, Union[Dict, List, bool]]:
        """Obtain a dictionary describing Instruction object instance (self)"""
        return {
            "Opeartion": {
                "function": self.operation_function.__name__,
                "parity": self.operation_parity,
            },
            "In operands": [
                {"register": reg, "indice": idx} for reg, idx in zip(self.input_registers, self.input_register_indices)
            ],
            "Out register": {"register": self.output_register, "indices": self.output_register_indices},
            "Area operation": self.area_operation_function,
        }

    def __repr__(self) -> str:
        """String representation of the istruction"""
        # Create the instruction destination
        destination = f'{self.output_register}[{", ".join(map(str, self.output_register_indices))}]'

        # Make this the line below readable ...
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
        if self.is_area_opearation:
            instruction = f"{instruction}.{self.area_operation_function.__name__}()"

        return instruction
