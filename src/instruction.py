#######################
# @$%&             &%$@#
#!    Michal Glos    !#
#!     EVO - 2023    !#
#!        __         !#
#!      <(o )___     !#
#!       ( ._> /     !#
# @$%&     `---'   &%$@#
#######################

from typing import Any, Dict, Iterable, Union, List
from pprint import pformat

import torch
import numpy as np

################################################################################
#####                       Easy access constants                          #####
################################################################################


def _id(_input: Any) -> Any:
    """
    Function returning it's only argument

    Used as unary function when area instructions is initiated so
    area opearations could be performed without unary op. preceeding it
    """
    return _input


def _safe_div(_input_1: torch.Tensor, _input_2: torch.Tensor) -> torch.Tensor:
    """Divide tensors safely, when tried to divide by 0, set the result to 0"""
    # Perform the division (nan != nan, hence all nans are replaced by 0.)
    _result = torch.div(_input_1, _input_2)
    _result[_result != _result] = 0
    return _result


# Binary operations
BINARY = [torch.add, torch.sub, torch.mul, torch.pow, _safe_div]

# Unary operations
UNARY = [torch.exp, torch.log, torch.sin, torch.cos, torch.tan, torch.tanh, torch.sqrt]

# Area operations (Reducing area to single value)
AREA = [torch.mean, torch.median, torch.sum, torch.max, torch.min, torch.prod]

IN_REGS = ["input_registers", "hidden_registers"]
OUT_REGS = ["hidden_registers", "result_registers"]

################################################################################
#####                          Instruction class                           #####
################################################################################


class Instruction:
    """
    Single instruction to be executed

    important properties:
        parent:     Program instance which contains this Instruction instance
        op_par:     Instruction parity (integer: 1, 2, ...)
        op_fn:      Instruction function
        in_reg:     Instruction input registers - would be passed into the op_fn (list of strings)
        out_reg:    Output register - output of op_fn is stored there  (string)
        in_idx:     Indices into input registers (pair-tuples if are slices are used)
        out_id:     Index into output register
        area:       Does Instruction instance work with tensor slices? (bool)
        area_op:    Area operation, None if area == False
    """

    def __init__(self, parent) -> None:
        """
        Initialize the instruction as object

        @args:
            parent:     Program class instance (parent program)
        """
        # Keep pointing to the parent program
        self.parent = parent
        # Initialize other properties
        self.op_par = None
        self.op_fn = None
        self.in_reg = None
        self.in_idx = None
        self.out_id = None
        self.out_reg = None
        self.area = None
        self.area_op = None

    ################################################################################
    #####                  Sampling from instruction space                     #####
    ################################################################################

    def execute(self) -> None:
        """Execute vectorized instruction on whole dataset"""
        # Obtain the register fields (Tensors)
        in_regs = [getattr(self.parent, _in_reg) for _in_reg in self.in_reg]
        out_reg = getattr(self.parent, self.out_reg)
        # Obtain the operands (move first axis to the end for easy indexing
        # and slicing single register value for the whole dataset)
        operands = []
        for _idx, _reg in zip(self.in_idx, in_regs):
            # This is a little hacky, I want to choose slice for each dataset entry
            # with indexing tuples, therefore the first dimension (number of entries in dataset)
            # has to be moved to the end, so tuple indexing would index in registerfield for each entry
            # The result is swapped back - area functions would obtain different operands in shape
            operands.append(torch.moveaxis(torch.moveaxis(_reg, 0, -1)[_idx], -1, 0))

        # Calculate the result
        result = self.op_fn(*operands)

        # If area instruction, reduce tensor with area operation to scalar
        if self.area:
            result = self.area_op(result)

        # Save the result into the result register
        # The same process as with obtaining the operands has to be followed
        torch.moveaxis(out_reg, 0, -1)[self.out_id] = result

    ################################################################################
    #####                              Evolution                               #####
    ################################################################################

    def _mutate_operation(self) -> None:
        """Let's mutate the operation used in this Instruction instance"""
        # If mutating area operation, it's 50/50 which operation would be changed
        if self.area and np.random.random < 0.5:
            self.area_op = np.random.choice(AREA)
        # Otherwise, start with choosing parity of operation randomly with p proportional to their count
        else:
            unary = (np.random.randint(0, (len(UNARY) + len(BINARY))) < len(UNARY)) or self.area
            self.op_fn = np.random.choice(UNARY) if unary else np.random.choice(BINARY)

            # Changed from binary to unary, just get rid of the second operand
            if unary and self.op_par == 2:
                self.op_par = 1
                self.in_reg = self.in_reg[:1]
                self.in_idx = self.in_idx[:1]
            # Changed from unary to binary, we have to add one random input register with index
            elif not unary and self.op_par == 1:
                self.op_par = 2
                self.in_reg.append(np.random.choice(IN_REGS))
                self.in_idx.append(tuple(map(lambda x: np.random.randint(0, x), self.in_reg[1].shape)))

    def _mutate_in_reg(self) -> None:
        """Mutate one of inputs into Instruction operation"""
        # Choose register on which the mutation is performed
        in_reg_idx = np.random.randint(0, self.op_par)
        self.in_reg[in_reg_idx] = np.random.choice(IN_REGS)

        # Randomly choose new index
        if self.area:
            new_idx = []
            for dim in self.reg_shapes[self.in_reg[in_reg_idx]]:
                # Choose random 2 numbers for each dimension with correct values
                _idx, _ = torch.randperm(dim)[:2].sort()
                new_idx.append(tuple(_idx))
            self.in_idx[in_reg_idx] = tuple(new_idx)
        else:
            self.in_idx[in_reg_idx] = tuple(
                map(lambda x: np.random.randint(0, x), self.reg_shapes[self.in_reg[in_reg_idx]])
            )

    def _mutate_out_reg(self) -> None:
        """Change the destination of Instruction result"""
        # Randomly choose new register
        self.out_reg = np.random.choice(OUT_REGS)
        # Randomly generate index into that new register
        self.out_id = tuple(map(lambda x: np.random.randint(0, x), self.reg_shapes[self.out_reg]))

    def _mutate_area(self) -> None:
        """Mutate the selected area of operands"""
        # From area operation to scalar operation, unary parity is implicit here, could index with 0
        if self.area:
            self.in_idx[0] = tuple([np.random.randint(lb, hb) for lb, hb in self.in_idx[0]])
        else:
            # This instruction has to be unary in order to work now with area tensor slices
            if self.op_par == 2:
                self.op_par = 1
                self.op_fn = np.random.choice(UNARY)
                self.in_reg = self.in_reg[:1]
                self.in_idx = self.in_idx[:1]
            # Now we have unary operation, let's add random area operation and generate slice indices
            self.area_op = np.random.choice(AREA)
            new_idx = []
            for dim in self.reg_shapes[self.in_reg[0]]:
                _idx, _ = torch.randperm(dim)[:2].sort()
                new_idx.append(tuple(_idx))
            self.in_idx[0] = tuple(new_idx)
        # Flag the change 'officialy'
        self.area = not self.area

    def mutate(self) -> None:
        """
        Perform instruction mutation

        According to LGA convention, let's just really change only one thing here

        Could be one of following things:
            Operation
            Input index/register
            Output index/register
        Area instructions could mutate also on:
            Area of operands
        """
        # Area mutation has fixed probability which could differ from other
        not_area_p = (1 - self.parent.area_p) / 3.0
        # Just randomly choose one of mutation functions and execute it
        mutation_function = np.random.choice(
            [self._mutate_area, self._mutate_in_reg, self._mutate_out_reg, self._mutate_area],
            p=[not_area_p, not_area_p, not_area_p, self.parent.area_p],
        )
        # Execute the mutation function
        mutation_function()

    def copy(self, parent) -> "Instruction":
        """Create new instruction under provided parent"""
        # Create the object instance
        new_instruction = Instruction(parent)

        # Proceed with copying self properties into the new instruction
        new_instruction.op_par = self.op_par
        new_instruction.op_fn = self.op_fn
        new_instruction.out_reg = self.out_reg
        new_instruction.out_id = self.out_id
        new_instruction.area = self.area
        # List shall be copied
        new_instruction.in_reg = self.in_reg.copy()
        new_instruction.in_idx = self.in_idx.copy()

        return new_instruction

    @staticmethod
    def random(parent) -> "Instruction":
        """Generate a random instruction object instance"""
        # Create the object instance
        new_instruction = Instruction(parent)

        # Decide on area processing or single value processing
        new_instruction.area = np.random.random() < parent.area_p

        # Decide on parity (randomly, proportionally to number of operations)
        unary = (np.random.randint(0, (len(UNARY) + len(BINARY))) < len(UNARY)) or new_instruction.area
        new_instruction.op_par = 1 if unary else 2
        # Choose random function of chosen parity (If this is an area instruction, add possibility od _id unary fn)
        new_instruction.op_fn = (
            np.random.choice(UNARY + ([_id] if new_instruction.area else [])) if unary else np.random.choice(BINARY)
        )

        # Now choose input and output registers (just strings, getattr will be used on parent)
        new_instruction.in_reg = [np.random.choice(IN_REGS) for _ in range(new_instruction.op_par)]
        new_instruction.out_reg = np.random.choice(OUT_REGS)

        # If area instruction, generate slice ids
        if new_instruction.area:
            # For each register, for each dimenstion, get 2 unique indices in bounds of tensor shape
            # We can slice how many dimensions we want with tuple of par-tuples
            new_instruction.in_idx = []
            for dim in new_instruction.reg_shapes[new_instruction.in_reg[0]]:
                _idx, _ = tuple(torch.randperm(dim)[:2].sort())
                new_instruction.in_idx.append(_idx)
            new_instruction.in_idx = [tuple(new_instruction.in_idx)]

            # And finally, randomly obtain the area operation (Add identity - only area operation would be applied)
            new_instruction.area_op = np.random.choice(AREA)
        # Otherwise, generate only normal ids
        else:
            # Randomly generate indices into registers (Nasty!)
            new_instruction.in_idx = [
                tuple(map(lambda x: np.random.randint(0, x), new_instruction.reg_shapes[_reg]))
                for _reg in new_instruction.in_reg
            ]

        # Generate result register id
        new_instruction.out_id = tuple(
            [np.random.randint(0, x) for x in new_instruction.reg_shapes[new_instruction.out_reg]]
        )

        return new_instruction

    ################################################################################
    #####                                 Utils                                #####
    ################################################################################

    @property
    def reg_shapes(self) -> Dict[str, Iterable[int]]:
        """Shapes of registers given their names"""
        return {
            "input_registers": self.parent.input_reg_shape,
            "hidden_registers": self.parent.hidden_reg_shape,
            "result_registers": self.parent.result_reg_shape,
        }

    @property
    def _info_dict(self) -> Dict[str, Union[Dict, List, bool]]:
        """Obtain a dictionary describing Instruction object instance (self)"""
        return {
            "Opeartion": {
                "function": self.op_fn.__name__,
                "parity": self.op_par,
            },
            "In operands": [{"register": reg, "indice": idx} for reg, idx in zip(self.in_reg, self.in_idx)],
            "Out register": {"register": self.out_reg, "indices": self.out_id},
            "Area operation": self.area_op,
        }

    def __repr__(self) -> str:
        """String representation of instruction and it's instructions (developer friendly)"""
        return pformat(self._info_dict, width=120)
