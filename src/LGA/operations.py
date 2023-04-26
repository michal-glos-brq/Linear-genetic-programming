"""
This humble python module defines some tensor operations and groups operations into global constants according to their parity
"""
from typing import Any, Callable, List

import torch

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
    mask = torch.isclose(tensor2, ZERO_SCALAR_TENSOR)
    result = torch.div(tensor1, tensor2)
    result[mask] = 0
    return result


# Is any better way to do that?
ZERO_SCALAR_TENSOR = (torch.cuda if torch.cuda.is_available() else torch).FloatTensor(1).fill_(0)

# Binary operations
BINARY: List[Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = [torch.add, torch.sub, torch.mul, torch.pow, safe_div]

# Unary operations (identity is added only when sampling an operation for area instruction)
UNARY: List[Callable[[torch.Tensor], torch.Tensor]]  = [torch.exp, torch.log, torch.sin, torch.cos, torch.tan, torch.tanh, torch.sqrt]

# Area operations (Reducing area to single value, unary operation has to preceed it)
AREA: List[Callable[[], torch.Tensor]] = [torch.mean, torch.median, torch.sum, torch.max, torch.min, torch.prod]

UNARY_OP_RATIO = len(UNARY) / (len(BINARY) + len(UNARY))

INPUT_REGISTERS = ["input_registers", "hidden_registers"]
OUTPUT_REGISTERS = ["hidden_registers", "result_registers"]
