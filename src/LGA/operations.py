"""
This humble python module defines some tensor operations and groups operations into global constants according to their parity
"""
from typing import Any, Callable, List, Iterable

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


def times_minus1(tensor: torch.Tensor) -> torch.Tensor:
    """
    Divide tensors safely. If attempted to divide by 0, set the result to 0.

    Args:
        tensor (torch.Tensor): The input tensor.

    Returns:
        torch.Tensor: Tensor with switched sign
    """
    return tensor * (-1)


### This monstrosity here is only for instruction pickle purposes

# pylint disable=comparison-with-itself
def safe_division(tensor1: torch.Tensor, tensor2: torch.Tensor) -> torch.Tensor:
    """Safe division"""
    result = torch.div(tensor1, tensor2)
    return torch.nan_to_num(result, posinf=0.0, neginf=0.0)


# pylint disable=comparison-with-itself
def safe_power(tensor1: torch.Tensor, tensor2: torch.Tensor) -> torch.Tensor:
    """Safe power"""
    result = torch.pow(tensor1, tensor2)
    return torch.nan_to_num(result, posinf=0.0, neginf=0.0)


# pylint disable=comparison-with-itself
def safe_exp(tensor1: torch.Tensor) -> torch.Tensor:
    """Safe exp"""
    result = torch.exp(tensor1)
    return torch.nan_to_num(result, posinf=0.0, neginf=0.0)


# pylint disable=comparison-with-itself
def safe_logarithm(tensor: torch.Tensor) -> torch.Tensor:
    """Safe division"""
    result = torch.log(tensor)
    return torch.nan_to_num(result, posinf=0.0, neginf=0.0)


# pylint disable=comparison-with-itself
def safe_tan(tensor: torch.Tensor) -> torch.Tensor:
    """Safe division"""
    result = torch.tan(tensor)
    return torch.nan_to_num(result, posinf=0.0, neginf=0.0)


# pylint disable=comparison-with-itself
def safe_square_root(*tensors: Iterable[torch.Tensor]) -> torch.Tensor:
    """Safe division"""
    result = torch.sqrt(*tensors)
    return torch.nan_to_num(result, posinf=0.0, neginf=0.0)


# Binary operations
BINARY: List[Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = [
    torch.add,
    torch.sub,
    torch.mul,
    safe_power,
    torch.min,
    torch.max,
    safe_division,
]

# Unary operations (identity is added only when sampling an operation for area instruction)
UNARY: List[Callable[[torch.Tensor], torch.Tensor]] = [
    safe_exp,
    safe_logarithm,
    torch.sin,
    torch.cos,
    safe_tan,
    torch.tanh,
    safe_square_root,
    times_minus1,
]

# Area operations (Reducing area to single value, unary operation has to preceed it)
AREA: List[Callable[[], torch.Tensor]] = [torch.mean, torch.sum, torch.prod]

UNARY_OP_RATIO = len(UNARY) / (len(BINARY) + len(UNARY))

INPUT_REGISTERS = ["input_registers", "hidden_registers"]
OUTPUT_REGISTERS = ["hidden_registers", "result_registers"]
