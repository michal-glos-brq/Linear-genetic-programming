"""
This python module provides fitness functions and their mapping to string codes

Author: Michal Glos (xglosm01)
"""

from typing import Dict, Callable

import torch
from torch.nn.functional import cross_entropy, softmax


def accuracy_score(result_registers: torch.Tensor, ground_truth_labels: torch.Tensor) -> float:
    """
    Compute the percentage of correctly predicted classes.

    Args:
        result_registers (torch.Tensor[num_objects, num_classes]): Tensor containing the model's output.
        gt_labels (torch.Tensor[num_objects]): Ground truth labels.

    Returns:
        torch.Tensor: The percentage of correct predictions.
    """
    return (result_registers.argmax(dim=1) == ground_truth_labels).sum() / ground_truth_labels.size(0)


def cross_entropy_loss(result_registers: torch.Tensor, ground_truth_labels: torch.Tensor) -> float:
    """
    Compute the cross-entropy between ground truth labels and model predictions.

    Args:
        result_registers (torch.Tensor[num_objects, num_classes]): Tensor containing the model's output.
        ground_truth_labels (torch.Tensor[num_objects]): Ground truth labels.

    Returns:
        torch.Tensor: Cross-entropy result.
    """
    return torch.nan_to_num(-1 * cross_entropy(softmax(result_registers, dim=1), ground_truth_labels))


# Mapping of strings to funtions
FITNESS_FUNCTIONS: Dict[str, Callable[[torch.Tensor, torch.Tensor], float]] = {
    "ac": accuracy_score,
    "ce": cross_entropy_loss,
}
