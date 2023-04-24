from typing import Dict, Callable
from torch import Tensor
import torch.nn.functional.cross_entropy as ce

def accuracy_score(result_registers: Tensor, ground_truth_labels: Tensor) -> float:
    """
    Compute the percentage of correctly predicted classes.

    Args:
        result_registers (torch.Tensor[num_objects, num_classes]): Tensor containing the model's output.
        gt_labels (torch.Tensor[num_objects]): Ground truth labels.

    Returns:
        torch.Tensor: The percentage of correct predictions.
    """
    predicted_labels = result_registers.argmax(dim=1)
    correct_predictions = (predicted_labels == ground_truth_labels).sum()
    accuracy = (correct_predictions / len(ground_truth_labels)) * 100
    
    return accuracy.item()


def cross_entropy_loss(result_registers: Tensor, ground_truth_labels: Tensor) -> float:
    """
    Compute the cross-entropy between ground truth labels and model predictions.

    Args:
        result_registers (torch.Tensor[num_objects, num_classes]): Tensor containing the model's output.
        ground_truth_labels (torch.Tensor[num_objects]): Ground truth labels.

    Returns:
        torch.Tensor: Cross-entropy result.
    """
    # Normalize result registers with softmax
    return ce(result_registers, ground_truth_labels).item()


# Mapping of strings to funtions
FITNESS_FUNCTIONS: Dict[str, Callable[[Tensor, Tensor], float]] = {
    "ac": accuracy_score,
    "ce": cross_entropy_loss,
}
