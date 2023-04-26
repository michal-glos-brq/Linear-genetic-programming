"""
This humble python module implements all other utility functions used throughout the project
"""

from os.path import isfile
from random import random

def true_with_probability(probability: float) -> bool:
    """
    Returns True with a probability of 'p' and False with a probability of '1-p'.

    Args:
        probability (float): Probability value between 0 and 1.

    Returns:
        bool: True with probability 'p', False otherwise.
    """
    return random() < probability

def generate_unique_filename(file_path: str, file_suffix: str = "p") -> str:
    """
    Generate unique filename with provided suffix

    Args:
        file_path (str): Desired file path (directories included).
        file_suffix (str): Desired file suffix.

    Returns:
        str: Unique path to a file with suffix suffix :)
    """
    # If the file already exists, generate unique path with appending a number
    base_filename = f"{file_path}.{file_suffix}"

    if isfile(base_filename):
        counter = 1
        unique_filename = f"{file_path}_{counter}.{file_suffix}"

        while isfile(unique_filename):
            counter += 1
            unique_filename = f"{file_path}_{counter}.{file_suffix}"

        return unique_filename
    return base_filename
