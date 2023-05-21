"""
This humble python module implements all other utility functions used throughout the project

Author: Michal Glos (xglosm01)
"""

from random import random


class PopulationNotEvaluatedError(Exception):
    """Raised when the population was not evaluated and acessed based upon fitness values"""


class InstructionIndexError(Exception):
    """Tried to index an instruction of program outside of instruction array"""


class ProgramRegisterShapeError(Exception):
    """Program has invalid shape of hidden registers, could not be used with loaded dataset"""


def true_with_probability(probability: float) -> bool:
    """
    Returns True with a probability of 'p' and False with a probability of '1-p'.

    Args:
        probability (float): Probability value between 0 and 1.

    Returns:
        bool: True with probability 'p', False otherwise.
    """
    return random() < probability
