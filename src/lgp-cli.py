#! /usr/bin/env python3

"""
Linear Genetic Programming (LGP) for image classification
Author: Michal Glos, EVO - 2023
"""

import argparse
import sys
import pickle
import logging
from os.path import isfile, makedirs
from typing import Tuple, Dict, Union

import torch

from datasets import Dataset
from LGP import LGP
from program import Program


def main():
    """Main chunk of code executed only when this file is being executed"""
    args = parse_arguments()

    # Validate provided arguments, exit upon failure
    validate_arguments(args)

    logging.basicConfig(level=logging.DEBUG if args.debug else logging.INFO)

    # Obtain dictionaries with object configurations from CLI args
    (lgp_kwargs,) = prepare_lgp_configuration(args)
    dataset_kwargs = prepare_dataset_configuration(args)

    # Prepare directory structure
    prepare_directories(args)

    # Load dataset
    dataset = load_dataset(dataset_kwargs)

    # Try to load program if path was provided
    program = load_program(args.load, args.regs) if args.load else None

    # Instantiate algorithm object
    lgp = load_lgp(dataset, program, lgp_kwargs)

    # Run LGA
    if args.train:
        train_lgp(lgp, dataset)

    # Evaluate the algorithm
    if args.eval:
        evaluate_program(lgp, dataset)


def parse_arguments() -> argparse.Namespace:
    """Define and parse CLI arguments"""
    parser = argparse.ArgumentParser(
        description="Linear Genetic Algorithm (LGA) for image classification",
    )

    # Dataset-related arguments
    parser.add_argument(
        "-d", "--dataset", type=str, default="MNIST", help="Choose a torchvision dataset (e.g., MNIST, CIFAR10, etc.)"
    )
    parser.add_argument("--data-dir", type=str, default="data", help="Specify the directory to store the dataset")
    parser.add_argument("--resize", type=int, help="Resize images to the specified square (edge size in pixels)")
    parser.add_argument(
        "--split", type=int, default=10, help="Specify the percentage of the dataset used for training (0-100)"
    )
    parser.add_argument(
        "-n",
        "--normalize",
        type=int,
        nargs=2,
        default=[0, 1],
        help="Specify the interval to normalize the dataset (e.g., 0 1 or -1 1)",
    )
    parser.add_argument(
        "--test",
        type=int,
        nargs=2,
        help=(
            "Create a custom test dataset, specify the number of classes and data entries "
            "(e.g., 10 1000 for 10 classes and 1000 entries)"
        ),
    )

    # App flow
    parser.add_argument("-t", "--train", action="store_true", help="Train the LGA on the dataset")
    parser.add_argument("-e", "--eval", action="store_true", help="Evaluate the LGA on the evaluation dataset")
    # Loading program could get tricky, registerfield shapes have to match, is asseted in assertion part of this script
    parser.add_argument("-l", "--load", type=str, help="Load a pre-trained program from the specified path")
    parser.add_argument(
        "-md",
        "--model-dir",
        type=str,
        default="programs",
        help=(
            "Specify the directory to save the best programs in the following format: "
            "{--dataset}{--resize}_{generation}_{fitness}.p"
        ),
    )
    parser.add_argument("-log", "--logging-dir", default="logging", type=str, help="Specify the directory for logging")

    # LGP parameters
    parser.add_argument("-p", "--population", type=int, default=42, help="Specify the population size (upper bound)")
    parser.add_argument("-g", "--gens", type=int, default=60, help="Specify the number of generations for the LGP to evolve")
    parser.add_argument(
        "-mini",
        "--min-instructions",
        type=int,
        default=1,
        help="Specify the minimum number of instructions for evolved programs",
    )
    parser.add_argument(
        "-maxi",
        "--max-instructions",
        type=int,
        default=100,
        help="Specify the maximum number of instructions for evolved programs",
    )
    parser.add_argument(
        "-f",
        "--fitness",
        type=str,
        choices={"ce", "p"},
        default="p",
        help="Choose the fitness function: 'p' for accuracy percentage, 'ce' for cross-entropy",
    )
    parser.add_argument(
        "-pg",
        "--p-grow",
        type=int,
        default=25,
        help="Specify the chance (in %) to incrementally increase the instruction count of a program",
    )
    parser.add_argument(
        "-pm", "--p-mutate", type=int, default=25, help="Specify the chance (in %) of an individual program mutating"
    )
    parser.add_argument(
        "-pc", "--p-cross", type=int, default=25, help="Specify the chance (in %) of crossover when creating new offspring"
    )
    parser.add_argument(
        "-pa",
        "--p-area",
        type=int,
        default=10,
        help="Specify the probability (in %) of an instruction working with a tensor slice instead of singular value",
    )
    parser.add_argument("--mutate-regs", type=int, default=1, help="How many register values to mutate")
    parser.add_argument("--mutate-inst", type=int, default=1, help="How many isntructions to mutate")
    parser.add_argument("--elite", type=int, default=3, help="Elite to be kept after purge/elimination")
    parser.add_argument(
        "--elite--equal",
        action="store_true",
        help="Elite individuals are sampled with uniform distribution",
    )
    parser.add_argument("-r", "--regs", nargs="+", default=(42,), help="Shape of working registers - tuple of ints")

    # Utility
    parser.add_argument("--debug", action="store_true", help="Log debug information")

    # Return parsed arguments
    return parser.parse_args()


def validate_arguments(args: argparse.Namespace) -> None:
    """Validate arguments with asserts - exit upon failure"""
    # LGP configuration check
    assert 1 < args.elite < args.population, (
        f"Kept elite ({args.elite}) has to be lower then population ({args.population}) "
        f"and higher then 1 for eventual crossovers."
    )
    assert args.population > 0, f"Population size should be at least 3 (got {args.population})"
    assert args.gens > 0, f"Generations should be greater than 0 ( got {args.gens})"

    # Program mutation configuration check
    assert (
        0 <= args.p_crossover <= 100
    ), f"Crossover probability is in % and should be in interval <0,100> (got {args.p_crossover})"
    assert (
        0 <= args.mutation_p <= 100
    ), f"Mutation probability is in % and should be in interval <0,100> (got {args.p_crossover})"
    assert 0 <= args.p_grow <= 100, f"Growth probability is in % and should be in interval <0,100> (got {args.p_crossover})"
    assert (
        0 <= args.p_area <= 100
    ), f"Area instruction probability is in % and should be in interval <0,100> (got {args.p_crossover})"

    # Program configuration check
    assert args.min_instructions > 0, f"Minimum instructions should be greater than 0 (got {args.min_instructions})"
    assert args.max_instructions > 0, f"Maximum instructions should be greater than 0 (got {args.max_instructions})"
    assert args.min_instructions <= args.max_instructions, (
        f"Minimum number of instructions ({args.min_instructions}) should be lower or equal to "
        f"maximum number of instructions ({args.max_instructions})"
    )
    assert args.mutate_regs > 0, f"Mutate registers should be at lest 0, (got {args.mutate_regs})"
    assert args.mutate_inst > 0, f"Mutate instructions should be at least 0 (got {args.mutate_inst})"

    # Dataset checks
    assert args.test is None or (
        args.test[0] > 0 and args.test[1] > 0
    ), f"Test configuration values should be a tuple of 2 numbers greater than 0 (got {args.test})"
    assert (
        args.normalize[0] < args.normalize[1]
    ), f"Normalize values should be a pair of ascending numbers (got {args.normalize})"
    assert 0 <= args.split <= 100, f"Split percentage should be in interval <0,100> (got {args.split})"

    # Path checks
    assert args.load is None or isfile(args.load)
    assert not isfile(args.model_dir), f"Path to model directory {args.model_dir} is a file!"
    assert not isfile(args.logging_dir), f"Path to logging directory {args.logging_dir} is a file!"


def prepare_lgp_configuration(args: argparse.Namespace) -> Dict:
    """Map parsed arguments into dictionary \
        - keyword arguments for LGP class instance"""
    return {
        "population": args.population,
        "area_p": args.p_area,
        "grow_p": args.p_grow,
        "generations": args.gens,
        "min_inst": args.min_instructions,
        "max_inst": args.max_instructions,
        "fitness": args.fitness,
        "mutation_p": args.p_mutate,
        "mutate_reg": args.mutate_regs,
        "mutate_inst": args.mutate_inst,
        "crossover_p": args.p_cross,
        "elite": args.elite,
        "equal_elite": args.elite_equal,
        "hidden_regfield": args.regs,
        "logging_dir": args.logging_dir,
        "model_dir": args.model_dir,
    }


def prepare_dataset_configuration(args: argparse.Namespace) -> Tuple[Dict]:
    """Map parsed arguments into dictionary \
        - keyword arguments for Dataset class instance"""
    return {
        "name": args.dataset,
        "root": args.data_dir,
        "edge_size": args.resize,
        "data_split": args.split,
        "normalize": args.normalize,
        "test": args.test,
    }


def prepare_directories(args: argparse.Namespace) -> None:
    """
    Make sure provided paths to directories actually contain directories
    """
    makedirs(args.model_dir, exist_ok=True)
    # TODO: Dumping progress directory


def train_lgp(lgp: LGP, dataset: Dataset) -> None:
    """Train LGP on training part of dataset"""
    # Disable gradient computation for faster results
    with torch.no_grad():
        lgp.fit(dataset.test_X, dataset.test_y)


def evaluate_program(lgp: LGP, dataset: Dataset) -> None:
    """Evaluate the whole population of LGP"""
    # Disable gradient computation for faster results
    with torch.no_grad():
        lgp.evaluate_population(dataset.eval_X, dataset.eval_y, use_percent=True)

    logging.info(f"Best evaluated accuracy: {lgp.fitness[0]}")
    logging.info(f"Achieved by program:\n{lgp.best_program}")


def load_lgp(dataset: Dataset, program: Union[Program, None], lgp_kwargs: Dict) -> LGP:
    """Instantiate LGP class from provided dataset, program and kwargs"""
    lgp = LGP(dataset, program, **lgp_kwargs)

    logging.debug(f"LGP object initialized in configuration of:\n{lgp}")


def load_dataset(dataset_kwargs: Dict) -> Dataset:
    """Instantiate Dataset class with provided kwargs"""
    # Disable torch gradients for faster computation
    with torch.no_grad():
        dataset = Dataset(**dataset_kwargs)

    logging.debug(f"Dataset loaded with configuration:\n{dataset}")
    return dataset


def load_program(path: str, reg_shape: Tuple[int]):
    """
    Load program from pickled file and assert configuration \
    complience with current LGP configuration
    @args:
        path:   Path to pickled Program instance
    """
    with open(path, "rb") as file:
        program = pickle.load(file)

    # Program is trained with certain register dimensions, check current config for complience
    assert list(program.hidden_regfiled.shape) == reg_shape, (
        f'Loaded program\'s hidden register fields {", ".join(map(str, program.hidden_regfiled.shape))} must '
        f'be equal to the CLI option {", ".join(map(str, reg_shape))}!'
    )

    logging.info(f"Program loaded from path {path} in configuration of:\n{program}")


# Execute code
if __name__ == "__main__":

    main()
