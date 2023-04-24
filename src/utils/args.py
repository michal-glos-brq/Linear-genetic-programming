"""
This module is responsible for parsing CLI arguments and validating them with if keyword
Author: Michal Glos (xglosm01)
Project: EVO - 2023
"""

from argparse import ArgumentParser, Namespace
from os.path import isfile
from typing import Tuple, Dict

from torchvision.datasets import __dict__ as torchvision_datasets

from LP.operations import BINARY, UNARY, AREA
from utils.losses import FITNESS_FUNCTIONS

def parse_arguments() -> Namespace:
    """
    Parses command-line arguments using the argparse library.

    Returns:
        argparse.Namespace: The parsed command-line arguments.
    """
    parser = ArgumentParser(
        description="Linear Genetic Algorithm (LGA) for image classification",
    )

    # Prepare variables
    binary_operations_names = [fn.__name__ for fn in BINARY]
    unary_operations_names = [fn.__name__ for fn in UNARY]
    area_operations_names = [fn.__name__ for fn in AREA]

    fitness_function_descriptions = [f"{fit_code}: {fit_fn.__name__}" for fit_code, fit_fn in FITNESS_FUNCTIONS.items()]
    
    # Dataset-related arguments
    parser.add_argument(
        "-d", "--dataset", type=str, default="MNIST", help="Choose a torchvision dataset e.g., MNIST, CIFAR10, etc. (default MNIST)"
    )
    parser.add_argument("--data-dir", type=str, default="data", help="Specify the directory to store the dataset")
    parser.add_argument("--resize", type=int, help="Resize images to the specified square (edge size in pixels)")
    parser.add_argument(
        "--split", type=int, default=10, help="Specify the percentage of the dataset used for training (0-100)"
    )
    parser.add_argument(
        "-n", "--normalize", type=int, nargs=2, default=[0, 1],
        help="Specify the interval to normalize the dataset (e.g., 0 1 or -1 1)",
    )
    parser.add_argument(
        "--test", type=int, nargs=2, 
        help="Create a custom test dataset by specifying the number of classes and data entries (e.g., 10 1000)",
    )

    # App flow
    # Loading program could get tricky, registerfield shapes have to match, is validation part of this script
    parser.add_argument("-l", "--load", type=str, help="Load a pre-trained program from the specified path")
    parser.add_argument(
        "-md", "--model-dir", type=str, default="programs",
        help="Specify the directory to save the best programs in the format: {--dataset}{--resize}_{generation}_{fitness}.p"
    )
    parser.add_argument("--model-save-interval", type=int, default=1, help="Save best model each N generations")
    parser.add_argument("-log", "--logging-dir", default="logging", type=str, help="Specify the directory for logging")

    # LGA parameters
    parser.add_argument("-p", "--population", type=int, default=42, help="Specify the population size (default: 42)")
    parser.add_argument("-g", "--gens", type=int, default=60, help="Specify the number of generations for the LGA to evolve (default: 60)")
    parser.add_argument("--runs", type=int, default=10, help="Specify the number of times to run the algorithm (default: 10)")
    parser.add_argument(
        "-mini", "--min-instructions", type=int, default=1,
        help="Specify the minimum number of instructions for evolved programs",
    )
    parser.add_argument(
        "-maxi", "--max-instructions", type=int, default=100,
        help="Specify the maximum number of instructions for evolved programs",
    )
    parser.add_argument(
        "-f", "--fitness", type=str, choices=FITNESS_FUNCTIONS.keys(), default="ac",
        help=f"Choose the fitness function - {', '.join(fitness_function_descriptions)}",
    )
    parser.add_argument(
        "-pg", "--p-grow", type=int, default=25,
        help="Specify the chance (in %) to incrementally increase the instruction count of a program (default: 25)",
    )
    parser.add_argument(
        "-ps", "--p-shrink", type=int, default=25,
        help="Specify the chance (in %) to incrementally decrease the instruction count of a program (default: 25)",
    )
    parser.add_argument(
        "-pm", "--p-mutate", type=int, default=25, help="Specify the chance (in %) of an individual program mutating"
    )
    parser.add_argument(
        "-pc", "--p-cross", type=int, default=25, help="Specify the chance (in %) of crossover when creating new offspring"
    )
    parser.add_argument(
        "-pa", "--p-area", type=int, default=10,
        help="Specify the probability (in %) of an instruction working with a tensor slice instead of singular value",
    )
    parser.add_argument("--mutate-regs", type=int, default=1, help="Specify the number of register values to mutate (default: 1)")
    parser.add_argument("--mutate-inst", type=int, default=1, help="Specify the number of instructions to mutate (default: 1)")
    parser.add_argument("--elite", type=int, default=3, help="Elite to be kept after selection")
    parser.add_argument(
        "--elite-equal", action="store_true",
        help="Sample elite individuals for crossover and selection equally, regardless of their fitness",
    )
    parser.add_argument("-r", "--regs", nargs="+", default=(42,), help="Specify the shape of working registers as a tuple of integers (default: (42,))")
    parser.add_argument(
        "-b", "--binary", nargs="+", type=str, default=None, choices=binary_operations_names,
        help=f"Choose binary opeartions used in linear program from: {', '.join(binary_operations_names)}",
    )
    parser.add_argument(
        "-u", "--unary", nargs="+", type=str, default=None, choices=unary_operations_names,
        help=f"Choose unary opeartions used in linear program from: {', '.join(unary_operations_names)}",
    )
    parser.add_argument(
        "-a", "--area", nargs="+", type=str, default=None, choices=area_operations_names,
        help=f"Choose area opeartions used in linear program from: {', '.join(area_operations_names)}",
    )

    # Utility
    parser.add_argument("--debug", action="store_true", help="Enable loggings DEBUG level")

    # Return parsed arguments
    return parser.parse_args()

def validate_lga_configuration(args: Namespace) -> None:
    """Validate LGA configuration parameters."""
    if not (1 < args.elite < args.population):
        raise ValueError(
            f"Kept elite ({args.elite}) has to be lower than population ({args.population}) "
            f"and higher than 1 for eventual crossovers."
        )
    if args.population <= 3:
        raise ValueError(f"Population size should be at least 3 (got {args.population})")

    if args.gens <= 0:
        raise ValueError(f"Generations should be greater than 0 (got {args.gens})")

    if args.runs <= 0:
        raise ValueError(f"Number of runs has to be higher than 0 (got {args.runs}).")

def validate_program_mutation(args: Namespace) -> None:
    """Validate program mutation configuration parameters."""
    if not (0 <= args.p_cross <= 100):
        raise ValueError(
            f"Crossover probability should be a number between 0 and 100 (inclusive) (got {args.p_cross})"
        )

    if not (0 <= args.p_mutate <= 100):
        raise ValueError(
            f"Mutation probability should be a number between 0 and 100 (inclusive) (got {args.p_mutate})"
        )

    if not (0 <= args.p_grow <= 100):
        raise ValueError(
            f"Growth probability should be a number between 0 and 100 (inclusive) (got {args.p_grow})"
        )

    if not (0 <= args.p_shrink <= 100):
        raise ValueError(
            f"Shrink probability should be a number between 0 and 100 (inclusive) (got {args.p_shrink})"
        )

    if not (0 <= args.p_area <= 100):
        raise ValueError(
            f"Area instruction probability should be a number between 0 and 100 (inclusive) (got {args.p_area})"
        )

    if args.mutate_regs < 0:
        raise ValueError(f"Mutate registers should be at least 0 (got {args.mutate_regs})")

    if args.mutate_inst < 0:
        raise ValueError(f"Mutate instructions should be at least 0 (got {args.mutate_inst})")

def validate_program_configuration(args: Namespace) -> None:
    """Validate program configuration parameters."""
    if args.min_instructions <= 0:
        raise ValueError(f"Minimum instructions should be greater than 0 (got {args.min_instructions})")

    if args.max_instructions <= 0:
        raise ValueError(f"Maximum instructions should be greater than 0 (got {args.max_instructions})")

    if args.min_instructions > args.max_instructions:
        raise ValueError(
            f"Minimum number of instructions ({args.min_instructions}) should be lower or equal to "
            f"maximum number of instructions ({args.max_instructions})"
        )

def validate_dataset(args: Namespace) -> None:
    """Validate dataset configuration parameters."""
    if args.test is not None and args.test[0] < 2:
        raise ValueError(f"Number of classes in test dataset should be at least 2 (got {args.test})")

    if args.test is not None and args.test[1] < 2:
        raise ValueError(f"Number of entriers in test dataset should be at least 2 (got {args.test})")

    if (0 < args.split < 100):
        raise ValueError(f"Dataset of split point should be inbetween 0 and 100 (exclusive) (got {args.split})")

    if args.dataset not in torchvision_datasets:
        raise ValueError(f"Invalid dataset name ({args.dataset}). Choose from: {', '.join(torchvision_datasets.keys())}")

def validate_paths(args: Namespace) -> None:
    """Validate path parameters"""
    if args.load is not None and not isfile(args.load):
        raise FileNotFoundError(f"File with alleged pickled program{args.load} does not exist!")

    if isfile(args.model_dir):
        raise FileExistsError(f"Model directory already exists as a file: {args.dataset}")

    if isfile(args.logging_dir):
        raise FileExistsError(f"Logging directory already exists as a file: {args.output}")


def validate_arguments(args: Namespace) -> None:
    """Validate CLI arguments"""
    validate_lga_configuration(args)
    validate_program_configuration(args)
    validate_program_mutation(args)
    validate_dataset(args)
    validate_paths(args)


def prepare_configurations(args: Namespace) -> Tuple[Dict, Dict]:
    """
    Map parsed arguments into two dictionaries - keyword arguments for LGA and Dataset class instances.
    
    Args:
        args (Namespace): Parsed arguments from argparse.
    
    Returns:
        Tuple[Dict, Dict]: Two dictionaries containing configurations for LGA and Dataset instances.
    """

    lga_config = {
        "population": args.population,
        "area_instruction_p": args.p_area,
        "program_grow_p": args.p_grow,
        "program_shrink_p": args.p_shrink,
        "generations": args.gens,
        "min_instructions": args.min_instructions,
        "max_instructions": args.max_instructions,
        "fitness": args.fitness,
        "mutation_p": args.p_mutate,
        "mutate_registers": args.mutate_regs,
        "mutate_instructions": args.mutate_inst,
        "crossover_p": args.p_cross,
        "elite": args.elite,
        "equal_elite": args.elite_equal,
        "hidden_register_shape": tuple(args.regs),  # When used as shape, lists throw error
        "logging_dir": args.logging_dir,
        "model_dir": args.model_dir,
        "model_save_interval": args.model_save_interval,
        "binary": args.binary,
        "unary": args.unary,
        "area": args.area
    }

    dataset_config = {
        "name": args.dataset,
        "root": args.data_dir,
        "edge_size": args.resize,
        "data_split": args.split,
        "normalize": args.normalize,
        "test": args.test,
    }

    return lga_config, dataset_config
