#! /usr/bin/env python3

"""
Linear Genetic Programming (LGA) for image classification
Author: Michal Glos, EVO - 2023
"""

# TODO: Pickle the Best of Best programs

import logging
from pprint import pprint

from LGA.LGA import LGA
from utils.args import parse_arguments, validate_arguments, prepare_configurations
from utils.datasets import Dataset
from src.LP.program import Program


# Execute the code
if __name__ == "__main__":

    args = parse_arguments()

    # Validate provided arguments, exit upon failure
    validate_arguments(args)

    logging.basicConfig(level=logging.DEBUG if args.debug else logging.INFO)

    # Obtain dictionaries with object configurations from CLI args
    lga_kwargs, dataset_kwargs = prepare_configurations(args)

    # Load dataset
    dataset = Dataset(**dataset_kwargs)
    logging.info("Dataset loaded")
    logging.debug(f"Dataset configuration:\n{dataset}")

    # Try to load program if path was provided
    program = Program.load_program(args.load, args.regs) if args.load else None
    logging.info("Proto-program loaded" if program else "Proto-program not loaded")
    if program is not None:
        logging.debug(f"Proto-program:\n{program}")

    # Instantiate algorithm object
    lga = LGA(dataset, program, **lga_kwargs)
    logging.info("LGA class instance created!")
    logging.debug(f"LGA configuration:\n{lga}")

    # Run LGA
    lga.train(dataset, args.runs)

    # Print best programs and their fitness
    for fit, program in lga.best_programs:
        pprint({'fitness': fit, 'program': program})
