#! /usr/bin/env python3
                           #######################
                           #@$%&             &%$@#
                           #!    Michal Glos    !#
                           #!     EVO - 2023    !#
                           #!        __         !#
                           #!      <(o )___     !#
                           #!       ( ._> /     !#
                           #@$%&     `---'   &%$@#
                           #######################

import argparse
import sys
import pickle
import logging

import torch

from datasets import Dataset
from LGP import LGP



# Execute code
if __name__ == '__main__':

    ################################################################################
    #####                      Parsing arguments                               #####
    ################################################################################

    # Create the argument parser
    parser = argparse.ArgumentParser()

    # Dataset-related arguments
    parser.add_argument('--dataset', type=str, default='MNIST',
                        help='Choose dataset from torchvision datasets')
    parser.add_argument('--data-root', type=str, default='data',
                        help='Choose dataset directory')
    parser.add_argument('--resize', type=int,
                        help='Choose image (always square) edge size in pixels')
    parser.add_argument('--data-split', type=int, default=10,
                        help='Percentage of dataset for testing each individual, the rest is used for final evaluation')
    parser.add_argument('--normalize', type=int, nargs=2, default=[0,1], 
                        help='Provide interval in which data are normalized')
    parser.add_argument('--max-entries', type=int, help='Limit data entries')
    parser.add_argument('--test-dataset', type=int, nargs=2,
                        help=('Use test (one-hot vectors as data) dataset. Provide 2 ints - number '
                              'of classes and number of entries.')
    )

    # App flow
    parser.add_argument('--train', action='store_true', help='Perform training')
    parser.add_argument('--eval', action='store_true', help='Perform evaluation')
    parser.add_argument('--load-program', type=str,
                        help='Provide path to pickled Program class instance')
    parser.add_argument('--save-program', type=str, default='programs',
                        help=('Directory to store best programs in format:\n'
                              '\t{--dataset}{--resize}_{fitness}_{timestamp}.p')
    )

    # LGP parameters
    parser.add_argument('--population', type=int, default=42,
                        help='Population of linear programs')
    parser.add_argument('--gens',  type=int, default=60,
                        help='How many generations to evolve')
    parser.add_argument('--min-instr', type=int, default=1,
                        help='Minimal number of instructions per program')
    parser.add_argument('--max-instr', type=int, default=100,
                        help='Maximal number of instructions per program')
    parser.add_argument('--fitness', type=str, choices={'ce', 'p'}, default='p',
                        help=('Choose from following fitness functions. '
                              'Cross-entropy (ce), percentage of correctly classified (p).')
    )
    parser.add_argument('--grow-p', type=int, default=25,
                        help='Chance (in %) to incrementally increase instruction count of program')
    parser.add_argument('--mutation-p', type=int, default=25,
                        help='Chance of individual mutation in percents')
    parser.add_argument('--crossover-p', type=int, default=25,
                        help='Chance of crossover when creating new offspring')
    parser.add_argument('--elite', type=int, default=3,
                        help='Elite to be kept after purge/elimination')
    parser.add_argument('--equal-elite', action='store_true', 
                        help=('Choose from elite individuals to copy or crossover equally, '
                            'not according to their fitness')
    )
    # Utility
    parser.add_argument('--debug', action='store_true',
                        help='Log debug information')

    # Parse the arguments
    args = parser.parse_args()

    # Perform some assertions on our CLI options
    assert 1 < args.elite < args.population, \
        (f'Kept elite ({args.elite}) has to be lower then population ({args.population}) '
         f'and higher then 1 for eventuall crossovers.')

    assert 0 <= args.mutation_p <= 100 and 0 <= args.crossover_p <= 100, \
        (f'Mutation probability ({args.mutation_p} %) and crossover probability ({args.crossover_p} %)'
         ' are in percents, therefore choose an integer from the interval of <0, 100>.')

    assert args.min_instr <= args.max_instr, \
        (f'Minimum number of instructions ({args.min_instr}) should be lower or equal to '
         f'maximum number of instructions ({args.max_instr})')

    ################################################################################
    #####                             Utilities                                #####
    ################################################################################

    # Configure logging
    logging.basicConfig(level=logging.DEBUG if args.debug else logging.INFO)
    
    ################################################################################
    #####                              Dataset                                 #####
    ################################################################################
    
    # Obtain the dataset
    dataset = Dataset(name=args.dataset, root=args.data_root, edge_size=args.resize,
                      data_split=args.data_split, normalize=args.normalize,
                      test=args.test_dataset)
    logging.debug(f'Dataset loaded with configuration:\n{dataset}')

    ################################################################################
    #####                          Loading program                             #####
    ################################################################################

    # Try to load program if path was provided
    program = None
    # If there is a non-empty string, try to load it as pickled file
    if args.load_program:
        with open(args.load_program, 'rb') as file:
            program = pickle.loads(file)
        logging.info(f'Program loaded from path {args.load_program} in configuration of:\n{program}')

    ################################################################################
    #####                            LGP training                              #####
    ################################################################################
    # If --train flag -> start training
    if args.train:
        # Disable gradient computation for faster results
        with torch.no_grad():
            lgp = LGP(dataset, program, population=args.population, grow=args.grow_p, generations=args.gens,
                      min_inst=args.min_instr, max_inst=args.max_instr, fitness=args.fitness,
                      mutation_p=args.mutation_p, crossover_p=args.crossover_p, elite=args.elite,
                      equal_elite=args.equal_elite)
            lgp.fit(dataset.test_X, dataset.test_y)

    ################################################################################
    #####                           LGP evaluation                             #####
    ################################################################################

    # TODO --v
    # If --eval flag - start evaluating
    if args.eval:

        # Check if program obtained
        if program is None:
            sys.exit('Attempt to evaluate non-defined program (None). Either train a new one or provide path to a pickled one.')
        
        # Proceed to evaluation on the whole evaluation dataset while torch grads are turned off
        with torch.no_grad():
            results = program.eval(dataset.eval_X, dataset.eval_y)
