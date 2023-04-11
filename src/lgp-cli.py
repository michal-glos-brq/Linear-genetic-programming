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

import torch

from datasets import Dataset
from program import Program
from LGP import LGP

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
parser.add_argument('--grow', action='store_true', 
                    help='Incrementally increase the number of instructions instead of random lenght')
parser.add_argument('--min-instr', type=int, default=1,
                    help='Minimal number of instructions per program')
parser.add_argument('--max-instr', type=int, default=100,
                    help='Maximal number of instructions per program')
parser.add_argument('--fitness', type=str, choices={'ce', 'p'}, default='p',
                    help=('Choose from following fitness functions. '
                          'Cross-entropy (ce), percentage of correctly classified (p).')
)

# Utility
parser.add_argument('--verbose-training', action='store_true',
                    help='Dump training information inbetween epochs')


# Execute code
if __name__ == '__main__':

    # Parse the arguments
    args = parser.parse_args()
    
    # Obtain the dataset
    dataset = Dataset(name=args.dataset, root=args.data_root, edge_size=args.resize,
                      data_split=args.data_split, normalize=args.normalize,
                      test=args.test_dataset)

    import pdb; pdb.set_trace()

    # TODO: Implement program loading
    program = Program.load_program(args.load_program)

    # If --training flag - start training
    if args.train:
        # Disable gradient computation for faster results
        with torch.no_grad():
            lgp = LGP(args.population, args.gens, args.grow, args.min_instr, args.max_instr,
                      args.fitness, dataset.obj_shape, dataset.class_count)
            lgp.fit(dataset.test_X, dataset.test_y)


    # If --eval flag - start evaluating
    if args.eval:

        # Check if program obtained
        if program is None:
            sys.exit('Attempt to evaluate non-defined program (None). Either train a new one or provide path to a pickled one.')
        
        # Proceed to evaluation on the whole evaluation dataset while torch grads are turned off
        with torch.no_grad():
            results = program.eval(dataset.eval_X, dataset.eval_y)
