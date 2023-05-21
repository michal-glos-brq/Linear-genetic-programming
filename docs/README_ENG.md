
### Linear Genetic Programming Python Module

This module provides an implementation of Linear Genetic Algorithm (LGA) for image classification. It allows users to configure and run experiments using various options.

#### Requires Python3.8+
##### Anaconda is the preffered way to install the application
`pip install -r requirements.txt` 

CLI Options:

- Dataset-related arguments
  - `-d`, `--dataset`: Choose a torchvision dataset e.g., MNIST, CIFAR10, etc. (default: MNIST)
  - `--data-dir`: Specify the directory to store the dataset
  - `--resize`: Resize images to the specified square (edge size in pixels)
  - `--split`: Specify the percentage of the dataset used for training (0-100)
  - `-n`, `--normalize`: Specify the interval to normalize the dataset (e.g., 0 1 or -1 1)
  - `--test`: Create a custom test dataset by specifying the number of classes and data entries (e.g., 10 1000)

 - App flow
   - `-l`, `--load`: Load a pre-trained program from the specified path
   - `-md`, `--model-dir`: Specify the directory to save the best programs in the format: {dataset}_{resize}_{fitness}_{timestamp}.p
   - `-log`, `--logging-dir`: Specify the directory for logging

 - LGA parameters
   - `-p`, `--population`: Specify the population size (default: 42)
   - `-g`, `--gens`: Specify the number of generations for the LGA to evolve (default: 60)
   - `--runs`: Specify the number of times to run the algorithm (default: 10)
   - `-mini`, `--min-instructions`: Specify the minimum number of instructions for evolved programs
   - `-maxi`, `--max-instructions`: Specify the maximum number of instructions for evolved programs
   - `-f`, `--fitness`: Choose the fitness function (see below for available options)
   -  `-pg`, `--p-grow`: Specify the chance (in %) to incrementally increase the instruction count of a program (default: 25)
   -  `-ps`, `--p-shrink`: Specify the chance (in %) to incrementally decrease the instruction count of a program (default: 25)
   -  `-pm`, `--p-mutate`: Specify the chance (in %) of an individual program mutating
   -  `-pc`, `--p-cross`: Specify the chance (in %) of crossover when creating new offspring
   -  `-pa`, `--p-area`: Specify the probability (in %) of an instruction working with a tensor slice instead of singular value
   -  `--mutate-regs`: Specify the max. number of register values to mutate (default: 1)
   -  `--mutate-inst`: Specify the max. number of instructions to mutate (default: 1)
   -  `--elite`: Elite to be kept after selection
   -  `--elite-equal`: Sample elite individuals for crossover and selection equally, regardless of their fitness
   -  `-r`, `--regs`: Specify the shape of working registers as a tuple of integers (default: (42,))
   -  `-b`, `--binary`: Choose binary operations used in linear program
   -  `-u`, `--unary`: Choose unary operations used in linear program
   -  `-a`, `--area`: Choose area operations used in linear program

- Utility
   -  `--debug`: Enable loggings DEBUG level