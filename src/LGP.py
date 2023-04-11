                           #######################
                           #@$%&             &%$@#
                           #!    Michal Glos    !#
                           #!     EVO - 2023    !#
                           #!        __         !#
                           #!      <(o )___     !#
                           #!       ( ._> /     !#
                           #@$%&     `---'   &%$@#
                           #######################

from program import Program

import numpy as np
from tqdm import tqdm

class LGP:
    '''
    Linear Genetic Programming controll algorithm/object
    '''

    def __init__(self, population, generations, grow, min_instructions,
                 max_instructions, fitness, obj_shape, classes):
        '''
        Initialize LGP object

        @args:
            population:         Upper population bound
            generations:        For how many generations to evolve   
            grow:
            min_instructions:
            max_instructions:
            fitness:
            obj_shape:
            classes:
        '''
        # Save the configuration into properties
        self.population_bound = population          # Max population
        self.population = np.array([])              # Population of objects (Program)

        self.generations = generations              # How many generation to evolve
        self.actual_generation = 0                  # Generation evolving right now

        self.grow = grow                            # Let programs grow? (bool)
        self.min_instructions = min_instructions    # Minimal number of Program instructions
        self.max_instructions = max_instructions    # Maximal number of Program instructions

        self.fitness_fn = fitness                   # Fitness code (string)
        self.fitness = []                           # Fitness values for each individual

        self.obj_shape = obj_shape                  # Shape of data object
        self.classes = classes                      # How many classes to discriminate





    def fill_in_population(self):
        '''
        Generate Program instances to meet the population criteria
        
        Some based on cross-over, some based on random generator
        '''
        # How many individuals are missing from population
        pop_missing = self.population_bound - self.actual_population
        # Create new individuals
        self.population = np.append(self.population, [self.new_individual() for _ in range(pop_missing)])


    def new_individual(self):
        '''Create a new Program individual based on LGP object properties'''
        return Program(self.max_instructions, self.min_instructions, self.obj_shape,
                       self.classes, self.grow, self.fitness_fn)


    def evaluate_population(self, X, y_gt):
        '''
        Evaluate the whole population of programs and sort it from best to worse
        
        self.population is a list of Programs, Programs id to that list
        will correspond to id into result list returned by this method

        @args:
            X:      Data Tensor
            y_gt:   Labels Tensor
        '''
        # Rank programs with fitness fn
        self.fitness = [individual.eval(X, y_gt) for individual in self.population]
        # Sort population according to fitness value (Best first)
        self.population = self.population[np.argsort(self.fitness)]
        # Sort fitness evaluations in order to keep them relevant to their programs
        self.fitness.sort(reverse=True)


    def eliminate(self, keep_best=1):
        '''
        Eliminate inferior Programs, sort Porgrams according to fitness
        
        Makes sense only when population was already evaluated

        Play tournamen - let only top K live long enough to procreate

        @args:
            keep_best:  How many individuals to keep
        '''
        del self.population[keep_best:]


    def fit(self, X, y_gt):
        '''
        Evolve programs
        
        @args:
            X:
            y_gt:
        '''
        with tqdm(range(self.generations), desc='Evolving ...') as pbar:
            for _ in pbar:
                # Update generation iteration
                self.actual_generation += 1
                # Fill the whole population with individuals
                self.fill_in_population()
                # Rank them with fitness function
                self.evaluate_population(X, y_gt)
                # Keep only the elite
                self.eliminate()
                # Repeat till the last generation, update the progress bar
                pbar.set_description(f'Evolving ... ({self.fitness[0]} %)')
        
        # return the best program
        return self.population[0]


    @property
    def actual_population(self):
        '''Return the number of Programs in population'''
        return self.population.size
