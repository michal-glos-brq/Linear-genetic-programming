                           #######################
                           #@$%&             &%$@#
                           #!    Michal Glos    !#
                           #!     EVO - 2023    !#
                           #!        __         !#
                           #!      <(o )___     !#
                           #!       ( ._> /     !#
                           #@$%&     `---'   &%$@#
                           #######################

import numpy as np
from tqdm import tqdm
from pprint import pformat

from program import Program

################################################################################
#####                       Fitness functions                              #####
################################################################################
# Fitness fucntions:
#   take 2 args - array of predicted labels and GT labels
def fitness_correctly_classified(y_pred, y_gt):
    '''Compute the percentage of correctly predicted classes'''
    # Get boolean array with correctness of prediction for each object
    predictions = (y_pred.argmax(axis=0) == y_gt)
    # Return the percentage of correct predictions
    return (predictions.sum() / len(predictions)) * 100

# def fitness_cross_entropy(y_pred, y_gt):
#     '''Compute fitness based on classification cross-entropy'''
#     # TODO: Implement
#     pass

FITNESS = {
    'p': fitness_correctly_classified
}

################################################################################
#####                      LGP algorithm class                             #####
################################################################################


class LGP:
    '''
    Linear Genetic Programming controll algorithm/object

    important properties: (not mentioned in __init__ docstring, __init__ args with same names could differ, Achtung!)
        population:         NumPy array of Program instances (the whole population)
        population_bound:   Maximum number of individuals in population
        elite_size:         Number of individuals consisting elite
        hidden_reg_shape:   Shape of hidden registers
        fitness_fn:         Function computing fitness
        fitness:            Fitness ranking of individuals
        evaluated:          Current population was evaluated nad fitness values are current
    '''

    def __init__(self, dataset, program, population=42, elite=3, equal_elite=False, 
                 generations=60, min_inst=1, max_inst=100, fitness='p', hidden_reg_shape=(42,),
                 area_p=10, grow_p=25, mutation_p=25, crossover_p=25):
        '''
        Initialize LGP algorithm object

        @args:
            population:         Upper population bound
            elite:              Number of best individuals considered elite
            equal_elite:        Do not weight elite individuals chances to procreate by fitness
            generations:        For how many generations to evolve
            program:            Program instance or None (Proto Program ... :D )
            min_inst:           Minimum required lenght of program in instructions
            max_inst:           Maximum required lenght of program in instructions
            fitness:            Fitness function (choose from 'ce' and 'p'), see Program class
            dataset:            Dataset instance
            grow_p:               Incrementaly increase the lenght of program (in %)
            mutation_p:         Mutation probability (in %)
            crossover_p:        Crossover probability (in %)
            hidden_reg_shape:    Shape of hidden register field
        '''
        # Save the configuration into properties
        self.population_bound = population          # Max population
        # Initialize either empty population or single individual "population"
        self.population = np.array([program] if program else [])              

        self.generations = generations              # How many generation to evolve
        self.actual_generation = 0                  # Generation evolving right now

        self.grow_p = grow_p / 100.                     # Probability (in %) for a program to grow_p (an instruction)
        self.min_inst = min_inst                    # Minimal number of Program instructions
        self.max_inst = max_inst                    # Maximal number of Program instructions
        self.hidden_reg_shape = hidden_reg_shape    # Shape of hidden register field

        self.fitness_fn = FITNESS[fitness]          # Fitness code (string)
        self.fitness = []                           # Fitness values for each individual

        self.obj_shape = dataset.obj_shape          # Shape of data object
        self.classes = dataset.classes              # How many classes to discriminate

        self.mutation_p = mutation_p / 100.         # Main GA parameters, mutation and
        self.crossover_p = crossover_p / 100.       # crossover probabilities
        self.area_p = area_p / 100.                 # Probability of area-processing instruction
        self.elite_size = elite                     # Size of elite to keep to another gen
        self.equal_elite = equal_elite              # Distribute elite equally (not proportionally to fitness) when repopulating

        # Other utility configuration
        self.evaluated = False                      # Indicator whether the population was evaluated



    ################################################################################
    #####                     Adding on population                             #####
    ################################################################################

    def spawn_individuals(self):
        '''Generate Program instances to meet the population criteria'''
        # How many individuals are missing from population
        pop_missing = self.population_bound - len(self.actual_population)
        # Create new individuals
        self.population = np.append(self.population, [self.new_individual() for _ in range(pop_missing)])
        # Invalidate population evaluation, since new individuals were added 
        # fn new_individual requires fitness, therefore invalidate it at the end
        self.evaluated = False


    def new_individual(self):
        '''Create a new Program individual based on LGP object properties'''
        # If population was not evaluated, we do not have elite, generate individuals randomly
        # Mutation not even considered because it's randomly generated at the first place
        if not self.evaluated:
            return Program.creation(self.max_inst, self.min_inst, self.obj_shape,
                                    self.classes, self.hidden_reg_shape, self.area_p)

        # If population was evaluated, let's take the elite and repopulate
        elite_pop, elite_fitness = self.elite
        # Use fitness values as probability distribution of individuals or use uniform distribution among elite if self.equal_elite
        elite_probs = (np.ones_like(elite_fitness) / len(elite_pop)) if self.equal_elite else (elite_fitness / elite_fitness.sum())
        # Decide which actions to perform. If neither, program will be shrunk (it's already in the new population, so the new one gotta change a bit)
        crossover, mutate, grow_p = self.repopulation_probs >= np.random.random(3)
        
        ## Actual process of generating the new offspring begins here
        # Create new offspring either by crossover of 2 parents or by simple selection of elite individual
        if crossover:
            father, mother = np.random.choice(elite_pop, size=2, replace=False, p=elite_probs)
            offspring = Program.crossover(father, mother)
        else:
            parent = np.random.choice(elite_pop, p=elite_probs)
            offspring = Program.transcription(parent)
        
        # Mutate if randomly chosen to
        if mutate:
            offspring.mutate()
        
        # Grow_p if randomly chosen to
        if grow_p:
            offspring.grow_p()

        # When neither of following actions was chosen, let's delete individual's 
        # random instruction to decrease the probability of the same programs in population
        if not crossover and not mutate and not grow_p:
            offspring.shrink()
        
        # Finally return the newborn
        return offspring

    @property
    def repopulation_probs(self):
        '''Return numpy array of crossover, mutation and grow_pth probabilities'''
        return np.array([self.crossover_p, self.mutation_p, self.grow_p])


    ################################################################################
    #####             Evaluating and eliminating population                    #####
    ################################################################################

    def evaluate_population(self, X, y_gt):
        '''
        Evaluate the whole population of programs and sort it from best to worse
        
        This method is used for evolving and evaluating, therefore data are passed as arguments

        @args:
            X:      Data Tensor
            y_gt:   Labels Tensor
        '''
        # Rank each program with it's fitness
        self.fitness = [individual.eval(X, y_gt, self.fitness_fn) for individual in self.population]
        # Sort population according to fitness (Best first)
        self.population = self.population[np.argsort(self.fitness)]
        # Once we sorted the population, sort fitness values to weight individual
        self.fitness.sort(reverse=True)


    def eliminate(self):
        '''Eliminate inferior Programs'''
        del self.population[self.elite_size:]


    ################################################################################
    #####                     LGP algorithm controll                           #####
    ################################################################################

    def fit(self, X, y_gt):
        '''
        Execute the evolution in the name of The God itself
        
        @args:
            X:      Training dataset (data)
            y_gt:   Training dataset (labels)
        '''
        with tqdm(range(self.generations), desc='Evolving ...') as pbar:
            for _ in pbar:
                # Update generation iteration
                self.actual_generation += 1
                # Fill the whole population with individuals
                self.spawn_individuals()
                # Rank them with fitness function
                self.evaluate_population(X, y_gt)
                # Keep only the elite
                self.eliminate()
                # Repeat till the last generation, update the progress bar
                pbar.set_description(f'Evolving ... ({self.fitness[0]} %)')
        

    ################################################################################
    #####                                Utils                                 #####
    ################################################################################

    @property
    def elite(self):
        '''Return the array of elite individuals and theirs fitness'''
        return self.population[:self.elite_size], self.fitness[:self.elite_size]

    def _info_dict(self):
        '''Obtain information about Program instance in a dictionary'''
        return {
            'kokos': 'kokos'
        }

    def __repr__(self):
        '''String representation of program and it's instructions (developer friendly)'''
        return pformat(self._info_dict, width=120)
