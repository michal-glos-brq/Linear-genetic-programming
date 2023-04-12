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
import torch

from instruction import Instruction

################################################################################
#####                      Linear program class                            #####
################################################################################

class Program:
    '''
    Representation of linear program - a list of Instruction instances
    
    @properties:
        instructions:   List of linear instructions
    '''

    def __init__(self, max_len, min_len, obj_shape, classes):
        '''
        Initialize the Program class, initialize register init values,
        initialize instructions

        @args:
            max_len:        Maximal lenght of program (number of instructions)
            min_len:        Minimal lenght of program (number of instructions)
            obj_shape:      Shape of objects to be classified
            classes:        Number of classes (Output register shape)
        '''
        # Define max and min lenght of program
        self.max_len = max_len
        self.min_len = min_len
        self.grow = grow
        # Obtain dimensions of register fields (Used in Instruction.choose_operands)
        self.reg_images_shape, self.reg_hidden_shape = obj_shape, obj_shape
        self.reg_results_shape = (classes,)
        # Generate initialization values for classification and hidden registers
        self.reg_hidden_init = torch.normal(0, 1, self.reg_hidden_shape)
        self.reg_results_init = torch.normal(0, 1, self.reg_results_shape)
        # Generate the program
        self.generate_instructions()
        # Choose the fitness function
        self.fitness_fn = Program.fitness_dict[fitness]


    def generate_instructions(self):
        '''Create the new program generated by luck'''
        # Decide how many instructions to start with, if growth set to True, start on min_len
        # of program (in instructions), choose randomly from interval <min_len, max_len> otherwise
        instr_to_generate = self.min_len if self.grow else np.random.randint(self.min_len, self.max_len + 1)
        self.instructions = [Instruction(self) for _ in range(instr_to_generate)]



    def grow(self):
        '''Grow by 1 instruction, remove random one when limit is exceeded to stay within limits'''
        pass

    def shrink(self):
        '''Delete one random instruction, add one when lower limit exceeded to stay within limits'''
        # TODO
        pass

    def copy(self):
        '''Create a copy of itself'''
        pass


    def eval(self, X, y_gt, fitness_fn):
        '''
        Evaluate program on given dataset
        
        Returns classification success in percents

        @args:
            X:
            y_gt:

        '''
        # Start with register initialization
        # Dataset registers
        self.reg_images = X
        # Hidden registers initialize and set to init values
        self.reg_hidden = X.new_empty(X.shape)
        self.reg_hidden[:] = self.reg_hidden_init
        # Result registers initialize and set to init values 
        # TODO: Less hacky
        self.reg_results = y.new_empty((len(y), *self.reg_results_shape))
        self.reg_results[:] = self.reg_results_init

        # Now execute all instructions on data
        for instruction in self.instructions:
            instruction.execute()

        # TODO: Implement statistics
        import pdb; pdb.set_trace()
        # Return the value for chosen fitness functions
        return self.fitness_fn(self.reg_results, y)


    def mutate(self):
        '''Perform program mutation'''
        pass

    def __add__(self, other):
        '''Programs crossover'''
        assert type(self) is type(other), 'Crossover could be performed on two instances of class Program only!'
        
    def __repr__(self):
        '''String representation of program and it's instructions (developer friendly)'''
        return 'Program'

    def __str__(self):
        '''String representation of program and it's instructions (human friendly)'''
        return 'Program'
