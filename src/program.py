                           #######################
                           #@$%&             &%$@#
                           #!    Michal Glos    !#
                           #!     EVO - 2023    !#
                           #!        __         !#
                           #!      <(o )___     !#
                           #!       ( ._> /     !#
                           #@$%&     `---'   &%$@#
                           #######################

import torch
import numpy as np

from pprint import pformat

from instruction import Instruction

################################################################################
#####                      Linear program class                            #####
################################################################################

class Program:
    '''
    Representation of linear program - a list of Instruction instances
    
    @properties:
        instructions:       List of linear instructions
        hidden_reg_init:    Initialization values of hidden (working) memory
        result_reg_init:    Initialization values of result vector memory
    '''

    def __init__(self, max_instr, min_instr, obj_shape, classes, hidden_reg_shape):
        '''
        Initialize the Program class, initialize register init values,
        initialize instructions

        @args:
            max_instr:          Maximal lenght of program (number of instructions)
            min_instr:          Minimal lenght of program (number of instructions)
            obj_shape:          Shape of objects to be classified
            classes:            Number of classes (Output register shape)
            hidden_reg_shape:   Shape of hidden register field
        '''
        # Define max and min lenght of program
        self.max_instr = max_instr
        self.min_instr = min_instr

        # Obtain dimensions of register fields (Used in Instruction.choose_operands)
        self.input_reg_shape = obj_shape
        self.hidden_reg_shape = hidden_reg_shape
        self.result_reg_shape = (classes,)

        # Declare register init values tensors, would be set in one of classmethods
        self.hidden_reg_init = None
        self.result_reg_init = None

        # Declare the instruction field
        self.instructions = []


    ################################################################################
    #####                             Reproduction                             #####
    ################################################################################

    @staticmethod
    def creation(max_instr, min_instr, obj_shape, classes, hidden_reg_shape):
        '''
        Create random individual with no ancestors

        @args:
            max_instr:          Maximal lenght of program (number of instructions)
            min_instr:          Minimal lenght of program (number of instructions)
            obj_shape:          Shape of objects to be classified
            classes:            Number of classes (Output register shape)
            hidden_reg_shape:   Shape of hidden register field
        '''
        # Create the new individual
        new_individual = Program(max_instr, min_instr, obj_shape, classes, hidden_reg_shape)

        # Generate initial values of register fields
        new_individual.hidden_reg_init = torch.normal(0, 1, new_individual.hidden_reg_shape)
        new_individual.result_reg_init = torch.normal(0, 1, new_individual.result_reg_shape)

        # Generate the shortest possible program randomly
        for _ in range(min_instr):
            new_individual.instructions.append(Instruction(new_individual))


    @staticmethod
    def crossover(mother, father):
        '''
        Perform crossover of 2 parents, do not mutate yet
        
        @args:
            mother: Program object instance
            father: Program object instance
        '''
        # Create the new individual
        new_individual = Program(mother.max_instr, mother.min_instr, mother.obj_shape,
                                 mother.classes, mother.hidden_reg_shape)

        # Crossover of registers (random access crossover)
        # Create masks for result and hidden registers (apprx. 50% is False, 50% True)
        result_mask = torch.randint(0, 2, new_individual.result_reg_shape) == 1
        hidden_mask = torch.randint(0, 2, new_individual.hidden_reg_shape) == 1
        # Compose the offspring register initialization tensors
        new_individual.result_reg_init = (result_mask * mother.result_reg_init) + (~result_mask * father.result_reg_init)
        new_individual.hidden_reg_init = (hidden_mask * mother.hidden_reg_init) + (~hidden_mask * father.hidden_reg_init)

        # Now perform the instruction crossover
        # Ratio - how many (in fraction) instruction would be taken from mother and (1-ratio) from father
        ratio = np.random.random()
        # Index of crossover into mothers instructions
        mother_crossover_index = int(ratio * len(mother.instructions))
        # Index of crossover into fathers instructions
        father_crossover_index = int(ratio * len(father.instructions))
        # Now just slice parents instruction arrays and concatenate them into their offspring
        new_genome = mother.instructions[:mother_crossover_index] + father.instructions[father_crossover_index:]
        # Copy the new genome into the new individual (instructions are bound to parents)
        new_individual.instructions = [instr.copy(new_individual) for instr in new_genome]

        return new_individual


    @staticmethod
    def transcription(parent):
        '''
        Copy parents genome to it's offspring, no mutations yet
        
        @args:
            parent: Program instance
        '''
        # Create the new individual
        new_individual = Program(parent.max_instr, parent.min_instr, parent.obj_shape,
                                 parent.classes, parent.hidden_reg_shape)

        # Copy the register field init values
        new_individual.result_reg_init = parent.result_reg_init.clone()
        new_individual.hidden_reg_init = parent.hidden_reg_init.clone()

        # Copy the instructions from parent
        new_individual.instructions = [instr.copy(new_individual) for instr in parent.instructions]

        return new_individual


    ################################################################################
    #####                             Genome tweaks                            #####
    ################################################################################

    def mutate(self, registers, instructions):
        '''
        Perform program mutation
        
        Mutation is performed in both: register init values and instructions
            - registers get values added sampled from normal distribution (0,1)
            - instructions are instruction related, mutation command is passed upon instruction
 
        @args:
            registers:      How many registers to mutate?
            instructions:   How many instructions to mutate?
        '''
        # Calculate how many mutations would be performed on hidden registers
        register_sizes = torch.FloatTensor([self._hid_regs, self._res_regs])
        hid_reg_mutations = register_sizes.multinomial(registers, replacement=True).sum()
        res_reg_mutations = registers - hid_reg_mutations

        # Perform register mutations (This gets hacky ... makes perfect sense though)
        ## [torch.randperm(self._hid_regs)[:hid_reg_mutations]] - This indexes hid_reg_mutations 
        ## random elements of the Tensor without repeating, adds values sampled from normal dist.
        self.hidden_reg_init.view(-1)[torch.randperm(self._hid_regs)[:hid_reg_mutations]] += \
            torch.normal(0, 1, (hid_reg_mutations, ))
        self.result_reg_init.view(-1)[torch.randperm(self._res_regs)[:res_reg_mutations]] += \
            torch.normal(0, 1, (res_reg_mutations, ))

        # Choose random distinct instructions and perform mutations
        for instr in np.random.choice(self.instructions, size=instructions, repeat=False):
            instr.mutate()


    def _del_instruction(self, idx=None):
        '''Delete instruction on index idx, random if idx is None'''
        # Obtain valid index into instruction list
        if idx is None:
            idx = np.random.randint(0, len(self.instructions))
        else:
            assert 0 <= idx < len(self.instructions), f'Tried to delete instruction \
                on index {idx} from list of {len(self.instructions)} instructions.'
        # Delete the chosen instruction
        del self.instructions[idx]


    def grow(self):
        '''Add random instruction, remove another when max limit exceeded to compensate'''
        # Remove instruction first (when over limit) to not delete new instruction
        if len(self.instructions) >= self.max_instr:
            self._del_instruction()
        # Append new and random instruction at the end of instruction list
        self.instructions.append(Instruction(self))


    def shrink(self):
        '''Delete one random instruction, add one when lower limit exceeded to stay within limits'''
        # Delete random instruction
        self._del_instruction()
        # Add instruction if lower limit was exceeded
        if len(self.instructions) < self.min_instr:
            self.instructions.append(Instruction(self))


    ################################################################################
    #####                                 Utils                                #####
    ################################################################################

    def eval(self, X, y_gt, fitness_fn):
        '''
        Evaluate the program on the given dataset
        
        Returns ratio of correctly classified objects

        @args:
            X:          Tensor of objects to classify
            y_gt:       Tensor of ground truth labels
            fitness_fn: Fitness function
        '''
        # Start with register initialization
        self.input_registers = X.clone()        # Input registers could be manipulated
        self.hidden_registers = self.hidden_reg_init.repeat(X.shape[0], * (self.hidden_registers.sim() * [1]))
        self.result_registers = self.result_reg_init.repeat(X.shape[0], 1)

        # Now execute all instructions on data
        for instruction in self.instructions:
            instruction.execute()

        # Collapse the classification probabilities
        pred_y = self.result_registers.argmax(dim=1)

        # Return the value for chosen fitness functions
        return fitness_fn(pred_y, y_gt)

    def get_working_reg(self):
        '''Obtain results or hidden register with probability proportional to their sizes'''
        proportions = np.array(self._hid_regs, self._res_regs) / (self._hid_regs + self._res_regs)
        return np.random.choice(['hidden', 'result'], p=proportions)

    @property
    def _hid_regs(self):
        '''Number of hidden registers'''
        return self.hidden_reg_init.shape.numel()

    @property
    def _res_regs(self):
        '''Number of result registers'''
        return self.result_reg_init.shape.numel()

    @property
    def _info_dict(self):
        '''Obtain information about Program instance in a dictionary'''
        return {}

    def __repr__(self):
        '''String representation of program and it's instructions (developer friendly)'''
        return pformat(self._info_dict)
