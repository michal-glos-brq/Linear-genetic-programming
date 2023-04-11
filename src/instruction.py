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

# Binary operations for use
BINARY = [
    torch.add, torch.sub, torch.mul, torch.div
]

# Unary operations for use
UNARY = [
    torch.exp, torch.log, torch.sin, torch.cos
]

# For arbitrary index representing operation parity, there is a tuple of:
#  ((list of operations of given parity), number of operands)
instruction_parity = {
    0: (BINARY, 2),
    1: (UNARY, 1)
}

# Register names - properties of parent program
registers = [
    'reg_images', 'reg_hidden', 'reg_results'
]

class Instruction:
    '''
    Single instruction to be executed
    
    Importand properties:
        _parity:    index into instruction_parity dict      (immutable)
        parity:     value from instruction_parity dict
        program:    corresponding program instance
        operation:  torch function to apply
        registers:  indexes into registers array
        indices:    indices into corresponding registers in program instance 
    '''

    def __init__(self, program):
        '''
        Initialize the instruction
        
        @args:
            program
        '''
        # Keep pointing to the parent program
        self.program = program
        # Choose operation parity
        self.choose_parity()
        # Choose operation
        self.choose_operation()
        # Choose operands
        self.choose_operands()

    def choose_parity(self):
        '''Choose one of operation types'''
        # Randomly choose between binary and unary instruction type
        self._parity = np.random.randint(0, len(instruction_parity))

    @property
    def parity(self):
        '''Obtain parity operation and operant counts for self._parity'''
        return instruction_parity[self._parity]

    def choose_operation(self):
        '''When parity set, randomly choose one of corresponding operations'''
        # Randomly choose operation according to parity
        operation = np.random.randint(0, len(self.parity[0]))
        # Select it from array of operations
        self.operation = self.parity[0][operation]

    def choose_operands(self):
        '''Choose randomly register - Input image / Hidden registers / Result registers'''
        # Obtain register names
        self.operation_regs = [np.random.choice(registers) for _ in range(self.parity[1] + 1)]
        # Obtain register indices (must be tuple to choose single value and not slice)
        self.operation_idx = [tuple(np.random.randint(getattr(self.program, reg + '_shape')))
                              for reg in self.operation_regs]


    @torch.no_grad()
    def execute(self):
        '''Execute vectorized instruction on whole dataset'''
        # Unary operations
        if self.parity[1] == 1:
            # Obtain the source register
            src = getattr(self.program, self.operation_regs[0])
            # Compute the result
            result = self.operation(src[self.operation_idx[0]])
        
        # Binary operations
        elif self.parity[1] == 2:
            # Obtain the source registers
            src1 = getattr(self.program, self.operation_regs[0])
            src2 = getattr(self.program, self.operation_regs[1])
            # Compute the result
            result = self.operation(src1[self.operation_idx[0]], src2[self.operation_idx[1]])
        
        # Write the result
        result_register = getattr(self.program, self.operation_regs[-1])
        result_register[self.operation_idx[-1]] = result

    def mutate(self):
        '''
        Mutate instruction. Instruction keeps the same parity but could change operation,
        operands registers, operadns indices or result destination
        '''
        pass

    def __repr__(self):
        '''String representation of instruction and it's instructions (developer friendly)'''
        pass

    def __str__(self):
        '''String representation of instruction and it's instructions (human friendly)'''
        pass

