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


################################################################################
#####                       Easy access constants                          #####
################################################################################

def _id(_input):
    '''
    Function returning it's only argument

    Used as unary function when area instructions is initiated so
    area opearations could be performed without unary op. preceeding it
    '''
    return _input

# Binary operations
BINARY = [
    torch.add,
    torch.sub,      
    torch.mul,
    torch.div,
    torch.pow
]

# Unary operations
UNARY = [
    torch.exp,
    torch.log,
    torch.sin,
    torch.cos,
    torch.tan,
    torch.htan,
    torch.sqrt
]

# Area operations (Reducing area to single value)
AREA = [
    torch.mean,
    torch.median,
    torch.sum,
    torch.max,
    torch.min,
    torch.prod
]

IN_REGS = ['input_registers', 'hidden_registers']
OUT_REGS = ['hidden_registers', 'result_registers']

################################################################################
#####                          Instruction class                           #####
################################################################################

class Instruction:
    '''
    Single instruction to be executed
    
    important properties:
        parent:     Program instance which contains this Instruction instance
        op_par:     Instruction parity (integer: 1, 2, ...)
        op_fn:      Instruction function
        in_reg:     Instruction input registers - would be passed into the op_fn (list of strings)
        out_reg:    Output register - output of op_fn is stored there  (string)
        in_idx:     Indices into input registers (pair-tuples if are slices are used)
        out_id:     Index into output register
        area:       Does Instruction instance work with tensor slices? (bool)
        area_op:    Area operation, None if area == False
    '''

    def __init__(self, parent):
        '''
        Initialize the instruction as object
        
        @args:
            parent:     Program class instance (parent program)
        '''
        # Keep pointing to the parent program
        self.parent = parent
        # Initialize other properties
        self.op_par = None
        self.op_fn = None
        self.in_reg = None
        self.in_idx = None
        self.out_id = None
        self.out_reg = None
        self.area = None
        self.area_op = None


    ################################################################################
    #####                  Sampling from instruction space                     #####
    ################################################################################


    def execute(self):
        '''Execute vectorized instruction on whole dataset'''
        # Obtain the register fields (Tensors)
        in_regs = [getattr(self.parent, _in_reg) for _in_reg in self.in_reg]
        out_reg = getattr(self.parent, self.out_reg)

        # Obtain the operands
        operands = [_reg[_idx] for _idx, _reg in zip(self.in_idx, in_regs)]

        # Calculate the result
        result = self.op_fn(*operands)

        # If area instruction, reduce tensor with area operation to scalar
        if self.area:
            result = self.area_op(result)

        # Save the result into the result register
        out_reg[self.out_id] = result


    ################################################################################
    #####                              Evolution                               #####
    ################################################################################

    def _mutate_operation(self):
        '''Let's mutate the operation used in this Instruction instance'''
        # If mutating area operation, it's 50/50 which operation would be changed
        if self.area and np.random.random < 0.5:
            self.area_op = np.random.choice(AREA)
        # Otherwise, start with choosing parity of operation randomly with p proportional to their count
        else:
            unary = (np.random.randint(0, (len(UNARY) + len(BINARY))) < len(UNARY)) or self.area
            self.op_fn = np.random.choice(UNARY) if unary else np.random.choice(BINARY)
            
            # Changed from binary to unary, just get rid of the second operand
            if (unary and self.op_par == 2):
                self.op_par = 1
                self.in_reg = self.in_reg[:1]
                self.in_idx = self.in_idx[:1]
            # Changed from unary to binary, we have to add one random input register with index
            elif (not unary and self.op_par == 1):
                self.op_par = 2
                self.in_reg.append(np.random.choice(IN_REGS))
                self.in_idx.append(tuple(map(lambda x: np.random.randint(0, x), self.in_reg[1].shape)))

                
    
    def _mutate_in_reg(self):
        '''Mutate one of inputs into Instruction operation'''
        # Choose register on which the mutation is performed
        in_red_idx = np.random.randint(0, self.op_par)
        self.in_reg[in_red_idx] = np.random.choice(IN_REGS)
        new_register = getattr(self.parent, self.in_reg[in_red_idx])
        # Randomly choose new index
        if self.area:
            new_idx = []
            for dim in new_register.shape:
                # Choose random 2 numbers for each dimension with correct values
                _idx, _ = torch.randperm(dim)[:2].sort()
                new_idx.append(tuple(_idx))
            self.in_idx[in_red_idx] = tuple(new_idx)
        else:
            self.in_idx[in_red_idx] = tuple(map(lambda x: np.random.randint(0, x), new_register.shape))


    def _mutate_out_reg(self):
        '''Change the destination of Instruction result'''
        # Randomly choose new register
        self.out_reg = np.random.choice(OUT_REGS)
        new_register = getattr(self.parent, self.out_reg)
        # Randomly generate index into that new register
        self.out_id = tuple(map(lambda x: np.random.randint(0, x), new_register.shape))

    def _mutate_area(self):
        '''Mutate the selected area of operands'''
        # TODO: Maybe guard this with some probability? 25% of instructions would eventually be AREA
        # From area operation to scalar operation, unary parity is implicit here, could index with 0 
        if self.area:
            self.in_idx[0] = tuple([np.random.randint(lb, hb) for lb, hb in self.in_idx[0]])
        else:
            # This instruction has to be unary in order to work now with area tensor slices 
            if self.op_par == 2:
                self.op_par = 1
                self.op_fn = np.random.choice(UNARY)
                self.in_reg = self.in_reg[:1]
                self.in_idx = self.in_idx[:1]
            # Now we have unary operation, let's add random area operation and generate slice indices
            self.area_op = np.random.choice(AREA)
            register = getattr(self.parent, self.in_reg[0])
            new_idx = []
            for dim in register.shape:
                _idx, _ = torch.randperm(dim)[:2].sort()
                new_idx.append(tuple(_idx))
            self.in_idx[0] = tuple(new_idx)
        # Flag the change 'officialy'
        self.area = not self.area


    def mutate(self):
        '''
        Perform instruction mutation
        
        According to LGA convention, let's just really change only one thing here
        
        Could be one of following things:
            Operation
            Input index/register
            Output index/register
        Area instructions could mutate also on:
            Area of operands 
        '''
        # Just randomly choose one of mutation functions and execute it
        np.random.choice([self._mutate_area, self._mutate_in_reg, self._mutate_out_reg, self._mutate_area])()


    def copy(self, parent):
        '''Create new instruction under provided parent'''
        # Create the object instance
        new_instruction = Instruction(parent)

        # Proceed with copying self properties into the new instruction
        new_instruction.op_par = self.op_par
        new_instruction.op_fn = self.op_fn
        new_instruction.out_reg = self.out_reg
        new_instruction.area = self.area
        # List shall be copied
        new_instruction.in_reg = self.in_reg.copy()
        new_instruction.area_idx = [idx.copy() for idx in self.area_idx]
        

    @staticmethod
    def random(parent):
        '''Generate a random instruction object instance'''
        # Create the object instance
        new_instruction = Instruction(parent)
        
        # Decide on area processing or single value processing
        new_instruction.area = np.random.random() < parent.area_p
        
        # Decide on parity (randomly, proportionally to number of operations)
        unary = (np.random.randint(0, (len(UNARY) + len(BINARY))) < len(UNARY)) or new_instruction.area
        new_instruction.op_par = 1 if unary else 2
        new_instruction.op_fn = np.random.choice(UNARY) if unary else np.random.choice(BINARY)

        # Now choose input and output registers (just strings, getattr will be used on parent)
        new_instruction.in_reg = [np.random.choice(IN_REGS) for _ in new_instruction.op_par]
        new_instruction.out_reg = np.random.choice(OUT_REGS)

        # If area instruction, generate slice ids
        if new_instruction.area:
            # We are confident this is an unary operation, so index with 0
            register = getattr(parent, new_instruction.in_reg[0])

            # For each register, for each dimenstion, get 2 unique indices in bounds of tensor shape
            # We can slice how many dimensions we want with tuple of par-tuples
            new_instruction.in_idx = []
            for dim in register.shape:
                _idx, _ = tuple(torch.randperm(dim)[:2].sort())
                new_instruction.in_idx.append(_idx)
            new_instruction.in_idx = [tuple(new_instruction.in_idx)]

            # And finally, randomly obtain the area operation (Add identity - only area operation would be applied)
            new_instruction.area_op = np.random.choice(AREA + [_id])
        # Otherwise, generate only normal ids
        else:
            registers = [getattr(new_instruction.parent, _reg) for _reg in new_instruction.in_reg]
            # Randomly generate indices into registers (Nasty!)
            new_instruction.in_idx = [
                tuple(map(lambda x: np.random.randint(0, x), _reg.shape)) for _reg in registers
            ]

        # Generate result register id
        out_register = getattr(new_instruction.parent, new_instruction.out_reg)
        new_instruction.out_id = tuple([np.random.randint(0, x) for x in out_register.shape])

        return new_instruction


    ################################################################################
    #####                                 Utils                                #####
    ################################################################################

    @property
    def _info_dict(self):
        '''Obtain a dictionary describing Instruction object instance (self)'''
        return {}

    def __repr__(self):
        '''String representation of instruction and it's instructions (developer friendly)'''
        return pformat(self._info_dict)

