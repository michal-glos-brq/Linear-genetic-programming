import torch
import numpy as np

from LP.operations import UNARY, BINARY, AREA, INPUT_REGISTERS, OUTPUT_REGISTERS
from utils.other import true_with_probability

class Mutations:
    @staticmethod
    def mutate_operation(instruction) -> None:
        """Mutate the operation used in this Instruction instance"""
        # If mutating area operation, it's 50/50 which operation would be changed
        if instruction.is_area_operation and true_with_probability(0.5):
            instruction.area_operation_function = np.random.choice(AREA)
        # Otherwise, start with choosing parity of operation randomly with p proportional to their count
        else:
            unary = (np.random.randint(0, (len(UNARY) + len(BINARY))) < len(UNARY)) or instruction.is_area_operation
            instruction.operation_function = np.random.choice(UNARY) if unary else np.random.choice(BINARY)

            # Changed from binary to unary, just get rid of the second operand
            if unary and instruction.operation_parity == 2:
                instruction.operation_parity = 1
                instruction.input_registers = instruction.input_registers[:1]
                instruction.input_register_indices = instruction.input_register_indices[:1]
            # Changed from unary to binary, we have to add one random input register with index
            elif not unary and instruction.operation_parity == 1:
                instruction.operation_parity = 2
                instruction.input_registers.append(np.random.choice(INPUT_REGISTERS))
                instruction.input_register_indices.append(
                    tuple(map(lambda x: np.random.randint(0, x), instruction.input_registers[1].shape))
                )
        
    @staticmethod
    def mutate_input_registers(instruction) -> None:
        """Mutate one of inputs (change index or register and index)"""
        # Choose register on which the mutation is performed
        input_registers_indices = np.random.randint(0, instruction.operation_parity)
        instruction.input_registers[input_registers_indices] = np.random.choice(INPUT_REGISTERS)

        # Randomly choose new index
        if instruction.is_area_operation:
            new_indices = []
            for dim in instruction.reg_shapes[instruction.input_registers[input_registers_indices]]:
                # Choose random 2 numbers for each dimension with correct values
                indices = tuple([num.item() for num in torch.randperm(dim)[:2].sort().values])
                new_indices.append(tuple(indices))
            instruction.input_register_indices[input_registers_indices] = tuple(new_indices)
        else:
            instruction.input_register_indices[input_registers_indices] = tuple(
                map(lambda x: np.random.randint(0, x), instruction.reg_shapes[instruction.input_registers[input_registers_indices]])
            )

    @staticmethod
    def mutate_output_register(instruction) -> None:
        """Change the destination of Instruction result"""
        # Randomly choose new register
        instruction.output_register = np.random.choice(OUTPUT_REGISTERS)
        # Randomly generate index into that new register
        instruction.output_register_indices = tuple(map(lambda x: np.random.randint(0, x), instruction.reg_shapes[instruction.output_register]))

    @staticmethod
    def mutate_area(instruction) -> None:
        """Change index into output register or output register and its' index"""
        # From area operation to scalar operation, unary parity is implicit here, could index with 0
        if instruction.is_area_operation:
            instruction.input_register_indices[0] = tuple([np.random.randint(lb, hb) for lb, hb in instruction.input_register_indices[0]])
        else:
            # This instruction has to be unary in order to work now with area tensor slices
            if instruction.operation_parity == 2:
                instruction.operation_parity = 1
                instruction.operation_function = np.random.choice(UNARY)
                instruction.input_registers = instruction.input_registers[:1]
                instruction.input_register_indices = instruction.input_register_indices[:1]
            # Now we have unary operation, let's add random area operation and generate slice indices
            instruction.area_operation_function = np.random.choice(AREA)
            new_indices = []
            for dim in instruction.reg_shapes[instruction.input_registers[0]]:
                indices = tuple([num.item() for num in torch.randperm(dim)[:2].sort().values])
                new_indices.append(tuple(indices))
            instruction.input_register_indices[0] = tuple(new_indices)
        # Flag the change 'officialy'
        instruction.is_area_operation = not instruction.is_area_operation

    @staticmethod
    def mutate_instruction(instruction) -> None:
        """
        Perform instruction mutation

        According to LGA convention, let's just really change only one thing here

        Could be one of following things:
            Operation
            Input index/register
            Output index/register
            Area operation added or removed
        Area instructions could mutate also on:
            Area of operands
        """
        # Area mutation has fixed probability
        # (1 - area prob.) will be equally distributed alongside other mutation types
        not_area_instruction_p = (1 - instruction.parent.area_instruction_p) / 3.0
        # Randomly choose one of mutation functions and execute it
        mutation_function = np.random.choice(
            [instruction._mutate_area, instruction._mutate_input_registers, instruction._mutate_output_register, instruction._mutate_area],
            p=[not_area_instruction_p, not_area_instruction_p, not_area_instruction_p, instruction.parent.area_instruction_p],
        )
        # Execute the mutation function
        mutation_function()


    @staticmethod
    def mutate_program(program, registers_to_mutate: int, instructions_to_mutate: int) -> None:
        """
        Perform program mutation

        Mutation is performed in both: register init values and instructions
            - registers get values added sampled from normal distribution (0,1)
            - instructions are instruction related, mutation command is passed upon instruction

        Args:
            registers_to_mutate (int):  How many registers to mutate?
            instructions_to_mutate (int):   How many instructions to mutate?
        """
        # Calculate how many mutations would be performed on hidden registers
        register_sizes = torch.FloatTensor([program._hid_regs, program._res_regs])
        hid_reg_mutations = register_sizes.multinomial(registers_to_mutate, replacement=True).sum()
        res_reg_mutations = registers_to_mutate - hid_reg_mutations

        # Perform register mutations (This gets hacky ... makes perfect sense though)
        ## [torch.randperm(self._hid_regs)[:hid_reg_mutations]] - This indexes hid_reg_mutations
        ## random elements of the Tensor without repeating, adds values sampled from normal dist.
        # Apply minimum to not mutate more registers than there are registers ...
        program.hidden_register_initial_values.view(-1)[torch.randperm(program._hid_regs)[:hid_reg_mutations]] += torch.normal(
            0, 1, (min(hid_reg_mutations, program._hid_regs),)
        ).to(program.torch_device)
        program.result_register_initial_values.view(-1)[torch.randperm(program._res_regs)[:res_reg_mutations]] += torch.normal(
            0, 1, (min(res_reg_mutations, program._res_regs),)
        ).to(program.torch_device)

        # Choose random distinct instructions and perform mutations (Do not exceed number of instructions)
        for instruction in np.random.choice(
            program.instructions, size=min(instructions_to_mutate, len(program.instructions)), replace=False
        ):
            Mutations.mutate_instruction(instruction)
