"""
This python module provides all necessary operations for linear program mutation - 
    it's registers' initial values and instructions.
"""

from random import random, choice, randint, sample

import torch

from LGA.program import Program
from LGA.instruction import Instruction
from LGA.operations import UNARY, BINARY, AREA, INPUT_REGISTERS, OUTPUT_REGISTERS, UNARY_OP_RATIO
from utils.other import true_with_probability


class Mutations:
    """Class implementing static methods for mutation operations."""

    @staticmethod
    def mutate_operation(instr: Instruction) -> None:
        """
        Mutate instruction operation function.
        Randomly (uniformly) sample either new operation function or area operation
        function in case of instruction having area operation.

        Args:
            instr (Instruction): Instruction to undergo mutation
        """
        if instr.is_area_operation and true_with_probability(0.5):  # 50/50 chance
            instr.area_operation_function = choice(AREA)

        else:
            is_unary_operation = random() < UNARY_OP_RATIO or instr.is_area_operation
            instr.operation_function = choice(UNARY if is_unary_operation else BINARY)

            # Changed from binary to unary, thus remove 2nd operand
            if is_unary_operation and instr.operation_parity == 2:
                instr.operation_parity = 1
                instr.input_registers = instr.input_registers[:1]
                instr.input_register_indices = instr.input_register_indices[:1]
            # Changed from unary to binary, thus add 2nd operand
            elif not is_unary_operation and instr.operation_parity == 1:
                instr.operation_parity = 2
                instr.input_registers.append(choice(INPUT_REGISTERS))
                instr.input_register_indices.append(
                    Instruction.get_random_index(instr.parent.lga.register_shapes_dict[instr.input_registers[1]])
                )

    @staticmethod
    def mutate_input_registers(instr: Instruction) -> None:
        """
        Mutate instruction operands.
        Randomly change one of input register or/and indexing/slicing indices.

        Args:
            instr (Instruction): Instruction to undergo mutation
        """
        # Choose register on which the mutation is performed
        input_registers_index = randint(0, instr.operation_parity - 1)
        instr.input_registers[input_registers_index] = choice(INPUT_REGISTERS)
        input_register_shape = instr.parent.lga.register_shapes_dict[instr.input_registers[input_registers_index]]

        # Randomly choose new index
        if instr.is_area_operation:
            instr.input_register_indices[input_registers_index] = Instruction.get_random_slice(
                input_register_shape, instr.parent.lga.torch_device
            )
        else:
            instr.input_register_indices[input_registers_index] = Instruction.get_random_index(input_register_shape)

    @staticmethod
    def mutate_output_register(instr: Instruction) -> None:
        """Randomly choose output register from all possible output registers"""
        # Randomly choose new register
        instr.output_register = choice(OUTPUT_REGISTERS)

        # Randomly generate index into that new register
        register_shape = instr.parent.lga.register_shapes_dict[instr.output_register]
        instr.output_register_indices = Instruction.get_random_index(register_shape)

    @staticmethod
    def mutate_area(instr: Instruction) -> None:
        """From area instruction, create non-area instr and vice versa"""
        if instr.is_area_operation:
            # Choose new indices from within the slice indexing tuples
            instr.input_register_indices[0] = tuple([...] + [choice(ids).item() for ids in instr.input_register_indices[0]])
            instr.area_operation_function = None
        else:
            # Area operations require UNARY primary operation
            if instr.operation_parity == 2:
                instr.operation_parity = 1
                instr.operation_function = choice(UNARY)
                instr.input_registers = instr.input_registers[:1]
                instr.input_register_indices = instr.input_register_indices[:1]

            instr.area_operation_function = choice(AREA)
            register_shape = instr.parent.lga.register_shapes_dict[instr.input_registers[0]]
            instr.input_register_indices[0] = Instruction.get_random_slice(register_shape, instr.parent.lga.torch_device)

        instr.is_area_operation = not instr.is_area_operation

    @staticmethod
    def mutate_instruction(instr: Instruction) -> None:
        """
        Perform instruction mutation
        One of following mutations will be performed:
             - Operation function
             - Change of input or output register indices
             - Change of input or output register
             - Area operation added or removed
             - Area operation function (if instr operates with tensor slices)
        """
        # Area mutation has fixed probability in order to keep area operations in desired number
        if true_with_probability(instr.parent.lga.area_instruction_p):
            Mutations.mutate_area(instr)
        else:
            mutation_function = choice(
                [Mutations.mutate_input_registers, Mutations.mutate_output_register, Mutations.mutate_operation]
            )
            mutation_function(instr)

    @staticmethod
    def mutate_program(program: Program, registers_to_mutate: int, instructions_to_mutate: int) -> None:
        """
        Perform mutation of register initialization values and instructions

        Args:
            registers_to_mutate (int):      Number of registers to mutate
            instructions_to_mutate (int):   Number of instructions to mutate
        """
        # Introduce random element
        registers_to_mutate = randint(1, registers_to_mutate)
        instructions_to_mutate = randint(1, instructions_to_mutate)
        # Obtain number of registers for each registerfield initialization values to be mutated
        register_sizes = torch.FloatTensor([program.lga.hidden_register_count, program.lga.result_register_count])
        res_reg_mutations = min(
            register_sizes.multinomial(registers_to_mutate, replacement=True).sum(), program.lga.result_register_count
        )
        hid_reg_mutations = min(registers_to_mutate - res_reg_mutations, program.lga.hidden_register_count)

        # Add samples from normal(0,1) to selected registers
        if hid_reg_mutations:
            hidden_register_indices = torch.randint(
                program.lga.hidden_register_count, (hid_reg_mutations,), device=program.lga.torch_device
            )
            hidden_register_mutation_values = torch.randn((hid_reg_mutations,), device=program.lga.torch_device)
            program.hidden_register_initial_values.view(-1)[hidden_register_indices] += hidden_register_mutation_values

        if res_reg_mutations:
            result_register_indices = torch.randint(
                program.lga.result_register_count, (res_reg_mutations,), device=program.lga.torch_device
            )
            result_register_mutation_values = torch.randn((res_reg_mutations,), device=program.lga.torch_device)
            program.result_register_initial_values.view(-1)[result_register_indices] += result_register_mutation_values

        # Randomly choose instructions_to_mutate Instruction instances for mutation
        instructions_to_mutate = min(instructions_to_mutate, len(program.instructions))
        for instr in sample(program.instructions, k=instructions_to_mutate):
            Mutations.mutate_instruction(instr)
