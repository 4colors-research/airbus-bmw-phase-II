'''
This file is part of Garona. 
(c) 4colors Research Ltd, 2024

This file contains the implementation of the amplitude amplification routine.
'''

import numpy as np
import itertools, copy 
from colorama import Fore, Back

from qiskit import QuantumCircuit, transpile
from qiskit.visualization import plot_histogram
from qiskit_braket_provider import BraketProvider
from qiskit_braket_provider import BraketLocalBackend



def linear_combination_oracle(qc, input_qubits, output_qubit, weights, b):        
    '''Applies a series of controlled rotations to encode a linear combination of input values (represented by qubits) onto an output qubit.'''
    for i, weight in enumerate(weights):
        angle = 2 * np.arcsin(abs(weight))  # Convert weight magnitude to an angle
        if weight >= 0:
            qc.crx(angle, input_qubits[i], output_qubit)  # Controlled rotation for positive weight
        else:
            qc.crx(-angle, input_qubits[i], output_qubit)  # Controlled rotation for positive weight

        #Account for the constant term.
        angle = 2 * np.arcsin(abs(b))  
        if b >= 0:
            qc.rx(angle, output_qubit) 
        else:
            qc.rx(-angle, output_qubit)


def custom_diffusion_operator(qc, input_qubits, last_qubit):
    '''Creates a custom diffusion (or inversion about the mean) operator on a set of qubits.'''
    # Apply H and X to all qubits except the last one
    qc.h(input_qubits)
    qc.x(input_qubits)
    
    # Reflect only the states where the last qubit is 1
    qc.h(last_qubit)
    qc.mcx(input_qubits, last_qubit)  # Multi-controlled X targeting the last qubit
    qc.h(last_qubit)
    
    # Undo the X and H gates to restore original states
    qc.x(input_qubits)
    qc.h(input_qubits)


def classical_local_search(weights, b, obj_coeffs):
    '''Performs an exhaustive local search (clasically).'''
    weights = copy.copy(weights)
    b = copy.copy(b)
    obj_coeffs = copy.copy(obj_coeffs)

    n_constraints = weights.shape[0]
    n_coeffs = weights.shape[1]
        
    for i in range(n_constraints):
        W = np.sum(np.abs(weights[i])) + np.abs(b[i])
        weights[i] /= W
        b[i] /= W

    ranges = [range(2)] * n_coeffs
    cartesian_product = list(itertools.product(*ranges))

    assignemnts = [list(i) for i in list(cartesian_product)]    
    return assignemnts



def quantum_local_search(weights, b, obj_coeffs, n_shots, device, n_of_iterations=1):    
    '''
    Performs a quantum-based local search algorithm to optimize a problem with constraints represented by the input parameters. 
    It normalizes the constraints and objective function coefficients, sets up a quantum circuit with qubits representing constraints and objectives, 
    and applies custom oracles and diffusion operations to find the best configurations; finally, 
    it measures and returns the highest-ranking bitstrings (solutions) and their objective values.
    '''    
    weights = copy.copy(weights)
    b = copy.copy(b)
    obj_coeffs = copy.copy(obj_coeffs)

    n_constraints = weights.shape[0]
    n_coeffs = weights.shape[1]

    for i in range(n_constraints):
        W = np.sum(np.abs(weights[i])) + np.abs(b[i])
        weights[i] /= W
        b[i] /= W

    W = np.sum(np.abs(obj_coeffs))
    nonscaled_obj_coeffs = copy.copy(obj_coeffs)
    obj_coeffs /= W
    
    n_qubits = n_coeffs
    n_ancilla_qubits = 2*n_constraints
    total_n_qubits = n_qubits + n_ancilla_qubits
    obj_function_qubit = total_n_qubits - 2
    last_qubit = total_n_qubits - 1
    
    qc = QuantumCircuit(total_n_qubits)  
    qc.h(range(n_qubits))

    # Apply the oracle on all the constraints 
    for i in range(n_constraints):
        constraint_qubit =  n_qubits + i
        linear_combination_oracle(qc, range(n_coeffs), constraint_qubit, weights[i], b[i])
        qc.rx(np.pi/4, constraint_qubit)
        qc.cx(constraint_qubit, last_qubit)

    # Apply the oracle on objective function
    linear_combination_oracle(qc, range(n_coeffs), obj_function_qubit, obj_coeffs, 0)
    qc.rx(np.pi/4, obj_function_qubit)
    qc.cx(obj_function_qubit, last_qubit)
    
    # Apply the diffusion operator
    for _ in range(n_of_iterations):   
        qc.x(last_qubit)  
        qc.cz(0, last_qubit) 
        qc.x(last_qubit)  
        custom_diffusion_operator(qc, list(range(total_n_qubits-1)), last_qubit)

    # print(f'Number of gates = {qc.size()}, depth = {qc.depth()}, width = {qc.width()}, non-local gates = {qc.num_nonlocal_gates()}')

    qc.measure_all()
    qc_braket = transpile(qc, backend=device)
    job = device.run(qc_braket, shots=n_shots)
    result = job.result()
    counts = result.get_counts()

    sorted_measurements = sorted(counts.items(), key=lambda item: item[1], reverse=True)

    results = []

    for bit in ['0', '1']:
        for measurement, count in sorted_measurements[0:10]:
            if measurement[0]!=bit:
                continue
                    
            bits = [int(b) for b in measurement[-n_qubits:]]
            bits.reverse()
            results.append(bits)            

    obj_values = []
    for measurement, count in sorted_measurements[0:10]:
        obj = 0
        for j in range(n_qubits):
            obj += int(measurement[total_n_qubits-1-j])*nonscaled_obj_coeffs[j]    
        obj_values.append(obj)

    return results
