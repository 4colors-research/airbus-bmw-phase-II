'''
This file is part of Garona. 
(c) 4colors Research Ltd, 2024

This file contains code for Experiment 1.
'''

import quantum, qaoa

import numpy as np
import pandas as pd

import os, time, pickle
from tqdm import tqdm
from collections import Counter

import warnings
warnings.filterwarnings('ignore')

from qiskit_braket_provider import BraketProvider
from qiskit_braket_provider import BraketLocalBackend


def print_value_percentages(input_list):
    counts = Counter(input_list)
    total_count = len(input_list)
    results = []
    for value, count in counts.items():
        percentage = (count / total_count) * 100
        results.append(f"{value}: {percentage:.2f}%")  # Format to 2 decimal places
    print(", ".join(results))


def build_sorted_dataframe(classical_results, c, A, b):
    # Prepare data for DataFrame
    data = []
    for x in classical_results:
        # Calculate objective value
        obj_value = c @ x

        # Calculate infeasibility
        infeasibility = A @ x + b
        
        # Count violated constraints (positive infeasibility)
        violated_constraints = np.sum(infeasibility > 0)
        max_violation = np.max(infeasibility) if violated_constraints > 0 else 0

        # Append data (objective value, number of violated constraints, max violation)
        data.append({
            'x' : x,
            'Objective Value': obj_value,
            'Violated Constraints': violated_constraints,
            'Max Violation': max_violation
        })

    # Create DataFrame
    df = pd.DataFrame(data)

    # Sort the DataFrame based on the specified criteria
    sorted_df = df.sort_values(
        by=['Max Violation', 'Violated Constraints', 'Objective Value'],
        ascending=[True, True, True]  # All are ascending
    ).reset_index(drop=True)

    return sorted_df


def perform_experiment(n_coeffs, n_constraints, n_shots, results_filename):
    classical_times = []
    quantum_times = []
    indices = []

    obj_val_diff = []
    viol_contsr_diff = []
    max_viol_diff = []
    
    for i in tqdm(range(1)):
        filename = path_to_data + 'instance_' + str(n_coeffs) + '_' + str(n_constraints) + '_' + str(i)
        with open(filename, 'rb') as f:        
            [A, b, c] = pickle.load(f)

        t0=time.time()
        quantum_results = qaoa.local_search(A, b, c, 1)
        t1=time.time()
        quantum_times.append(t1-t0)

        # t0=time.time()
        # quantum_results = quantum.quantum_local_search(A, b, c, n_shots, device)
        # t1=time.time()
        # quantum_times.append(t1-t0)

        t0=time.time()
        classical_results = quantum.classical_local_search(A, b, c)
    
        t1=time.time()
        classical_times.append(t1-t0)

        classical_df = build_sorted_dataframe(classical_results, c, A, b)
        quantum_df = build_sorted_dataframe(quantum_results, c, A, b)

        value_to_find = quantum_df.loc[0, 'x']
        matching_index = classical_df.index[classical_df['x'].apply(lambda x: x == value_to_find)].tolist()
    
        obj_val_diff.append(classical_df.loc[0, 'Objective Value']-quantum_df.loc[0, 'Objective Value'])
        viol_contsr_diff.append(classical_df.loc[0, 'Violated Constraints']-quantum_df.loc[0, 'Violated Constraints'])
        max_viol_diff.append(classical_df.loc[0, 'Max Violation']-quantum_df.loc[0, 'Max Violation'])
        
        indices.append(matching_index[0])
        with open(results_filename, 'wb') as f:
            pickle.dump([indices, classical_times, quantum_times], f)

        with open(results_filename+"_copy", 'wb') as f:
            pickle.dump([indices, classical_times, quantum_times], f)    

        print('Means', np.mean(obj_val_diff), np.mean(viol_contsr_diff), np.mean(max_viol_diff))

    print_value_percentages(indices)
    print(f'Classical times = {np.mean(classical_times):.3f}. Quantum times = {np.mean(quantum_times):.3f}')
    print()
    print(np.mean(obj_val_diff), np.mean(viol_contsr_diff), np.mean(max_viol_diff))
    print() 


provider = BraketProvider()
device = BraketLocalBackend()
# device = provider.get_backend("Aria 1")
# Providers : [BraketBackend[Ankaa-2], BraketBackend[Aria 1], BraketBackend[Aria 2], BraketBackend[Forte 1], BraketBackend[Garnet], BraketBackend[SV1], BraketBackend[TN1], BraketBackend[dm1]]

for n_coeffs in [4, 5, 6]:
    for n_constraints in [2, 3]:    
        for instance_type in ["medium"]:
            # for n_shots in [10, 100, 1000]:
            for n_shots in [100]:
                path_to_data = "../experiments/1/" + instance_type + "/"                
                results_filename =  "../experiments/1/results/" + path_to_data.replace("/", "_").replace("..", "") + 'instance_' + str(n_coeffs) + '_' + str(n_constraints) + '_' + str(n_shots) + '_'+ "qaoa"
                if os.path.exists(results_filename):
                    continue
                print(results_filename)
                perform_experiment(n_coeffs, n_constraints, n_shots, results_filename)
