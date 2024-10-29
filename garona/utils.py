'''
This file is part of Garona. 
(c) 4colors Research Ltd, 2024

This file contains the auxiliary functions used by Garona.
'''

import numpy as np

def mean_coefficient_by_row(A):
    mean_nonzero = []
    for row in range(A.shape[0]):
        # Get the start and end indices of the non-zero values for this row
        start_idx = A.indptr[row]
        end_idx = A.indptr[row + 1]
        
        # Get the non-zero values for this row
        non_zero_values = np.abs(A.data[start_idx:end_idx])
        
        if len(non_zero_values) > 0:
            # Calculate the mean of non-zero entries
            mean_value = np.mean(non_zero_values)
        else:
            mean_value = 1  # If no non-zero entries, you can decide how to handle it
        
        mean_nonzero.append(mean_value)
    return mean_nonzero


def normalize_matrix_and_vector(A, b):
    mean_nonzero = mean_coefficient_by_row(A)

    for row in range(A.shape[0]):
        start_idx = A.indptr[row]
        end_idx = A.indptr[row + 1]
        
        if mean_nonzero[row] != 0:  # Avoid division by zero            
            A.data[start_idx:end_idx] /= mean_nonzero[row]  # Divide the non-zero values of this row by the mean

    b /= np.array(mean_nonzero)


def is_nearly_integer(value, tolerance):
    return abs(value - round(value)) < tolerance
