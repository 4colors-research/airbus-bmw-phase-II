'''
This file is part of Garona. 
(c) 4colors Research Ltd, 2024

This file contains the implementation of the feasibility pump.
'''

import pulp
from collections import defaultdict


def feasibility_pump(model, mip_vars, values):
    ''' mip_variables that have been rounded
        integer values of each variable (list)
    '''
    q_pos = defaultdict()
    q_neg = defaultdict()
        
    var_names = []
    for i, var in enumerate(mip_vars):
        var_names.append(var.name)
        q_pos[var.name] = pulp.LpVariable(f'q_pos_{var.name}', lowBound=0, cat='Continous')
        q_neg[var.name] = pulp.LpVariable(f'q_neg_{var.name}', lowBound=0, cat='Continous')        
        model.prob += ( var-values[i] == q_pos[var.name]-q_neg[var.name])

    model.prob.objective = pulp.lpSum((q_pos[var] + q_neg[var]) for var in var_names)
    model.solve()    
    return model
    


