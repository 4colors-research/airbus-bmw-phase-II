# **Garona**

**Garona** is an integrated production and transportation planning hybrid solver developed to optimize the Airbus industrial system for the Airbus-BMW Group Quantum-Powered Logistic Challenge. Its primary component is a hybrid matheuristic that combines mixed-integer linear programming (MILP) with heuristics, quantum search techniques, and machine learning.

---

## **Running Garona**

To run Garona, use:
```bash
python3 garona.py
```

To save the output of the run, consider:

```python3 garona.py | tee garona-output.txt```
 
## Configuration files

The Garona configuration files include several customizable settings to adjust the solver's parameters and behaviors.

### Instance parameters

* `random_seed`: sets the random seed for reproducibility.
* `numerical_tolerance`: specifies the numerical tolerance for use in numerical algorithms.
    
## Instance input

The following settings allow for tuning the problem-specific requirements and objective function priorities:

* `default_production_split`: defines the lower bound for production split between sites producing the same part. For example, setting this to 0.2 means that production is split between sites at a minimum of 20% and a maximum of 80%.


The next four parameters represent the coefficients in the objective function's linear combination, each controlling the contribution of a specific element to the overall optimization objective. You can interpret this function in terms of monetary value:

* `production_cost_obj_function_weight`
* `transport_cost_obj_function_weight`
* `emission_cost_obj_function_weight`
* `target_workshare_obj_function_weight`

For example: 

* The `emission_cost_obj_function_weight` could represent the current price of CO₂ certificates per gram.

* The `target_workshare_obj_function_weight` applies a penalty to deviations from workload targets, with its coefficient representing the penalty cost per euro that deviates from the target workload.

Additional workload-related parameters:

* `site_workshare_target_lambda_penalty`: controls the importance of meeting workload targets for production sites.
* `supplier_workshare_target_lambda_penalty`: controls the importance of workload targets for suppliers.

The `demand` parameter specifies the total number of aircraft that all Final Assembly Lines (FALs) must produce. `demand_per_FAL` allows for defining production targets for individual FALs, specifying how many aircraft each should produce. The sum of all `demand_per_FAL` entries should not exceed the demand value. If the total is less than demand and some FALs do not have specified values, the solver will automatically distribute the remaining production optimally across the unscheduled FALs.

For example, in the setup below, production is set as follows, Hamburg: 20, Mirabel: 30, Mobile: 20. The remaining production demand (30 units) will be optimally allocated across Tianjin and Toulouse.


```
"demand": 100
"demand_per_FAL":   {                                                              
                        "Hamburg FAL" : 20,
                        "Mirabel FAL" : 30,
                        "Mobile FAL" : 20
                    }               

```
    
