# Parameter optimization

Hyperparameter optimization is the process of finding the best hyperparameters
for a machine learning model. Hyperparameters are parameters that are set
before training a model and can significantly affect the performance of the
model.

Optuna is a Python framework for automated hyperparameter optimization.


## What is it?

1. `simple.py`: minimal script showing the objective/study/trial workflow.
1. `experiments.ipynb`: methodology-first introduction to Optuna, including a
   controlled comparison of stochastic samplers and a machine-learning example
   with separate validation and test data.
1. `parameterized_heat_equation_optimization.ipynb`: advanced case study using
   a parameterized heat-equation PINN. It compares random search with Optuna,
   confirms a shortlist across fresh training seeds, and reserves a final audit
   suite until after model selection.
