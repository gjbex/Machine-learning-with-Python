# PINNs

Physics-Informed Neural Networks (PINNs) are (deep) neural networks that are
trained using physics properties, e.g., ordinary or partial differential
equations in the loss function.


## What is it?

1. `logistic_de.ipynb`: Jupyter notebook solving the logistic differential
   equation using a PINN.
1. `pendulum.ipynb`: Jupyter notebook solving the equation of a pendulum
   with damping using a PINN.
1. `parameterized_heat_equation.ipynb`: Jupyter notebook training one PINN
   across a family of heat equations with varying initial conditions and
   diffusivities. It uses hard initial and boundary constraints and validates
   interpolation and extrapolation against the analytical solution.

All notebooks in this directory run in the repository's standard
[CPU](../../environment.yml) or [GPU](../../environment_gpu.yml) environment.

The [PINN hyperparameter-optimization case study](../parameter-optimization/parameterized_heat_equation_optimization.ipynb)
is kept with the parameter-optimization material because its primary subject is
experimental methodology; it uses the parameterized heat equation as its
realistic example.
