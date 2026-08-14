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
1. `environment.yml`: conda environment specification file.
