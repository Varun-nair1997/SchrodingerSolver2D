# Finite-Difference Solver for the 2D Time-Independent Schrödinger Equation
#### Author: Varun Nair

This repository contains a Python implementation of a finite-difference solver for the time-independent two-dimensional Schrödinger equation on a periodic square domain.
The solver computes ground and excited states using an iterative residual-minimization method (imaginary-time–like evolution) combined with Gram–Schmidt orthogonalization.

## Mathematical Formulation
We solve the eigenvalue problem:

H ψ(x, y) = E ψ(x, y)

where the Hamiltonian is defined as:

H = −(1/2) ∇² + V(x, y)

The Laplacian is discretized using second-order central finite differences on a uniform 2D grid with periodic boundary conditions.
The energy expectation value is computed as:

E = ⟨ψ | H | ψ⟩

### Eigenstates are obtained by iteratively minimizing the residual:

r = Hψ − Eψ

### Convergence is achieved when:

‖Hψ − Eψ‖ < tolerance

## Numerical Method
Uniform 2D spatial grid
Second-order finite-difference Laplacian
Periodic boundary conditions (implemented using numpy.roll)
Residual-based iterative solver
Gram–Schmidt orthogonalization for excited states
Wavefunction normalization at each iteration

## Update rule:
ψₙ₊₁ = normalize( ψₙ + α (Hψₙ − Eₙψₙ) )

where α is a step-size parameter.


## Features
- Multiple potential types:
    - Cosine lattice
    - Sine lattice
    - Mixed sine/cosine lattices
    - Finite square well
- Computation of multiple eigenstates
- Energy tracking during convergence
- 2D visualization of eigenfunctions

## Requirements
- Python 3.8+
- NumPy
- Matplotlib

Install dependencies with:
```pip install numpy matplotlib```

## Usage
Run the script directly:
python schrodinger2d.py
Example configuration:

```
solver = Schrodinger2D(
    mesh=50,
    x_length=4*np.pi,
    y_length=4*np.pi,
    pot_type='cosine',
    n_atoms=1
)

states, energies = solver.solve_n_states(
    n_states=4,
    n_steps=2500,
    step=-0.01,
    tol=1e-3
)

solver.plot_states(states, energies)
```

## Output
For each computed eigenstate, the solver returns:

- The eigenfunction ψ(x, y)
- The corresponding energy eigenvalue E

## Limitations
- Real-valued wavefunctions only
- Dense grid representation (no sparse matrices)
- Convergence sensitive to step size
- Not optimized for very large grids


_This implementation was developed for academic and instructional purposes in computational quantum mechanics during a course on electronic structure methods @ the Max Plank Insittute for Sustainable Materials._
