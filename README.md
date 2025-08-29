# Particle Swarm Optimization (PSO) Implementation

This repository contains an implementation of Particle Swarm Optimization (PSO) compatible with the `softpy` library. The implementation adheres to the specifications provided in the project brief and is suitable for experimentation and integration with the `softpy` meta-heuristics framework.

## Features

* `ParticleCandidate` class that extends `FloatVectorCandidate` and includes PSO-specific attributes and operations.
* `ParticleSwarmOptimizer` class that extends `MetaHeuristicsAlgorithm` and implements the PSO procedure in full.
* Velocity and position updates, neighbor selection, and fitness tracking implemented according to standard PSO methodology.
* Designed for compatibility and straightforward integration with `softpy`.

## Repository layout

```
softpy-pso/
├── pso_implementation.py      # Main PSO implementation
├── requirements.txt           # Python dependencies
└── README.md                 # This file
```

## Installation

1. Clone the repository:

```bash
git clone <https://github.com/Pablo-Rimoldi/softpy-pso>
cd softpy-pso
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

Dependencies typically include:

* `numpy`
* `scipy`
* `softpy` (version 0.1.2 or later)

## Implementation overview

### ParticleCandidate

`ParticleCandidate` is implemented as a subclass of `FloatVectorCandidate` and includes the following:

**Attributes**

* `size`: dimensionality of the solution vector.
* `lower`, `upper`: arrays specifying per-dimension bounds.
* `candidate`: current position vector (numpy array).
* `velocity`: current velocity vector (numpy array).
* `inertia`: inertia weight for velocity update.
* `wl`, `wn`, `wg`: weights for local, neighborhood and global influence (enforced to sum to 1).

**Methods**

* `__init__`: initializes the object.
* `generate`: class method that returns randomly-initialized particles within the specified bounds.
* `mutate`: updates the particle position according to its velocity and enforces bounds.
* `recombine`: computes the new velocity using the PSO velocity update formula.

### ParticleSwarmOptimizer

`ParticleSwarmOptimizer` is implemented as a subclass of `MetaHeuristicsAlgorithm` and provides the PSO algorithm.

**Attributes**

* `pop_size`: number of particles.
* `population`: list of `ParticleCandidate` instances.
* `fitness_func`: callable that evaluates particle fitness.
* `n_neighbors`: neighbor count used for local neighborhood best calculations.
* `best`: array of best positions per particle.
* `fitness_best`: best fitness value per particle.
* `global_best`: best position found by the swarm.
* `global_fitness_best`: best fitness value found by the swarm.

**Methods**

* `__init__`: constructs the optimizer with the provided configuration.
* `fit`: executes the PSO loop, performing initialization, fitness evaluation, neighbor selection, velocity and position updates, and best-value tracking until termination.

## Algorithm details

The implemented PSO follows the standard procedure:

1. Initialize a population of particles with random positions and velocities within the specified bounds.
2. Evaluate the fitness of each particle using `fitness_func`.
3. Update personal bests and the global best.
4. For each particle, select a set of neighbors and determine the neighborhood best.
5. Update particle velocities using the formula:

```
v(t+1) = inertia * v(t)
         + rl * wl * (local_best - x)
         + rn * wn * (neighborhood_best - x)
         + rg * wg * (global_best - x)
```

where `rl`, `rn`, `rg` are independent random coefficients sampled per update.

6. Update particle positions from the new velocities and enforce boundary constraints.
7. Repeat until the termination criterion is met (for example, a maximum number of iterations).

## Usage example

```python
import numpy as np
from pso_implementation import ParticleSwarmOptimizer

# Define a simple fitness function (sphere, negative for maximization)
def sphere_function(particle):
    return -np.sum(particle.candidate ** 2)

pso = ParticleSwarmOptimizer(
    fitness_func=sphere_function,
    pop_size=30,
    n_neighbors=5,
    size=5,
    lower=np.full(5, -5.0),
    upper=np.full(5, 5.0)
)

best_particle = pso.fit(n_iters=100)

print('Best solution found:', best_particle.candidate)
print('Best fitness value:', pso.global_fitness_best)
```

## Testing

A simple two-dimensional test function can be used to verify behavior:

```python
# Simple 2D test
def simple_2d_function(particle):
    x, y = particle.candidate[0], particle.candidate[1]
    return -(x ** 2 + y ** 2)

pso_2d = ParticleSwarmOptimizer(
    fitness_func=simple_2d_function,
    pop_size=20,
    n_neighbors=4,
    size=2,
    lower=np.array([-3.0, -3.0]),
    upper=np.array([3.0, 3.0])
)

best_2d = pso_2d.fit(n_iters=50)
print(f"2D solution: ({best_2d.candidate[0]:.6f}, {best_2d.candidate[1]:.6f})")
```

## Configuration and tuning

* **Population size**: typical values are 20–50.
* **Neighbors**: typically 3–8.
* **Inertia**: governs exploration versus exploitation (commonly 0.4–0.9).
* **Weights (wl, wn, wg)**: determine the influence of personal, neighborhood, and global bests.
* **Bounds**: define the search space and should be set appropriately for the problem domain.

Tuning these parameters affects convergence speed and solution quality.

## Integration with `softpy`

* Classes inherit from `softpy` base classes and follow its naming conventions.
* The implementation is designed to be compatible with the `softpy` evolutionary framework and can be extended where necessary.

## License

This implementation is provided for research and educational purposes. For reuse in other projects, please comply with the licensing terms of `softpy` and include an appropriate license header in source files.

---

If additional adjustments are required (for example: alternative velocity formulas, concurrency support, or metric/logging integrations), specify the desired changes and an updated version can be prepared.
