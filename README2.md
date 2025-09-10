# Particle Swarm Optimization (PSO) Library in Python

## Overview

This project implements a **Particle Swarm Optimization (PSO)** algorithm in Python. Desinged as integration of the library **softpy**.

The project includes:

* `ParticleCandidate`: a class representing a particle with position, velocity, and PSO-specific attributes.
* `ParticleSwarmOptimizer`: the main PSO class for optimizing a given fitness function.
* Example usage and visualization of the optimization process.

---

## Features

* Fully object-oriented design.
* Configurable population size, number of neighbors, and PSO weights.
* Handles multidimensional optimization problems.
* Fitness tracking over iterations.
* Support for visualizing optimization progress.

---

## Installation

You can clone the repository and install the dependencies:

```bash
git clone <repository-url>
cd particle-swarm-optimizer
pip install -r requirements.txt
```

**Dependencies:**

* Python >= 3.8
* NumPy
* SoftPy

---

## Testing

* Lo script `test_pso.py` è il punto di partenza per testare rapidamente il comportamento.

---

## Contributing

Contributions are welcome! Feel free to submit pull requests for:

* New PSO variants (e.g., constriction factor, velocity clamping).
* Additional visualization tools.
* Performance improvements.

---

## License

This project is released under the **MIT License**.