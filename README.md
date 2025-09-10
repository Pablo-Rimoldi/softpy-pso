# softpy-PSO

Minimal implementation of **Particle Swarm Optimization (PSO)** designed to integrate with the **softpy** library.

---

## Description

This project contains a compact PSO implementation (classes `ParticleCandidate` and `ParticleSwarmOptimizer`) intended to be easily tested and adapted. A practical example is provided in the `test_pso.py` file included in the repository.

---

## Key Features

* Representation of a single agent (`ParticleCandidate`) with position and velocity.
* PSO algorithm (`ParticleSwarmOptimizer`) with:

  * velocity/position updates,
  * neighborhood selection for local bests.
* Designed for integration with `softpy` (extends `FloatVectorCandidate` and `MetaHeuristicsAlgorithm`).

---

## Repository Structure

```
softpy-pso/
├── pso_implementation.py  
├── test_pso.py            
├── requirements.txt        
└── README.md               
```

---

## Requirements

* numpy >= 1.21.0
* scipy >= 1.7.0
* softpy == 0.1.2

---

## Installation

Clone the repository and install dependencies:

```bash
git clone https://github.com/<your-username>/softpy-pso.git
cd softpy-pso
pip install -r requirements.txt
```

---

## Tuning Tips

* `pop_size`: typically 20–50 (depends on problem dimension).
* `n_neighbors`: 3–8 for local structures; fewer neighbors encourages global exploration.
* `inertia`: 0.4–0.9 (lower for faster convergence).
* weights `wl/wn/wg`: balance personal/neighborhood/global influence according to the problem.

---

## Testing

* The script `test_pso.py` is the starting point to quickly test behavior.

---

## Contributing

PRs and issues are welcome. For changes:

1. Open an issue to discuss major features.
2. Create a branch and add tests.
3. Submit a PR describing the changes.

---

## License

MIT