import numpy as np
from typing import Callable, List, Optional
from softpy import FloatVectorCandidate
from softpy.evolutionary.singlestate import MetaHeuristicsAlgorithm


class ParticleCandidate(FloatVectorCandidate):
    """
    Representation of a particle in the PSO algorithm.
    Extends FloatVectorCandidate by adding velocity and PSO-specific update logic.
    """

    def __init__(self, size: int, lower: np.ndarray, upper: np.ndarray,
                 candidate: np.ndarray, velocity: np.ndarray):

        # Initialize base candidate structure
        super().__init__(size, candidate, None, lower, upper)

        self.size = size
        self.lower = lower.astype(float)
        self.upper = upper.astype(float)
        self.candidate = candidate.astype(float)
        self.velocity = velocity.astype(float)

        # PSO update coefficients (must sum to 1)
        self.inertia = 0.7  # weight applied to previous velocity
        self.wl = 0.4       # cognitive weight (personal experience)
        self.wn = 0.3       # social-neighborhood weight
        self.wg = 0.3       # global social weight

        # Guarantee that the velocity update formula is well-defined
        assert abs(self.wl + self.wn + self.wg - 1.0) < 1e-10, "Weights must sum to 1"

    @classmethod
    def generate(cls, size: int, lower: np.ndarray, upper: np.ndarray):
        """
        Factory method: creates a particle with random position and velocity.
        Velocity is initialized relative to the search space size.
        """
        lower = np.asarray(lower, dtype=float)
        upper = np.asarray(upper, dtype=float)

        candidate = np.random.uniform(lower, upper, size).astype(float)

        # Velocity range proportional to domain width
        velocity_range = np.abs(upper - lower)
        velocity = np.random.uniform(-velocity_range, velocity_range, size).astype(float)

        return cls(size, lower, upper, candidate, velocity)

    def mutate(self):
        """
        Update position based on velocity.
        Handles boundary constraints by clipping and prevents divergence
        by restricting velocity magnitude relative to the search domain.
        """
        self.candidate = self.candidate + self.velocity
        self.candidate = np.clip(self.candidate, self.lower, self.upper)

        # Velocity clamping ensures stable exploration
        velocity_max = 0.5 * np.abs(self.upper - self.lower)
        self.velocity = np.clip(self.velocity, -velocity_max, velocity_max)

    def recombine(self, local_best: 'ParticleCandidate',
                neighborhood_best: 'ParticleCandidate',
                global_best: 'ParticleCandidate'):
        """
        Update velocity according to PSO dynamics.
        Combines inertia with cognitive, neighborhood, and global influences.

        Parameters:
        -----------
        local_best : ParticleCandidate
            Particle’s historical best position.
        neighborhood_best : ParticleCandidate
            Best position found among selected neighbors.
        global_best : ParticleCandidate
            Overall best position across the swarm.
        """
        rl = np.random.uniform(0, 1, self.size)
        rn = np.random.uniform(0, 1, self.size)
        rg = np.random.uniform(0, 1, self.size)

        # Velocity update rule (vectorized per dimension)
        self.velocity = (self.inertia * self.velocity +
                        rl * self.wl * (self.candidate - local_best.candidate) +
                        rn * self.wn * (self.candidate - neighborhood_best.candidate) +
                        rg * self.wg * (self.candidate - global_best.candidate))


class ParticleSwarmOptimizer(MetaHeuristicsAlgorithm):
    """
    PSO metaheuristic optimizer.
    Maintains a swarm of particles, tracks personal/global/neighborhood bests,
    and iteratively applies velocity/position updates.
    """

    def __init__(self, fitness_func: Callable, pop_size: int, n_neighbors: int, **kwargs):

        super().__init__(fitness_func, pop_size)

        self.fitness_func = fitness_func
        self.pop_size = pop_size
        self.n_neighbors = n_neighbors
        self.kwargs = kwargs

        # Core data structures for swarm tracking
        self.population = []  # active particles
        self.best = []  # each particle’s historical best
        self.fitness_best = np.full(pop_size, -np.inf, dtype=float)  # fitness of best[i]
        self.global_best = None
        self.global_fitness_best = -np.inf

        # Ensure required generation parameters exist
        required_params = ['size', 'lower', 'upper']
        for param in required_params:
            if param not in kwargs:
                raise ValueError(f"Missing required parameter: {param}")

    def fit(self, n_iters: int):
        """
        Execute PSO for a given number of iterations.
        Updates personal, neighborhood, and global bests at each step.

        Returns:
        --------
        ParticleCandidate
            Best solution found after all iterations.
        """
        # Initialize swarm randomly
        self.population = [ParticleCandidate.generate(**self.kwargs)
                          for _ in range(self.pop_size)]

        # Start with each particle’s own state as its personal best
        self.best = [ParticleCandidate(p.size, p.lower, p.upper, 
                                       p.candidate.copy(), p.velocity.copy()) 
                    for p in self.population]

        for iteration in range(n_iters):
            # Evaluate all particles
            fitness_values = np.array([self.fitness_func(particle) 
                                       for particle in self.population])

            # Update personal and global bests
            for i, (particle, fitness) in enumerate(zip(self.population, fitness_values)):
                if fitness > self.fitness_best[i]:
                    self.fitness_best[i] = fitness
                    self.best[i] = ParticleCandidate(
                        particle.size, particle.lower, particle.upper,
                        particle.candidate.copy(), particle.velocity.copy()
                    )

                if fitness > self.global_fitness_best:
                    self.global_fitness_best = fitness
                    self.global_best = ParticleCandidate(
                        particle.size, particle.lower, particle.upper,
                        particle.candidate.copy(), particle.velocity.copy()
                    )

            # Assign best neighbor for each particle
            best_neighbors = []
            for i in range(self.pop_size):
                other_indices = [j for j in range(self.pop_size) if j != i]
                if len(other_indices) > 0:
                    num_neighbors_to_select = min(self.n_neighbors, len(other_indices))
                    neighbor_indices = np.random.choice(other_indices,
                                                       size=num_neighbors_to_select,
                                                       replace=False)
                    best_neighbor_idx = neighbor_indices[
                        np.argmax([self.fitness_best[idx] for idx in neighbor_indices])
                    ]
                    best_neighbors.append(self.best[best_neighbor_idx])
                else:
                    # Single-particle case: fallback to global or own best
                    best_neighbors.append(self.global_best if self.global_best else self.best[i])

            # Apply velocity update and move particles
            for i, particle in enumerate(self.population):
                particle.recombine(
                    self.best[i],
                    best_neighbors[i],
                    self.global_best
                )
                particle.mutate()

        return self.global_best


# Quick usage example
def test_pso():

    def sphere_function(particle):
        # Sphere benchmark: f(x) = -Σ x_i² (maximum at 0)
        return -np.sum(particle.candidate ** 2)
    
    dim = 5
    lower_bounds = np.full(dim, -10.0)
    upper_bounds = np.full(dim, 10.0)
    
    optimizer = ParticleSwarmOptimizer(
        fitness_func=sphere_function,
        pop_size=30,
        n_neighbors=5,
        size=dim,
        lower=lower_bounds,
        upper=upper_bounds
    )
    
    best_particle = optimizer.fit(n_iters=100)
    
    print(f"Best position found: {best_particle.candidate}")
    print(f"Best fitness: {sphere_function(best_particle)}")
    print(f"Distance from optimum: {np.linalg.norm(best_particle.candidate)}")
    
    return best_particle


if __name__ == "__main__":
    test_pso()
