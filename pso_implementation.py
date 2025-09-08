import numpy as np
from typing import Callable, List
from softpy import FloatVectorCandidate
from softpy.evolutionary.singlestate import MetaHeuristicsAlgorithm


class ParticleCandidate(FloatVectorCandidate):
    """
    Particle candidate for Particle Swarm Optimization.
    Inherits from FloatVectorCandidate and adds PSO-specific attributes and methods.
    """

    def __init__(self, size: int, lower: np.ndarray, upper: np.ndarray,
                 candidate: np.ndarray, velocity: np.ndarray):
        """
        Initialize a particle candidate.

        Parameters:
        -----------
        size : int
            Number of components in the position
        lower : np.ndarray
            Lower bounds for each position component
        upper : np.ndarray
            Upper bounds for each position component
        candidate : np.ndarray
            Current position of the particle
        velocity : np.ndarray
            Current velocity of the particle
        """
        # Initialize parent class
        super().__init__(size, candidate, None, lower, upper)

        self.size = size
        self.lower = lower
        self.upper = upper
        self.candidate = candidate
        self.velocity = velocity

        # PSO parameters with constraint wl + wn + wg = 1
        self.inertia = 0.7
        self.wl = 0.2  # local best weight
        self.wn = 0.3  # neighborhood best weight
        self.wg = 0.5  # global best weight

        # Ensure weights sum to 1
        total_weight = self.wl + self.wn + self.wg
        self.wl /= total_weight
        self.wn /= total_weight
        self.wg /= total_weight

    @classmethod
    def generate(cls, size: int, lower: np.ndarray, upper: np.ndarray):
        """
        Generate a new particle candidate with random position and velocity.

        Parameters:
        -----------
        size : int
            Number of components in the position
        lower : np.ndarray
            Lower bounds for each position component
        upper : np.ndarray
            Upper bounds for each position component

        Returns:
        --------
        ParticleCandidate
            New particle with random position and velocity
        """
        # Generate random position between lower and upper bounds
        candidate = np.random.uniform(lower, upper, size)

        # Generate random velocity between -|upper - lower| and |upper - lower|
        velocity_range = np.abs(upper - lower)
        velocity = np.random.uniform(-velocity_range, velocity_range, size)

        return cls(size, lower, upper, candidate, velocity)

    def mutate(self):
        """
        Update particle position based on current velocity.

        Returns:
        --------
        ParticleCandidate
            New particle with updated position
        """
        new_candidate = self.candidate + self.velocity

        # Ensure position stays within bounds
        new_candidate = np.clip(new_candidate, self.lower, self.upper)

        return ParticleCandidate(self.size, self.lower, self.upper, new_candidate, self.velocity)

    def recombine(self, local_best: 'ParticleCandidate',
                  neighborhood_best: 'ParticleCandidate',
                  global_best: 'ParticleCandidate'):
        """
        Update particle velocity using PSO velocity update formula.

        Parameters:
        -----------
        local_best : ParticleCandidate
            Best position found by this particle
        neighborhood_best : ParticleCandidate
            Best position found by neighbors
        global_best : ParticleCandidate
            Best position found by any particle

        Returns:
        --------
        ParticleCandidate
            New particle with updated velocity
        """
        # Generate random coefficients
        rl = np.random.uniform(0, 1)
        rn = np.random.uniform(0, 1)
        rg = np.random.uniform(0, 1)

        # Update velocity using PSO formula
        new_velocity = (self.inertia * self.velocity +
                        rl * self.wl * (local_best.candidate - self.candidate) +
                        rn * self.wn * (neighborhood_best.candidate - self.candidate) +
                        rg * self.wg * (global_best.candidate - self.candidate))

        return ParticleCandidate(self.size, self.lower, self.upper, self.candidate, new_velocity)


class ParticleSwarmOptimizer(MetaHeuristicsAlgorithm):
    """
    Particle Swarm Optimization algorithm implementation.
    Inherits from MetaHeuristicsAlgorithm.
    """

    def __init__(self, fitness_func: Callable, pop_size: int, n_neighbors: int, **kwargs):
        """
        Initialize the PSO optimizer.

        Parameters:
        -----------
        fitness_func : Callable
            Fitness function that takes a ParticleCandidate and returns a number
        pop_size : int
            Size of the particle population
        n_neighbors : int
            Number of neighbors for each particle
        **kwargs : dict
            Additional arguments for particle generation (size, lower, upper)
        """
        self.fitness_func = fitness_func
        self.pop_size = pop_size
        self.n_neighbors = n_neighbors
        self.kwargs = kwargs

        # Initialize population and tracking arrays
        self.population = []
        self.best = []
        self.fitness_best = np.full(pop_size, -np.inf)
        self.global_best = None
        self.global_fitness_best = -np.inf

        # Validate required kwargs
        required_params = ['size', 'lower', 'upper']
        for param in required_params:
            if param not in kwargs:
                raise ValueError(f"Missing required parameter: {param}")

    def fit(self, n_iters: int = 10):
        """
        Run the PSO optimization for specified number of iterations.

        Parameters:
        -----------
        n_iters : int
            Number of iterations to run

        Returns:
        --------
        ParticleCandidate
            Best particle found during optimization
        """
        # Step 1: Create initial population
        self.population = [ParticleCandidate.generate(**self.kwargs)
                           for _ in range(self.pop_size)]

        # Initialize best array with current population
        self.best = [particle for particle in self.population]

        # Main optimization loop
        for iteration in range(n_iters):
            # Step 2a: Compute fitness for each particle
            fitness_values = [self.fitness_func(
                particle) for particle in self.population]

            # Step 2b: Update best positions and global best
            for i, (particle, fitness) in enumerate(zip(self.population, fitness_values)):
                if fitness > self.fitness_best[i]:
                    self.fitness_best[i] = fitness
                    self.best[i] = particle

                # Update global best
                if fitness > self.global_fitness_best:
                    self.global_fitness_best = fitness
                    self.global_best = particle

            # Step 2c: Select neighbors and find best neighbor for each particle
            best_neighbors = []
            for i in range(self.pop_size):
                # Randomly select n_neighbors other particles
                other_indices = [j for j in range(self.pop_size) if j != i]
                if len(other_indices) > 0:
                    neighbor_indices = np.random.choice(other_indices,
                                                        size=min(self.n_neighbors, len(
                                                            other_indices)),
                                                        replace=False)

                    # Find best neighbor
                    best_neighbor_idx = max(neighbor_indices,
                                            key=lambda idx: self.fitness_best[idx])
                    best_neighbors.append(self.best[best_neighbor_idx])
                else:
                    # If no neighbors, use global best
                    best_neighbors.append(self.global_best)

            # Step 2d: Update velocities and positions
            new_population = []
            for i, particle in enumerate(self.population):
                # Recombine to update velocity
                updated_particle = particle.recombine(
                    self.best[i],
                    best_neighbors[i],
                    self.global_best
                )

                # Mutate to update position
                final_particle = updated_particle.mutate()
                new_population.append(final_particle)

            self.population = new_population

        return self.global_best
