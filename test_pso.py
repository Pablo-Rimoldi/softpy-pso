import numpy as np
import matplotlib.pyplot as plt
from pso_implementation import *
from mpl_toolkits.mplot3d import Axes3D  # needed for 3D plots

# --- Rastrigin function ---
def rastrigin(x, A=10):
    n = len(x)
    return A * n + np.sum(x**2 - A * np.cos(2 * np.pi * x))

def rastrigin_fitness(particle):
    return -rastrigin(particle.candidate)  # PSO maximizes

# Problem parameters
dim = 2
lower = -5.12 * np.ones(dim)
upper = 5.12 * np.ones(dim)

# Initialize PSO (use your original code!)
pso = ParticleSwarmOptimizer(
    fitness_func=rastrigin_fitness,
    pop_size=40,
    n_neighbors=5,
    size=dim,
    lower=lower,
    upper=upper
)

# Run for 100 iterations
for _ in range(100):
    pso.fit(n_iters=1)

best_particle = pso.global_best

print("Best fitness found:", rastrigin_fitness(best_particle))
print("Best solution found:", best_particle.candidate)

# --- Create Rastrigin surface ---
X = np.linspace(-5.12, 5.12, 200)
Y = np.linspace(-5.12, 5.12, 200)
X, Y = np.meshgrid(X, Y)

Z = np.zeros_like(X)
for i in range(X.shape[0]):
    for j in range(X.shape[1]):
        Z[i, j] = rastrigin(np.array([X[i, j], Y[i, j]]))

# --- 3D Plot ---
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')

# surface
ax.plot_surface(X, Y, Z, cmap="viridis", alpha=0.7, linewidth=0, antialiased=True)

# final particles
xs = [p.candidate[0] for p in pso.population]
ys = [p.candidate[1] for p in pso.population]
zs = [rastrigin(p.candidate) for p in pso.population]
ax.scatter(xs, ys, zs, color="black", s=30, label="Particles")

# best particle
ax.scatter(best_particle.candidate[0],
           best_particle.candidate[1],
           rastrigin(best_particle.candidate),
           color="red", s=200, marker="*", label="Best")

ax.set_xlabel("x1")
ax.set_ylabel("x2")
ax.set_zlabel("Rastrigin(x)")
ax.set_title("PSO on Rastrigin (final population)")
ax.legend()

plt.show()





import numpy as np
import matplotlib.pyplot as plt
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

# Track fitness over iterations
fitness_history = []
for _ in range(100):
    pso.fit(n_iters=1)
    fitness_history.append(pso.global_fitness_best)

best_particle = pso.global_best

print('Best solution found:', best_particle.candidate)
print('Best fitness value:', pso.global_fitness_best)

# Plot fitness over time
plt.figure()
plt.plot(fitness_history, marker='o')
plt.xlabel("Iteration")
plt.ylabel("Fitness (negative of Sphere)")
plt.title("PSO performance on Sphere (dim=5)")
plt.grid()
plt.show()


## 2D testing
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

fitness_history_2d = []
for _ in range(50):
    pso_2d.fit(n_iters=1)
    fitness_history_2d.append(pso_2d.global_fitness_best)

best_2d = pso_2d.global_best
print(f"2D solution: ({best_2d.candidate[0]:.6f}, {best_2d.candidate[1]:.6f})")

# Plot fitness over time for the 2D case
plt.figure()
plt.plot(fitness_history_2d, marker='x', color='orange')
plt.xlabel("Iteration")
plt.ylabel("Fitness (negative of x^2+y^2)")
plt.title("PSO performance on simple 2D function")
plt.grid()
plt.show()

# Final 2D population scatter
xs = [p.candidate[0] for p in pso_2d.population]
ys = [p.candidate[1] for p in pso_2d.population]

plt.figure()
plt.scatter(xs, ys, label="Particles")
plt.scatter([best_2d.candidate[0]],
            [best_2d.candidate[1]],
            color="red", marker="*", s=200, label="Best")
plt.xlabel("x")
plt.ylabel("y")
plt.title("Final PSO population (dim=2)")
plt.legend()
plt.show()
