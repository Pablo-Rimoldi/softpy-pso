import numpy as np
from pso_implementation import *

# Definiamo la funzione di fitness (massimizzazione)
def sphere_fitness(particle):
    # Valore negativo della somma dei quadrati (perché la tua PSO massimizza)
    return -np.sum(particle.candidate ** 2)

# Parametri del problema
size = 5  # dimensione del vettore candidato
lower = np.full(size, -5.0)  # limiti inferiori
upper = np.full(size, 5.0)   # limiti superiori

# Inizializziamo l'ottimizzatore
pso = ParticleSwarmOptimizer(
    fitness_func=sphere_fitness,
    pop_size=20,        # popolazione di 20 particelle
    n_neighbors=3,      # 3 vicini per particella
    size=size,
    lower=lower,
    upper=upper
)

# Lanciamo l'ottimizzazione per 50 iterazioni
best_particle = pso.fit(n_iters=50)

# Risultati
print("Miglior candidato trovato:", best_particle.candidate)
print("Valore di fitness:", sphere_fitness(best_particle))