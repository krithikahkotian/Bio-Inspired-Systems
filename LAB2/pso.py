import numpy as np

# Rastrigin Function
def rastrigin(x):
    A = 10
    return A * len(x) + sum([(xi**2 - A * np.cos(2 * np.pi * xi)) for xi in x])

# Particle class
class Particle:
    def __init__(self, dim, bounds):
        self.position = np.random.uniform(bounds[0], bounds[1], dim)
        self.velocity = np.zeros(dim)
        self.best_position = np.copy(self.position)
        self.best_score = rastrigin(self.position)

    def update_velocity(self, global_best, w=0.5, c1=1.5, c2=1.5):
        r1, r2 = np.random.rand(len(self.position)), np.random.rand(len(self.position))
        cognitive = c1 * r1 * (self.best_position - self.position)
        social = c2 * r2 * (global_best - self.position)
        self.velocity = w * self.velocity + cognitive + social

    def update_position(self, bounds):
        self.position += self.velocity
        self.position = np.clip(self.position, bounds[0], bounds[1])
        score = rastrigin(self.position)
        if score < self.best_score:
            self.best_score = score
            self.best_position = np.copy(self.position)

# PSO Algorithm
def pso(objective_func, dim=2, num_particles=30, max_iter=100, bounds=(-5.12, 5.12)):
    swarm = [Particle(dim, bounds) for _ in range(num_particles)]
    global_best = min(swarm, key=lambda p: p.best_score).best_position
    global_best_score = rastrigin(global_best)

    for _ in range(max_iter):
        for particle in swarm:
            particle.update_velocity(global_best)
            particle.update_position(bounds)

        current_best = min(swarm, key=lambda p: p.best_score)
        if current_best.best_score < global_best_score:
            global_best = np.copy(current_best.best_position)
            global_best_score = current_best.best_score

    return global_best, global_best_score

# Run PSO
best_position, best_score = pso(rastrigin, dim=2)
print(f"Best Position: {best_position}")
print(f"Best Score: {best_score}")
