import numpy as np
import math

# ------------------------------
# Step 1: Example financial data
# ------------------------------
# Expected returns of 4 assets
returns = np.array([0.12, 0.10, 0.15, 0.09])  

# Covariance matrix of asset returns (risk relationships)
cov_matrix = np.array([
    [0.010, 0.002, 0.001, 0.003],
    [0.002, 0.008, 0.002, 0.002],
    [0.001, 0.002, 0.012, 0.004],
    [0.003, 0.002, 0.004, 0.009]
])

num_assets = len(returns)

# ------------------------------
# Step 2: Fitness function
# ------------------------------
def portfolio_fitness(weights, alpha=0.5, beta=0.5):
    weights = np.array(weights)
    weights = np.clip(weights, 0, 1)  # bounds [0,1]
    weights /= np.sum(weights)        # normalize (budget constraint)

    expected_return = np.dot(weights, returns)
    risk = np.dot(weights.T, np.dot(cov_matrix, weights))

    # lower fitness = better (we minimize risk - return)
    fitness = alpha * risk - beta * expected_return
    return fitness, expected_return, risk

# ------------------------------
# Step 3: Cuckoo Search Algorithm
# ------------------------------
def levy_flight(Lambda):
    sigma = (math.gamma(1 + Lambda) * np.sin(np.pi * Lambda / 2) /
             (math.gamma((1 + Lambda)/2) * Lambda * 2**((Lambda-1)/2)))**(1/Lambda)
    u = np.random.normal(0, sigma, num_assets)
    v = np.random.normal(0, 1, num_assets)
    step = u / np.abs(v)**(1/Lambda)
    return step

def cuckoo_search(n=20, max_iter=100, pa=0.25):
    nests = np.random.dirichlet(np.ones(num_assets), size=n)
    fitness = [portfolio_fitness(w)[0] for w in nests]
    best_idx = np.argmin(fitness)
    best = nests[best_idx]

    for _ in range(max_iter):
        for i in range(n):
            step_size = levy_flight(1.5)
            new_solution = nests[i] + step_size * np.random.randn(num_assets)
            new_solution = np.clip(new_solution, 0, 1)
            new_solution /= np.sum(new_solution)
            
            new_fitness = portfolio_fitness(new_solution)[0]
            if new_fitness < fitness[i]:
                nests[i] = new_solution
                fitness[i] = new_fitness

        # Abandon some nests with probability pa
        for i in range(n):
            if np.random.rand() < pa:
                nests[i] = np.random.dirichlet(np.ones(num_assets))
                fitness[i] = portfolio_fitness(nests[i])[0]

        # Update best nest
        best_idx = np.argmin(fitness)
        best = nests[best_idx]

    return best, portfolio_fitness(best)

# ------------------------------
# Step 4: Run optimization
# ------------------------------
best_weights, (fitness_value, best_return, best_risk) = cuckoo_search()

print("Optimal Portfolio Allocation:")
for i, w in enumerate(best_weights):
    print(f"  Asset {i+1}: {w:.2f}")

print(f"\nExpected Return: {best_return:.4f}")
print(f"Risk (Variance): {best_risk:.4f}")
