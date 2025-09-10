import math
import random
import matplotlib.pyplot as plt

# ---------------- Problem Setup ----------------
NUM_CUSTOMERS = 20     # number of customers
VEHICLE_CAPACITY = 30  # max load per vehicle
POP_SIZE = 80          # population size
GENERATIONS = 300
TOURNAMENT_K = 3
CROSSOVER_RATE = 0.9
MUTATION_RATE = 0.2
ELITE_SIZE = 2

random.seed(1)

# Depot
depot = (50, 50)

# Customers (randomly generated)
customers = [(random.uniform(0, 100), random.uniform(0, 100)) for _ in range(NUM_CUSTOMERS)]
demands = [random.randint(1, 10) for _ in range(NUM_CUSTOMERS)]

# Distance matrix
points = [depot] + customers
n = len(points)
dist = [[0.0]*n for _ in range(n)]
for i in range(n):
    for j in range(n):
        dx, dy = points[i][0] - points[j][0], points[i][1] - points[j][1]
        dist[i][j] = math.hypot(dx, dy)

# ---------------- Decoder ----------------
def decode_permutation(perm):
    routes, cur_route, cur_load = [], [], 0
    for c in perm:
        d = demands[c-1]
        if cur_load + d <= VEHICLE_CAPACITY:
            cur_route.append(c)
            cur_load += d
        else:
            routes.append(cur_route)
            cur_route, cur_load = [c], d
    if cur_route:
        routes.append(cur_route)
    return routes

def route_cost(route):
    if not route:
        return 0
    cost = dist[0][route[0]]
    for i in range(len(route)-1):
        cost += dist[route[i]][route[i+1]]
    cost += dist[route[-1]][0]
    return cost

def total_cost(routes):
    return sum(route_cost(r) for r in routes)

# ---------------- Genetic Operators ----------------
def random_permutation():
    perm = list(range(1, NUM_CUSTOMERS+1))
    random.shuffle(perm)
    return perm

def tournament_selection(pop, fitness):
    best = None
    for _ in range(TOURNAMENT_K):
        ind = random.choice(pop)
        if best is None or fitness[tuple(ind)] < fitness[tuple(best)]:
            best = ind
    return best[:]

def ordered_crossover(p1, p2):
    n = len(p1)
    a, b = sorted(random.sample(range(n), 2))
    def ox(x, y):
        child = [-1]*n
        child[a:b+1] = x[a:b+1]
        pos = (b+1) % n
        for elem in y:
            if elem not in child:
                child[pos] = elem
                pos = (pos+1) % n
        return child
    return ox(p1, p2), ox(p2, p1)

def swap_mutation(perm):
    a, b = random.sample(range(len(perm)), 2)
    perm[a], perm[b] = perm[b], perm[a]

# ---------------- GA ----------------
population = [random_permutation() for _ in range(POP_SIZE)]
fitness = {}

best_cost = float("inf")
best_solution = None
history = []

for gen in range(GENERATIONS):
    # evaluate
    for ind in population:
        if tuple(ind) not in fitness:
            routes = decode_permutation(ind)
            fitness[tuple(ind)] = total_cost(routes)
    # track best
    for ind in population:
        cost = fitness[tuple(ind)]
        if cost < best_cost:
            best_cost = cost
            best_solution = ind[:]
    history.append(best_cost)

    # elitism
    sorted_pop = sorted(population, key=lambda x: fitness[tuple(x)])
    new_pop = sorted_pop[:ELITE_SIZE]

    # reproduction
    while len(new_pop) < POP_SIZE:
        p1, p2 = tournament_selection(population, fitness), tournament_selection(population, fitness)
        if random.random() < CROSSOVER_RATE:
            c1, c2 = ordered_crossover(p1, p2)
        else:
            c1, c2 = p1, p2
        if random.random() < MUTATION_RATE: swap_mutation(c1)
        if random.random() < MUTATION_RATE: swap_mutation(c2)
        new_pop.extend([c1, c2])
    population = new_pop[:POP_SIZE]

# ---------------- Results ----------------
best_routes = decode_permutation(best_solution)
print(f"Best cost: {best_cost:.2f}")
for i, r in enumerate(best_routes, 1):
    load = sum(demands[c-1] for c in r)
    print(f"Route {i}: {r}, load={load}, cost={route_cost(r):.2f}")

# Plot solution
plt.figure(figsize=(8,8))
plt.scatter([p[0] for p in customers], [p[1] for p in customers], c='blue')
plt.scatter(*depot, c='red', marker='s', s=100)
for route in best_routes:
    xs = [depot[0]] + [points[c][0] for c in route] + [depot[0]]
    ys = [depot[1]] + [points[c][1] for c in route] + [depot[1]]
    plt.plot(xs, ys, marker='o')
plt.title(f"Best Solution (Cost={best_cost:.2f})")
plt.show()

# Plot convergence
plt.plot(history)
plt.title("Convergence")
plt.xlabel("Generation")
plt.ylabel("Best Cost")
plt.show()
