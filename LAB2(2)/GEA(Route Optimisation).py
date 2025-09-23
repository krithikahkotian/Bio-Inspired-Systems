import random
import math
import matplotlib.pyplot as plt

# --- Problem Setup ---
# Coordinates of delivery houses (x, y)
houses = [
    (0, 0),   # Depot (start and end point)
    (2, 3),
    (5, 4),
    (1, 6),
    (7, 2),
    (6, 6)
]

NUM_HOUSES = len(houses)
POP_SIZE = 50
GENERATIONS = 200
MUTATION_RATE = 0.1
CROSSOVER_RATE = 0.8

# --- Distance Calculation ---
def distance(a, b):
    return math.sqrt((a[0]-b[0])**2 + (a[1]-b[1])**2)

def route_distance(route):
    total = 0
    for i in range(len(route)-1):
        total += distance(houses[route[i]], houses[route[i+1]])
    return total

# --- Fitness Function ---
def fitness(route):
    return 1 / (route_distance(route) + 1e-6)

# --- Initialize Population ---
def create_route():
    route = list(range(1, NUM_HOUSES))  # houses except depot
    random.shuffle(route)
    return [0] + route + [0]  # start and end at depot

def init_population():
    return [create_route() for _ in range(POP_SIZE)]

# --- Selection (Tournament) ---
def selection(population):
    tournament = random.sample(population, 5)
    tournament.sort(key=lambda r: fitness(r), reverse=True)
    return tournament[0]

# --- Crossover (Order Crossover - OX) ---
def crossover(parent1, parent2):
    if random.random() > CROSSOVER_RATE:
        return parent1[:]

    start, end = sorted(random.sample(range(1, NUM_HOUSES), 2))
    child = [None] * len(parent1)

    # Copy segment from parent1
    child[start:end] = parent1[start:end]

    # Fill remaining positions from parent2 in order
    fill_values = [g for g in parent2 if g not in child]
    fill_positions = [i for i in range(1, NUM_HOUSES) if child[i] is None]

    for pos, val in zip(fill_positions, fill_values):
        child[pos] = val

    # Ensure start and end are depot
    child[0] = 0
    child[-1] = 0
    return child

# --- Mutation (Swap) ---
def mutate(route):
    if random.random() < MUTATION_RATE:
        i, j = random.sample(range(1, NUM_HOUSES), 2)
        route[i], route[j] = route[j], route[i]
    # Ensure depot at start and end
    route[0] = 0
    route[-1] = 0
    return route

# --- GEA Algorithm ---
def gene_expression_algorithm():
    population = init_population()
    best_route = min(population, key=lambda r: route_distance(r))
    best_distance = route_distance(best_route)

    for gen in range(GENERATIONS):
        new_population = []
        for _ in range(POP_SIZE):
            parent1 = selection(population)
            parent2 = selection(population)
            child = crossover(parent1, parent2)
            child = mutate(child)
            # Ensure all genes are integers (no None)
            if None in child:
                child = create_route()
            new_population.append(child)

        population = new_population
        current_best = min(population, key=lambda r: route_distance(r))
        current_dist = route_distance(current_best)

        if current_dist < best_distance:
            best_distance = current_dist
            best_route = current_best

        if gen % 20 == 0:
            print(f"Generation {gen} | Best Distance: {best_distance:.2f}")

    return best_route, best_distance

# --- Run the Algorithm ---
best_route, best_distance = gene_expression_algorithm()
print("\nBest Route Found:", best_route)
print("Best Distance:", round(best_distance, 2))

# --- Plot the Best Route ---
x = [houses[i][0] for i in best_route]
y = [houses[i][1] for i in best_route]

plt.figure(figsize=(6,6))
plt.plot(x, y, marker='o', color='blue')
for idx, (xi, yi) in enumerate(houses):
    plt.text(xi+0.1, yi+0.1, f"{idx}", fontsize=12)
plt.title("Optimized Delivery Route")
plt.xlabel("X")
plt.ylabel("Y")
plt.grid(True)
plt.show()
