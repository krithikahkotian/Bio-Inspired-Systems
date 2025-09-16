import random

class ACO_TSP:
    def __init__(self, graph, pheromone, n_ants=10, n_iterations=100, alpha=1, beta=5, rho=0.5, Q=100):
        """
        graph      : adjacency matrix of distances (2D list)
        pheromone  : initial pheromone matrix (2D list)
        n_ants     : number of ants
        n_iterations : number of iterations
        alpha      : pheromone importance
        beta       : heuristic (1/distance) importance
        rho        : evaporation rate
        Q          : pheromone deposit factor
        """
        self.graph = graph
        self.pheromone = pheromone
        self.n = len(graph)
        self.n_ants = n_ants
        self.n_iterations = n_iterations
        self.alpha = alpha
        self.beta = beta
        self.rho = rho
        self.Q = Q

    def run(self):
        best_length = float('inf')
        best_path = None

        for it in range(self.n_iterations):
            all_paths = []
            all_lengths = []

            for ant in range(self.n_ants):
                path = self.construct_solution()
                length = self.path_length(path)
                all_paths.append(path)
                all_lengths.append(length)

                if length < best_length:
                    best_length = length
                    best_path = path

            self.update_pheromones(all_paths, all_lengths)
            print(f"Iteration {it+1}/{self.n_iterations} - Best Length: {best_length:.2f}")

        return best_path, best_length

    def construct_solution(self):
        start = random.randint(0, self.n - 1)
        path = [start]
        visited = set(path)

        while len(path) < self.n:
            current = path[-1]
            next_city = self.choose_next_city(current, visited)
            path.append(next_city)
            visited.add(next_city)

        return path

    def choose_next_city(self, current, visited):
        probabilities = []
        denominator = 0

        for j in range(self.n):
            if j not in visited:
                tau = self.pheromone[current][j] ** self.alpha
                eta = (1 / self.graph[current][j]) ** self.beta if self.graph[current][j] > 0 else 0
                denominator += tau * eta

        for j in range(self.n):
            if j not in visited:
                tau = self.pheromone[current][j] ** self.alpha
                eta = (1 / self.graph[current][j]) ** self.beta if self.graph[current][j] > 0 else 0
                probabilities.append((j, (tau * eta) / denominator))

        r = random.random()
        cumulative = 0
        for city, prob in probabilities:
            cumulative += prob
            if r <= cumulative:
                return city

        return probabilities[-1][0]  # fallback

    def path_length(self, path):
        length = 0
        for i in range(len(path) - 1):
            length += self.graph[path[i]][path[i + 1]]
        length += self.graph[path[-1]][path[0]]  # return to start
        return length

    def update_pheromones(self, all_paths, all_lengths):
        # Evaporation
        for i in range(self.n):
            for j in range(self.n):
                self.pheromone[i][j] *= (1 - self.rho)

        # Deposit
        for path, length in zip(all_paths, all_lengths):
            deposit = self.Q / length
            for i in range(len(path) - 1):
                a, b = path[i], path[i + 1]
                self.pheromone[a][b] += deposit
                self.pheromone[b][a] += deposit
            # closing edge
            self.pheromone[path[-1]][path[0]] += deposit
            self.pheromone[path[0]][path[-1]] += deposit


# ---------------- Example Usage ----------------
if __name__ == "__main__":
    # Example Graph (distance matrix)
    graph = [
        [0, 10, 12, 11],
        [10, 0, 13, 15],
        [12, 13, 0, 9],
        [11, 15, 9, 0]
    ]

    # Example Initial Pheromone Matrix
    pheromone = [
        [0, 1, 1, 1],
        [1, 0, 1, 1],
        [1, 1, 0, 1],
        [1, 1, 1, 0]
    ]

    aco = ACO_TSP(graph, pheromone, n_ants=5, n_iterations=20, alpha=1, beta=5, rho=0.5, Q=100)
    best_path, best_length = aco.run()

    print("\nBest Path Found:", best_path)
    print("Best Path Length:", best_length)
