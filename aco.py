import numpy as np

class ACO:
    def __init__(self, n_ants, n_iter, alpha, beta, rho, Q, dist_matrix):
        self.n_ants = n_ants
        self.n_iter = n_iter
        self.alpha = alpha
        self.beta = beta
        self.rho = rho
        self.Q = Q
        self.dist = dist_matrix
        self.n = dist_matrix.shape[0]

        # initial pheromone
        self.tau = np.ones((self.n, self.n))

        # visibility = 1 / distance
        self.eta = 1 / (self.dist + 1e-10)

    def probability(self, i, visited):
        """Compute probability distribution to choose next city."""
        allowed = np.array([c for c in range(self.n) if c not in visited])
        tau_allowed = self.tau[i][allowed]
        eta_allowed = self.eta[i][allowed]

        numer = (tau_allowed ** self.alpha) * (eta_allowed ** self.beta)
        denom = np.sum(numer)
        return allowed, numer / denom

    def construct_solution(self):
        solutions = []
        distances = []

        for ant in range(self.n_ants):
            visited = [np.random.randint(self.n)]  # starting city

            while len(visited) < self.n:
                i = visited[-1]
                allowed, probs = self.probability(i, visited)
                next_city = np.random.choice(allowed, p=probs)
                visited.append(next_city)

            # return to start city
            visited.append(visited[0])

            # compute tour length
            distance = sum(self.dist[visited[i], visited[i+1]] for i in range(self.n))
            solutions.append(visited)
            distances.append(distance)

        return solutions, distances

    def update_pheromone(self, solutions, distances):
        # evaporation
        self.tau *= (1 - self.rho)

        for sol, dist in zip(solutions, distances):
            pheromone_to_add = self.Q / dist

            for i in range(self.n):
                a, b = sol[i], sol[i+1]
                self.tau[a][b] += pheromone_to_add
                self.tau[b][a] += pheromone_to_add  # symmetric

    def run(self):
        best_dist = float("inf")
        best_path = None

        for _ in range(self.n_iter):
            solutions, distances = self.construct_solution()

            # update best
            min_idx = np.argmin(distances)
            if distances[min_idx] < best_dist:
                best_dist = distances[min_idx]
                best_path = solutions[min_idx]

            # update pheromones
            self.update_pheromone(solutions, distances)

        return best_path, best_dist


# ------------------ EXAMPLE RUN ------------------

# Distance matrix
dist_matrix = np.array([
    [0, 2, 2, 5],
    [2, 0, 3, 4],
    [2, 3, 0, 2],
    [5, 4, 2, 0]
])

aco = ACO(
    n_ants=5,
    n_iter=1000,
    alpha=1,
    beta=2,
    rho=0.5,
    Q=100,
    dist_matrix=dist_matrix
)

best_path, best_distance = aco.run()
print("Best Path:", best_path)
print("Best Distance:", best_distance)
