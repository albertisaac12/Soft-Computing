import numpy as np

class ACO:
    
    def __init__(self,n_ants,n_iters,alpha, beta, distance_matrix, Q, rho):
        self.n_ants = n_ants
        self.n_iters = n_iters
        self.alpha = alpha
        self.beta = beta
        self.rho = rho

        self.Q = Q
        self.dist_matrix = distance_matrix
        self.cities = distance_matrix.shape[0]

        self.tau = np.ones((self.cities,self.cities)) # will be the same for each city
        self.eta = 1 / (self.dist_matrix + 1e-10) # will be the same size as distance matrix

        print(self.eta.shape == self.dist_matrix.shape)
        

    def calculatue_probablity(self,i,visited):
        """Compute probability distribution to choose next city."""

        # i is the index of the last visited city and visited is the list of all the visited cities
        # we need to make sure to get the probability of all the unvisited cities

        not_visited = [c for c in range(self.cities) if c not in visited]

        current_tau = self.tau[i][not_visited]
        eta = self.eta[i][not_visited]

        num = (current_tau ** self.alpha) * (eta ** self.beta)

        den = np.sum(num)
        return not_visited, num / den


    def construct_solution(self):
        solutions = []
        distances = []

        for ant in range(self.n_ants):
            visited = [np.random.randint(self.cities)]

            while len(visited) < self.cities:
                i = visited[-1]
                not_visited, probs = self.calculatue_probablity(i,visited)
                next_city = np.random.choice(not_visited,p=probs)
                visited.append(next_city)

            # return to start city
            visited.append(visited[0])

            distance = sum(self.dist_matrix[visited[i], visited[i+1]] for i in range(self.cities))
            solutions.append(visited)
            distances.append(distance)

        return solutions, distances

    def update_pheromone(self,solutions,distances):
        
        # evaporation
        self.tau*=(1-self.rho)

        for sol, dist in zip(solutions, distances):
            pheromone_to_add = self.Q / dist

            for i in range(self.cities):
                a, b = sol[i], sol[i+1]
                self.tau[a][b] += pheromone_to_add
                self.tau[b][a] += pheromone_to_add

    def run(self):
        best_dist = float("inf")
        best_path = None

        for _ in range(self.n_iters):
            solutions, distances = self.construct_solution()

            # update best
            min_idx = np.argmin(distances)
            if distances[min_idx] < best_dist:
                best_dist = distances[min_idx]
                best_path = solutions[min_idx]

            # update pheromones
            self.update_pheromone(solutions, distances)

        return best_path, best_dist


dist_matrix = np.array([
    [0, 2, 2, 5],
    [2, 0, 3, 4],
    [2, 3, 0, 2],
    [5, 4, 2, 0]
])

aco = ACO(
    n_ants=5,
    n_iters=1000,
    alpha=1,
    beta=2,
    rho=0.5,
    Q=100,
    distance_matrix=dist_matrix
)

best_path, best_distance = aco.run()
print("Best Path:", best_path)
print("Best Distance:", best_distance)


"""

Number of ants
Number of iterations
Number of cities , can be get from the distance matrix
distance matrix
alpha => the strength of phermone
beta => the  infuencle on visibility eta
eta => visibility
phermone values(tau) => always one when we init them
eta = 1 / dij
p = Tau^alpha * eta^ beta / sum(Tau^alpha * eta^ beta) this is the transistion brobablity


what i forgot
Q = phermone per ant initally set to 100

"""