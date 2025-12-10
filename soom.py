import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

class SOM:
    def __init__(self, grid_x, grid_y, dims, lr, iters):
        self.grid_x = grid_x
        self.grid_y = grid_y
        self.lr = lr
        self.iters = iters

        self.weights = np.random.rand(grid_x, grid_y, dims)
        print("Initial weights:\n", self.weights)

    def find_bmu(self, inputs):
        diff = self.weights - inputs                      # (5,5,3)
        dist = np.sqrt(np.sum(diff**2, axis=2))           # (5,5)
        return np.unravel_index(np.argmin(dist), dist.shape)

    def train(self, data):
        for t in range(self.iters):
            x = data[np.random.randint(len(data))]
            i, j = self.find_bmu(x)
            self.weights[i, j] += self.lr * (x - self.weights[i, j])

    def map(self, data):
        return [self.find_bmu(x) for x in data]


# --------------------- RUN TEST ------------------------

# np.random.seed(25)
som = SOM(5, 5, 3, 0.1, 1000)

print("\n\nRandom training data:")
x = np.array([
 
 [0.77520317,0.16904146,0.76699433],
 [0.33536585,0.47239795,0.21506437],
 [0.91209456,0.75920765,0.67656136],
 [0.02137628,0.66087433,0.09443959],
 [0.83116257,0.11274904,0.56682961],
 qa
 ])
# print(x[0])

som.train(x)
# 
mapped = som.map(x)
print("\nMapped BMUs:", mapped)

# print("\nFinal weights shape:", som.weights.shape)
print("Final weights:\n", som.weights)

print("\nSample → Cluster assignment:")
for i, x in enumerate(x):
    row, col = mapped[i]
    cluster_id = row * som.grid_y + col
    print(f"Sample {x} → Cluster {cluster_id}")


