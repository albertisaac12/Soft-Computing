import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


class SOM:
    def __init__(self, grid_x, grid_y, dims, lr=0.1, iters=1000):
        self.grid_x = grid_x
        self.grid_y = grid_y
        self.dims = dims
        self.lr = lr
        self.iters = iters

        # Random weights for each neuron (5,5,3)
        self.weights = np.random.rand(grid_x, grid_y, dims)
        print("Initial weights:\n", self.weights)

    # ---------- BMU SEARCH ----------
    def find_bmu(self, inputs):
        diff = self.weights - inputs                # (5,5,3)
        dist = np.sqrt(np.sum(diff**2, axis=2))     # (5,5)
        return np.unravel_index(np.argmin(dist), dist.shape)

    # ---------- TRAINING ----------
    def train(self, data):
        for t in range(self.iters):
            x = data[np.random.randint(len(data))]  # pick random sample
            i, j = self.find_bmu(x)
            self.weights[i, j] += self.lr * (x - self.weights[i, j])

    # ---------- MAP EACH DATA POINT TO BMU ----------
    def map(self, data):
        return [self.find_bmu(x) for x in data]

    # ---------- U-MATRIX (DISTANCE HEATMAP) ----------
    def plot_distance_map(self):
        dist_map = np.zeros((self.grid_x, self.grid_y))

        for i in range(self.grid_x):
            for j in range(self.grid_y):
                w = self.weights[i, j]
                dist_map[i, j] = np.mean(np.sqrt(np.sum((self.weights - w)**2, axis=2)))

        plt.figure(figsize=(6, 5))
        sns.heatmap(dist_map, cmap="viridis")
        plt.title("SOM Distance Heatmap (U-Matrix)")
        plt.show()

    # ---------- HIT MAP ----------
    def plot_hit_map(self, mapped):
        hit_map = np.zeros((self.grid_x, self.grid_y))

        for (i, j) in mapped:
            hit_map[i, j] += 1

        plt.figure(figsize=(6, 5))
        sns.heatmap(hit_map, annot=True, cmap="coolwarm")
        plt.title("BMU Hit Map")
        plt.show()

    # ---------- WEIGHT HEATMAPS (ONE PER DIMENSION) ----------
    def plot_weights(self):
        fig, axes = plt.subplots(1, self.dims, figsize=(14, 4))

        for d in range(self.dims):
            sns.heatmap(self.weights[:, :, d], ax=axes[d], cmap="inferno")
            axes[d].set_title(f"Weight Dimension {d}")

        plt.show()


# ---------------------------------------------------------
# ------------------------ TEST ----------------------------
# ---------------------------------------------------------

som = SOM(5, 5, 3, lr=0.1, iters=1000)

data = np.array([
    [0.77520317, 0.16904146, 0.76699433],
    [0.33536585, 0.47239795, 0.21506437],
    [0.91209456, 0.75920765, 0.67656136],
    [0.02137628, 0.66087433, 0.09443959],
    [0.83116257, 0.11274904, 0.56682961]
])

print("\nTraining SOM...")
som.train(data)

mapped = som.map(data)
print("\nMapped BMUs:", mapped)

print("\nFinal weights:\n", som.weights)

# ------------------- PLOTTING ----------------------------
print("\nShowing U-Matrix...")
som.plot_distance_map()

print("\nShowing Hit Map...")
som.plot_hit_map(mapped)

print("\nShowing Weight Heatmaps...")
som.plot_weights()

print("\nSample → Cluster assignment:")
for i, sample in enumerate(data):
    print(f"Sample {sample} → BMU {mapped[i]}")
