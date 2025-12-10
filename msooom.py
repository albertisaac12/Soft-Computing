from minisom import MiniSom
from sklearn.datasets import load_iris
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import numpy as np

# Load data
iris = load_iris()
X = MinMaxScaler().fit_transform(iris.data)

# Train SOM
som = MiniSom(x=10, y=10, input_len=4, sigma=1.0, learning_rate=0.5)
som.random_weights_init(X)
som.train_random(X, 1000)

# Compute U-Matrix
u_matrix = som.distance_map()  # 10x10 matrix

# Plot simple heatmap
plt.figure(figsize=(6, 6))
plt.imshow(u_matrix.T, cmap="viridis")   # just a heatmap
plt.colorbar(label="Distance")
plt.title("Simple SOM Heatmap (U-Matrix)")
plt.show()
