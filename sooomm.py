# som_minisom_example.py

import numpy as np
import matplotlib.pyplot as plt
from minisom import MiniSom
from sklearn import datasets
from sklearn.preprocessing import MinMaxScaler


# 1. Load some data (Iris dataset)
iris = datasets.load_iris()
X = iris.data        # shape (150, 4)
y = iris.target      # class labels (0,1,2)


# 2. Scale features to [0, 1] – SOMs work better when data is normalized
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)


# 3. Create the SOM
#    x,y = map size, input_len = number of features
som_x, som_y = 10, 10
som = MiniSom(
    x=som_x,
    y=som_y,
    input_len=X_scaled.shape[1],
    sigma=1.0,
    learning_rate=0.5,
    neighborhood_function='gaussian',
    random_seed=42
)

# 4. Initialize and train the SOM
som.random_weights_init(X_scaled)
print("Training SOM...")
som.train_random(data=X_scaled, num_iteration=1000)
print("Training done!")


# 5. Get U-matrix (distance map) and plot it
plt.figure(figsize=(7, 7))
u_matrix = som.distance_map()  # returns a (som_x, som_y) matrix
plt.imshow(u_matrix, interpolation='nearest')
plt.title("SOM U-Matrix (Distance Map)")
plt.colorbar(label="Distance")

# 6. Optionally overlay the data points with their class
markers = ['o', 's', 'D']        # one marker per class
colors = ['r', 'g', 'b']

for i, x in enumerate(X_scaled):
    w = som.winner(x)           # BMU coordinates (i,j)
    plt.plot(
        w[1] + 0.5,
        w[0] + 0.5,
        markers[y[i]],
        markerfacecolor='none',
        markeredgecolor=colors[y[i]],
        markersize=8,
        markeredgewidth=1.5
    )

plt.tight_layout()
plt.show()


# 7. Example: print BMU for first 5 samples
print("\nBMUs for first 5 samples:")
for i in range(100):
    print(f"Sample {i}, class {y[i]}, BMU:", som.winner(X_scaled[i]))
