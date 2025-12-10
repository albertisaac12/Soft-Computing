import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

X = np.array([[0,0],
              [0,1],
              [1,0],
              [1,1]], dtype=np.float32)

y = np.array([[0],
              [1],
              [1],
              [0]], dtype=np.float32)

# Build the model
model = keras.Sequential([
    layers.Dense(2, input_dim=2, activation='sigmoid'),  # hidden layer
    layers.Dense(1, activation='sigmoid')               # output layer
])


# Compile the model (defines loss and optimizer)
model.compile(optimizer=keras.optimizers.SGD(learning_rate=0.1),
              loss='mean_squared_error',
              metrics=['accuracy'])


history = model.fit(X, y, epochs=3000)
# Evaluate
loss, acc = model.evaluate(X, y)
print(f"Final Loss: {loss:.4f}, Accuracy: {acc:.4f}")


# Predictions
print("\nPredictions:")
print(model.predict(X))