import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Perceptron
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split


iris = load_iris()
X = iris.data[:, :2]
y = iris.target


x_train, x_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=20
)


model = Perceptron()
model.fit(x_train, y_train)


x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(
    np.arange(x_min, x_max, 0.02),
    np.arange(y_min, y_max, 0.02)
)


Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)


plt.contourf(xx, yy, Z, alpha=0.4, cmap='coolwarm')


plt.scatter(X[:, 0], X[:, 1], c=y, cmap='coolwarm', edgecolors='k')

plt.xlabel("Sepal Length")
plt.ylabel("Sepal Width")
plt.title("Perceptron Decision Boundary on Iris Dataset (2 Features)")
plt.show()
