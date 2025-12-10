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


print(model.predict(x_test))