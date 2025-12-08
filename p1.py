from sklearn.linear_model import Perceptron
import numpy as np

# or gate training data
x = np.array([
    [0,0],
    [0,1],
    [1,0],
    [1,1]
])

y = np.array([0,1,1,1])

model = Perceptron(max_iter = 1000,random_state= 30 , eta0= 0.1)

model.fit(x,y)

for input_vector in x:
    pridiction = model.predict([input_vector])[0]
    print(f"{input_vector} -> {pridiction}")



print("The weights are :", model.coef_)
print("The model bias is : ", model.intercept_)