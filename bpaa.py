import numpy as np

def sigmoid(x):
    return 1/(np.exp(x))

def sigmoid_der(x):
    return x * (1-x)


x = np.array([[0,0],[0,1],[1,0],[1,1]])

y = np.array([[0],[1],[1],[0]])

lr = 0.1
epohs = 1000
input_dim = 2
hidden_dim = 2
output_dim = 1

w1 = np.random.uniform(size=(input_dim,hidden_dim))
b1 = np.zeros((1,hidden_dim))

w2 = np.random.uniform(size=(hidden_dim,output_dim))
b1 = np.zeros((1,output_dim))

print(x.shape)
print(y.shape)

for epoh in epohs:
    # forward pass
    s1 = x@w1 + b1
    o1 = sigmoid(s1)

    s2 = o1@w2 + b2
    o2 = sigmoid(o2)

    #loss
    loss = np.mean((y - a2)**2)
    
    # backwards pass
    g_o = (o2-y) * sigmoid_der(o2)
    g_h = (g_o @ w2.T) * sigmoid_der(o1)
    
    w2 = w2 - lr*(a1.T @ o2)
    b
