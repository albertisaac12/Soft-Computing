import numpy as np

def trimf(x, params):
    a, b, c = params
    return np.maximum(0, np.minimum((x-a)/(b-a+1e-6), (c-x)/(c-b+1e-6)))

def fuzzify(s, f):
    S = [trimf(s,(0,0,5)), trimf(s,(0,5,10)), trimf(s,(5,10,10))]
    F = [trimf(f,(0,0,5)), trimf(f,(0,5,10)), trimf(f,(5,10,10))]
    return np.array([S[i]*F[j] for i in range(3) for j in range(3)])

class ANFIS:
    def __init__(self): self.W = np.random.randn(9,3)   # p,q,r for 9 rules

    def forward(self, s, f):
        r = fuzzify(s,f)
        w = r / (r.sum() + 1e-6)
        out = self.W[:,0]*s + self.W[:,1]*f + self.W[:,2]
        return (w * out).sum(), r, w

    def train(self, S, F, T, lr=1e-3, epochs=50):
        for _ in range(epochs):
            for s, f, t in zip(S, F, T):
                y, r, w = self.forward(s,f)
                e = t - y
                grad = -2*e*w[:,None] * np.array([s,f,1])
                self.W -= lr*grad

# --- DATA ---
np.random.seed(0)
N = 200
S = np.random.uniform(0,10,N)
F = np.random.uniform(0,10,N)
T = 2 + 0.8*S + 0.4*F + np.random.randn(N)*0.5

# --- TRAIN ---
m = ANFIS()
m.train(S,F,T)

# --- TEST ---
print(m.forward(8,9)[0])
