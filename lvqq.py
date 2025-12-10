import numpy as np

class LVQ:
    def __init__(self, n_prototypes, lr=0.1):
        self.lr = lr
        self.n_classes = n_prototypes
        self.prototypes = None
        self.proto_labels = None

    def fit(self, X, y, epochs=20):
        # pick one random sample per class as prototype
        self.prototypes = np.zeros((self.n_classes, X.shape[1]))
        self.proto_labels = np.arange(self.n_classes)

        for c in range(self.n_classes):
            idx = np.where(y == c)[0][0]
            self.prototypes[c] = X[idx]

        # training
        for _ in range(epochs):
            for x, label in zip(X, y):
                # find nearest prototype (winner)
                dists = np.linalg.norm(self.prototypes - x, axis=1)
                w = np.argmin(dists)

                # update rule
                if self.proto_labels[w] == label:
                    self.prototypes[w] += self.lr * (x - self.prototypes[w])
                else:
                    self.prototypes[w] -= self.lr * (x - self.prototypes[w])

    def predict(self, X):
        preds = []
        for x in X:
            dists = np.linalg.norm(self.prototypes - x, axis=1)
            preds.append(self.proto_labels[np.argmin(dists)])
        return np.array(preds)



# Simple 2-class dataset
X = np.array([[1,1],[1.2,1.1],[8,8],[7.8,8.3]])
y = np.array([0,0,1,1])

model = LVQ(n_prototypes=2, lr=0.2)
model.fit(X, y, epochs=20)

print("Prototypes:\n", model.prototypes)
print("Prediction:", model.predict([[1,1],[8,8],[7.5,8.1]]))
