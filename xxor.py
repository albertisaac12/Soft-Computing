import numpy as np

i_o = {
    (0,0) : np.array([0]),
    (0,1) : np.array([1]),
    (1,0) : np.array([1]),
    (1,1) : np.array([0]),
    "lr" : 0.1,
    "iter": 1000,
    "weights_or": np.array([1,1]),
    "bias_or":np.array([-1]),
    "weights_nand": np.array([-1,-1]),
    "bias_nand":np.array([1]),
    "weights_and": np.array([1,1]),
    "bias_and":np.array([-1.5])
}



# Training settings
lr = i_o["lr"]
epochs = i_o["iter"]

def train(inputs, targets, w_key, b_key, gate_name):
    print(f"\nTraining {gate_name} Gate...")
    weights = i_o[w_key].astype(float)
    bias = i_o[b_key].astype(float)
    
    for _ in range(epochs):
        total_error = 0
        for x, y, target in zip(inputs[:,0], inputs[:,1], targets):
            z = x * weights[0] + y * weights[1] + bias[0]
            pred = 1 if z >= 0 else 0
            
            error = target - pred
            total_error += abs(error)
            
            weights[0] += lr * error * x
            weights[1] += lr * error * y
            bias[0] += lr * error
            
        if total_error == 0:
            break
            
    i_o[w_key] = weights
    i_o[b_key] = bias
    print(f"Final {gate_name} Weights: {weights}, Bias: {bias}")

    





# Data for training
X = np.array([[0,0], [0,1], [1,0], [1,1]])
y_or = np.array([0, 1, 1, 1])
y_nand = np.array([1, 1, 1, 0])
y_and = np.array([0, 0, 0, 1])

# Initialize weights to small random values/zeros for training demonstration
i_o["weights_or"] = np.array([0.0, 0.0])
i_o["bias_or"] = np.array([0.0])
i_o["weights_nand"] = np.array([0.0, 0.0])
i_o["bias_nand"] = np.array([0.0])
i_o["weights_and"] = np.array([0.0, 0.0])
i_o["bias_and"] = np.array([0.0])

# Train the gates
train(X, y_or, "weights_or", "bias_or", "OR")
train(X, y_nand, "weights_nand", "bias_nand", "NAND")
train(X, y_and, "weights_and", "bias_and", "AND")

# Test the full XOR logic
print("\nFinal XOR Truth Table Results:")
for x, y in X:
    # Use trained weights
    z_or = x * i_o["weights_or"][0] + y * i_o["weights_or"][1] + i_o["bias_or"][0]
    o_out = 1 if z_or >= 0 else 0
    
    z_nand = x * i_o["weights_nand"][0] + y * i_o["weights_nand"][1] + i_o["bias_nand"][0]
    n_out = 1 if z_nand >= 0 else 0
    
    # Final AND takes OR and NAND outputs as input
    z_and = o_out * i_o["weights_and"][0] + n_out * i_o["weights_and"][1] + i_o["bias_and"][0]
    final = 1 if z_and >= 0 else 0
    
    print(f"[{x}, {y}] -> OR:{o_out}, NAND:{n_out} -> XOR:{final}")