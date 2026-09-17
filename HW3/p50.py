import numpy as np

# Define your transition matrix P
P = np.array([
    [0.9, 0.1, 0.0],
    [0.0, 0.75, 0.25],
    [0.5, 0.0, 0.5]
])

# Compute P^100 numerically
P_50 = np.linalg.matrix_power(P, 50)
print(P_50)












