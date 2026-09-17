import numpy as np

# Define your transition matrix P
P = np.array([
    [0.5, 0.5, 0.0, 0.0, 0.0, 0.0],
    [0.3, 0.0, 0.3, 0.0, 0.3, 0.0],
    [0.5, 0.0, 0.25, 0.75, 0.0, 0.0],
    [0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
    [0.0, 0.0, 0.0, 0.0, 0.0, 1],
    [0.0, 0.0, 0.0, 0.0, 1.0, 0.0]
])

# Compute P^100 numerically
P_21 = np.linalg.matrix_power(P, 21)
print(P_21)
