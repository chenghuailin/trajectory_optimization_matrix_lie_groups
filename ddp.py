import numpy as np

# Define system dynamics and cost function parameters
T = 10  # Number of time steps
x0 = 0  # Initial position
x_target = 10  # Target position
R = 1  # Control effort weight

# Initialize trajectories
x = np.zeros(T+1)
u = np.zeros(T)

def forward_pass(x0, u):
    x = np.zeros(T+1)
    x[0] = x0
    for t in range(T):
        x[t+1] = x[t] + u[t]
    return x

def cost(x, u, x_target, R):
    J = 0
    for t in range(T):
        J += 0.5 * (x[t] - x_target)**2 + 0.5 * R * u[t]**2
    J += 0.5 * (x[T] - x_target)**2
    return J

def backward_pass(x, u, x_target, R):
    Vx = np.zeros(T+1)
    Vxx = np.zeros(T+1)
    k = np.zeros(T)
    K = np.zeros(T)
    
    # Final cost-to-go
    Vx[T] = x[T] - x_target
    Vxx[T] = 1
    
    for t in reversed(range(T)):
        Qx = x[t] - x_target + Vx[t+1]
        Qu = R * u[t]
        Qxx = 1 + Vxx[t+1]
        Quu = R
        Qux = 0
        
        k[t] = -Qu / Quu
        K[t] = -Qux / Quu
        
        Vx[t] = Qx + K[t] * Qu
        Vxx[t] = Qxx + K[t] * Qux
    
    return k, K


# Main iLQR loop
for i in range(100):  # Number of iterations
    # Forward pass
    x = forward_pass(x0, u)
    
    # Backward pass
    k, K = backward_pass(x, u, x_target, R)
    
    # Update control inputs
    for t in range(T):
        u[t] += k[t] + K[t] * (x[t] - x0)

# Print final trajectories
print("Final state trajectory:", x)
print("Final control inputs:", u)