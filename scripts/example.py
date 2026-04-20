import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from gurobipy import Model, GRB

# --- 1. Differentiable Wrapper for Gurobi ---
# In a real implementation, you would use a library like 'Blackbox-backprop' 
# or 'PyVat'. Here we simulate the gradient of the solver.
class DifferentiableKnapsack(torch.autograd.Function):
    @staticmethod
    def forward(ctx, weights, capacity, c_coeffs):
        """
        Solves: argmin c^T * x 
        s.t. weights^T * x <= capacity, x in {0,1}
        """
        n = len(c_coeffs)
        m = Model("Knapsack")
        m.setParam('OutputFlag', 0)
        
        # Define binary variables
        x = m.addVars(n, vtype=GRB.BINARY, name="x")
        
        # Set linear surrogate objective
        m.setObjective(sum(c_coeffs[i].item() * x[i] for i in range(n)), GRB.MINIMIZE)
        
        # Add Knapsack constraint
        m.addConstr(sum(weights[i] * x[i] for i in range(n)) <= capacity)
        
        m.optimize()
        
        # Extract solution
        x_sol = torch.tensor([x[i].X for i in range(n)], dtype=torch.float32)
        
        # Save for backward pass (using Pogančić et al. 2019 blackbox method)
        ctx.save_for_backward(c_coeffs, x_sol)
        ctx.weights = weights
        ctx.capacity = capacity
        return x_sol

    @staticmethod
    def backward(ctx, grad_output):
        c_coeffs, x_sol = ctx.saved_tensors
        lambda_val = 0.1 # Perturbation hyperparameter
        
        # Compute "improved" coefficients by moving in direction of gradient
        c_prime = c_coeffs + lambda_val * grad_output
        
        # Solve solver again with perturbed costs to estimate gradient
        # (Simplified Pogančić-style differentiation)
        with torch.no_grad():
            x_prime = DifferentiableKnapsack.apply(ctx.weights, ctx.capacity, c_prime)
            
        # Gradient of solver is (x_prime - x_sol) / lambda
        grad_c = -(x_sol - x_prime) / lambda_val
        return None, None, grad_c

# --- 2. SurCo Components ---

class SurCoPriorNet(nn.Module):
    """Predicts initial surrogate coefficients c from instance data y."""
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 16),
            nn.ReLU(),
            nn.Linear(16, output_dim)
        )
    
    def forward(self, y):
        return self.net(y)

def nonlinear_objective(x, target):
    """Example nonlinear cost: f(x) = ||x - target||^2."""
    return torch.sum((x - target) ** 2)

# --- 3. Execution Script ---

# Problem setup
N = 5 # Number of items
weights = np.array([2, 3, 4, 5, 9])
capacity = 10
y_instance = torch.randn(8) # Simulated instance metadata (y)
target_x = torch.tensor([1, 1, 0, 0, 0], dtype=torch.float32) # Target we want to reach

# Initialize prior model
model = SurCoPriorNet(input_dim=8, output_dim=N)

# --- SurCo-Hybrid Online Fine-Tuning ---

def run_hybrid_example():
    # Step A: Get prior prediction (Inference)
    with torch.no_grad():
        c_initial = model(y_instance)
        
    # Step B: Optimize surrogate coefficients c directly
    c_coeffs = c_initial.clone().detach().requires_grad_(True)
    optimizer = optim.Adam([c_coeffs], lr=0.1)
    
    print(f"Targeting solution: {target_x.tolist()}")
    
    for step in range(20):
        optimizer.zero_grad()
        
        # 1. Map surrogate cost c to feasible solution x using Gurobi
        x_sol = DifferentiableKnapsack.apply(weights, capacity, c_coeffs)
        
        # 2. Compute nonlinear loss
        loss = nonlinear_objective(x_sol, target_x)
        
        # 3. Step
        loss.backward()
        optimizer.step()
        
        if step % 5 == 0:
            print(f"Step {step} | Loss: {loss.item():.4f} | Current X: {x_sol.tolist()}")

    return x_sol

final_solution = run_hybrid_example()
print(f"\nFinal SurCo-Hybrid Solution: {final_solution.tolist()}")