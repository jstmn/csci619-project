import torch
import torch.optim as optim

def train_surco_hybrid(train_data, model, solver, objective_f, num_epochs):
    """
    Implements the SurCo-hybrid logic.
    
    Args:
        train_data: Training set {y_i} for offline learning.
        test_instance: The specific instance y_test to solve at deployment.
        model: Neural network theta that predicts coefficients c_hat(y; theta).
        solver: Differentiable combinatorial solver g_omega(c).
        nonlinear_f: The original nonlinear cost function f(x; y).
    """
    
    # --- STAGE 1: Offline SurCo-prior Training ---
    # Goal: Distill domain knowledge into model parameters theta
    
    optimizer_theta = optim.Adam(model.parameters(), lr=0.001)
    
    for epoch in range(num_epochs):
        for y_batch in train_data:
            optimizer_theta.zero_grad()
            
            # Predict surrogate coefficients for the batch
            c_hat = model(y_batch)
            
            # Solve the linear surrogate problem differentiably
            # x_star = argmin c^T * x s.t. x in Omega
            x_sol = solver(c_hat, y_batch) 
            
            # Compute nonlinear loss and backpropagate through the solver
            loss = objective_f(x_sol, y_batch)
            loss.backward()
            optimizer_theta.step()


def finetune_surco_hybrid(test_instance, model, diff_solver, nonlinear_f, 
                          lr=0.01, threshold=1e-6, max_steps=200):
    """
    Actual implementation of the SurCo-hybrid online refinement phase.
    
    Args:
        test_instance: The specific y parameters for the current problem[cite: 54, 59].
        model: The pre-trained SurCo-prior neural network[cite: 113, 136].
        diff_solver: A differentiable wrapper around a solver (e.g., PyVat, CVXPYLayers)[cite: 42, 88].
        nonlinear_f: The differentiable nonlinear objective function[cite: 71, 74].
    """
    
    # 1. Warm-start: Predict initial surrogate costs from the offline model [cite: 47, 136]
    # We detach from the model's graph because we are now optimizing 'c' directly.
    with torch.no_grad():
        c_hat = model(test_instance)
    
    # Define the surrogate coefficients as the parameters to be optimized
    c_coeffs = torch.tensor(c_hat, requires_grad=True)
    
    # The paper often uses Adam for these gradient updates [cite: 91, 621]
    optimizer = optim.Adam([c_coeffs], lr=lr)
    
    previous_loss = float('inf')
    final_x = None

    # 2. Refinement Loop: Iteratively improve the surrogate for this specific instance [cite: 137, 139]
    for step in range(max_steps):
        optimizer.zero_grad()
        
        # solver(c) returns x_star = argmin c^T * x s.t. x in Omega
        # This must be a differentiable wrapper to provide gradients
        x_sol = diff_solver(c_coeffs, test_instance)
        
        # Calculate the actual nonlinear objective value
        loss = nonlinear_f(x_sol, test_instance)
        
        # Check for convergence based on the absolute change in loss
        current_loss_val = loss.item()
        if abs(previous_loss - current_loss_val) < threshold:
            print(f"Converged at step {step}: Loss = {current_loss_val:.4f}")
            final_x = x_sol
            break
            
        previous_loss = current_loss_val
        final_x = x_sol
        
        # 3. Backpropagate: Grad(Loss) -> Grad(x) -> Grad(c) 
        loss.backward()
        optimizer.step()
        
    # Return the final combinatorially feasible solution [cite: 41, 45]
    return final_x