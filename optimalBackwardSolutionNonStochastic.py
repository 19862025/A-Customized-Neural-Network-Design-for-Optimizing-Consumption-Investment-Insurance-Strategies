#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 27 12:15:57 2025

To do:  compare the solution from this code with the cii_nn solution when all of the sigmas are set equal to 0.

@author: marcial
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import root_scalar

# Define problem constants and parameters (these can be updated as per the problem's data)
T = 40  # Time horizon (age 25 to 65)
r_t = np.full(T, 0.02)  # Risk-free interest rate
eta_t = np.full(T, 0.05)  # Deterministic investment interest rates @@@ not used
R_t = np.linspace(5,7,T)  # Yearly income (in $10000 units)
X_init = 1 # Initial wealth (in $10000 units)

def generate_scenarios():
    """Generate multiple scenarios for alpha_t and pD_t."""
    pd_t = np.linspace(0.001, 0.01, T+1)  # Probability of death per year # @@@ NOTE T+1
    Pbard_t = 1-np.cumsum(pd_t)  # Survival probability
    scenarios = []
    D_t = 0.995**np.arange(T+1) # Discount factor
    # @@@ Discount the probabilities
    pD_t = pd_t*D_t
    PbarD_t = Pbard_t*D_t
    
    for scale in np.linspace(1,1.04,5):  # Scaling factors for scenarios  (note should be > 1 for insurance company to make profit)
        alpha_t = pd_t * scale  # Scaled insurance return parameter

        scenarios.append((alpha_t, pD_t, PbarD_t))
    return scenarios

scenarios = generate_scenarios()

delta = 0.5  # Consumption exponent
epsilon = 0.9  # Legacy exponent -- should be larger than consumption exponent.
# @@@ To calibrate legacy versus consumption
C0,X0 = 1,1
K = (epsilon/delta)*C0**delta/X0**epsilon  # Legacy-consumption scaling factor

betaMax = 1 # @@@ Maximum insurance payment

# Utility and derivative functions
def VC_prime(C):
    return ( (C>=0)*C + (C<=0)*1E-3 ) ** (delta - 1)# Avoid singularity

def VX_prime(X):
    return ( (X>=0)*X + (X<=0)*1E-3 ) ** (epsilon - 1) / K# Avoid singularity

def WC(V_prime):
    return V_prime ** (1 / (delta - 1))

def WX(V_prime):
    return (K*V_prime) ** (1 / (epsilon - 1)) # @@@ Note correction

# Wealth equation
def wealth_equation(X_prev, C, beta, r_t, R_t):
    return X_prev * (1 + r_t) - C - beta + R_t

# Backward solve
def backward_solve(X_T, r_t, alpha_t, pD_t, PbarD_t):
    C = np.zeros(T)
    beta = np.zeros(T)
    X = np.zeros(T + 1)
    Y = np.zeros(T)

    X[-1] = X_T

    for t in range(T - 1, -1, -1):
        if t == T - 1:
            # @@@ NOTE T instead of t as per overleaf document
            C[t] = WC(VX_prime(X[T]) * PbarD_t[T] / PbarD_t[T - 1])
            Y[t] = WX(VX_prime(X[T]) * PbarD_t[T] * alpha_t[T-1] / pD_t[T - 1])
        else:
            Qt = VX_prime(X[t + 1]) * PbarD_t[t + 1] * (1 + r_t[t])
            C[t] = WC(Qt / PbarD_t[t])
            Y[t] = WX(Qt * alpha_t[t] / pD_t[t])

        beta[t] = (Y[t] * (1 + r_t[t]) - X[t + 1] - C[t] + R_t[t]) / (1 + (1 + r_t[t]) / alpha_t[t])
        
        beta[t] = min(beta[t], alpha_t[t]*Y[t])# @@@ Prevent negative wealth
        if beta[t] > betaMax or beta[t]<0: # @@@ Limit maximum payment and ensure beta is not negative
            beta[t] = betaMax * (beta[t]>=0) # @@@ Set equal to betaMax or 0
            # Adjust X[t] and Y[t] accordingly
            X[t] = (X[t+1] + C[t] + beta[t] - R_t[t]) / (1 + r_t[t])
            Y[t] = X[t] + beta[t] / alpha_t[t]
        else:
            X[t] = max(Y[t] - beta[t] / alpha_t[t],0) # @@@ ensure X is positive (avoid roundoff)
            Y[t] = X[t] + beta[t] / alpha_t[t] # @@@ ensure compatibility

    return X, C, beta

# Forward solve
def forward_solve(X0, C, beta, r_t, R_t):
    X = np.zeros(T + 1)
    X[0] = X0

    for t in range(T):
        X[t + 1] = wealth_equation(X[t], C[t], beta[t], r_t[t], R_t[t])

    return X

# Define fixed-point function
def fixed_point_function(X_T, X_init, r_t, alpha_t, pD_t, PbarD_t):
    X_backward, C, beta = backward_solve(X_T, r_t, alpha_t, pD_t, PbarD_t)
    X_forward = forward_solve(X_init, C, beta, r_t, R_t)
    return X_forward[-1] - X_T

# Plot results for multiple scenarios
X_T_guess = 1
plt.figure(figsize=(16, 12))

for idx, (alpha_t, pD_t, PbarD_t) in enumerate(scenarios):
    # Solve using root_scalar with Newton's method
    result = root_scalar(fixed_point_function, args=(X_init, r_t, alpha_t, pD_t, PbarD_t), x0=X_T_guess, method='secant', xtol=1e-2)

    if result.converged:
        print(f"Scenario {idx + 1}: Converged to X[T]: {result.root}")
        X_solution, C_solution, beta_solution = backward_solve(result.root, r_t, alpha_t, pD_t, PbarD_t)
        time = np.arange(T + 1)

        # Plot wealth trajectory
        plt.subplot(4, 1, 1)
        plt.plot(time, X_solution, label=f"Scenario {idx + 1}", marker="o")
        plt.title("Optimal Wealth Trajectory")
        plt.xlabel("Time")
        plt.ylabel("Wealth (X_t)")
        plt.grid(True)
        plt.legend()

        # Plot consumption trajectory
        plt.subplot(4, 1, 2)
        plt.plot(time[:-1], C_solution, label=f"Scenario {idx + 1}", marker="o")
        plt.title("Optimal Consumption Trajectory")
        plt.xlabel("Time")
        plt.ylabel("Consumption (C_t)")
        plt.grid(True)
        plt.legend()

        # Plot insurance payments
        plt.subplot(4, 1, 3)
        plt.plot(time[:-1], beta_solution, label=f"Scenario {idx + 1}", marker="o")
        plt.title("Optimal Insurance Payments")
        plt.xlabel("Time")
        plt.ylabel("Insurance Payments (β_t)")
        plt.grid(True)
        plt.legend()

        # Plot probability of death and insurance return
        plt.subplot(4, 1, 4)
        plt.plot(np.arange(T+1), pD_t, label=f"pD_t Scenario {idx + 1}", marker="o")
        plt.plot(np.arange(T+1), alpha_t, label=f"alpha_t Scenario {idx + 1}", marker="x")
        plt.title("Probability of Death and Insurance Return Parameter")
        plt.xlabel("Time")
        plt.ylabel("Value")
        plt.grid(True)
        plt.legend()

    else:
        print(f"Scenario {idx + 1}: Newton's method did not converge.")

plt.tight_layout()
plt.show()
