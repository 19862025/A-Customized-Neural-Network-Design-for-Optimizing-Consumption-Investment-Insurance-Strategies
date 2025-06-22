#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 11 17:05:09 2025

@author: marcial
"""

import numpy as np
import matplotlib.pyplot as plt

# Parameters
N = 1000  # Number of Monte Carlo samples ()
T = 70      # Time horizon
r = 0.02    # Risk-free rate
volatilities_Z = [0.05, 0.1, 0.2]  # Different volatilities for Z
volatilities_S = [0.05, 0.1, 0.2]  # Different volatilities for S

# Plot paths for different volatilities
plt.figure(figsize=(12, 5))
for sigma_Z in volatilities_Z:
    Z_paths = np.zeros((N, T))
    for t in range(1, T):
        Z_paths[:, t] = Z_paths[:, t-1] + np.random.normal(0, sigma_Z, N)
    for i in range(min(N, 10)):  # Plot only 5 sample paths per volatility for clarity
        plt.plot(range(T), Z_paths[i, :], alpha=0.6, label=f'sigma_Z={sigma_Z}' if i == 0 else "")
plt.title("Evolution of Z_u over Time for Different Volatilities")
plt.xlabel("Time Steps")
plt.ylabel("Z_u")
plt.legend()
plt.grid()
plt.show()

plt.figure(figsize=(12, 5))
for sigma_S in volatilities_S:
    S_paths = np.ones((N, T))  # Start with S = 1
    for t in range(1, T):
        S_paths[:, t] = S_paths[:, t-1] * (1 + np.random.normal(0, sigma_S, N))
    for i in range(min(N, 10)):
        plt.plot(range(T), S_paths[i, :], alpha=0.6, label=f'sigma_S={sigma_S}' if i == 0 else "")
plt.title("Evolution of S_u over Time for Different Volatilities")
plt.xlabel("Time Steps")
plt.ylabel("S_u")
plt.legend()
plt.grid()
plt.show()
