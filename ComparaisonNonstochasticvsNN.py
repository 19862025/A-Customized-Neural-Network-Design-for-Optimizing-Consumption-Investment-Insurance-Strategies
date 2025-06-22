#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun 22 20:03:43 2025

@author: marcial
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import root_scalar
import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.layers import Layer, Input, Dense, Concatenate, Normalization

# Backward-forward model functions (from optimalBackwardSolutionNonStochastic.py)
T = 40
r_t = np.full(T, 0.02)
R_t = np.linspace(5, 7, T)
X_init = 1
pd_t = np.linspace(0.001, 0.01, T+1)
Pbard_t = 1 - np.cumsum(pd_t)
D_t = 0.995**np.arange(T+1)
pD_t = (pd_t * D_t)[:-1]  # Shape (T,)
PbarD_t = Pbard_t * D_t
alpha_t = pd_t[:-1] * 1.0  # Scenario with scale = 1.0
delta = 0.5
epsilon = 0.9
C0, X0 = 1, 1
K = (epsilon/delta) * C0**delta / X0**epsilon
betaMax = 1

def VC_prime(C):
    return ((C >= 0) * C + (C <= 0) * 1e-3) ** (delta - 1)

def VX_prime(X):
    return ((X >= 0) * X + (X <= 0) * 1e-3) ** (epsilon - 1) / K

def WC(V_prime):
    return V_prime ** (1 / (delta - 1))

def WX(V_prime):
    return (K * V_prime) ** (1 / (epsilon - 1))

def wealth_equation(X_prev, C, beta, r_t, R_t):
    return X_prev * (1 + r_t) - C - beta + R_t

def backward_solve(X_T, r_t, alpha_t, pD_t, PbarD_t):
    C = np.zeros(T)
    beta = np.zeros(T)
    X = np.zeros(T + 1)
    Y = np.zeros(T)
    X[-1] = X_T
    for t in range(T - 1, -1, -1):
        if t == T - 1:
            C[t] = WC(VX_prime(X[T]) * PbarD_t[T] / PbarD_t[T - 1])
            Y[t] = WX(VX_prime(X[T]) * PbarD_t[T] * alpha_t[T-1] / pD_t[T - 1])
        else:
            Qt = VX_prime(X[t + 1]) * PbarD_t[t + 1] * (1 + r_t[t])
            C[t] = WC(Qt / PbarD_t[t])
            Y[t] = WX(Qt * alpha_t[t] / pD_t[t])
        beta[t] = (Y[t] * (1 + r_t[t]) - X[t + 1] - C[t] + R_t[t]) / (1 + (1 + r_t[t]) / alpha_t[t])
        beta[t] = min(beta[t], alpha_t[t] * Y[t])
        if beta[t] > betaMax or beta[t] < 0:
            beta[t] = betaMax * (beta[t] >= 0)
            X[t] = (X[t+1] + C[t] + beta[t] - R_t[t]) / (1 + r_t[t])
            Y[t] = X[t] + beta[t] / alpha_t[t]
        else:
            X[t] = max(Y[t] - beta[t] / alpha_t[t], 0)
            Y[t] = X[t] + beta[t] / alpha_t[t]
    return X, C, beta

def forward_solve(X0, C, beta, r_t, R_t):
    X = np.zeros(T + 1)
    X[0] = X0
    for t in range(T):
        X[t + 1] = wealth_equation(X[t], C[t], beta[t], r_t[t], R_t[t])
    return X

def fixed_point_function(X_T, X_init, r_t, alpha_t, pD_t, PbarD_t):
    X_backward, C, beta = backward_solve(X_T, r_t, alpha_t, pD_t, PbarD_t)
    X_forward = forward_solve(X_init, C, beta, r_t, R_t)
    return X_forward[-1] - X_T

# Run backward-forward model
result = root_scalar(fixed_point_function, args=(X_init, r_t, alpha_t, pD_t, PbarD_t), x0=1, method='secant', xtol=1e-2)
X_bf, C_bf, beta_bf = backward_solve(result.root, r_t, alpha_t, pD_t, PbarD_t)

# NN model parameters (aligned with backward-forward)
J = 3200
M = 3
sigma_Z = 0.0
sigma_S = 0.0
r = 0.02
Z0 = 0.025
alpha_scale = 1.0
utility_factor = 1/T**2
R = R_t  # Use backward-forward income

# Generate non-stochastic data
np.random.seed(42)
B_zt = np.random.normal(0, sigma_Z, (J, T+M))
B_St = np.random.normal(0, sigma_S, (J, T+M))
B_zt[:, 0] = 0
Z = Z0 + np.cumsum(B_zt, axis=-1)
Irel = Z + B_St - r
x_train = np.copy(Irel)
normalizer = Normalization()
normalizer.adapt(x_train)
y_train = np.ones((J,))

# NN tensor variables
pD_t_tf = tf.Variable(pD_t.reshape(1, -1), dtype=tf.float32, trainable=False)
PbarD_t_tf = tf.Variable(PbarD_t.reshape(1, -1), dtype=tf.float32, trainable=False)
pd_t_tf = tf.Variable(pd_t[:-1].reshape(1, -1), dtype=tf.float32, trainable=False)

# NN utility functions (from previous code)
def VC(C):
    C = tf.clip_by_value(C, 1e-6, 1e6)
    return tf.pow(C, delta) / delta

def VX(X):
    X = tf.clip_by_value(X, 1e-6, 1e6)
    return tf.pow(X, epsilon) / (K * epsilon)

def tf_fn(concatenated_outputs, T, utility_factor):
    batch_size = tf.shape(concatenated_outputs)[0]
    cVals = concatenated_outputs[:, 0:4*T:4]
    bVals = concatenated_outputs[:, 1:4*T:4]
    iVals = concatenated_outputs[:, 2:4*T:4]
    rhoVals = concatenated_outputs[:, 3:4*T:4]
    IrelVals = concatenated_outputs[:, 4*T:]
    cbi_stacked = tf.stack([cVals, bVals, iVals], axis=-1)
    cbi_softmax = tf.nn.softmax(cbi_stacked, axis=-1)
    c = tf.clip_by_value(cbi_softmax[..., 0], 1e-6, 1-1e-6)
    b = tf.clip_by_value(cbi_softmax[..., 1], 1e-6, 1-1e-6)
    rho = tf.clip_by_value(tf.nn.sigmoid(rhoVals), 1e-6, 1-1e-6)
    x_init = tf.ones((batch_size, 1), dtype=tf.float32)
    wealth_factor = tf.clip_by_value(1 - c - b + r + rho * IrelVals, 1e-6, 1e6)
    x = tf.TensorArray(dtype=tf.float32, size=T+1, dynamic_size=False, clear_after_read=False)
    x = x.write(0, x_init[:, 0])
    for n in range(T):
        x_n = x.read(n) * wealth_factor[:, n] + R[n]
        x = x.write(n + 1, x_n)
    x = x.stack()
    x = tf.transpose(x)
    consumption_utility = VC(c * x[:, :-1]) * PbarD_t_tf[:, :-1]
    legacy_term = x[:, :-1] * (1 + b / (alpha_scale * pd_t_tf + 1e-6))
    legacy_utility = VX(legacy_term) * pD_t_tf
    utility = utility_factor * (consumption_utility + legacy_utility)
    total_utility = tf.reduce_sum(utility, axis=1) + VX(x[:, -1]) * PbarD_t_tf[:, -1] * utility_factor
    return tf.expand_dims(total_utility, axis=-1)

class CustomComputationLayer(Layer):
    def __init__(self, T, utility_factor, **kwargs):
        super(CustomComputationLayer, self).__init__(**kwargs)
        self.T = T
        self.utility_factor = utility_factor
    def build(self, input_shape):
        pass
    def call(self, inputs):
        return tf_fn(inputs, self.T, self.utility_factor)
    def compute_output_shape(self, input_shape):
        return (input_shape[0], 1)

def build_custom_model(T, M, nodeConfig, utility_factor):
    inputs = Input(shape=(T + M,))
    normalized_inputs = normalizer(inputs)
    layer_outputs = []
    for tt in range(T):
        subset_input = normalized_inputs[:, :(M + tt)]
        dense = Dense(nodeConfig[tt], activation="relu")(subset_input)
        output = Dense(4, activation="linear")(dense)
        layer_outputs.append(output)
    concatenated = Concatenate()(layer_outputs + [inputs[:, M:]])
    computed_output = CustomComputationLayer(T, utility_factor)(concatenated)
    final_output = Dense(1, activation="sigmoid", use_bias=False)(computed_output)
    model = Model(inputs=inputs, outputs=[final_output, layer_outputs])
    final_dense_layer = model.layers[-1]
    final_dense_layer.set_weights([np.ones_like(final_dense_layer.get_weights()[0])])
    final_dense_layer.trainable = False
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss=["mse", None])
    return model, normalizer

def extract_policy2(concatenated_outputs, T):
    batch_size = tf.shape(concatenated_outputs)[0]
    cVals = concatenated_outputs[:, 0:4*T:4]
    bVals = concatenated_outputs[:, 1:4*T:4]
    iVals = concatenated_outputs[:, 2:4*T:4]
    rhoVals = concatenated_outputs[:, 3:4*T:4]
    IrelVals = concatenated_outputs[:, 4*T:]
    cbi_stacked = tf.stack([cVals, bVals, iVals], axis=-1)
    cbi_softmax = tf.nn.softmax(cbi_stacked, axis=-1)
    c = tf.clip_by_value(cbi_softmax[..., 0], 1e-6, 1-1e-6)
    b = tf.clip_by_value(cbi_softmax[..., 1], 1e-6, 1-1e-6)
    rho = tf.clip_by_value(tf.nn.sigmoid(rhoVals), 1e-6, 1-1e-6)
    x_init = tf.ones((batch_size, 1), dtype=tf.float32)
    r_broadcasted = tf.broadcast_to(tf.constant(r, dtype=tf.float32), tf.shape(c))
    wealth_factor = tf.clip_by_value(1 - c - b + r_broadcasted + rho * IrelVals, 1e-6, 1e6)
    x = tf.TensorArray(dtype=tf.float32, size=T+1, dynamic_size=False, clear_after_read=False)
    x = x.write(0, x_init[:, 0])
    for n in range(T):
        x_n = x.read(n) * wealth_factor[:, n] + R[n]
        x_n = tf.clip_by_value(x_n, 1e-6, 1e6)
        x = x.write(n + 1, x_n)
    x = x.stack()
    x = tf.transpose(x)
    return c.numpy(), b.numpy(), rho.numpy(), x.numpy()

# Run NN model
nodeParams = [15, 0]
nodeConfig = (nodeParams[0] * np.ones(T) + nodeParams[1] * np.arange(T)).astype(int)
model, normalizer = build_custom_model(T, M, nodeConfig, utility_factor)
history = model.fit(x_train, [y_train, [tf.zeros((J, 4)) for _ in range(T)]], epochs=30, batch_size=32, verbose=0)
_, layer_outputs = model(x_train)
concatenated_outputs = tf.keras.layers.Concatenate()(layer_outputs + [x_train[:, M:]])
c_nn, b_nn, rho_nn, x_nn = extract_policy2(concatenated_outputs, T)
c_nn_mean = np.mean(c_nn * x_nn[:, :-1], axis=0)  # Absolute consumption
b_nn_mean = np.mean(b_nn * x_nn[:, :-1], axis=0)  # Absolute bequest
x_nn_mean = np.mean(x_nn, axis=0)

# Compute utilities
def compute_utility_bf(X, C, beta):
    consumption_utility = np.sum(VC(C) * PbarD_t[:-1])
    legacy_term = X[:-1] * (1 + beta / (alpha_t + 1e-6))
    legacy_utility = np.sum(VX(legacy_term) * pD_t)
    terminal_utility = VX(X[-1]) * PbarD_t[-1]
    return consumption_utility + legacy_utility + terminal_utility

def compute_utility_nn(c, b, x):
    consumption_utility = np.mean(np.sum(VC(c * x[:, :-1]) * PbarD_t[:-1], axis=1))
    legacy_term = x[:, :-1] * (1 + b / (alpha_scale * pd_t[:-1] + 1e-6))
    legacy_utility = np.mean(np.sum(VX(legacy_term) * pD_t, axis=1))
    terminal_utility = np.mean(VX(x[:, -1]) * PbarD_t[-1])
    return consumption_utility + legacy_utility + terminal_utility

utility_bf = compute_utility_bf(X_bf, C_bf, beta_bf)
utility_nn = compute_utility_nn(c_nn, b_nn, x_nn)

# Plot comparison
plt.figure(figsize=(12, 8))
plt.subplot(3, 1, 1)
plt.plot(range(T), C_bf, label='Backward-Forward (C_t)', marker='o')
plt.plot(range(T), c_nn_mean, label='NN (c_t x_t)', marker='x')
plt.title('Consumption Comparison')
plt.xlabel('Time')
plt.ylabel('Consumption')
plt.legend()
plt.grid(True)

plt.subplot(3, 1, 2)
plt.plot(range(T), beta_bf, label='Backward-Forward (β_t)', marker='o')
plt.plot(range(T), b_nn_mean, label='NN (b_t x_t)', marker='x')
plt.title('Bequest/Insurance Comparison')
plt.xlabel('Time')
plt.ylabel('Bequest/Insurance')
plt.legend()
plt.grid(True)

plt.subplot(3, 1, 3)
plt.plot(range(T+1), X_bf, label='Backward-Forward (X_t)', marker='o')
plt.plot(range(T+1), x_nn_mean, label='NN (x_t)', marker='x')
plt.title('Wealth Comparison')
plt.xlabel('Time')
plt.ylabel('Wealth')
plt.legend()
plt.grid(True)

plt.suptitle('Backward-Forward vs Neural Network (Non-Stochastic)', fontsize=16)
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.show()

# Print results
print("=== Backward-Forward vs NN Comparison ===")
print("Utility (Backward-Forward):", utility_bf)
print("Utility (NN):", utility_nn)
print("\nMean Initial Strategies (t=0):")
print("Backward-Forward - C[0]:", C_bf[0], "β[0]:", beta_bf[0], "X[0]:", X_bf[0])
print("NN - c[0]x[0]:", c_nn_mean[0], "b[0]x[0]:", b_nn_mean[0], "x[0]:", x_nn_mean[0])
