#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May  4 17:01:44 2025

@author: marcial

Sensitivity analysis for delta, epsilon, and alpha_scale
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Model, Input
from tensorflow.keras.layers import Dense, Concatenate, Normalization, Layer
import matplotlib.pyplot as plt

# Parameters
J = 3200
M = 3
T = 20
r = 0.02
C0, X0 = 1, 1
R = np.ones(T)
utility_factor = 1 / T**2

# Survival probabilities and discount factors
pd_t = np.linspace(0.001, 0.01, T + 1)
Pbard_t = 1 - np.cumsum(pd_t)
D_t = 0.995 ** np.arange(T + 1)
pD_t = (pd_t * D_t).reshape(1, -1)[:, :-1]
PbarD_t = (Pbard_t * D_t).reshape(1, -1)
PbarD_t_tf = tf.Variable(PbarD_t, dtype=tf.float32, trainable=False)
pD_t_tf = tf.Variable(pD_t, dtype=tf.float32, trainable=False)
pd_t_tf = tf.Variable(pd_t, dtype=tf.float32, trainable=False)

# Utility functions
def VC(C, delta_val):
    C = tf.clip_by_value(C, 1e-6, 1e6)
    return tf.pow(C, delta_val) / delta_val

def VX(X, epsilon_val):
    X = tf.clip_by_value(X, 1e-6, 1e6)
    return tf.pow(X, epsilon_val) / (K * epsilon_val)

# Custom computation layer
class CustomComputationLayer(Layer):
    def __init__(self, T, utility_factor, delta_val, epsilon_val, alpha_scale_val, **kwargs):
        super().__init__(**kwargs)
        self.T = T
        self.utility_factor = utility_factor
        self.delta_val = delta_val
        self.epsilon_val = epsilon_val
        self.alpha_scale_val = alpha_scale_val

    def call(self, inputs):
        batch_size = tf.shape(inputs)[0]
        cVals = inputs[:, 0:4*T:4]
        bVals = inputs[:, 1:4*T:4]
        iVals = inputs[:, 2:4*T:4]
        rhoVals = inputs[:, 3:4*T:4]
        IrelVals = inputs[:, 4*T:]

        cbi_stacked = tf.stack([cVals, bVals, iVals], axis=-1)
        cbi_softmax = tf.nn.softmax(cbi_stacked, axis=-1)
        c = tf.clip_by_value(cbi_softmax[..., 0], 1e-6, 1-1e-6)
        b = tf.clip_by_value(cbi_softmax[..., 1], 1e-6, 1-1e-6)
        rho = tf.clip_by_value(tf.nn.sigmoid(rhoVals), 1e-6, 1-1e-6)

        x_init = tf.ones((batch_size, 1), dtype=tf.float32)
        wealth_factor = tf.clip_by_value(1 - c - b + r + rho * IrelVals, 1e-6, 1e6)

        x = tf.TensorArray(dtype=tf.float32, size=T+1, clear_after_read=False)
        x = x.write(0, x_init[:, 0])
        for n in range(T):
            x_n = x.read(n) * wealth_factor[:, n] + R[n]
            x = x.write(n + 1, x_n)
        x = tf.transpose(x.stack())
        pd_t_tf_reshaped = tf.expand_dims(pd_t_tf, axis=0)
        consumption_utility = VC(c * x[:, :-1], self.delta_val) * PbarD_t_tf[:, :-1]
        legacy_term = x[:, :-1] * (1 + b / (self.alpha_scale_val * pd_t_tf_reshaped[:, :-1] + 1e-6))
        legacy_utility = VX(legacy_term, self.epsilon_val) * pD_t_tf
        utility = self.utility_factor * (consumption_utility + legacy_utility)
        total_utility = tf.reduce_sum(utility, axis=1) + VX(x[:, -1], self.epsilon_val) * PbarD_t_tf[:, -1] * self.utility_factor
        return tf.expand_dims(total_utility, axis=-1)

# Build model
def build_model(T, M, nodeConfig, utility_factor, delta_val, epsilon_val, alpha_scale_val):
    global K
    K = (delta_val / epsilon_val) * X0**epsilon_val / C0**delta_val
    inputs = Input(shape=(T + M,))
    normalizer = Normalization()
    normalizer.adapt(x_train)
    normalized_inputs = normalizer(inputs)
    layer_outputs = []

    for tt in range(T):
        subset_input = normalized_inputs[:, :(M + tt)]
        dense = Dense(nodeConfig[tt], activation="relu")(subset_input)
        output = Dense(4, activation="linear")(dense)
        layer_outputs.append(output)

    concatenated = Concatenate()(layer_outputs + [inputs[:, M:]])
    computed_output = CustomComputationLayer(T, utility_factor, delta_val, epsilon_val, alpha_scale_val)(concatenated)
    final_output = Dense(1, activation="sigmoid", use_bias=False)(computed_output)

    model = Model(inputs=inputs, outputs=[final_output, layer_outputs])
    final_dense_layer = model.layers[-1]
    final_dense_layer.set_weights([np.ones_like(final_dense_layer.get_weights()[0])])
    final_dense_layer.trainable = False

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                  loss=["mse", None])
    return model, normalizer

# Extract policy
def extract_policy(concatenated_outputs, T):
    cVals = concatenated_outputs[:, 0:4*T:4]
    bVals = concatenated_outputs[:, 1:4*T:4]
    iVals = concatenated_outputs[:, 2:4*T:4]
    rhoVals = concatenated_outputs[:, 3:4*T:4]
    cbi_stacked = tf.stack([cVals, bVals, iVals], axis=-1)
    cbi_softmax = tf.nn.softmax(cbi_stacked, axis=-1)
    c = tf.clip_by_value(cbi_softmax[..., 0], 1e-6, 1-1e-6)
    b = tf.clip_by_value(cbi_softmax[..., 1], 1e-6, 1-1e-6)
    rho = tf.clip_by_value(tf.nn.sigmoid(rhoVals), 1e-6, 1-1e-6)
    return c.numpy(), b.numpy(), rho.numpy()

# Sensitivity analysis
def run_sensitivity_analysis():
    global x_train
    np.random.seed(42)
    nodeConfig = 15 * np.ones(T).astype(int)

    # Parameters to test
    params = {
        'delta': [0.16, 0.18, 0.2, 0.22, 0.24],
        'epsilon': [0.72, 0.81, 0.9, 0.99, 1.08],
        'alpha_scale': [0.8, 0.9, 1.0, 1.1, 1.2]
    }

    results = {param: {
        'c[0] mean': [], 'b[0] mean': [], 'rho[0] mean': [],
        'c[0] std': [], 'b[0] std': [], 'rho[0] std': []} for param in params}

    # Baseline values
    baseline_Z0 = 0.025
    baseline_sigma_Z = 0.01
    baseline_sigma_S = 0.01
    baseline_delta = 0.2
    baseline_epsilon = 0.9
    baseline_alpha_scale = 1.0

    # Generate training data (fixed for all runs)
    B_zt = np.random.normal(0, baseline_sigma_Z, (J, T + M))
    B_St = np.random.normal(0, baseline_sigma_S, (J, T + M))
    B_zt[:, 0] = 0
    Z = baseline_Z0 + np.cumsum(B_zt, axis=-1)
    Irel = Z + B_St - r
    x_train = np.copy(Irel)
    y_train = np.ones((J,))

    for param_name, param_values in params.items():
        for value in param_values:
            # Set parameters
            delta_val = value if param_name == 'delta' else baseline_delta
            epsilon_val = value if param_name == 'epsilon' else baseline_epsilon
            alpha_scale_val = value if param_name == 'alpha_scale' else baseline_alpha_scale

            # Build and train model
            model, normalizer = build_model(T, M, nodeConfig, utility_factor, delta_val, epsilon_val, alpha_scale_val)
            model.fit(x_train, [y_train, [np.zeros((J, 4)) for _ in range(T)]], 
                      epochs=30, batch_size=32, verbose=0)

            # Generate test data
            B_zt = np.random.normal(0, baseline_sigma_Z, (J, T + M))
            B_St = np.random.normal(0, baseline_sigma_S, (J, T + M))
            B_zt[:, 0] = 0
            Z = baseline_Z0 + np.cumsum(B_zt, axis=-1)
            x_test = Z + B_St - r
            y_test = np.ones((J,))

            # Extract policy
            _, layer_outputs = model(x_test)
            concatenated_outputs = Concatenate()(layer_outputs + [x_test[:, M:]])
            c, b, rho = extract_policy(concatenated_outputs, T)

            # Record averages and std for t=0
            results[param_name]['c[0] mean'].append(np.mean(c[:, 0]))
            results[param_name]['b[0] mean'].append(np.mean(b[:, 0]))
            results[param_name]['rho[0] mean'].append(np.mean(rho[:, 0]))
            results[param_name]['c[0] std'].append(np.std(c[:, 0]))
            results[param_name]['b[0] std'].append(np.std(b[:, 0]))
            results[param_name]['rho[0] std'].append(np.std(rho[:, 0]))

    # Plotting
    fig, axes = plt.subplots(3, 1, figsize=(10, 15))
    titles = ['delta', 'epsilon', 'alpha_scale']
    for ax, param_name, title in zip(axes, params.keys(), titles):
        c0_mean = results[param_name]['c[0] mean']
        b0_mean = results[param_name]['b[0] mean']
        rho0_mean = results[param_name]['rho[0] mean']
        c0_error = results[param_name]['c[0] std']
        b0_error = results[param_name]['b[0] std']
        rho0_error = results[param_name]['rho[0] std']

        ax.errorbar(params[param_name], c0_mean, yerr=c0_error, label='c[0]', fmt='o', capsize=3)
        ax.errorbar(params[param_name], b0_mean, yerr=b0_error, label='b[0]', fmt='o', capsize=3)
        ax.errorbar(params[param_name], rho0_mean, yerr=rho0_error, label='rho[0]', fmt='o', capsize=3)

        ax.set_title(f'Sensitivity to {title}')
        ax.set_xlabel(title)
        ax.set_ylabel('Average Value at t=0')
        ax.legend()
        ax.grid(True)

    plt.tight_layout()
    plt.show()

    return results

# Run analysis
if __name__ == "__main__":
    sensitivity_results = run_sensitivity_analysis()
    for param in ['delta', 'epsilon', 'alpha_scale']:
        print(f"\nSensitivity to {param}:")
        print(f"c[0] mean: {sensitivity_results[param]['c[0] mean']}")
        print(f"b[0] mean: {sensitivity_results[param]['b[0] mean']}")
        print(f"rho[0] mean: {sensitivity_results[param]['rho[0] mean']}")
        print(f"c[0] std: {sensitivity_results[param]['c[0] std']}")
        print(f"b[0] std: {sensitivity_results[param]['b[0] std']}")
        print(f"rho[0] std: {sensitivity_results[param]['rho[0] std']}")