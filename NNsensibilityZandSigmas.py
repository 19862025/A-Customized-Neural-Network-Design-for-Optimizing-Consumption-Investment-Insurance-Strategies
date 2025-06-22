#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 31 08:44:30 2025

@author: marcial
"""


import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Model, Input
from tensorflow.keras.layers import Dense, Concatenate, Normalization, Layer
import matplotlib.pyplot as plt

# Parameters
J = 3200  # Number of vectors
M = 3     # Number of previous stock prices known
T = 20    # Number of years
r = 0.02  # Risk-free interest rate
C0, X0 = 1, 1  # Initial consumption and wealth
delta = 0.2    # Consumption preference
epsilon = 0.9  # Legacy preference
R = np.ones(T)  # Income
utility_factor = 1 / T**2

# Survival probabilities and discount factors
pd_t = np.linspace(0.001, 0.01, T + 1)
Pbard_t = 1 - np.cumsum(pd_t)
D_t = 0.995 ** np.arange(T + 1)
pD_t = (pd_t * D_t).reshape(1, -1)[:, :-1]
PbarD_t = (Pbard_t * D_t).reshape(1, -1)
PbarD_t_tf = tf.Variable(PbarD_t, dtype=tf.float32, trainable=False)
pD_t_tf = tf.Variable(pD_t, dtype=tf.float32, trainable=False)

# Utility functions
def VC(C):
    C = tf.clip_by_value(C, 1e-6, 1e6)
    return tf.pow(C, delta) / delta

def VX(X):
    X = tf.clip_by_value(X, 1e-6, 1e6)
    return tf.pow(X, epsilon) / (K * epsilon)

# Custom computation layer
class CustomComputationLayer(Layer):
    def __init__(self, T, utility_factor, **kwargs):
        super().__init__(**kwargs)
        self.T = T
        self.utility_factor = utility_factor

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

        consumption_utility = VC(c * x[:, :-1]) * PbarD_t_tf[:, :-1]
        legacy_term = x[:, :-1] * pD_t_tf * (1 + b / (alpha_scale * pD_t_tf + 1e-6))
        legacy_utility = VX(legacy_term)
        utility = self.utility_factor * (consumption_utility + legacy_utility)
        total_utility = tf.reduce_sum(utility, axis=1) + VX(x[:, -1]) * PbarD_t_tf[:, -1] * self.utility_factor
        return tf.expand_dims(total_utility, axis=-1)

# Build model
def build_model(T, M, nodeConfig, utility_factor, alpha_scale):
    global K
    K = (epsilon / delta) * C0**delta / X0**epsilon * 0.1
    inputs = Input(shape=(T + M,))
    normalizer = Normalization()
    normalizer.adapt(x_train)  # Will be set later
    normalized_inputs = normalizer(inputs)
    layer_outputs = []

    for K_idx in range(T):
        subset_input = normalized_inputs[:, :(M + K_idx)]
        dense = Dense(nodeConfig[K_idx], activation="relu")(subset_input)
        output = Dense(4, activation="linear")(dense)
        layer_outputs.append(output)

    concatenated = Concatenate()(layer_outputs + [inputs[:, M:]])
    computed_output = CustomComputationLayer(T, utility_factor)(concatenated)
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
    global x_train, alpha_scale
    np.random.seed(42)
    nodeConfig = 15 * np.ones(T).astype(int)

    # Parameters to test
    params = {
        'Z0': [0.015, 0.02, 0.025, 0.03],
        'sigma_Z': [0.005, 0.01, 0.015, 0.02],
        'sigma_S': [0.005, 0.01, 0.015, 0.02],
        'alpha_scale': [10, 15, 20, 25]
    }

    results = {param: {'c[0]': [], 'b[0]': [], 'rho[0]': []} for param in params}

    # Baseline values
    baseline_sigma_Z = 0.01
    baseline_sigma_S = 0.01
    baseline_Z0 = 0.025
    baseline_alpha_scale = 10.0

    for param_name, param_values in params.items():
        for value in param_values:
            # Set parameter
            Z0 = value if param_name == 'Z0' else baseline_Z0
            sigma_Z = value if param_name == 'sigma_Z' else baseline_sigma_Z
            sigma_S = value if param_name == 'sigma_S' else baseline_sigma_S
            alpha_scale = value if param_name == 'alpha_scale' else baseline_alpha_scale

            # Generate training data
            B_zt = np.random.normal(0, sigma_Z, (J, T + M))
            B_St = np.random.normal(0, sigma_S, (J, T + M))
            B_zt[:, 0] = 0
            Z = Z0 + np.cumsum(B_zt, axis=-1)
            Irel = Z + B_St - r
            x_train = np.copy(Irel)
            y_train = np.ones((J,))

            # Build and train model
            model, normalizer = build_model(T, M, nodeConfig, utility_factor, alpha_scale)
            model.fit(x_train, [y_train, [np.zeros((J, 4)) for _ in range(T)]], 
                      epochs=30, batch_size=32, verbose=0)

            # Extract policy
            _, layer_outputs = model(x_train)
            concatenated_outputs = Concatenate()(layer_outputs + [x_train[:, M:]])
            c, b, rho = extract_policy(concatenated_outputs, T)

            # Record averages for t=0
            results[param_name]['c[0]'].append(np.mean(c[:, 0]))
            results[param_name]['b[0]'].append(np.mean(b[:, 0]))
            results[param_name]['rho[0]'].append(np.mean(rho[:, 0]))

    # Plotting
    fig, axes = plt.subplots(4, 1, figsize=(10, 20))
    titles = ['Z0', 'sigma_Z', 'sigma_S', 'alpha_scale']
    for ax, param_name, title in zip(axes, params.keys(), titles):
        ax.plot(params[param_name], results[param_name]['c[0]'], label='c[0]', marker='o')
        ax.plot(params[param_name], results[param_name]['b[0]'], label='b[0]', marker='o')
        ax.plot(params[param_name], results[param_name]['rho[0]'], label='rho[0]', marker='o')
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
    for param, data in sensitivity_results.items():
        print(f"\nSensitivity to {param}:")
        print(f"c[0]: {data['c[0]']}")
        print(f"b[0]: {data['b[0]']}")
        print(f"rho[0]: {data['rho[0]']}")