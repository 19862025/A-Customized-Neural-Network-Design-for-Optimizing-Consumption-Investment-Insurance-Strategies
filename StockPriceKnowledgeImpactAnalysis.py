
# -*- coding: utf-8 -*-
"""
Stock Price Knowledge Impact Analysis
Created on March 30, 2025
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
sigma_Z = 0.01
sigma_S = 0.01
r = 0.02  # Risk-free interest rate
Z0 = 0.025  # Initial random process
C0, X0 = 1, 1  # Initial consumption and wealth
delta = 0.2    # Consumption preference
epsilon = 0.9  # Legacy preference
alpha_scale = 10.0  # Bequest influence
R = np.ones(T)  # Income
utility_factor = 1 / T**2
K = (delta / epsilon) * C0**epsilon / X0**delta 

# Generate training data
np.random.seed(42)
B_zt = np.random.normal(0, sigma_Z, (J, T + M))
B_St = np.random.normal(0, sigma_S, (J, T + M))
B_zt[:, 0] = 0
Z = Z0 + np.cumsum(B_zt, axis=-1)
Irel = Z + B_St - r
x_train = np.copy(Irel)
y_train = np.ones((J,))

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
        #self.alpha_scale = alpha_scale 

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
        consumption_utility = VC(c * x[:, :-1]) * PbarD_t_tf[:, :-1]
        legacy_term = x[:, :-1] * (1 + b / (alpha_scale * pd_t_tf_reshaped[:, :-1] + 1e-6))
        legacy_utility = VX(legacy_term)
        utility = self.utility_factor * (consumption_utility + legacy_utility)
        total_utility = tf.reduce_sum(utility, axis=1) + VX(x[:, -1]) * PbarD_t_tf[:, -1] * self.utility_factor
        return tf.expand_dims(total_utility, axis=-1)

# Build model
def build_model(T, M, nodeConfig, utility_factor, include_current=False):
    inputs = Input(shape=(T + M,))
    normalizer = Normalization()
    normalizer.adapt(x_train)
    normalized_inputs = normalizer(inputs)
    layer_outputs = []

    for t in range(T):
        if include_current:
            # Include current stock price (M + t)
            subset_input = normalized_inputs[:, :M + t + 1]
        else:
            # Only past stock prices (up to M + t - 1, excluding current)
            subset_input = normalized_inputs[:, :M + t]
        dense = Dense(nodeConfig[t], activation="relu")(subset_input)
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

# Extract policy and utility
def extract_policy_and_utility(concatenated_outputs, T):
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

    # Compute utility (same as CustomComputationLayer but for evaluation)
    batch_size = tf.shape(concatenated_outputs)[0]
    x_init = tf.ones((batch_size, 1), dtype=tf.float32)
    wealth_factor = tf.clip_by_value(1 - c - b + r + rho * IrelVals, 1e-6, 1e6)
    x = tf.TensorArray(dtype=tf.float32, size=T+1, clear_after_read=False)
    x = x.write(0, x_init[:, 0])
    for n in range(T):
        x_n = x.read(n) * wealth_factor[:, n] + R[n]
        x = x.write(n + 1, x_n)
    x = tf.transpose(x.stack())
    pd_t_tf_reshaped = tf.expand_dims(pd_t_tf, axis=0)
    consumption_utility = VC(c * x[:, :-1]) * PbarD_t_tf[:, :-1]
    legacy_term = x[:, :-1] * pD_t_tf * (1 + b / ( alpha_scale * pd_t_tf_reshaped[:, :-1] + 1e-6))
    legacy_utility = VX(legacy_term)
    utility = utility_factor * (consumption_utility + legacy_utility)
    total_utility = tf.reduce_sum(utility, axis=1) + VX(x[:, -1]) * PbarD_t_tf[:, -1] * utility_factor

    return c.numpy(), b.numpy(), rho.numpy(), total_utility.numpy()

# Main analysis
def analyze_stock_price_knowledge():
    nodeConfig = 15 * np.ones(T).astype(int)

    # Baseline model (no current stock price knowledge)
    baseline_model, baseline_normalizer = build_model(T, M, nodeConfig, utility_factor, include_current=False)
    baseline_model.fit(x_train, [y_train, [np.zeros((J, 4)) for _ in range(T)]], 
                       epochs=30, batch_size=32, verbose=0)
    _, baseline_layer_outputs = baseline_model(x_train)
    baseline_concat = Concatenate()(baseline_layer_outputs + [x_train[:, M:]])
    baseline_c, baseline_b, baseline_rho, baseline_utility = extract_policy_and_utility(baseline_concat, T)

    # Modified model (with current stock price knowledge)
    modified_model, modified_normalizer = build_model(T, M, nodeConfig, utility_factor, include_current=True)
    modified_model.fit(x_train, [y_train, [np.zeros((J, 4)) for _ in range(T)]], 
                       epochs=30, batch_size=32, verbose=0)
    _, modified_layer_outputs = modified_model(x_train)
    modified_concat = Concatenate()(modified_layer_outputs + [x_train[:, M:]])
    modified_c, modified_b, modified_rho, modified_utility = extract_policy_and_utility(modified_concat, T)

    # Compute mean strategies and utilities
    baseline_c_mean = np.mean(baseline_c, axis=0)
    baseline_b_mean = np.mean(baseline_b, axis=0)
    baseline_rho_mean = np.mean(baseline_rho, axis=0)
    baseline_mean_utility = np.mean(baseline_utility) / utility_factor

    modified_c_mean = np.mean(modified_c, axis=0)
    modified_b_mean = np.mean(modified_b, axis=0)
    modified_rho_mean = np.mean(modified_rho, axis=0)
    modified_mean_utility = np.mean(modified_utility) / utility_factor

    # Print results
    print("Baseline Model (No Current Stock Price Knowledge):")
    print(f"Mean c: {baseline_c_mean}")
    print(f"Mean b: {baseline_b_mean}")
    print(f"Mean rho: {baseline_rho_mean}")
    print(f"Mean Utility: {baseline_mean_utility:.4f}\n")

    print("Modified Model (With Current Stock Price Knowledge):")
    print(f"Mean c: {modified_c_mean}")
    print(f"Mean b: {modified_b_mean}")
    print(f"Mean rho: {modified_rho_mean}")
    print(f"Mean Utility: {modified_mean_utility:.4f}")
    print(f"Utility Improvement: {(modified_mean_utility - baseline_mean_utility):.4f} ({((modified_mean_utility - baseline_mean_utility) / baseline_mean_utility * 100):.2f}%)")

    # Plotting mean strategies
    fig, axes = plt.subplots(3, 1, figsize=(10, 15))
    time_steps = np.arange(T)

    axes[0].plot(time_steps, baseline_c_mean, label='Baseline c', marker='o')
    axes[0].plot(time_steps, modified_c_mean, label='Modified c', marker='x')
    axes[0].set_title('Mean Consumption (c)')
    axes[0].set_xlabel('Time Step')
    axes[0].set_ylabel('c')
    axes[0].legend()
    axes[0].grid(True)

    axes[1].plot(time_steps, baseline_b_mean, label='Baseline b', marker='o')
    axes[1].plot(time_steps, modified_b_mean, label='Modified b', marker='x')
    axes[1].set_title('Mean Bequest (b)')
    axes[1].set_xlabel('Time Step')
    axes[1].set_ylabel('b')
    axes[1].legend()
    axes[1].grid(True)

    axes[2].plot(time_steps, baseline_rho_mean, label='Baseline rho', marker='o')
    axes[2].plot(time_steps, modified_rho_mean, label='Modified rho', marker='x')
    axes[2].set_title('Mean Risk Exposure (rho)')
    axes[2].set_xlabel('Time Step')
    axes[2].set_ylabel('rho')
    axes[2].legend()
    axes[2].grid(True)

    plt.tight_layout()
    plt.show()

# Run analysis
if __name__ == "__main__":
    analyze_stock_price_knowledge()