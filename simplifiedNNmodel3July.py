import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense, Input
import numpy as np


# Placeholder for variables
# --------------------------

T = 40

x_init = 1

# Lifetime parameters
pd_t = np.linspace(0.001, 0.01, T)
Pbard_t = np.ones(T+1)
Pbard_t[1:] = 1 - np.cumsum(pd_t)
D_t = 0.995**np.arange(T+1)
pD_t = pd_t * D_t[:-1]  # Shape (T,)
PbarD_t = Pbard_t * D_t

# Utility parameters
delta = 0.5
epsilon = 0.9
C0, X0 = 1, 1
K = (delta/epsilon) *  X0**epsilon / C0**delta
ki = 1/25.# scaling of interest rate inputs
# Compensation parameters (agrees with notation in paper)
a = 0.6
alpha_t = pd_t*a  # Scenario with scale = 1.0
R_t = np.linspace(5, 7, T) # salary

# --------------------------
# Interest rate parameters
r_t = 0.02# Interest rate
M = 3# Known previous interest rates
N_nodes = [5]*T

sigma_Z = 0.00
sigma_S = 0.00
Z0 = 0.00

J = 4800
nn_epoch  = 40

# Generate non-stochastic data
np.random.seed(42)
B_zt = np.random.normal(0, sigma_Z, (J, T+M))
B_St = np.random.normal(0, sigma_S, (J, T+M))
B_zt[:, 0] = 0
# Relative interest rate, based on hidden process
Z = Z0 + np.cumsum(B_zt, axis=-1)
Irel = Z + B_St - r_t
x_train = Irel/ki# Rescale inputs
y_train = np.ones((J,))

# Sample differentiable VCfn and VXfn implementations
def VCfn(x,delta):
    return tf.math.pow(tf.nn.relu(x), delta)/delta

def VXfn(x,epsilon,K):
    return tf.math.pow(tf.nn.relu(x), epsilon)/(K*epsilon)


def eval_totUtil_numpy(c, b, i, rho, s, M, ki, r, R, PbarD, pd, pD, delta, epsilon, K, x_init):
    T = c.shape[1] if c.ndim == 2 else len(c)
    batch = c.shape[0] if c.ndim == 2 else 1

    # If single sample, convert to batch of 1 for uniformity
    if c.ndim == 1:
        c = c[None, :]
        b = b[None, :]
        i = i[None, :]
        rho = rho[None, :]
        s = s[None, :]

    s_tail = s[:, M:]  # shape: (batch, T)
    w = 1 - c - b + r + rho * ki * s_tail  # shape: (batch, T)

    # Compute x iteratively
    x_list = [np.ones((batch, 1)) * x_init]
    for j in range(T):
        prev_x = x_list[-1]  # shape: (batch, 1)
        wj = w[:, j:j+1]     # shape: (batch, 1)
        Rj = R[j]            # scalar
        next_x = prev_x * wj + Rj
        x_list.append(next_x)
    x = np.concatenate(x_list, axis=1)  # shape: (batch, T+1)

    # Apply VCfn
    cx = c * x[:, :-1]
    consumUtil = VCfn(cx, delta) * PbarD[:-1]

    # Apply VXfn
    legacyVal = x[:, :-1] * (1 + b / alpha_t)
    legacyUtil = VXfn(legacyVal, epsilon, K) * pD

    finalUtil = VXfn(x[:, -1], epsilon, K) * PbarD[-1]
    totUtil = np.sum(legacyUtil + consumUtil, axis=1) + finalUtil  # shape: (batch,)

    return x,consumUtil,legacyUtil,totUtil


class CustomModel(Model):
    def __init__(self, T, M, N_nodes, ki, alpha, r, R, PbarD, pD, delta, epsilon,K,x_init):
        super(CustomModel, self).__init__()
        self.T = T
        self.M = M
        self.N_nodes = N_nodes
        self.ki = ki# Scaling for interest rate
        self.alpha = tf.constant(alpha, dtype=tf.float32)
        self.r = tf.constant(r, dtype=tf.float32)
        self.R = tf.constant(R, dtype=tf.float32)
        self.PbarD = tf.constant(PbarD, dtype=tf.float32)
        self.pD = tf.constant(pD, dtype=tf.float32)
        self.delta = delta
        self.epsilon = epsilon
        self.K = K
        self.x_init = x_init

        # Build custom subnet for each j
        self.hidden_nets = []
        for j in range(T):
            net = [
                Dense(N_nodes[j], activation='relu'),
                Dense(4, activation='softmax')
            ]
            self.hidden_nets.append(net)          

    def call(self, s):
        batch_size = tf.shape(s)[0]
        c_list, b_list, i_list, rho_list = [], [], [], []

        for j in range(self.T):
            slice_input = s[:, :self.M + j]
            slice_input = s[:, :self.M + j+1]# Current knowledge
            h = self.hidden_nets[j][0](slice_input)   # ReLU layer
            out = self.hidden_nets[j][1](h)           # Softmax layer
            c_list.append(out[:, 0])
            b_list.append(out[:, 1])
            i_list.append(out[:, 2])
            rho_list.append(out[:, 3])


        # Stack into shape (batch_size, T)
        c = tf.stack(c_list, axis=1)
        b = tf.stack(b_list, axis=1)
        i = tf.stack(i_list, axis=1)
        rho = tf.stack(rho_list, axis=1)

        s_tail = s[:, self.M:]  # shape (batch_size, T)
        w = 1 - c - b + self.r + rho * (self.ki * s_tail)  # shape (batch_size, T)
        # test = self.r
        # tf.print("total", tf.reduce_max(test))

        # Compute x iteratively
        x0 = tf.ones((batch_size, 1))*x_init
        x_list = [x0]
        for j in range(self.T):
            prev_x = x_list[-1]
            wj = tf.expand_dims(w[:, j], axis=1)
            Rj = self.R[j]
            next_x = prev_x * wj + Rj
            x_list.append(next_x)
        x = tf.concat(x_list, axis=1)  # shape (batch_size, T+1)

        # consumUtil = VCfn(c * x[:-1]) * PbarD[:-1]
        cx = c * x[:, :-1]
        consumUtil = VCfn(cx,delta) * self.PbarD[:-1]

        # legacyVal = x[:-1] * (1 + b / (alpha * pd + 1e-6))
        legacyVal = x[:, :-1] * (1 + b / self.alpha)
        legacyUtil = VXfn(legacyVal,epsilon,K) * self.pD

        # Final value
        finalUtil = VXfn(x[:, -1],epsilon,K) * self.PbarD[-1]
        totUtil = tf.reduce_sum(legacyUtil + consumUtil, axis=1) + finalUtil

        return -tf.expand_dims(totUtil, axis=1)  # shape (batch_size, 1)
    
    
    def get_cbirho(self, s):
        c_list, b_list, i_list, rho_list = [], [], [], []

        for j in range(self.T):
            slice_input = s[:, :self.M + j + 1]
            h = self.hidden_nets[j][0](slice_input)
            out = self.hidden_nets[j][1](h)
            c_list.append(out[:, 0])
            b_list.append(out[:, 1])
            i_list.append(out[:, 2])
            rho_list.append(out[:, 3])

        # Stack into shape (batch_size, T)
        c = tf.stack(c_list, axis=1)
        b = tf.stack(b_list, axis=1)
        i = tf.stack(i_list, axis=1)
        rho = tf.stack(rho_list, axis=1)

        return c, b, i, rho


class MeanLossLogger(tf.keras.callbacks.Callback):
    def __init__(self, s_train):
        super().__init__()
        self.s_train = s_train

    def on_epoch_end(self, epoch, logs=None):
        # Predict on full training set
        neg_util = self.model.predict(self.s_train, verbose=0)
        mean_util = -tf.reduce_mean(neg_util).numpy()
        print(f"Epoch {epoch + 1}: Mean utility = {mean_util:.4f}")


# 1. Build full model
input_len = T + M
inputs = Input(shape=(input_len,))
model = CustomModel(T, M, N_nodes, ki, alpha_t, r_t, R_t, PbarD_t, pD_t, delta, epsilon, K, x_init)
outputs = model(inputs)
full_model = Model(inputs, outputs)

# 2. Compile
full_model.compile(optimizer='adam', loss=lambda y_true, y_pred: y_pred)

# 3. Prepare training data
# s_train must have shape (num_samples, T + M)
# y_train is dummy zeros, because model returns the loss directly
y_train = np.zeros((x_train.shape[0], 1), dtype=np.float32)

# 4. Fit the model
# Instantiate the callback
mean_loss_logger = MeanLossLogger(x_train)

# Fit the model with the callback
full_model.fit(x_train, y_train, 
               epochs=nn_epoch, batch_size=32, 
               callbacks=[mean_loss_logger])


# 5. Get utility predictions (positive values)
neg_util_values = full_model.predict(x_train)   # shape (num_samples, 1)
util_values = -neg_util_values                  # shape (num_samples, 1)



sample_input = x_train[0:1]  # Shape (1, T + M)

# Get c, b, i, rho from the model
c_vals, b_vals, i_vals, rho_vals = model.get_cbirho(tf.constant(sample_input, dtype=tf.float32))

# Convert to numpy arrays for inspection
c = c_vals.numpy().flatten()
b = b_vals.numpy().flatten()
i = i_vals.numpy().flatten()
rho = rho_vals.numpy().flatten()


s = x_train[0,:]
[x,consumUtil,legacyUtil,totUtil] = eval_totUtil_numpy(c,b,i,rho,s,M,ki,r_t,R_t,PbarD_t,pd_t,pD_t,delta,epsilon,K,x_init)
