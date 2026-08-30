from __future__ import annotations

import argparse
import copy
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import tensorflow as tf


# Reproducibility

def set_seed(seed: int = 0) -> None:
    """Seeds every source of randomness to ensure reproducible weight initialization."""
    random.seed(seed)                # Seed Python's built-in random module
    np.random.seed(seed)             # Seed NumPy's global random state
    tf.random.set_seed(seed)         # Seed TensorFlow's graph-level and op-level randomness


# Configuration

@dataclass
class CaseConfig:
    """Stores all hyperparameters mapping to the paper's experimental setup."""
    learning_rate: float
    lr_decay_every: int
    epochs: int
    batch_size: int
    derivative_order: int
    segment_window: int
    eta: float = 100.0               # Weight for physics PDE residual constraint
    mc_samples: int = 10             # Monte Carlo samples drawn during training
    n_segments: int = 4              # Number of sequence segments per sample window
    hgrr_levels: int = 2             # Hierarchy levels in the complex recurrent network
    encoder_projection_dim: int = 32 # Dimension to project raw sensor segments into
    lstm_hidden_dim: int = 32        # Hidden state size for the Bidirectional LSTM
    hgrr_hidden_dim: int = 32        # Hidden state size for the Bayesian HGRR
    physics_hidden_dim: int = 64     # Hidden size for the non-linear physics MLP
    prior_sigma: float = 1.0         # Standard deviation for Bayesian prior N(0, 1)
    max_beta: float = 20.0           # Clipping safeguard for Weibull shape parameter
    max_power: float = 20.0          # Clipping safeguard to prevent overflow in exponentiation
    grad_clip: float = 1.0           # Global norm clipping threshold for stable gradients

# Hyperparameters defined in Table 2 for the two experimental case studies
CASE_I = CaseConfig(
    learning_rate=5e-4, lr_decay_every=40, epochs=150, 
    batch_size=64, derivative_order=2, segment_window=30
)
CASE_II = CaseConfig(
    learning_rate=1e-3, lr_decay_every=60, epochs=100, 
    batch_size=128, derivative_order=2, segment_window=50
)


# BayesianDense: Bayes-by-Backprop Gaussian variational layer

class BayesianDense(tf.Module):
    """
    Gaussian variational fully-connected layer. 
    Weights are probability distributions sampled via the reparameterization trick.
    """
    def __init__(self, in_features: int, out_features: int, prior_sigma: float = 1.0, rho_init: float = -5.0, name: Optional[str] = None):
        super().__init__(name=name)
        self.prior_sigma = float(prior_sigma) # Standard deviation of the prior

        # Initialize means (mu) using a uniform distribution matching standard Kaiming bounds
        bound = 1.0 / math.sqrt(in_features) if in_features > 0 else 0.0
        self.weight_mu = tf.Variable(tf.random.uniform([out_features, in_features], -bound, bound), name="weight_mu")
        self.bias_mu = tf.Variable(tf.random.uniform([out_features], -bound, bound), name="bias_mu")

        # Initialize standard deviation parameters (rho). rho_init = -5 forces initial std to be tiny.
        self.weight_rho = tf.Variable(tf.fill([out_features, in_features], float(rho_init)), name="weight_rho")
        self.bias_rho = tf.Variable(tf.fill([out_features], float(rho_init)), name="bias_rho")

    def sigma_from_rho(self, rho: tf.Tensor) -> tf.Tensor:
        # Maps unconstrained rho to a strictly positive standard deviation using softplus
        return tf.nn.softplus(rho) + 1e-8

    def __call__(self, x: tf.Tensor, sample: bool = True) -> tf.Tensor:
        if sample:
            # Reparameterization Trick: Draw standard normal noise (eps) to sample weights
            weight_sigma = self.sigma_from_rho(self.weight_rho)
            bias_sigma = self.sigma_from_rho(self.bias_rho)
            eps_w = tf.random.normal(tf.shape(self.weight_mu))
            eps_b = tf.random.normal(tf.shape(self.bias_mu))
            weight = self.weight_mu + weight_sigma * eps_w
            bias = self.bias_mu + bias_sigma * eps_b
        else:
            # Deterministic pass using only the means (used during evaluation if requested)
            weight, bias = self.weight_mu, self.bias_mu

        # Apply the linear transformation: y = x * W^T + b
        return tf.matmul(x, weight, transpose_b=True) + bias

    def _gaussian_kl(self, mu: tf.Tensor, sigma: tf.Tensor) -> tf.Tensor:
        # Computes the analytical KL divergence between the posterior N(mu, sigma) and prior N(0, prior_sigma)
        var_ratio = tf.square(sigma / self.prior_sigma)
        kl = 0.5 * (var_ratio + tf.square(mu / self.prior_sigma) - 1.0 - tf.math.log(var_ratio))
        return tf.reduce_sum(kl)

    def kl_divergence(self) -> tf.Tensor:
        # Returns the total KL divergence for this layer's weights and biases
        weight_sigma = self.sigma_from_rho(self.weight_rho)
        bias_sigma = self.sigma_from_rho(self.bias_rho)
        return self._gaussian_kl(self.weight_mu, weight_sigma) + self._gaussian_kl(self.bias_mu, bias_sigma)


def collect_kl(module: tf.Module) -> tf.Tensor:
    """Recursively aggregates the KL divergence from all BayesianDense submodules."""
    total = tf.zeros(())
    for child in module.submodules:
        if isinstance(child, BayesianDense):
            total = total + child.kl_divergence()
    return total


# SegmentedBiLSTM

class SegmentedBiLSTM(tf.Module):
    """Processes sequential data by chunking it into segments and encoding bidirectionally."""
    def __init__(self, n_features: int, segment_window: int, projection_dim: int, lstm_hidden: int, name: Optional[str] = None):
        super().__init__(name=name)
        self.segment_window = segment_window
        # Initial Dense projection to embed raw features (Eq. 9-13)
        self.segment_dense = tf.keras.layers.Dense(projection_dim, activation="relu")
        # Bidirectional LSTM to capture temporal context flowing forwards and backwards
        self.bilstm = tf.keras.layers.Bidirectional(
            tf.keras.layers.LSTM(lstm_hidden, return_sequences=True, return_state=True),
            merge_mode="concat",
        )
        self.out_dim = 2 * lstm_hidden

    def segment(self, x: tf.Tensor) -> tf.Tensor:
        # Chops the full sequence (batch, S, features) into discrete segments (batch, d, window*features)
        x = tf.convert_to_tensor(x, dtype=tf.float32)
        seq_len, n_features = x.shape[1], x.shape[2]
        d = seq_len // self.segment_window
        usable = d * self.segment_window
        x = x[:, :usable, :]
        batch = tf.shape(x)[0]
        return tf.reshape(x, [batch, d, self.segment_window * n_features])

    def __call__(self, x: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        segments = self.segment(x)                               # Split sequence
        projected = self.segment_dense(segments)                 # Project into embedding space
        seq_out, fwd_h, fwd_c, bwd_h, bwd_c = self.bilstm(projected) # Process through Bi-LSTM
        h_bl = tf.concat([fwd_h, bwd_h], axis=-1)                # Concatenate final forward and backward states (Eq. 14)
        return h_bl, seq_out


# HGRULevel / BayesianHGRR

class HGRULevel(tf.Module):
    """A single hierarchical complex-valued recurrent cell with phase-rotation dynamics."""
    def __init__(self, in_dim: int, hidden_dim: int, kappa_init: float, prior_sigma: float, name: Optional[str] = None):
        super().__init__(name=name)
        self.hidden_dim = hidden_dim
        # Bayesian gates for the complex recurrent cell (Eq. 15-21)
        self.w_g = BayesianDense(in_dim, hidden_dim, prior_sigma)
        self.w_c = BayesianDense(in_dim, hidden_dim, prior_sigma)
        self.w_eps = BayesianDense(in_dim, hidden_dim, prior_sigma)
        self.w_theta = BayesianDense(in_dim, hidden_dim, prior_sigma)
        # Learnable lower bound (kappa) restricting how aggressively this specific layer forgets
        self.kappa_raw = tf.Variable(float(kappa_init), name="kappa_raw")
        self.layer_norm = tf.keras.layers.LayerNormalization()
        self.w_o = BayesianDense(2 * hidden_dim, hidden_dim, prior_sigma)

    def step(self, v_t: tf.Tensor, h_prev: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        g_t = tf.sigmoid(self.w_g(v_t))                               # Output gate magnitude (Eq. 15)
        c_t = tf.nn.silu(self.w_c(v_t))                               # Candidate vector (Eq. 16)
        kappa = tf.sigmoid(self.kappa_raw)                            # Squash kappa to (0, 1)
        eps_t = kappa + (1.0 - kappa) * tf.sigmoid(self.w_eps(v_t))   # Forget gate magnitude bounded by kappa (Eq. 20)
        theta_t = self.w_theta(v_t)                                   # Phase shift angle (Eq. 21)

        # Convert phase and magnitude into complex domain via Euler's formula
        rotation = tf.complex(tf.cos(theta_t), tf.sin(theta_t))
        eps_complex = tf.complex(eps_t, tf.zeros_like(eps_t))
        c_complex = tf.complex(c_t, tf.zeros_like(c_t))

        # Rotate previous memory and blend with new candidate data (Eq. 22)
        h_t = eps_complex * rotation * h_prev + (1.0 - eps_complex) * c_complex

        # Split complex state back into real and imaginary parts, apply layer norm (Eq. 23)
        h_real_imag = tf.concat([tf.math.real(h_t), tf.math.imag(h_t)], axis=-1)
        g_real_imag = tf.concat([g_t, g_t], axis=-1)
        o_tilde = self.layer_norm(g_real_imag * h_real_imag)

        o_t = self.w_o(o_tilde)                                       # Final output projection (Eq. 24)
        return o_t, h_t

    def __call__(self, x_seq: tf.Tensor) -> tf.Tensor:
        # Unrolls the recurrent step function over the temporal sequence
        batch = tf.shape(x_seq)[0]
        time_steps = x_seq.shape[1]
        h = tf.zeros([batch, self.hidden_dim], dtype=tf.complex64)
        outputs = []
        for t_idx in range(time_steps):
            out, h = self.step(x_seq[:, t_idx, :], h)
            outputs.append(out)
        return tf.stack(outputs, axis=1)


class BayesianHGRR(tf.Module):
    """Stacks multiple complex HGRU layers to map time-series to Weibull distribution parameters."""
    def __init__(self, in_dim: int, hidden_dim: int, num_levels: int, prior_sigma: float, name: Optional[str] = None):
        super().__init__(name=name)
        self.hidden_dim = hidden_dim
        # Spread kappa initialization so lower levels learn short-term traits and higher levels learn long-term traits
        if num_levels == 1:
            kappa_inits = [0.0]
        else:
            kappa_inits = np.linspace(-3.0, 3.0, num_levels).tolist()

        self.levels = []
        for level_idx in range(num_levels):
            level_in_dim = in_dim if level_idx == 0 else hidden_dim
            self.levels.append(
                HGRULevel(in_dim=level_in_dim, hidden_dim=hidden_dim, kappa_init=kappa_inits[level_idx], prior_sigma=prior_sigma)
            )

        # Regresses the final temporal representation into Weibull Scale (alpha) and Shape (beta) parameters (Eq. 26)
        self.regressor = BayesianDense(hidden_dim, 2, prior_sigma)

    def __call__(self, v_seq: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
        x = v_seq
        for level in self.levels:
            x = level(x)
        top_output = x[:, -1, :] # Extract the final state of the highest level

        raw_params = self.regressor(top_output)
        params = tf.nn.softplus(raw_params) # Ensures strictly positive distribution parameters

        # Add tiny floor value to prevent NaN math downstream (log(0) avoidance)
        alpha = params[:, 0] + 1e-4
        beta = params[:, 1] + 1e-4
        return top_output, alpha, beta

    def kl_divergence(self) -> tf.Tensor:
        return collect_kl(self)


# Weibull distribution Math

def weibull_log_pdf(y: tf.Tensor, alpha: tf.Tensor, beta: tf.Tensor, max_beta: float = 20.0, max_power: float = 20.0) -> tf.Tensor:
    """Computes the numerically stable Log Probability Density Function of the Weibull distribution (Eq. 27)."""
    eps = 1e-6
    y = tf.clip_by_value(y, eps, tf.float32.max)
    alpha = tf.clip_by_value(alpha, eps, tf.float32.max)
    beta = tf.clip_by_value(beta, eps, max_beta)

    log_y = tf.math.log(y)
    log_alpha = tf.math.log(alpha)
    log_beta = tf.math.log(beta)
    log_ratio = log_y - log_alpha

    # Exponent is clamped to prevent float32 overflow
    power = tf.clip_by_value(beta * log_ratio, -max_power, max_power)
    return log_beta - log_alpha + (beta - 1.0) * log_ratio - tf.exp(power)

def weibull_nll(y: tf.Tensor, alpha: tf.Tensor, beta: tf.Tensor, max_beta: float = 20.0, max_power: float = 20.0) -> tf.Tensor:
    """Mean Negative Log-Likelihood loss for the current mini-batch (Eq. 35)."""
    return -tf.reduce_mean(weibull_log_pdf(y, alpha, beta, max_beta=max_beta, max_power=max_power))

def weibull_mean(alpha: tf.Tensor, beta: tf.Tensor) -> tf.Tensor:
    """Computes the statistical expected value (mean RUL) of the Weibull distribution (Eq. 28)."""
    beta = tf.clip_by_value(beta, 1e-6, 20.0)
    # Computed in log space using lgamma to prevent Gamma function overflow
    return alpha * tf.exp(tf.math.lgamma(1.0 + 1.0 / beta))

def weibull_variance(alpha: tf.Tensor, beta: tf.Tensor) -> tf.Tensor:
    """Computes the statistical variance of the Weibull distribution (Eq. 43-44)."""
    beta = tf.clip_by_value(beta, 1e-6, 20.0)
    gamma1 = tf.exp(tf.math.lgamma(1.0 + 1.0 / beta))
    gamma2 = tf.exp(tf.math.lgamma(1.0 + 2.0 / beta))
    value = tf.square(alpha) * (gamma2 - tf.square(gamma1))
    return tf.maximum(value, 0.0)

def weibull_pdf(y: tf.Tensor, alpha: tf.Tensor, beta: tf.Tensor, max_beta: float = 20.0, max_power: float = 20.0) -> tf.Tensor:
    """Exponentiates the log PDF, used strictly for computing epistemic uncertainty (Eq. 42)."""
    log_pdf = weibull_log_pdf(y, alpha, beta, max_beta=max_beta, max_power=max_power)
    return tf.exp(tf.clip_by_value(log_pdf, -50.0, 20.0))


# Deep Hidden Physics Model (Deep-HPM)

class NonlinearDynamicNetwork(tf.Module):
    """The MLP representing the arbitrary nonlinear function G() governing degradation physics."""
    def __init__(self, h_bl_dim: int, hidden_dim: int = 64, name: Optional[str] = None) -> None:
        super().__init__(name=name)
        input_dim = 1 + h_bl_dim + 1 + h_bl_dim + h_bl_dim
        # Maps time, state, current RUL, and derivatives to the expected RUL change rate
        self.dense1 = tf.keras.layers.Dense(hidden_dim, activation="relu", input_shape=(input_dim,))
        self.dense2 = tf.keras.layers.Dense(1)

    def __call__(self, t: tf.Tensor, h_bl: tf.Tensor, rul: tf.Tensor, drul_dh: tf.Tensor, d2rul_dh2: tf.Tensor) -> tf.Tensor:
        x = tf.concat([t, h_bl, rul, drul_dh, d2rul_dh2], axis=-1)
        return tf.squeeze(self.dense2(self.dense1(x)), axis=-1)


class DeepHPM(tf.Module):
    """Extracts physical constraints by enforcing partial differential equation (PDE) compliance."""
    def __init__(self, h_bl_dim: int, hidden_dim: int = 64, name: Optional[str] = None) -> None:
        super().__init__(name=name)
        self.h_bl_dim = h_bl_dim
        self.dynamic_net = NonlinearDynamicNetwork(h_bl_dim, hidden_dim)

    def residual(self, solution_fn, h_bl: tf.Tensor, t: tf.Tensor, derivative_order: int = 2) -> tf.Tensor:
        # Generates copies to safely observe inside the gradient tape without mutating graph inputs
        h = tf.identity(h_bl)
        time = tf.identity(t)

        # Outer persistent tape records operations to calculate 2nd order derivatives
        with tf.GradientTape(persistent=True) as outer_tape:
            outer_tape.watch(h)
            outer_tape.watch(time)

            # Inner persistent tape calculates 1st order derivatives (gradient mapping)
            with tf.GradientTape(persistent=True) as inner_tape:
                inner_tape.watch(h)
                inner_tape.watch(time)
                alpha, beta = solution_fn(h, time)
                rul = weibull_mean(alpha, beta)[:, tf.newaxis] # Target variable (Eq. 28)

            drul_dt = inner_tape.gradient(rul, time)           # Rate of change over time (Eq. 29)
            drul_dh = inner_tape.gradient(rul, h)              # Rate of change across feature state

            if derivative_order >= 2:
                # Calculates diagonal second derivatives dynamically
                second_components = []
                for j in range(self.h_bl_dim): # Loop unrolled safely because h_bl_dim is a fixed integer
                    first_j = drul_dh[:, j:j + 1]
                    second_j = outer_tape.gradient(first_j, h)[:, j:j + 1]
                    second_components.append(second_j)
                d2rul_dh2 = tf.concat(second_components, axis=-1)
            else:
                d2rul_dh2 = tf.zeros_like(drul_dh)

        del inner_tape, outer_tape # Free memory footprint of persistent tapes immediately

        # Feed the PDE inputs into G() to predict what the temporal rate of change SHOULD be
        g_pred = self.dynamic_net(time, h, rul, drul_dh, d2rul_dh2)
        # Returns the error between the mathematical time-derivative and the physics-network prediction (Eq. 30)
        return tf.squeeze(drul_dt, axis=-1) - g_pred

    def residual_loss(self, solution_fn, h_bl: tf.Tensor, t: tf.Tensor, derivative_order: int = 2) -> tf.Tensor:
        # Squares the residual so optimization bounds it at zero (avoids pushing error arbitrarily negative)
        residual = self.residual(solution_fn, h_bl, t, derivative_order)
        return tf.reduce_mean(tf.square(residual))


# PCBNN Wrapper Class

class PCBNN(tf.Module):
    """Primary model encapsulating the Encoder, Bayesian HGRR, and Physics Model."""
    def __init__(self, n_features: int, case: CaseConfig, name: Optional[str] = None) -> None:
        super().__init__(name=name)
        self.case = case
        self.n_features = n_features

        self.encoder = SegmentedBiLSTM(
            n_features=n_features, segment_window=case.segment_window,
            projection_dim=case.encoder_projection_dim, lstm_hidden=case.lstm_hidden_dim,
        )
        self.h_bl_dim = self.encoder.out_dim

        self.hgrr = BayesianHGRR(
            in_dim=self.h_bl_dim + 1, hidden_dim=case.hgrr_hidden_dim,
            num_levels=case.hgrr_levels, prior_sigma=case.prior_sigma,
        )

        self.deep_hpm = DeepHPM(h_bl_dim=self.h_bl_dim, hidden_dim=case.physics_hidden_dim)

    def encode(self, x: tf.Tensor) -> tf.Tensor:
        h_bl, _ = self.encoder(x)
        return h_bl

    def solution_fn(self, h_bl: tf.Tensor, t: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        # Formats input and feeds it into Bayesian layers for PDE extraction
        if len(t.shape) == 1:
            t = t[:, tf.newaxis]
        v = tf.concat([h_bl, t], axis=-1)[:, tf.newaxis, :]
        _, alpha, beta = self.hgrr(v)
        beta = tf.clip_by_value(beta, 1e-4, self.case.max_beta)
        return alpha, beta

    def __call__(self, x: tf.Tensor, t: tf.Tensor, compute_physics: bool = True) -> Dict[str, tf.Tensor]:
        # Full forward pass entry point
        h_bl = self.encode(x)
        alpha, beta = self.solution_fn(h_bl, t)

        if compute_physics:
            residual_loss = self.deep_hpm.residual_loss(
                self.solution_fn, h_bl, t, derivative_order=self.case.derivative_order
            )
        else:
            residual_loss = tf.zeros(())

        return {"h_bl": h_bl, "alpha": alpha, "beta": beta, "residual_loss": residual_loss}

    def kl_divergence(self) -> tf.Tensor:
        return self.hgrr.kl_divergence()


# Data Loading & Preprocessing

REMOVED_CMAPPSS_SENSORS = {1, 5, 10, 16, 18, 19} # Sensors identified as redundant in paper

def read_cmapss_file(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path, sep=r"\s+", header=None)

def load_cmapss_training_trajectories(data_root: str | Path, dataset: str) -> List[Tuple[int, np.ndarray, np.ndarray]]:
    """Loads text files and drops uninformative sensors."""
    root = Path(data_root)
    train = read_cmapss_file(root / f"train_{dataset}.txt")

    unit = train.iloc[:, 0].to_numpy(dtype=np.int64)
    cycle = train.iloc[:, 1].to_numpy(dtype=np.float32)
    sensors_all = train.iloc[:, 2:23].to_numpy(dtype=np.float32)

    keep_indices = [i for i in range(21) if (i + 1) not in REMOVED_CMAPPSS_SENSORS]
    sensors = sensors_all[:, keep_indices]

    trajectories = []
    max_cycle_by_unit = {int(uid): float(cycle[unit == uid].max()) for uid in np.unique(unit)}

    for uid in np.unique(unit):
        mask = unit == uid
        order = np.argsort(cycle[mask])
        x = sensors[mask][order]
        cyc = cycle[mask][order]
        raw_rul = max_cycle_by_unit[int(uid)] - cyc
        rul = np.minimum(raw_rul, 125.0).astype(np.float32) # Cap RUL at 125 (Eq. 49)
        trajectories.append((int(uid), x.astype(np.float32), rul))

    return trajectories

def minmax_fit(trajectories: Sequence[Tuple[int, np.ndarray, np.ndarray]]) -> Tuple[np.ndarray, np.ndarray]:
    values = np.concatenate([x for _, x, _ in trajectories], axis=0)
    return values.min(axis=0, keepdims=True), values.max(axis=0, keepdims=True)

def minmax_apply(x: np.ndarray, x_min: np.ndarray, x_max: np.ndarray) -> np.ndarray:
    denom = np.clip(x_max - x_min, 1e-8, None)
    return (x - x_min) / denom

class CMapssWindowDataset:
    """Slides a temporal window across engine histories for training sample generation."""
    def __init__(self, trajectories: Sequence[Tuple[int, np.ndarray, np.ndarray]], segment_window: int, n_segments: int, stride: int, x_min: np.ndarray, x_max: np.ndarray):
        self.seq_len = segment_window * n_segments
        self.samples = []

        for _, sensors, rul in trajectories:
            normalized = minmax_apply(sensors, x_min, x_max).astype(np.float32)
            L = len(rul)
            if L < self.seq_len:
                continue

            for start in range(0, L - self.seq_len + 1, stride):
                end = start + self.seq_len
                x_win = normalized[start:end]
                current_time = np.array([float(end - 1) / max(L - 1, 1)], dtype=np.float32)
                y = float(max(rul[end - 1], 1e-6))
                self.samples.append((x_win, current_time, y))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> Tuple[np.ndarray, np.ndarray, float]:
        return self.samples[index]

def split_trajectories(trajectories: Sequence[Tuple[int, np.ndarray, np.ndarray]], val_fraction: float = 0.1, seed: int = 0) -> Tuple[List, List]:
    """Isolates validation data by entire engine unit to prevent data leakage."""
    rng = np.random.default_rng(seed)
    idx = np.arange(len(trajectories))
    rng.shuffle(idx)
    n_val = max(1, int(round(len(idx) * val_fraction)))
    val_idx = set(idx[:n_val].tolist())
    train = [item for i, item in enumerate(trajectories) if i not in val_idx]
    val = [item for i, item in enumerate(trajectories) if i in val_idx]
    return train, val

class ArrayBatcher:
    """Lightweight mini-batch generator yielding TensorFlow tensors."""
    def __init__(self, dataset, batch_size: int, shuffle: bool) -> None:
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle

    def __iter__(self):
        n = len(self.dataset)
        order = np.arange(n)
        if self.shuffle:
            np.random.shuffle(order)

        for start in range(0, n, self.batch_size):
            batch_idx = order[start:start + self.batch_size]
            xs, ts, ys = zip(*[self.dataset[int(i)] for i in batch_idx])
            yield (
                tf.convert_to_tensor(np.stack(xs), dtype=tf.float32),
                tf.convert_to_tensor(np.stack(ts), dtype=tf.float32),
                tf.convert_to_tensor(np.array(ys, dtype=np.float32)),
            )

    def __len__(self) -> int:
        return (len(self.dataset) + self.batch_size - 1) // self.batch_size


# Total Loss Function

def total_loss(y: tf.Tensor, alpha: tf.Tensor, beta: tf.Tensor, kl: tf.Tensor, residual_loss: tf.Tensor, dataset_size, batch_size, eta: float, max_beta: float, max_power: float) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
    """Combines Variational ELBO scaling and Physics constraints into the final objective."""
    nll_batch = weibull_nll(y, alpha, beta, max_beta=max_beta, max_power=max_power)
    
    # Use TensorFlow graph math instead of Python max() and float()
    dataset_size_tf = tf.cast(dataset_size, tf.float32)
    batch_size_tf = tf.cast(tf.maximum(batch_size, 1), tf.float32)
    scale = dataset_size_tf / batch_size_tf
    
    expected_full_nll = nll_batch * scale # Scale up to represent full dataset size (Eq. 35)
    elbo = kl + expected_full_nll
    total = elbo + eta * residual_loss    # Final Loss combining Bayes and Physics (Eq. 37)
    return total, elbo, expected_full_nll


# Optimized Training & Validation Functions

def fit(model: PCBNN, train_dataset: CMapssWindowDataset, val_dataset: CMapssWindowDataset, case: CaseConfig, patience: Optional[int] = None) -> PCBNN:
    train_loader = ArrayBatcher(train_dataset, batch_size=case.batch_size, shuffle=True)
    val_loader = ArrayBatcher(val_dataset, batch_size=case.batch_size, shuffle=False)
    optimizer = tf.keras.optimizers.Adam(learning_rate=case.learning_rate)

    dataset_size_tf = tf.constant(len(train_dataset), dtype=tf.float32)
    
    # COMPILED TRAINING STEP: Runs entirely on GPU, bypassing Python execution

    @tf.function(reduce_retracing=True)
    def train_step(x, t, y):
        accumulated_grads = [tf.zeros_like(v) for v in model.trainable_variables]
        batch_total, batch_elbo, batch_nll, batch_physics = 0.0, 0.0, 0.0, 0.0
        
        # Python loop unrolls inside the graph compiler to evaluate 10 Monte Carlo iterations concurrently
        for _ in range(case.mc_samples):
            with tf.GradientTape() as tape:
                out = model(x, t, compute_physics=True)
                alpha, beta, physics = out["alpha"], out["beta"], out["residual_loss"]
                kl = model.kl_divergence()
                
                loss, elbo, nll = total_loss(
                    y=y, alpha=alpha, beta=beta, kl=kl, residual_loss=physics,
                    dataset_size=len(train_dataset), batch_size=tf.size(y), eta=case.eta,
                    max_beta=case.max_beta, max_power=case.max_power,
                )
                scaled_loss = loss / float(case.mc_samples)
            
            # Extract gradients and accumulate them iteratively across all MC samples
            grads = tape.gradient(scaled_loss, model.trainable_variables)
            accumulated_grads = [acc if g is None else acc + g for acc, g in zip(accumulated_grads, grads)]
            
            batch_total += loss
            batch_elbo += elbo
            batch_nll += nll
            batch_physics += physics
        
        # Clip gradient explosion and apply updates
        clipped_grads, _ = tf.clip_by_global_norm(accumulated_grads, case.grad_clip)
        optimizer.apply_gradients(zip(clipped_grads, model.trainable_variables))
        
        # Calculate current batch prediction RMSE
        pred_mean = weibull_mean(alpha, beta)
        rmse_sq = tf.reduce_sum(tf.square(pred_mean - y))
        
        return batch_total / float(case.mc_samples), batch_elbo / float(case.mc_samples), batch_nll / float(case.mc_samples), batch_physics / float(case.mc_samples), rmse_sq

    # COMPILED VALIDATION STEP

    @tf.function(reduce_retracing=True)
    def val_step(x, t, y):
        # Forward pass without calculating physics derivatives
        out = model(x, t, compute_physics=False)
        alpha, beta, physics = out["alpha"], out["beta"], out["residual_loss"]
        kl = model.kl_divergence()
        
        loss, elbo, nll = total_loss(
            y=y, alpha=alpha, beta=beta, kl=kl, residual_loss=physics,
            dataset_size=len(val_dataset), batch_size=tf.size(y), eta=case.eta,
            max_beta=case.max_beta, max_power=case.max_power,
        )
        pred_mean = weibull_mean(alpha, beta)
        rmse_sq = tf.reduce_sum(tf.square(pred_mean - y))
        return loss, elbo, nll, physics, rmse_sq

    best_val_loss = float("inf")
    best_weights = None
    epochs_without_improvement = 0

    for epoch in range(1, case.epochs + 1):
        if epoch > 1 and (epoch - 1) % case.lr_decay_every == 0:
            optimizer.learning_rate.assign(optimizer.learning_rate * 0.5)

        train_loss, train_rmse_sq, train_samples = 0.0, 0.0, 0
        for x, t, y in train_loader:
            l, _, _, _, r_sq = train_step(x, t, y)
            train_loss += float(l)
            train_rmse_sq += float(r_sq)
            train_samples += int(tf.size(y))
        
        train_loss /= len(train_loader)
        train_rmse = math.sqrt(train_rmse_sq / train_samples)

        val_loss, val_rmse_sq, val_samples = 0.0, 0.0, 0
        for x, t, y in val_loader:
            l, _, _, _, r_sq = val_step(x, t, y)
            val_loss += float(l)
            val_rmse_sq += float(r_sq)
            val_samples += int(tf.size(y))
        
        val_loss /= len(val_loader)
        val_rmse = math.sqrt(val_rmse_sq / val_samples)

        print(f"Epoch {epoch:03d} | train loss {train_loss:.4f} | train RMSE {train_rmse:.4f} | val loss {val_loss:.4f} | val RMSE {val_rmse:.4f} | lr {float(optimizer.learning_rate):.3e}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_weights = [v.numpy().copy() for v in model.trainable_variables]
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if patience is not None and epochs_without_improvement >= patience:
            print(f"Early stopping at epoch {epoch}.")
            break

    if best_weights is not None:
        for var, value in zip(model.trainable_variables, best_weights):
            var.assign(value)

    return model


# Test Prediction & Uncertainty Bounds

def predict_with_uncertainty(model: PCBNN, x: tf.Tensor, t: tf.Tensor, n_samples: int = 50) -> Dict[str, tf.Tensor]:
    """Generates stochastic samples for a given engine to estimate final predictive bounds."""
    alpha_samples, beta_samples, mean_samples = [], [], []

    for _ in range(n_samples):
        h_bl = model.encode(x)
        alpha, beta = model.solution_fn(h_bl, t)
        alpha_samples.append(alpha)
        beta_samples.append(beta)
        mean_samples.append(weibull_mean(alpha, beta))

    alpha = tf.stack(alpha_samples, axis=0)
    beta = tf.stack(beta_samples, axis=0)
    y_m = tf.stack(mean_samples, axis=0)

    y_pred = tf.reduce_mean(y_m, axis=0) # Expectation of RUL predictions (Eq. 40)
    
    var_aleatoric = tf.reduce_mean(weibull_variance(alpha, beta), axis=0) # Intrinsic system variance (Eq. 44)
    
    likelihood_at_prediction = weibull_pdf(y_m, alpha, beta)
    var_epistemic = tf.reduce_mean(tf.square(likelihood_at_prediction), axis=0) - tf.square(tf.reduce_mean(likelihood_at_prediction, axis=0)) # Model ignorance variance (Eq. 42)

    var_total = tf.maximum(var_aleatoric + var_epistemic, 0.0)
    std_total = tf.sqrt(var_total)

    # 95% Confidence Interval using standard Z-score (Algorithm 1, Line 14)
    ci_lower = y_pred - 1.96 * std_total
    ci_upper = y_pred + 1.96 * std_total

    return {
        "y_pred": y_pred, "var_aleatoric": var_aleatoric, "var_epistemic": var_epistemic,
        "var_total": var_total, "std_total": std_total, "ci_lower": ci_lower, "ci_upper": ci_upper,
    }


# Custom Evaluation Metrics

def rmse_metric(y_pred: tf.Tensor, y_true: tf.Tensor) -> float:
    return float(tf.sqrt(tf.reduce_mean(tf.square(y_pred - y_true))))

def score_metric(y_pred: tf.Tensor, y_true: tf.Tensor) -> float:
    """Asymmetric penalty heavily penalizing late predictions (overestimating health)."""
    d = y_pred - y_true
    early_mask = d < 0
    late_mask = tf.logical_not(early_mask)

    d_early = tf.boolean_mask(d, early_mask)
    d_late = tf.boolean_mask(d, late_mask)

    score_early = tf.exp(tf.clip_by_value(-d_early / 13.0, -50.0, 50.0)) - 1.0
    score_late = tf.exp(tf.clip_by_value(d_late / 10.0, -50.0, 50.0)) - 1.0
    return float(tf.reduce_sum(score_early) + tf.reduce_sum(score_late))

def picp_metric(y_true: tf.Tensor, lower: tf.Tensor, upper: tf.Tensor) -> float:
    """Percentage of actual failure times caught inside the 95% CI."""
    inside = tf.logical_and(y_true >= lower, y_true <= upper)
    return float(tf.reduce_mean(tf.cast(inside, tf.float32))) * 100.0


# Test Processing & Output

def load_cmapss_test_trajectories(data_root: str | Path, dataset: str) -> Tuple[List[Tuple[int, np.ndarray]], np.ndarray]:
    root = Path(data_root)
    test = read_cmapss_file(root / f"test_{dataset}.txt")
    rul_file = root / f"RUL_{dataset}.txt"
    
    unit = test.iloc[:, 0].to_numpy(dtype=np.int64)
    cycle = test.iloc[:, 1].to_numpy(dtype=np.float32)
    sensors_all = test.iloc[:, 2:23].to_numpy(dtype=np.float32)

    keep_indices = [i for i in range(21) if (i + 1) not in REMOVED_CMAPPSS_SENSORS]
    sensors = sensors_all[:, keep_indices]
    final_rul = np.loadtxt(rul_file).astype(np.float32)

    trajectories = []
    for uid in np.unique(unit):
        mask = unit == uid
        order = np.argsort(cycle[mask])
        trajectories.append((int(uid), sensors[mask][order].astype(np.float32)))

    return trajectories, np.minimum(final_rul, 125.0)

class CMapssTestDataset:
    def __init__(self, trajectories: Sequence[Tuple[int, np.ndarray]], final_rul: np.ndarray, segment_window: int, n_segments: int, x_min: np.ndarray, x_max: np.ndarray):
        self.samples = []
        self.seq_len = segment_window * n_segments

        for idx, (uid, sensors) in enumerate(trajectories):
            normalized = minmax_apply(sensors, x_min, x_max).astype(np.float32)
            if len(normalized) < self.seq_len:
                pad = self.seq_len - len(normalized)
                first = np.repeat(normalized[:1], pad, axis=0)
                x_window = np.concatenate([first, normalized], axis=0)
            else:
                x_window = normalized[-self.seq_len:]
            self.samples.append((int(uid), x_window.astype(np.float32), np.array([1.0], dtype=np.float32), float(max(final_rul[idx], 1e-6))))

def evaluate_test_set(model: PCBNN, dataset: CMapssTestDataset, n_mc: int) -> Tuple[Dict[str, float], pd.DataFrame]:
    rows = []
    for uid, x_np, t_np, y in dataset.samples:
        x = tf.convert_to_tensor(x_np[np.newaxis, ...], dtype=tf.float32)
        t = tf.convert_to_tensor(t_np[np.newaxis, ...], dtype=tf.float32)
        result = predict_with_uncertainty(model, x, t, n_samples=n_mc)

        pred = float(result["y_pred"][0])
        error = pred - y
        rows.append({
            "engine_id": uid, "real_rul": y, "predicted_rul": pred, "error": error, "absolute_error": abs(error),
            "ci_lower": float(result["ci_lower"][0]), "ci_upper": float(result["ci_upper"][0]),
            "ci_width": float(result["ci_upper"][0] - result["ci_lower"][0]), "std_total": float(result["std_total"][0]),
            "var_total": float(result["var_total"][0]), "var_aleatoric": float(result["var_aleatoric"][0]), "var_epistemic": float(result["var_epistemic"][0]),
        })

    results_df = pd.DataFrame(rows).sort_values("engine_id").reset_index(drop=True)
    y_pred, y_true = tf.constant(results_df["predicted_rul"].to_numpy(), dtype=tf.float32), tf.constant(results_df["real_rul"].to_numpy(), dtype=tf.float32)
    lower, upper = tf.constant(results_df["ci_lower"].to_numpy(), dtype=tf.float32), tf.constant(results_df["ci_upper"].to_numpy(), dtype=tf.float32)

    metrics = {
        "RMSE": rmse_metric(y_pred, y_true), "MAE": float(tf.reduce_mean(tf.abs(y_pred - y_true))),
        "MaxAbsoluteError": float(tf.reduce_max(tf.abs(y_pred - y_true))), "Score": score_metric(y_pred, y_true),
        "PICP_percent": picp_metric(y_true, lower, upper), "Mean_CI_width": float(tf.reduce_mean(upper - lower)),
        "Mean_aleatoric_variance": float(results_df["var_aleatoric"].mean()), "Mean_epistemic_variance": float(results_df["var_epistemic"].mean()),
        "Mean_total_variance": float(results_df["var_total"].mean()),
    }
    return metrics, results_df

def print_detailed_results(results_df: pd.DataFrame, metrics: Dict[str, float]) -> None:
    print("\n" + "=" * 118)
    print(f"{'Engine':>8} {'Real RUL':>12} {'Predicted RUL':>16} {'Error':>12} {'Abs Error':>12} {'95% CI Lower':>14} {'95% CI Upper':>14}")
    for row in results_df.itertuples(index=False):
        print(f"{int(row.engine_id):>8} {row.real_rul:>12.2f} {row.predicted_rul:>16.2f} {row.error:>12.2f} {row.absolute_error:>12.2f} {row.ci_lower:>14.2f} {row.ci_upper:>14.2f}")
    print("=" * 118)
    for k, v in metrics.items(): print(f"{k:>42}: {v:.4f}")

def save_detailed_results(results_df: pd.DataFrame, output_path: str | Path) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(path, index=False)
    return path


# Execution Entry Point

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--dataset", choices=["FD001", "FD002", "FD003", "FD004"], default="FD001")
    parser.add_argument("--case", choices=["I", "II"], default="II")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--stride", type=int, default=5)
    parser.add_argument("--hgrr-levels", type=int, default=None)
    parser.add_argument("--n-segments", type=int, default=None)
    parser.add_argument("--mc-train", type=int, default=None)
    parser.add_argument("--mc-test", type=int, default=50)
    parser.add_argument("--results-csv", default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--eta", type=float, default=None)
    parser.add_argument("--max-beta", type=float, default=None)
    parser.add_argument("--max-power", type=float, default=None)
    parser.add_argument("--grad-clip", type=float, default=None)
    parser.add_argument("--patience", type=int, default=None)
    return parser.parse_args()

def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    case = copy.deepcopy(CASE_I if args.case == "I" else CASE_II)
    if args.hgrr_levels is not None: case.hgrr_levels = args.hgrr_levels
    if args.n_segments is not None: case.n_segments = args.n_segments
    if args.mc_train is not None: case.mc_samples = args.mc_train
    if args.batch_size is not None: case.batch_size = args.batch_size
    if args.lr is not None: case.learning_rate = args.lr
    if args.epochs is not None: case.epochs = args.epochs
    if args.eta is not None: case.eta = args.eta

    print(f"TensorFlow version: {tf.__version__}")
    all_train = load_cmapss_training_trajectories(args.data_root, args.dataset)
    train_traj, val_traj = split_trajectories(all_train, val_fraction=args.val_fraction, seed=args.seed)

    x_min, x_max = minmax_fit(train_traj)
    train_ds = CMapssWindowDataset(train_traj, case.segment_window, case.n_segments, args.stride, x_min, x_max)
    val_ds = CMapssWindowDataset(val_traj, case.segment_window, case.n_segments, args.stride, x_min, x_max)

    model = PCBNN(n_features=train_ds[0][0].shape[-1], case=case)
    model = fit(model=model, train_dataset=train_ds, val_dataset=val_ds, case=case, patience=args.patience)

    test_traj, test_rul = load_cmapss_test_trajectories(args.data_root, args.dataset)
    test_ds = CMapssTestDataset(test_traj, test_rul, case.segment_window, case.n_segments, x_min, x_max)
    metrics, results_df = evaluate_test_set(model=model, dataset=test_ds, n_mc=args.mc_test)

    saved_path = save_detailed_results(results_df, args.results_csv or f"results_{args.dataset}.csv")
    print_detailed_results(results_df, metrics)
    print(f"\nDetailed results CSV: {saved_path.resolve()}")

if __name__ == "__main__":
    main()