"""
====================================================================================
 PCBNN — Physics-Constrained Bayesian Neural Network (TensorFlow port)
====================================================================================
TensorFlow/Keras reconstruction of the PCBNN model described in:

    E. Wang et al., "A physics-constrained Bayesian neural network for machinery
    remaining useful life prediction and uncertainty quantification",
    Reliability Engineering and System Safety 266 (2026) 111778.

This file is a direct, equation-by-equation port of a PyTorch reconstruction of
the same paper. Every class, function, and comment has been re-derived for
TensorFlow (tf.Module / tf.GradientTape / Keras layers), not just mechanically
translated -- a few things had to change shape because PyTorch and TensorFlow
handle autograd, complex numbers, and RNN state very differently. Every place
that changed is called out explicitly in a comment.

IMPORTANT (same caveats as the original reconstruction):
- This is a mathematical reconstruction from the published paper, not the
  authors' original source code (the paper states Keras/TensorFlow was used
  for the ORIGINAL experiments, but the authors did not publish that code).
  This file happens to now also be TensorFlow, but it is still an independent
  reconstruction built only from the equations and text in the paper.
- Values explicitly reported by the paper are used where available (Table 2,
  Section 4.1, Eq. 49, etc).
- Details the paper does not fully specify (hidden sizes, exact HGRR level
  count, total window length, etc.) are exposed as configuration parameters,
  not silently presented as exact paper facts.

Core paper equations implemented, and where to find them in this file:
    Bi-LSTM segment encoder ................ Eqs. (9)-(14)  -> SegmentedBiLSTM
    HGRR / complex recurrence .............. Eqs. (15)-(24) -> HGRULevel, BayesianHGRR
    Weibull likelihood / parameters ........ Eqs. (25)-(27) -> weibull_* functions
    Deep hidden physics model (deep-HPM) ... Eqs. (28)-(31) -> DeepHPM
    ELBO / Bayes-by-Backprop ............... Eqs. (32)-(37) -> BayesianDense, total_loss
    Prediction & uncertainty decomposition . Eqs. (38)-(44) -> predict_with_uncertainty
    Evaluation metrics ..................... Eqs. (45)-(47) -> rmse_metric / score_metric / picp_metric
    C-MAPSS RUL label cap ................... Eq. (49)       -> load_cmapss_training_trajectories

Requirements:
    pip install tensorflow numpy pandas

Example:
    python pcbnn_tensorflow.py --data-root ./CMAPSSData --dataset FD001 --case II
====================================================================================
"""

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


# ====================================================================================
# SECTION 0 — Reproducibility
# ====================================================================================

def set_seed(seed: int = 0) -> None:
    """Seeds every source of randomness this script touches (Python, NumPy, TF)."""
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    # Unlike PyTorch's separate torch.cuda.manual_seed_all(), a single
    # tf.random.set_seed() call also seeds TensorFlow's GPU ops, so there is
    # no separate "seed the GPU" step needed here.


# ====================================================================================
# SECTION 1 — Configuration
# ====================================================================================
# Unchanged from the original reconstruction: this is plain Python bookkeeping,
# framework-agnostic, so nothing here needed to change for TensorFlow.
# ====================================================================================

@dataclass
class CaseConfig:
    # Values from the paper's Table 2.
    learning_rate: float
    lr_decay_every: int
    epochs: int
    batch_size: int
    derivative_order: int
    segment_window: int

    # The paper's sensitivity study identifies eta around 100 as a good
    # intermediate setting. The paper does not put eta in Table 2 itself.
    eta: float = 100.0

    # Paper text: "10 Monte Carlo samples ... during training" / "50 samples
    # ... for testing". This field controls TRAINING MC samples; test-time MC
    # samples are a separate CLI flag (--mc-test, default 50) since the paper
    # uses two different values for the two phases.
    mc_samples: int = 10

    # Not explicitly specified in the paper. This is a reconstruction choice:
    # how many segments make up one fixed-length training window.
    n_segments: int = 4

    # Not explicitly specified in the paper. A minimum of two levels is used
    # here to make the term "hierarchical" in "Hierarchical Gated Recurrent
    # Regressor" operational (one level alone has no hierarchy to speak of).
    hgrr_levels: int = 2

    # Hidden sizes are not reported in the paper; exposed as reconstruction
    # parameters rather than claimed paper values.
    encoder_projection_dim: int = 32
    lstm_hidden_dim: int = 32
    hgrr_hidden_dim: int = 32
    physics_hidden_dim: int = 64

    # Standard-normal prior for the Gaussian Bayes-by-Backprop surrogate.
    # Exact prior scale is not explicitly reported in the paper.
    prior_sigma: float = 1.0

    # Numerical safeguards. These are implementation safeguards, not paper
    # parameters -- see the comments at their point of use for why each one
    # is needed to keep training numerically stable.
    max_beta: float = 20.0
    max_power: float = 20.0
    grad_clip: float = 1.0


CASE_I = CaseConfig(
    learning_rate=5e-4,
    lr_decay_every=40,
    epochs=150,
    batch_size=64,
    derivative_order=2,
    segment_window=30,
)

CASE_II = CaseConfig(
    learning_rate=1e-3,
    lr_decay_every=60,
    epochs=100,
    batch_size=128,
    derivative_order=2,
    segment_window=50,
)


# ====================================================================================
# SECTION 2 — BayesianDense: Bayes-by-Backprop Gaussian variational layer
# ====================================================================================
# WHAT CHANGED FROM THE PYTORCH VERSION:
#   - Subclasses tf.Module instead of nn.Module. tf.Module gives us the same
#     two things nn.Module gave the original: (a) automatic tracking of
#     tf.Variable attributes so the optimizer can find them via
#     model.trainable_variables, and (b) a `.submodules` property that
#     recursively walks every nested tf.Module -- this is what collect_kl()
#     below uses instead of PyTorch's `.modules()`.
#   - PyTorch's version CACHED the KL from the most recent forward() call.
#     Here, kl_divergence() recomputes the KL directly from (mu, rho) on
#     demand every time it's called. This is simpler and removes an entire
#     class of "which forward call does this KL correspond to" bugs -- the KL
#     only depends on the CURRENT variational parameters, not on any
#     particular sampled noise, so there was never a reason to cache it.
# ====================================================================================

class BayesianDense(tf.Module):
    """
    Gaussian variational fully-connected layer (the TF equivalent of
    torch.nn.Linear, but with a full posterior distribution over every
    weight instead of one fixed value per weight).

        q(w) = Normal(mu, sigma^2),   sigma = softplus(rho)
        p(w) = Normal(0, prior_sigma^2)

    Sampling uses the reparameterization trick (Section 3.1.4 / Eq. 36's
    "Bayes by Backprop"): w = mu + sigma * eps, eps ~ N(0,1). This is what
    lets gradients flow back into mu and rho through an operation that LOOKS
    random -- the randomness (eps) is generated fresh each call but is never
    itself a trainable quantity, so TF's GradientTape can differentiate
    straight through it.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        prior_sigma: float = 1.0,
        rho_init: float = -5.0,
        name: Optional[str] = None,
    ) -> None:
        super().__init__(name=name)
        if prior_sigma <= 0:
            raise ValueError("prior_sigma must be > 0")

        self.in_features = in_features
        self.out_features = out_features
        self.prior_sigma = float(prior_sigma)

        # --- weight/bias MEANS ---
        # PyTorch's nn.init.kaiming_uniform_(weight, a=sqrt(5)) is, for a
        # plain Linear layer, mathematically identical to sampling uniformly
        # from [-1/sqrt(fan_in), 1/sqrt(fan_in)] -- i.e. exactly PyTorch's own
        # DEFAULT nn.Linear initialization. We reproduce that same uniform
        # range directly here so the starting distribution of weights matches
        # the original reconstruction, rather than relying on a different
        # framework's different default initializer (which could subtly bias
        # early training behavior).
        bound = 1.0 / math.sqrt(in_features) if in_features > 0 else 0.0
        self.weight_mu = tf.Variable(
            tf.random.uniform([out_features, in_features], -bound, bound),
            name="weight_mu",
        )
        self.bias_mu = tf.Variable(
            tf.random.uniform([out_features], -bound, bound),
            name="bias_mu",
        )

        # --- weight/bias STANDARD DEVIATIONS (stored as unconstrained "rho") ---
        # rho_init=-5 => softplus(-5) ~= 0.0067, i.e. every weight starts out
        # almost deterministic (tiny posterior std). This lets the network
        # first learn good MEAN weights, similar to an ordinary network,
        # before the posterior variance has much room to grow and start
        # injecting meaningful uncertainty.
        self.weight_rho = tf.Variable(
            tf.fill([out_features, in_features], float(rho_init)),
            name="weight_rho",
        )
        self.bias_rho = tf.Variable(
            tf.fill([out_features], float(rho_init)),
            name="bias_rho",
        )

    @staticmethod
    def sigma_from_rho(rho: tf.Tensor) -> tf.Tensor:
        # softplus(x) = log(1+e^x) maps any real number to a POSITIVE number,
        # so "rho" can move freely during optimization without ever producing
        # an invalid (negative) standard deviation. +1e-8 keeps sigma from
        # ever being exactly zero (avoids log(0) inside the KL formula).
        return tf.nn.softplus(rho) + 1e-8

    def __call__(self, x: tf.Tensor, sample: bool = True) -> tf.Tensor:
        """
        x: (..., in_features)  ->  (..., out_features)

        sample=True  (the normal/training/inference mode used everywhere in
                       this file): draws a FRESH random weight matrix and
                       bias every single call, via the reparameterization
                       trick. Two calls with the same input WILL generally
                       give two different outputs -- that is the entire
                       mechanism this whole model's uncertainty estimates
                       are built on.
        sample=False (exposed for completeness/debugging): uses just the
                       posterior mean, i.e. a deterministic "MAP" pass.
        """
        if sample:
            weight_sigma = self.sigma_from_rho(self.weight_rho)
            bias_sigma = self.sigma_from_rho(self.bias_rho)

            eps_w = tf.random.normal(tf.shape(self.weight_mu))
            eps_b = tf.random.normal(tf.shape(self.bias_mu))

            weight = self.weight_mu + weight_sigma * eps_w
            bias = self.bias_mu + bias_sigma * eps_b
        else:
            weight = self.weight_mu
            bias = self.bias_mu

        # x @ weight.T + bias -- TF's tf.matmul(..., transpose_b=True) is the
        # direct analogue of PyTorch's F.linear(x, weight, bias).
        return tf.matmul(x, weight, transpose_b=True) + bias

    def _gaussian_kl(self, mu: tf.Tensor, sigma: tf.Tensor) -> tf.Tensor:
        """Closed-form KL[N(mu,sigma^2) || N(0,prior_sigma^2)], summed over every element."""
        var_ratio = tf.square(sigma / self.prior_sigma)
        kl = 0.5 * (
            var_ratio
            + tf.square(mu / self.prior_sigma)
            - 1.0
            - tf.math.log(var_ratio)
        )
        return tf.reduce_sum(kl)

    def kl_divergence(self) -> tf.Tensor:
        """
        Total KL for this layer's weight AND bias posteriors, computed fresh
        from the CURRENT (mu, rho) -- independent of any particular sampled
        noise, so it's safe to call this at any point without first calling
        __call__.
        """
        weight_sigma = self.sigma_from_rho(self.weight_rho)
        bias_sigma = self.sigma_from_rho(self.bias_rho)
        return self._gaussian_kl(self.weight_mu, weight_sigma) + self._gaussian_kl(
            self.bias_mu, bias_sigma
        )


def collect_kl(module: tf.Module) -> tf.Tensor:
    """
    Walks every submodule inside `module` (tf.Module.submodules recurses
    through nested tf.Modules automatically, INCLUDING ones stored inside a
    plain Python list attribute -- this was verified empirically before
    relying on it, since that's exactly how BayesianHGRR stores its stacked
    HGRULevel instances) and sums the .kl_divergence() of every BayesianDense
    found. This is the direct TF analogue of PyTorch's
    `for m in module.modules(): if isinstance(m, BayesianLinear): ...`.
    """
    total = tf.zeros(())
    for child in module.submodules:
        if isinstance(child, BayesianDense):
            total = total + child.kl_divergence()
    return total


# ====================================================================================
# SECTION 3 — SegmentedBiLSTM
# Eqs. (9)-(14)
# ====================================================================================
# WHAT CHANGED FROM THE PYTORCH VERSION:
#   - PyTorch's nn.LSTM(bidirectional=True) returns h_n stacked as
#     [layer0_fwd, layer0_bwd, ...], and we read h_n[-2]/h_n[-1] for the
#     final forward/backward states.
#   - Keras' tf.keras.layers.Bidirectional(LSTM(..., return_state=True))
#     instead returns a flat tuple:
#         (sequence_output, forward_h, forward_c, backward_h, backward_c)
#     This was verified directly against the installed Keras version before
#     relying on it (Bidirectional's return signature has changed across
#     Keras versions historically). We use forward_h and backward_h directly
#     -- no index arithmetic needed, which is arguably clearer than the
#     PyTorch h_n[-2]/h_n[-1] indexing trick it replaces.
# ====================================================================================

class SegmentedBiLSTM(tf.Module):
    def __init__(
        self,
        n_features: int,
        segment_window: int,
        projection_dim: int,
        lstm_hidden: int,
        name: Optional[str] = None,
    ) -> None:
        super().__init__(name=name)
        if segment_window <= 0:
            raise ValueError("segment_window must be > 0")

        self.segment_window = segment_window
        self.n_features = n_features

        # Eq. (9)-(13)'s "learnable encoder projection, followed by a ReLU
        # activation" that turns each flattened raw segment into a fixed-size
        # X_{H_L} embedding. A plain Keras Dense+ReLU is used here (not a
        # BayesianDense) because the paper's Bayesian treatment is specific
        # to the HGRR/Weibull stage (Section 3.1.2), not this encoder.
        self.segment_dense = tf.keras.layers.Dense(projection_dim, activation="relu")

        # A standard bidirectional LSTM realizes Eqs. (9)-(13) (run once
        # forward, once backward) and Eq. (14) is the subsequent
        # concatenation of the two directions' final hidden states, done
        # explicitly below in __call__.
        self.bilstm = tf.keras.layers.Bidirectional(
            tf.keras.layers.LSTM(lstm_hidden, return_sequences=True, return_state=True),
            merge_mode="concat",
        )
        self.out_dim = 2 * lstm_hidden

    def segment(self, x: tf.Tensor) -> tf.Tensor:
        """
        x: (batch, S, n_features) raw sequence.
        Chops it into (batch, d, window*n_features), d = S // window.
        Matches Section 3.1.1-A exactly: "d = S/W_in ... segments X_Win".
        """
        x = tf.convert_to_tensor(x, dtype=tf.float32)
        if len(x.shape) != 3:
            raise ValueError(f"Expected x=(batch,time,features), got {tuple(x.shape)}")

        seq_len = x.shape[1]
        n_features = x.shape[2]
        d = seq_len // self.segment_window
        if d < 1:
            raise ValueError(
                f"Sequence length {seq_len} is shorter than segment window {self.segment_window}."
            )

        usable = d * self.segment_window
        x = x[:, :usable, :]
        batch = tf.shape(x)[0]  # dynamic: the only dimension we don't assume is static
        return tf.reshape(x, [batch, d, self.segment_window * n_features])

    def __call__(self, x: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        """x: (batch, S, n_features) -> h_BL: (batch, 2*lstm_hidden), seq_out: full sequence."""
        segments = self.segment(x)                          # (b, d, window*F)
        projected = self.segment_dense(segments)             # (b, d, projection_dim)  -- X_{H_L}

        seq_out, fwd_h, fwd_c, bwd_h, bwd_c = self.bilstm(projected)
        # fwd_h / bwd_h are each the FINAL hidden state of their direction,
        # shape (batch, lstm_hidden) -- exactly h_forward / h_backward from
        # Eq. (14). fwd_c / bwd_c (final CELL states) are unused here, same
        # as the original PyTorch reconstruction only used the hidden state.

        h_bl = tf.concat([fwd_h, bwd_h], axis=-1)             # Eq. (14): h_BL = [h_forward, h_backward]
        return h_bl, seq_out


# ====================================================================================
# SECTION 4 — HGRULevel / BayesianHGRR
# Eqs. (15)-(26)
# ====================================================================================
# WHAT CHANGED FROM THE PYTORCH VERSION:
#   - torch.complex(re, im) / h_t.real / h_t.imag  ->  tf.complex(re, im) /
#     tf.math.real(h_t) / tf.math.imag(h_t). Verified beforehand that TF
#     correctly backpropagates gradients through complex64 intermediate
#     tensors back into the REAL-valued weight matrices that produced them
#     (this is the part of the whole port most likely to silently break, so
#     it was tested in isolation first).
#   - dtype is tf.complex64 (TF's name for the same 32-bit-real +
#     32-bit-imaginary complex type PyTorch calls torch.cfloat).
# Structurally, everything else (the GLU gate, the candidate value, the
# learned-lower-bound forget magnitude, the phase rotation, the final
# LayerNorm + projection) is an unchanged, direct translation.
# ====================================================================================

class HGRULevel(tf.Module):
    """
    ONE LEVEL of the hierarchy. The paper's HGRR equations (15)-(24) describe
    a single gated complex recurrence; the paper's own prose says forgetting
    behaviour should differ "at various levels of the network hierarchy",
    which implies a STACK of these levels (see BayesianHGRR below, which
    builds that stack -- this class is just one rung of it).
    """

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        kappa_init: float,
        prior_sigma: float,
        name: Optional[str] = None,
    ) -> None:
        super().__init__(name=name)
        self.hidden_dim = hidden_dim

        # Eq. (15): GLU output gate.
        self.w_g = BayesianDense(in_dim, hidden_dim, prior_sigma)
        # Eq. (16): candidate value c_t.
        self.w_c = BayesianDense(in_dim, hidden_dim, prior_sigma)
        # Eqs. (20)-(21): forget-gate magnitude (epsilon) and phase (theta).
        self.w_eps = BayesianDense(in_dim, hidden_dim, prior_sigma)
        self.w_theta = BayesianDense(in_dim, hidden_dim, prior_sigma)

        # This level's own learned lower bound kappa on the forgetting
        # magnitude (paper: "adjusted by a learned lower bound kappa").
        # A plain tf.Variable, not wrapped in BayesianDense, because the
        # paper treats kappa as one scalar per level, not a distribution.
        self.kappa_raw = tf.Variable(float(kappa_init), name="kappa_raw")

        self.layer_norm = tf.keras.layers.LayerNormalization()
        self.w_o = BayesianDense(2 * hidden_dim, hidden_dim, prior_sigma)

    def step(self, v_t: tf.Tensor, h_prev: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        """
        ONE recurrent update.
        v_t:    (batch, in_dim) real
        h_prev: (batch, hidden_dim) COMPLEX (tf.complex64)
        returns (o_t real, h_t complex)
        """
        g_t = tf.sigmoid(self.w_g(v_t))                       # Eq. (15)
        c_t = tf.nn.silu(self.w_c(v_t))                        # Eq. (16)

        kappa = tf.sigmoid(self.kappa_raw)                      # squash into (0,1)
        eps_t = kappa + (1.0 - kappa) * tf.sigmoid(self.w_eps(v_t))  # Eq. (20), in [kappa,1)
        theta_t = self.w_theta(v_t)                              # Eq. (21) phase, unbounded angle

        # Eq. (21): exp(i*theta) = cos(theta) + i*sin(theta)
        rotation = tf.complex(tf.cos(theta_t), tf.sin(theta_t))
        eps_complex = tf.complex(eps_t, tf.zeros_like(eps_t))
        c_complex = tf.complex(c_t, tf.zeros_like(c_t))

        # Eq. (22): the actual forget-gate recurrence update.
        h_t = eps_complex * rotation * h_prev + (1.0 - eps_complex) * c_complex

        # Eq. (23): split back into real/imag, gate, normalize.
        h_real_imag = tf.concat([tf.math.real(h_t), tf.math.imag(h_t)], axis=-1)
        g_real_imag = tf.concat([g_t, g_t], axis=-1)
        o_tilde = self.layer_norm(g_real_imag * h_real_imag)

        o_t = self.w_o(o_tilde)                                  # Eq. (24)
        return o_t, h_t

    def __call__(self, x_seq: tf.Tensor) -> tf.Tensor:
        """
        x_seq: (batch, T, in_dim) -- this level's full input sequence.
        Returns o_seq: (batch, T, hidden_dim), this level's output at every
        step -- which becomes the NEXT level's x_seq (standard stacked-RNN
        wiring, matching the paper's "various levels of the network
        hierarchy").
        """
        if len(x_seq.shape) != 3:
            raise ValueError("x_seq must have shape (batch,time,features)")

        batch = tf.shape(x_seq)[0]
        time_steps = x_seq.shape[1]
        h = tf.zeros([batch, self.hidden_dim], dtype=tf.complex64)

        outputs: List[tf.Tensor] = []
        for t_idx in range(time_steps):
            out, h = self.step(x_seq[:, t_idx, :], h)
            outputs.append(out)
        return tf.stack(outputs, axis=1)


class BayesianHGRR(tf.Module):
    """
    Stacks `num_levels` HGRULevel instances, each starting from a DIFFERENT
    kappa (the paper: "lower levels maintain smaller forgetting gate values
    ... higher levels use values closer to 1"). Level 0's raw kappa starts
    around -3 (sigmoid(-3)~=0.05, i.e. allowed to forget aggressively -> a
    short-term/local focus); the top level starts around +3
    (sigmoid(3)~=0.95, forced to remember -> a long-term/global focus).
    """

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        num_levels: int,
        prior_sigma: float,
        name: Optional[str] = None,
    ) -> None:
        super().__init__(name=name)
        if num_levels < 1:
            raise ValueError("num_levels must be >= 1")

        self.hidden_dim = hidden_dim
        self.num_levels = num_levels

        if num_levels == 1:
            kappa_inits = [0.0]
        else:
            kappa_inits = np.linspace(-3.0, 3.0, num_levels).tolist()

        levels: List[HGRULevel] = []
        for level_idx in range(num_levels):
            level_in_dim = in_dim if level_idx == 0 else hidden_dim
            levels.append(
                HGRULevel(
                    in_dim=level_in_dim,
                    hidden_dim=hidden_dim,
                    kappa_init=kappa_inits[level_idx],
                    prior_sigma=prior_sigma,
                )
            )
        # A plain Python list -- verified beforehand that tf.Module's
        # automatic tracking (used by both collect_kl() above and the
        # optimizer's `.trainable_variables`) correctly discovers tf.Module
        # instances stored inside a list attribute like this one.
        self.levels = levels

        # Eq. (26): two outputs -> alpha, beta.
        self.regressor = BayesianDense(hidden_dim, 2, prior_sigma)

    def __call__(self, v_seq: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
        """
        v_seq: (batch, T, in_dim) -- usually T=1 in this project (one
        [h_BL, time] pair per training sample, matching Fig. 2).
        """
        x = v_seq
        for level in self.levels:
            x = level(x)
        top_output = x[:, -1, :]                                 # top level's last time step

        raw_params = self.regressor(top_output)
        params = tf.nn.softplus(raw_params)                       # Eq. (26): forces > 0

        # Tiny additive floors purely to guarantee alpha,beta are never
        # numerically exactly zero (which would break log(alpha)/log(beta) /
        # divisions by alpha downstream in the Weibull formulas).
        alpha = params[:, 0] + 1e-4
        beta = params[:, 1] + 1e-4
        return top_output, alpha, beta

    def kl_divergence(self) -> tf.Tensor:
        return collect_kl(self)


# ====================================================================================
# SECTION 5 — Weibull distribution: Eqs. (25)-(28), (39)-(44)
# ====================================================================================
# Pure math, framework-agnostic in spirit -- only the function calls changed
# (torch.log/torch.exp/torch.lgamma/torch.clamp -> their tf.* equivalents).
# All numerical-stability safeguards (clamping, log-space computation) are
# carried over unchanged from the original reconstruction.
# ====================================================================================

def weibull_log_pdf(
    y: tf.Tensor,
    alpha: tf.Tensor,
    beta: tf.Tensor,
    max_beta: float = 20.0,
    max_power: float = 20.0,
) -> tf.Tensor:
    """Numerically stable log Weibull density, corresponding to Eq. (27)."""
    eps = 1e-6
    y = tf.clip_by_value(y, eps, tf.float32.max)
    alpha = tf.clip_by_value(alpha, eps, tf.float32.max)
    beta = tf.clip_by_value(beta, eps, max_beta)

    log_y = tf.math.log(y)
    log_alpha = tf.math.log(alpha)
    log_beta = tf.math.log(beta)
    log_ratio = log_y - log_alpha

    # (y/alpha)^beta = exp(beta * log(y/alpha)); the exponent is clamped
    # purely to prevent floating-point overflow during optimization, not
    # because the paper's formula has any such cap.
    power = tf.clip_by_value(beta * log_ratio, -max_power, max_power)
    return log_beta - log_alpha + (beta - 1.0) * log_ratio - tf.exp(power)


def weibull_nll(
    y: tf.Tensor,
    alpha: tf.Tensor,
    beta: tf.Tensor,
    max_beta: float = 20.0,
    max_power: float = 20.0,
) -> tf.Tensor:
    """Mean minibatch negative log-likelihood, corresponding to Eq. (35)."""
    return -tf.reduce_mean(
        weibull_log_pdf(y, alpha, beta, max_beta=max_beta, max_power=max_power)
    )


def weibull_mean(alpha: tf.Tensor, beta: tf.Tensor) -> tf.Tensor:
    """Eq. (28): E[Weibull(alpha,beta)] = alpha * Gamma(1 + 1/beta)."""
    beta = tf.clip_by_value(beta, 1e-6, 20.0)
    # tf.math.lgamma(x) = log(Gamma(x)) -- computing in log-space and then
    # exponentiating is the numerically safe route, since Gamma(x) itself can
    # become astronomically large for some inputs.
    return alpha * tf.exp(tf.math.lgamma(1.0 + 1.0 / beta))


def weibull_variance(alpha: tf.Tensor, beta: tf.Tensor) -> tf.Tensor:
    """Eqs. (43)-(44): Var[Weibull] = alpha^2 * [Gamma(1+2/beta) - Gamma(1+1/beta)^2]."""
    beta = tf.clip_by_value(beta, 1e-6, 20.0)
    gamma1 = tf.exp(tf.math.lgamma(1.0 + 1.0 / beta))
    gamma2 = tf.exp(tf.math.lgamma(1.0 + 2.0 / beta))
    value = tf.square(alpha) * (gamma2 - tf.square(gamma1))
    return tf.maximum(value, 0.0)  # guards against a tiny negative value from float rounding


def weibull_pdf(
    y: tf.Tensor,
    alpha: tf.Tensor,
    beta: tf.Tensor,
    max_beta: float = 20.0,
    max_power: float = 20.0,
) -> tf.Tensor:
    """Weibull probability density, used by Eq. (42)'s epistemic-uncertainty formula."""
    log_pdf = weibull_log_pdf(y, alpha, beta, max_beta=max_beta, max_power=max_power)
    return tf.exp(tf.clip_by_value(log_pdf, -50.0, 20.0))


# ====================================================================================
# SECTION 6 — Deep hidden physics model (deep-HPM)
# Eqs. (28)-(31)
# ====================================================================================
# WHAT CHANGED FROM THE PYTORCH VERSION (this is the most structurally
# different section in the whole port):
#   - PyTorch computes higher-order derivatives with torch.autograd.grad(...,
#     create_graph=True, retain_graph=True), calling it repeatedly against the
#     SAME underlying computation graph.
#   - TensorFlow's equivalent is NESTED tf.GradientTape contexts: an "inner"
#     tape records the forward pass and gives the FIRST derivatives; an
#     "outer" tape, wrapped around the inner one, can then differentiate
#     THOSE derivatives again to get the SECOND derivatives. Both tapes must
#     be created with persistent=True (since each is asked for more than one
#     gradient), and -- this was confirmed empirically before writing this,
#     because it's an easy way to silently get None gradients -- the
#     second-derivative loop must stay INSIDE the outer tape's `with` block;
#     calling `outer_tape.gradient(...)` after the block has exited does not
#     see the gradient-of-gradient operations, even with persistent=True.
#   - Persistent tapes are not garbage-collected automatically; they are
#     explicitly deleted (`del inner_tape, outer_tape`) once we're done with
#     them, matching TF's documented guidance for persistent tapes.
# ====================================================================================

class NonlinearDynamicNetwork(tf.Module):
    """The learned function G(t, hBL, RUL, dRUL/dhBL, d2RUL/dhBL2) from Fig. 2/5."""

    def __init__(self, h_bl_dim: int, hidden_dim: int = 64, name: Optional[str] = None) -> None:
        super().__init__(name=name)
        # input = [t, h_BL, rul, drul_dh (h_bl_dim), d2rul_dh2 (h_bl_dim)]
        input_dim = 1 + h_bl_dim + 1 + h_bl_dim + h_bl_dim
        # A plain (non-Bayesian) two-layer MLP, matching Fig. 2's "Nonlinear
        # dynamic network" box (Linear -> ReLU -> Linear).
        self.dense1 = tf.keras.layers.Dense(hidden_dim, activation="relu", input_shape=(input_dim,))
        self.dense2 = tf.keras.layers.Dense(1)

    def __call__(
        self,
        t: tf.Tensor,
        h_bl: tf.Tensor,
        rul: tf.Tensor,
        drul_dh: tf.Tensor,
        d2rul_dh2: tf.Tensor,
    ) -> tf.Tensor:
        x = tf.concat([t, h_bl, rul, drul_dh, d2rul_dh2], axis=-1)
        return tf.squeeze(self.dense2(self.dense1(x)), axis=-1)


class DeepHPM(tf.Module):
    def __init__(self, h_bl_dim: int, hidden_dim: int = 64, name: Optional[str] = None) -> None:
        super().__init__(name=name)
        self.dynamic_net = NonlinearDynamicNetwork(h_bl_dim, hidden_dim)

    def residual(
        self,
        solution_fn,
        h_bl: tf.Tensor,
        t: tf.Tensor,
        derivative_order: int = 2,
    ) -> tf.Tensor:
        """
        solution_fn: callable (h_bl, t) -> (alpha, beta) -- this is
                     BayesianHGRR.__call__ wrapped by PCBNN.solution_fn,
                     passed in so this class never needs to know how alpha/
                     beta are actually produced.
        Returns f = "real rate of change" minus "G's predicted rate of
        change", per sample -- Eq. (30).
        """
        if derivative_order not in (1, 2, 3):
            raise ValueError("The paper's derivative search scope is {1,2,3}.")

        h = tf.identity(h_bl)   # plain tensors (not tf.Variable), so we must
        time = tf.identity(t)   # explicitly `.watch()` them below to track gradients.

        with tf.GradientTape(persistent=True) as outer_tape:
            outer_tape.watch(h)
            outer_tape.watch(time)

            with tf.GradientTape(persistent=True) as inner_tape:
                inner_tape.watch(h)
                inner_tape.watch(time)
                alpha, beta = solution_fn(h, time)
                rul = weibull_mean(alpha, beta)[:, tf.newaxis]   # rul~(h_BL,t), Eq. (28), shape (batch,1)

            # dRUL/dt : Eq. (29)'s time-derivative term.
            drul_dt = inner_tape.gradient(rul, time)
            # dRUL/dh_BL : Eq. (29)'s state-derivative term.
            drul_dh = inner_tape.gradient(rul, h)

            if derivative_order >= 2:
                # The paper writes d^2(rul)/d(h_BL)^2 in vector notation.
                # Because rul is scalar and h_BL is a vector, this
                # reconstruction supplies the DIAGONAL second derivatives
                # [d2(rul)/d(h_i)^2] -- the usual compact vector
                # representation when G is built to accept the same
                # dimensionality as h_BL (Fig. 5's diagram). This loop MUST
                # remain inside the `with outer_tape:` block -- verified
                # empirically that calling outer_tape.gradient() after the
                # block exits returns None instead of the correct gradient.
                second_components: List[tf.Tensor] = []
                for j in range(h.shape[-1]):
                    first_j = drul_dh[:, j:j + 1]
                    second_j = outer_tape.gradient(first_j, h)[:, j:j + 1]
                    second_components.append(second_j)
                d2rul_dh2 = tf.concat(second_components, axis=-1)
            else:
                d2rul_dh2 = tf.zeros_like(drul_dh)

            if derivative_order >= 3:
                # The paper's search scope (Table 2) includes order 3, but
                # both reported experimental cases use order 2. Third-order
                # derivatives are not computed here, matching the original
                # reconstruction's choice not to feed a larger, unspecified
                # tensor shape into G for a configuration the paper never
                # actually reports results for.
                pass

        del inner_tape, outer_tape  # persistent tapes must be deleted explicitly

        g_pred = self.dynamic_net(time, h, rul, drul_dh, d2rul_dh2)
        return tf.squeeze(drul_dt, axis=-1) - g_pred

    def residual_loss(
        self,
        solution_fn,
        h_bl: tf.Tensor,
        t: tf.Tensor,
        derivative_order: int = 2,
    ) -> tf.Tensor:
        residual = self.residual(solution_fn, h_bl, t, derivative_order)
        # The paper writes the residual aggregation as a plain mean in
        # Eq. (31). As a MINIMIZATION objective this is problematic: a raw
        # signed mean has no lower bound (gradient descent could push it
        # arbitrarily negative forever rather than toward zero). The standard
        # fix -- used here, as in the original reconstruction -- is to
        # minimize the MEAN SQUARED residual instead, so positive and
        # negative PDE errors cannot cancel out and the objective is bounded
        # below by zero exactly when the physics constraint is satisfied.
        return tf.reduce_mean(tf.square(residual))


# ====================================================================================
# SECTION 7 — PCBNN: wires SegmentedBiLSTM + BayesianHGRR + DeepHPM together
# ====================================================================================

class PCBNN(tf.Module):
    def __init__(self, n_features: int, case: CaseConfig, name: Optional[str] = None) -> None:
        super().__init__(name=name)
        self.case = case
        self.n_features = n_features

        self.encoder = SegmentedBiLSTM(
            n_features=n_features,
            segment_window=case.segment_window,
            projection_dim=case.encoder_projection_dim,
            lstm_hidden=case.lstm_hidden_dim,
        )
        self.h_bl_dim = self.encoder.out_dim

        self.hgrr = BayesianHGRR(
            in_dim=self.h_bl_dim + 1,          # +1 for the concatenated time scalar
            hidden_dim=case.hgrr_hidden_dim,
            num_levels=case.hgrr_levels,
            prior_sigma=case.prior_sigma,
        )

        self.deep_hpm = DeepHPM(
            h_bl_dim=self.h_bl_dim,
            hidden_dim=case.physics_hidden_dim,
        )

    def encode(self, x: tf.Tensor) -> tf.Tensor:
        h_bl, _ = self.encoder(x)
        return h_bl

    def solution_fn(self, h_bl: tf.Tensor, t: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        """
        (h_bl, t) -> (alpha, beta). Used both for the "real" forward pass and
        as the differentiable callable DeepHPM.residual() needs.
        """
        if len(t.shape) == 1:
            t = t[:, tf.newaxis]
        v = tf.concat([h_bl, t], axis=-1)[:, tf.newaxis, :]   # add a size-1 time-step axis
        _, alpha, beta = self.hgrr(v)
        beta = tf.clip_by_value(beta, 1e-4, self.case.max_beta)
        return alpha, beta

    def __call__(
        self,
        x: tf.Tensor,
        t: tf.Tensor,
        compute_physics: bool = True,
    ) -> Dict[str, tf.Tensor]:
        """
        THE MAIN ENTRY POINT, called once per training/eval step.
        x: (batch, S, n_features) raw sensor window
        t: (batch, 1) current time (as a fraction of the machine's life)
        """
        h_bl = self.encode(x)
        alpha, beta = self.solution_fn(h_bl, t)

        if compute_physics:
            # h_bl is NOT detached here: the physics constraint participates
            # in end-to-end optimization of the whole PCBNN (this matches the
            # most recent PyTorch reconstruction's choice; an earlier
            # reconstruction had detached it, trading off encoder-gradient
            # purity for less physics-driven regularization of the encoder).
            residual_loss = self.deep_hpm.residual_loss(
                self.solution_fn, h_bl, t, derivative_order=self.case.derivative_order
            )
        else:
            residual_loss = tf.zeros(())

        return {"h_bl": h_bl, "alpha": alpha, "beta": beta, "residual_loss": residual_loss}

    def kl_divergence(self) -> tf.Tensor:
        return self.hgrr.kl_divergence()


# ====================================================================================
# SECTION 8 — C-MAPSS data loading
# ====================================================================================
# Framework-agnostic: this section is almost byte-for-byte unchanged from the
# original reconstruction, since pandas/NumPy don't care whether the model
# downstream is PyTorch or TensorFlow. The only change is that dataset
# classes now return plain NumPy arrays (not torch tensors) from
# __getitem__, since batching/tensor conversion happens in ArrayBatcher
# (Section 9) using tf.convert_to_tensor instead of torch.from_numpy.
# ====================================================================================

# C-MAPSS has 21 sensor columns. The paper removes sensors 1,5,10,16,18,19
# (Fig. 14's correlation analysis on FD001/FD003). Sensor numbering here is
# 1-based, exactly as in the paper.
REMOVED_CMAPPSS_SENSORS = {1, 5, 10, 16, 18, 19}


def read_cmapss_file(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path, sep=r"\s+", header=None)


def load_cmapss_training_trajectories(
    data_root: str | Path,
    dataset: str,
) -> List[Tuple[int, np.ndarray, np.ndarray]]:
    """
    Returns (unit_id, sensors, rul) for each training engine.
    RUL is capped at 125 as specified by Eq. (49).
    """
    root = Path(data_root)
    train = read_cmapss_file(root / f"train_{dataset}.txt")

    unit = train.iloc[:, 0].to_numpy(dtype=np.int64)
    cycle = train.iloc[:, 1].to_numpy(dtype=np.float32)
    sensors_all = train.iloc[:, 2:23].to_numpy(dtype=np.float32)

    keep_indices = [i for i in range(21) if (i + 1) not in REMOVED_CMAPPSS_SENSORS]
    sensors = sensors_all[:, keep_indices]

    trajectories: List[Tuple[int, np.ndarray, np.ndarray]] = []
    max_cycle_by_unit: Dict[int, float] = {
        int(uid): float(cycle[unit == uid].max()) for uid in np.unique(unit)
    }

    for uid in np.unique(unit):
        mask = unit == uid
        order = np.argsort(cycle[mask])
        x = sensors[mask][order]
        cyc = cycle[mask][order]
        raw_rul = max_cycle_by_unit[int(uid)] - cyc
        rul = np.minimum(raw_rul, 125.0).astype(np.float32)   # Eq. (49)
        trajectories.append((int(uid), x.astype(np.float32), rul))

    return trajectories


def minmax_fit(trajectories: Sequence[Tuple[int, np.ndarray, np.ndarray]]) -> Tuple[np.ndarray, np.ndarray]:
    values = np.concatenate([x for _, x, _ in trajectories], axis=0)
    return values.min(axis=0, keepdims=True), values.max(axis=0, keepdims=True)


def minmax_apply(x: np.ndarray, x_min: np.ndarray, x_max: np.ndarray) -> np.ndarray:
    denom = np.clip(x_max - x_min, 1e-8, None)
    return (x - x_min) / denom


class CMapssWindowDataset:
    """
    Slides a fixed-length window across every engine's history, normalizes
    it, and pairs each window with a time fraction and the true RUL at the
    window's last time step. Plain Python container (no framework
    dependency) -- ArrayBatcher (Section 9) turns groups of these into
    TensorFlow tensors at batch time.
    """

    def __init__(
        self,
        trajectories: Sequence[Tuple[int, np.ndarray, np.ndarray]],
        segment_window: int,
        n_segments: int,
        stride: int,
        x_min: np.ndarray,
        x_max: np.ndarray,
    ) -> None:
        if n_segments < 1:
            raise ValueError("n_segments must be >= 1")
        if stride < 1:
            raise ValueError("stride must be >= 1")

        self.seq_len = segment_window * n_segments
        self.samples: List[Tuple[np.ndarray, np.ndarray, float]] = []

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


def split_trajectories(
    trajectories: Sequence[Tuple[int, np.ndarray, np.ndarray]],
    val_fraction: float = 0.1,
    seed: int = 0,
) -> Tuple[List[Tuple[int, np.ndarray, np.ndarray]], List[Tuple[int, np.ndarray, np.ndarray]]]:
    """Splits by ENGINE (not by overlapping window) so validation engines are
    never seen, even partially, during training."""
    if not 0.0 < val_fraction < 1.0:
        raise ValueError("val_fraction must be between 0 and 1")

    rng = np.random.default_rng(seed)
    idx = np.arange(len(trajectories))
    rng.shuffle(idx)

    n_val = max(1, int(round(len(idx) * val_fraction)))
    val_idx = set(idx[:n_val].tolist())

    train = [item for i, item in enumerate(trajectories) if i not in val_idx]
    val = [item for i, item in enumerate(trajectories) if i in val_idx]
    return train, val


# ====================================================================================
# SECTION 9 — ArrayBatcher: minimal TensorFlow replacement for torch's DataLoader
# ====================================================================================

class ArrayBatcher:
    """
    Iterates a CMapssWindowDataset (or any object with __len__/__getitem__
    returning (x, t, y) NumPy triples) in shuffled or sequential mini-batches,
    converting each batch to TensorFlow tensors on the fly. This exists
    purely because torch.utils.data.DataLoader has no direct TensorFlow
    equivalent that matches this exact "list of numpy triples" data shape as
    simply -- tf.data.Dataset is TensorFlow's own (more heavyweight) answer
    to the same problem, but a plain Python generator keeps this port's
    control flow a line-for-line match with the original training loop.
    """

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
            xs, ts, ys = [], [], []
            for i in batch_idx:
                x, t, y = self.dataset[int(i)]
                xs.append(x)
                ts.append(t)
                ys.append(y)
            yield (
                tf.convert_to_tensor(np.stack(xs), dtype=tf.float32),
                tf.convert_to_tensor(np.stack(ts), dtype=tf.float32),
                tf.convert_to_tensor(np.array(ys, dtype=np.float32)),
            )

    def __len__(self) -> int:
        return (len(self.dataset) + self.batch_size - 1) // self.batch_size


# ====================================================================================
# SECTION 10 — ELBO + physics loss: Eqs. (34)-(37)
# ====================================================================================

def total_loss(
    y: tf.Tensor,
    alpha: tf.Tensor,
    beta: tf.Tensor,
    kl: tf.Tensor,
    residual_loss: tf.Tensor,
    dataset_size: int,
    batch_size: int,
    eta: float,
    max_beta: float,
    max_power: float,
) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
    """
    Stochastic ELBO + physics penalty (Eqs. 34, 37).

    The paper writes the full-dataset NLL as a SUM over N_train samples
    (Eq. 35). We instead compute the MEAN minibatch NLL (weibull_nll already
    averages), then rescale by N/B so its EXPECTATION tracks what the
    full-dataset sum-based NLL would contribute to the objective. The KL term
    is counted once per objective evaluation (it doesn't scale with batch
    size -- it's a property of the whole variational posterior, not of any
    particular minibatch of data).
    """
    nll_batch = weibull_nll(y, alpha, beta, max_beta=max_beta, max_power=max_power)
    scale = float(dataset_size) / float(max(batch_size, 1))
    expected_full_nll = nll_batch * scale
    elbo = kl + expected_full_nll
    total = elbo + eta * residual_loss
    return total, elbo, expected_full_nll


# ====================================================================================
# SECTION 11 — Training loop
# ====================================================================================
# WHAT CHANGED FROM THE PYTORCH VERSION:
#   - PyTorch's pattern is: call .backward() once per Monte Carlo sample with
#     the loss already divided by mc_samples, and gradients silently
#     ACCUMULATE into each parameter's .grad attribute across those calls
#     (that's just how PyTorch autograd works by default) until
#     optimizer.step() is finally called once per batch.
#   - TensorFlow's GradientTape has no such implicit accumulation: each
#     `tape.gradient(...)` call returns a fresh, independent list of
#     gradients, and nothing is added to the model's variables until
#     optimizer.apply_gradients() is explicitly called. So here, gradient
#     accumulation across Monte Carlo samples is done BY HAND: a running list
#     of per-variable gradient sums is kept, one fresh (small, per-sample)
#     GradientTape is opened for each MC sample, and only after all MC
#     samples for a batch are done do we clip and apply the accumulated
#     gradients once.
#   - This also naturally solves the memory problem PyTorch's version avoids
#     by dividing-and-backward-ing per sample: each MC sample's tape
#     (including its own nested physics tapes) is opened, used, and released
#     before starting the next MC sample, so at most one Monte Carlo sample's
#     worth of computation graph exists in memory at a time.
# ====================================================================================

def run_epoch(
    model: PCBNN,
    loader: ArrayBatcher,
    optimizer: Optional[tf.keras.optimizers.Optimizer],
    dataset_size: int,
    eta: float,
    mc_samples: int,
    train: bool,
) -> Dict[str, float]:
    """Runs ONE pass over `loader`. If train=True, also updates the model's weights."""
    total_running = elbo_running = nll_running = physics_running = 0.0
    rmse_sq_sum = 0.0
    n_seen = 0
    n_batches = 0

    trainable_vars = model.trainable_variables if train else None

    for x, t, y in loader:
        n_batches += 1

        # --- running sums of gradients across this batch's Monte Carlo samples ---
        if train:
            accumulated_grads = [tf.zeros_like(v) for v in trainable_vars]

        batch_total = batch_elbo = batch_nll = batch_physics = 0.0
        last_alpha = last_beta = None

        for _ in range(mc_samples):
            if train:
                # A fresh tape per MC sample -- see the section banner above
                # for why gradients are accumulated by hand here instead of
                # relying on implicit accumulation like the PyTorch version.
                with tf.GradientTape() as tape:
                    out = model(x, t, compute_physics=True)
                    alpha, beta = out["alpha"], out["beta"]
                    physics = out["residual_loss"]
                    kl = model.kl_divergence()
                    loss, elbo, nll = total_loss(
                        y=y, alpha=alpha, beta=beta, kl=kl, residual_loss=physics,
                        dataset_size=dataset_size, batch_size=int(tf.size(y)), eta=eta,
                        max_beta=model.case.max_beta, max_power=model.case.max_power,
                    )
                    if not tf.math.is_finite(loss):
                        raise FloatingPointError(
                            f"Non-finite loss: loss={float(loss)}, ELBO={float(elbo)}, "
                            f"NLL={float(nll)}, physics={float(physics)}"
                        )
                    scaled_loss = loss / float(mc_samples)

                grads = tape.gradient(scaled_loss, trainable_vars)
                # Some variables can legitimately receive no gradient on a
                # given call (e.g. an HGRR level whose output this particular
                # sample's compute graph didn't route through under some
                # future architecture change) -- treat a None gradient as a
                # zero contribution rather than crashing, mirroring how
                # PyTorch simply leaves .grad untouched for unused params.
                accumulated_grads = [
                    acc if g is None else acc + g
                    for acc, g in zip(accumulated_grads, grads)
                ]
            else:
                out = model(x, t, compute_physics=False)
                alpha, beta = out["alpha"], out["beta"]
                physics = out["residual_loss"]
                kl = model.kl_divergence()
                loss, elbo, nll = total_loss(
                    y=y, alpha=alpha, beta=beta, kl=kl, residual_loss=physics,
                    dataset_size=dataset_size, batch_size=int(tf.size(y)), eta=eta,
                    max_beta=model.case.max_beta, max_power=model.case.max_power,
                )

            if not tf.math.is_finite(alpha).numpy().all() or not tf.math.is_finite(beta).numpy().all():
                raise FloatingPointError(
                    f"Non-finite Weibull parameters detected: "
                    f"alpha=[{float(tf.reduce_min(alpha))}, {float(tf.reduce_max(alpha))}], "
                    f"beta=[{float(tf.reduce_min(beta))}, {float(tf.reduce_max(beta))}]"
                )

            batch_total += float(loss)
            batch_elbo += float(elbo)
            batch_nll += float(nll)
            batch_physics += float(physics)
            last_alpha, last_beta = alpha, beta

        if train:
            clipped_grads, grad_norm = tf.clip_by_global_norm(accumulated_grads, model.case.grad_clip)
            if not tf.math.is_finite(grad_norm):
                raise FloatingPointError(f"Non-finite gradient norm detected: {float(grad_norm)}")
            optimizer.apply_gradients(zip(clipped_grads, trainable_vars))

        pred_mean = weibull_mean(last_alpha, last_beta)
        rmse_sq_sum += float(tf.reduce_sum(tf.square(pred_mean - y)))
        n_seen += int(tf.size(y))

        total_running += batch_total / mc_samples
        elbo_running += batch_elbo / mc_samples
        nll_running += batch_nll / mc_samples
        physics_running += batch_physics / mc_samples

    rmse = math.sqrt(rmse_sq_sum / max(n_seen, 1))
    return {
        "loss": total_running / max(n_batches, 1),
        "elbo": elbo_running / max(n_batches, 1),
        "nll": nll_running / max(n_batches, 1),
        "physics": physics_running / max(n_batches, 1),
        "rmse": rmse,
    }


def fit(
    model: PCBNN,
    train_dataset: CMapssWindowDataset,
    val_dataset: CMapssWindowDataset,
    case: CaseConfig,
    patience: Optional[int] = None,
) -> PCBNN:
    """The full training driver: loops over epochs, applies the step-decay LR
    schedule, tracks the best-so-far checkpoint, and (optionally) stops early."""
    train_loader = ArrayBatcher(train_dataset, batch_size=case.batch_size, shuffle=True)
    val_loader = ArrayBatcher(val_dataset, batch_size=case.batch_size, shuffle=False)

    optimizer = tf.keras.optimizers.Adam(learning_rate=case.learning_rate)

    best_val_loss = float("inf")
    best_weights: Optional[List[np.ndarray]] = None
    epochs_without_improvement = 0

    for epoch in range(1, case.epochs + 1):
        # --- step-decay learning rate schedule (paper: "reducing the
        # learning rate by a factor of 0.5 at fixed intervals") ---
        # Unlike PyTorch's torch.optim.lr_scheduler.StepLR object, Keras
        # optimizers don't ship a matching built-in scheduler object in this
        # imperative training-loop style, so the decay is applied by hand:
        # every `lr_decay_every` epochs, halve the optimizer's learning rate
        # directly.
        if epoch > 1 and (epoch - 1) % case.lr_decay_every == 0:
            optimizer.learning_rate.assign(optimizer.learning_rate * 0.5)

        train_stats = run_epoch(
            model=model, loader=train_loader, optimizer=optimizer,
            dataset_size=len(train_dataset), eta=case.eta, mc_samples=case.mc_samples, train=True,
        )
        val_stats = run_epoch(
            model=model, loader=val_loader, optimizer=None,
            dataset_size=len(val_dataset), eta=case.eta, mc_samples=1, train=False,
        )

        print(
            f"Epoch {epoch:03d} | "
            f"train loss {train_stats['loss']:.4f} | train RMSE {train_stats['rmse']:.4f} | "
            f"val loss {val_stats['loss']:.4f} | val RMSE {val_stats['rmse']:.4f} | "
            f"lr {float(optimizer.learning_rate):.3e}"
        )

        if val_stats["loss"] < best_val_loss:
            best_val_loss = val_stats["loss"]
            # tf.Variable objects don't support Python's copy.deepcopy the
            # way torch tensors do; instead we snapshot each variable's raw
            # NumPy value, which is what gets restored below once training
            # finishes.
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


# ====================================================================================
# SECTION 12 — Prediction and uncertainty decomposition: Eqs. (38)-(44)
# ====================================================================================
# WHAT CHANGED FROM THE PYTORCH VERSION:
#   - PyTorch wraps this whole function in @torch.no_grad(). TensorFlow has
#     no equivalent decorator to "turn off autograd" globally -- in eager
#     mode, simply NOT opening a tf.GradientTape around this code means no
#     computation graph is recorded at all, which is the natural TF
#     equivalent (there is nothing to disable; you just don't ask for it).
# Everything else (the Monte Carlo sampling loop, the point-prediction
# average, the aleatoric/epistemic variance decomposition, the confidence
# interval) is an unchanged, direct translation.
# ====================================================================================

def predict_with_uncertainty(
    model: PCBNN,
    x: tf.Tensor,
    t: tf.Tensor,
    n_samples: int = 50,
) -> Dict[str, tf.Tensor]:
    """Runs the model n_samples times (fresh random Bayesian weights each
    time) and returns a point prediction, a 95% CI, and the aleatoric/
    epistemic variance split."""
    alpha_samples: List[tf.Tensor] = []
    beta_samples: List[tf.Tensor] = []
    mean_samples: List[tf.Tensor] = []

    for _ in range(n_samples):
        h_bl = model.encode(x)
        alpha, beta = model.solution_fn(h_bl, t)
        alpha_samples.append(alpha)
        beta_samples.append(beta)
        mean_samples.append(weibull_mean(alpha, beta))

    alpha = tf.stack(alpha_samples, axis=0)      # (n_samples, batch)
    beta = tf.stack(beta_samples, axis=0)
    y_m = tf.stack(mean_samples, axis=0)

    y_pred = tf.reduce_mean(y_m, axis=0)                              # Eq. (40)

    var_aleatoric = tf.reduce_mean(weibull_variance(alpha, beta), axis=0)   # Eq. (44)

    likelihood_at_prediction = weibull_pdf(y_m, alpha, beta)
    var_epistemic = (
        tf.reduce_mean(tf.square(likelihood_at_prediction), axis=0)
        - tf.square(tf.reduce_mean(likelihood_at_prediction, axis=0))
    )                                                                   # Eq. (42)

    var_total = tf.maximum(var_aleatoric + var_epistemic, 0.0)          # Eq. (41)
    std_total = tf.sqrt(var_total)

    ci_lower = y_pred - 1.96 * std_total                                # Algorithm 1, line 14
    ci_upper = y_pred + 1.96 * std_total

    return {
        "y_pred": y_pred,
        "var_aleatoric": var_aleatoric,
        "var_epistemic": var_epistemic,
        "var_total": var_total,
        "std_total": std_total,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "alpha_samples": alpha,
        "beta_samples": beta,
    }


# ====================================================================================
# SECTION 13 — Evaluation metrics: Eqs. (45)-(47)
# ====================================================================================

def rmse_metric(y_pred: tf.Tensor, y_true: tf.Tensor) -> float:
    """Eq. (45)."""
    return float(tf.sqrt(tf.reduce_mean(tf.square(y_pred - y_true))))


def score_metric(y_pred: tf.Tensor, y_true: tf.Tensor) -> float:
    """
    Eq. (46): the paper's asymmetric Score function, which penalizes LATE
    predictions (telling an operator a machine is healthier than it really
    is) far more heavily than EARLY predictions (overly cautious but safe).
    """
    d = y_pred - y_true
    early_mask = d < 0
    late_mask = tf.logical_not(early_mask)

    d_early = tf.boolean_mask(d, early_mask)
    d_late = tf.boolean_mask(d, late_mask)

    score_early = tf.exp(tf.clip_by_value(-d_early / 13.0, -50.0, 50.0)) - 1.0
    score_late = tf.exp(tf.clip_by_value(d_late / 10.0, -50.0, 50.0)) - 1.0
    return float(tf.reduce_sum(score_early) + tf.reduce_sum(score_late))


def picp_metric(y_true: tf.Tensor, lower: tf.Tensor, upper: tf.Tensor) -> float:
    """Eq. (47): percentage of true RUL values that actually fell inside their own CI."""
    inside = tf.logical_and(y_true >= lower, y_true <= upper)
    return float(tf.reduce_mean(tf.cast(inside, tf.float32))) * 100.0


# ====================================================================================
# SECTION 14 — C-MAPSS test trajectory loading + full evaluation
# ====================================================================================

def load_cmapss_test_trajectories(
    data_root: str | Path,
    dataset: str,
) -> Tuple[List[Tuple[int, np.ndarray]], np.ndarray]:
    root = Path(data_root)
    test = read_cmapss_file(root / f"test_{dataset}.txt")
    rul_file = root / f"RUL_{dataset}.txt"
    if not rul_file.exists():
        raise FileNotFoundError(rul_file)

    unit = test.iloc[:, 0].to_numpy(dtype=np.int64)
    cycle = test.iloc[:, 1].to_numpy(dtype=np.float32)
    sensors_all = test.iloc[:, 2:23].to_numpy(dtype=np.float32)

    keep_indices = [i for i in range(21) if (i + 1) not in REMOVED_CMAPPSS_SENSORS]
    sensors = sensors_all[:, keep_indices]

    final_rul = np.loadtxt(rul_file).astype(np.float32)

    trajectories: List[Tuple[int, np.ndarray]] = []
    for uid in np.unique(unit):
        mask = unit == uid
        order = np.argsort(cycle[mask])
        trajectories.append((int(uid), sensors[mask][order].astype(np.float32)))

    if len(trajectories) != len(final_rul):
        raise ValueError(
            f"Number of test trajectories ({len(trajectories)}) does not match "
            f"RUL file length ({len(final_rul)})."
        )

    return trajectories, np.minimum(final_rul, 125.0)


class CMapssTestDataset:
    def __init__(
        self,
        trajectories: Sequence[Tuple[int, np.ndarray]],
        final_rul: np.ndarray,
        segment_window: int,
        n_segments: int,
        x_min: np.ndarray,
        x_max: np.ndarray,
    ) -> None:
        self.samples: List[Tuple[int, np.ndarray, np.ndarray, float]] = []
        self.seq_len = segment_window * n_segments

        if len(trajectories) != len(final_rul):
            raise ValueError("Test trajectories and final RUL length mismatch")

        for idx, (uid, sensors) in enumerate(trajectories):
            normalized = minmax_apply(sensors, x_min, x_max).astype(np.float32)
            if len(normalized) < self.seq_len:
                # Left-pad by repeating the first observed sample, so short
                # test trajectories still produce a full-length window.
                pad = self.seq_len - len(normalized)
                first = np.repeat(normalized[:1], pad, axis=0)
                x_window = np.concatenate([first, normalized], axis=0)
            else:
                x_window = normalized[-self.seq_len:]

            # Test trajectories are early-truncated (they don't run to actual
            # failure, unlike training trajectories), so the true total
            # lifetime is unknown. Using t=1.0 treats "the last observed
            # cycle" as the time input -- a reconstruction assumption the
            # paper does not explicitly specify for test-time normalization.
            current_cycle_fraction = 1.0

            self.samples.append((
                int(uid),
                x_window.astype(np.float32),
                np.array([current_cycle_fraction], dtype=np.float32),
                float(max(final_rul[idx], 1e-6)),
            ))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        return self.samples[index]


def evaluate_test_set(
    model: PCBNN,
    dataset: CMapssTestDataset,
    n_mc: int,
) -> Tuple[Dict[str, float], pd.DataFrame]:
    """
    Evaluates the complete C-MAPSS test set and returns both aggregate
    metrics and a row-by-row real-vs-predicted RUL table (one row per test
    engine, since C-MAPSS assigns one final RUL target per test trajectory).
    """
    rows: List[Dict[str, float]] = []

    for uid, x_np, t_np, y in dataset.samples:
        x = tf.convert_to_tensor(x_np[np.newaxis, ...], dtype=tf.float32)
        t = tf.convert_to_tensor(t_np[np.newaxis, ...], dtype=tf.float32)
        result = predict_with_uncertainty(model, x, t, n_samples=n_mc)

        pred = float(result["y_pred"][0])
        error = pred - y

        rows.append({
            "engine_id": uid,
            "real_rul": y,
            "predicted_rul": pred,
            "error": error,
            "absolute_error": abs(error),
            "ci_lower": float(result["ci_lower"][0]),
            "ci_upper": float(result["ci_upper"][0]),
            "ci_width": float(result["ci_upper"][0] - result["ci_lower"][0]),
            "std_total": float(result["std_total"][0]),
            "var_total": float(result["var_total"][0]),
            "var_aleatoric": float(result["var_aleatoric"][0]),
            "var_epistemic": float(result["var_epistemic"][0]),
        })

    results_df = pd.DataFrame(rows).sort_values("engine_id").reset_index(drop=True)

    y_pred = tf.constant(results_df["predicted_rul"].to_numpy(), dtype=tf.float32)
    y_true = tf.constant(results_df["real_rul"].to_numpy(), dtype=tf.float32)
    lower = tf.constant(results_df["ci_lower"].to_numpy(), dtype=tf.float32)
    upper = tf.constant(results_df["ci_upper"].to_numpy(), dtype=tf.float32)

    mae = float(tf.reduce_mean(tf.abs(y_pred - y_true)))
    max_abs_error = float(tf.reduce_max(tf.abs(y_pred - y_true)))

    metrics = {
        "RMSE": rmse_metric(y_pred, y_true),
        "MAE": mae,
        "MaxAbsoluteError": max_abs_error,
        "Score": score_metric(y_pred, y_true),
        "PICP_percent": picp_metric(y_true, lower, upper),
        "Mean_CI_width": float(tf.reduce_mean(upper - lower)),
        "Mean_aleatoric_variance": float(results_df["var_aleatoric"].mean()),
        "Mean_epistemic_variance": float(results_df["var_epistemic"].mean()),
        "Mean_total_variance": float(results_df["var_total"].mean()),
    }
    return metrics, results_df


def print_detailed_results(results_df: pd.DataFrame, metrics: Dict[str, float]) -> None:
    """Prints every test engine's real vs predicted RUL plus error information."""
    print("\n" + "=" * 118)
    print("REAL vs PREDICTED RUL — COMPLETE TEST SET")
    print("=" * 118)
    print(
        f"{'Engine':>8} {'Real RUL':>12} {'Predicted RUL':>16} {'Error':>12} "
        f"{'Abs Error':>12} {'95% CI Lower':>14} {'95% CI Upper':>14}"
    )
    print("-" * 118)
    for row in results_df.itertuples(index=False):
        print(
            f"{int(row.engine_id):>8} {row.real_rul:>12.2f} {row.predicted_rul:>16.2f} "
            f"{row.error:>12.2f} {row.absolute_error:>12.2f} "
            f"{row.ci_lower:>14.2f} {row.ci_upper:>14.2f}"
        )
    print("-" * 118)
    print(f"{'Number of test engines':>42}: {len(results_df)}")
    print(f"{'Mean absolute error (MAE)':>42}: {metrics['MAE']:.4f}")
    print(f"{'Maximum absolute error':>42}: {metrics['MaxAbsoluteError']:.4f}")
    print(f"{'Mean 95% CI width':>42}: {metrics['Mean_CI_width']:.4f}")
    print(f"{'Mean aleatoric variance':>42}: {metrics['Mean_aleatoric_variance']:.6f}")
    print(f"{'Mean epistemic variance':>42}: {metrics['Mean_epistemic_variance']:.6f}")
    print(f"{'Mean total variance':>42}: {metrics['Mean_total_variance']:.6f}")
    print(f"{'RMSE':>42}: {metrics['RMSE']:.4f}")
    print(f"{'Score':>42}: {metrics['Score']:.4f}")
    print(f"{'PICP':>42}: {metrics['PICP_percent']:.2f}%")
    print("=" * 118)


def save_detailed_results(results_df: pd.DataFrame, output_path: str | Path) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(path, index=False)
    return path


# ====================================================================================
# SECTION 15 — Main
# ====================================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Paper-faithful PCBNN reconstruction (TensorFlow)")
    parser.add_argument("--data-root", required=True, help="Path containing C-MAPSS files")
    parser.add_argument("--dataset", choices=["FD001", "FD002", "FD003", "FD004"], default="FD001")
    parser.add_argument(
        "--case", choices=["I", "II"], default="II",
        help="Case I: bearing-reconstruction settings; Case II: C-MAPSS settings (Table 2)",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--stride", type=int, default=5)
    parser.add_argument(
        "--hgrr-levels", type=int, default=None,
        help="Reconstruction choice; the exact number of levels is not specified in the paper.",
    )
    parser.add_argument(
        "--n-segments", type=int, default=None,
        help="Reconstruction choice; total sample length = segment_window*n_segments.",
    )
    parser.add_argument(
        "--mc-train", type=int, default=None,
        help="Monte Carlo samples per training batch. Paper text: 10.",
    )
    parser.add_argument(
        "--mc-test", type=int, default=50,
        help="Monte Carlo samples at test time. Paper text: 50.",
    )
    parser.add_argument("--results-csv", default=None,
                         help="Output CSV path. Defaults to results_<dataset>.csv.")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--eta", type=float, default=None)
    parser.add_argument("--max-beta", type=float, default=None)
    parser.add_argument("--max-power", type=float, default=None)
    parser.add_argument("--grad-clip", type=float, default=None)
    parser.add_argument("--patience", type=int, default=None,
                         help="Early-stopping patience in epochs. Default: no early stopping.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    case = copy.deepcopy(CASE_I if args.case == "I" else CASE_II)
    if args.hgrr_levels is not None:
        case.hgrr_levels = args.hgrr_levels
    if args.n_segments is not None:
        case.n_segments = args.n_segments
    if args.mc_train is not None:
        case.mc_samples = args.mc_train
    if args.batch_size is not None:
        case.batch_size = args.batch_size
    if args.lr is not None:
        case.learning_rate = args.lr
    if args.epochs is not None:
        case.epochs = args.epochs
    if args.eta is not None:
        case.eta = args.eta
    if args.max_beta is not None:
        case.max_beta = args.max_beta
    if args.max_power is not None:
        case.max_power = args.max_power
    if args.grad_clip is not None:
        case.grad_clip = args.grad_clip

    print(f"TensorFlow version: {tf.__version__}")
    print(f"Dataset: {args.dataset}")
    print(f"Segment window: {case.segment_window}")
    print(f"Segments per sample: {case.n_segments}")
    print(f"HGRR hierarchy levels: {case.hgrr_levels}")
    print(f"MC samples (train): {case.mc_samples}")
    print(f"MC samples (test): {args.mc_test}")
    print(f"Physics weight eta: {case.eta}")
    print(f"Batch size: {case.batch_size}")
    print(f"Learning rate: {case.learning_rate}")
    print(f"Max beta safeguard: {case.max_beta}")
    print(f"Max Weibull exponent safeguard: {case.max_power}")
    print(f"Gradient clip: {case.grad_clip}")

    all_train = load_cmapss_training_trajectories(args.data_root, args.dataset)
    train_traj, val_traj = split_trajectories(all_train, val_fraction=args.val_fraction, seed=args.seed)

    # Fit normalization only on the training engines, so validation
    # information never leaks into preprocessing statistics.
    x_min, x_max = minmax_fit(train_traj)

    train_ds = CMapssWindowDataset(
        trajectories=train_traj, segment_window=case.segment_window,
        n_segments=case.n_segments, stride=args.stride, x_min=x_min, x_max=x_max,
    )
    val_ds = CMapssWindowDataset(
        trajectories=val_traj, segment_window=case.segment_window,
        n_segments=case.n_segments, stride=args.stride, x_min=x_min, x_max=x_max,
    )

    if len(train_ds) == 0 or len(val_ds) == 0:
        raise RuntimeError(
            "No windows were generated. Reduce n_segments or increase the data window availability."
        )

    n_features = train_ds[0][0].shape[-1]
    print(f"Input sensor features after paper's sensor removal: {n_features}")
    print(f"Training windows: {len(train_ds)}")
    print(f"Validation windows: {len(val_ds)}")

    model = PCBNN(n_features=n_features, case=case)

    model = fit(
        model=model, train_dataset=train_ds, val_dataset=val_ds,
        case=case, patience=args.patience,
    )

    test_traj, test_rul = load_cmapss_test_trajectories(args.data_root, args.dataset)
    test_ds = CMapssTestDataset(
        trajectories=test_traj, final_rul=test_rul, segment_window=case.segment_window,
        n_segments=case.n_segments, x_min=x_min, x_max=x_max,
    )

    metrics, results_df = evaluate_test_set(model=model, dataset=test_ds, n_mc=args.mc_test)

    csv_path = args.results_csv or f"results_{args.dataset}.csv"
    saved_path = save_detailed_results(results_df, csv_path)

    print_detailed_results(results_df, metrics)
    print(f"\nDetailed results CSV: {saved_path.resolve()}")


if __name__ == "__main__":
    main()