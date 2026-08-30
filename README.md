# PCBNN — Physics-Constrained Bayesian Neural Network

PyTorch implementation of:

> Wang, Lei, Wen, Liu, Su, Zhang, Chen. *"A physics-constrained Bayesian
> neural network for machinery remaining useful life prediction and
> uncertainty quantification."* Reliability Engineering and System Safety,
> 266 (2026) 111778.

## Files → paper sections

| File | Paper section / equations | Contents |
|---|---|---|
| `bayesian_layers.py` | §3.1.2, §3.1.4 (Eqs. 32–37) | `BayesianLinear`: mean-field Gaussian variational weights (Bayes-by-Backprop), reparameterization trick, analytic KL vs. a standard-Gaussian prior |
| `segmented_bilstm.py` | §3.1.1-A (Eqs. 9–14) | Segmented Bi-LSTM: partitions a run-to-failure sequence into `d` segments, projects each, and runs a bidirectional LSTM to produce `h_BL` |
| `hgrr.py` | §3.1.1-B, §3.1.2 (Eqs. 15–27) | `BayesianHGRR`: GLU gate, complex-valued gated recurrence (`eps_t * exp(iθ_t)`), LayerNorm output projection, and the softplus regressor head producing Weibull `(alpha, beta)` |
| `deep_hpm.py` | §3.1.3 (Eqs. 28–31), Fig. 5 | `DeepHPM`: computes `∂rul/∂t`, `∂rul/∂h_BL`, `∂²rul/∂h_BL²` via autograd and feeds them to a small MLP (`NonlinearDynamicNetwork` = the learned `G(·)`) to form the physics residual `f` |
| `losses.py` | §3.1.4 (Eq. 27, 34, 35, 37) | Weibull NLL, ELBO loss (`KL + NLL`), and the total loss `L_ELBO + η·L_residual` |
| `model.py` | Fig. 2 | `PCBNN`: wires the encoder → HGRR → deep-HPM together |
| `train.py` | §4.1, Algorithm 1 (steps 1–7) | Adam + step-decay LR, early stopping on validation RMSE, Monte Carlo batch training |
| `predict.py` | §3.2, Algorithm 1 (steps 8–15), Eqs. 38–44 | MC sampling of `(alpha_m, beta_m)`, point prediction, aleatoric/epistemic variance decomposition, 95% CI, survival curve `S(t) = 1 − F_CDF(t)` |
| `data_utils.py` | §4.3.2 / §4.4.2 | Windowing dataset + a *synthetic* run-to-failure generator so the pipeline can be smoke-tested without downloading C-MAPSS / bearing data |
| `example.py` | — | End-to-end runnable demo (training + prediction + survival curve) |

## Quick start

```bash
pip install torch
python example.py
```

This trains on synthetic engines for a few epochs and prints a predicted
RUL, its 95% CI, and its aleatoric/epistemic variance split — just to
verify every piece (segmentation → complex recurrence → autograd PDE
residual → ELBO → uncertainty decomposition) is wired correctly.
**It is a plumbing smoke test, not a reproduction of the paper's reported
accuracy** — real results require the actual C-MAPSS / bearing datasets
and the paper's full training budget (§4.1, Table 2: 100–250 epochs,
10 MC training samples, η≈100, batch size 64–128).

## Using real data (C-MAPSS / bearings)

Replace `make_synthetic_engines()` with a loader that:
1. Reads `train_FD00X.txt` (C-MAPSS) or the bearing vibration CSVs.
2. Drops low-correlation sensors (paper excludes sensors 1, 5, 10, 16, 18,
   19 for FD001/FD003 — see Fig. 14; redo the correlation analysis per subset).
3. Min-max normalizes sensor channels (fit on train, apply to test).
4. Builds RUL labels:
   - C-MAPSS: piecewise-linear, capped at 125 (`Eq. 49`).
   - Bearings: `RUL = 1` before the fault-occurrence time (FOT, via a
     time-varying 3σ criterion on RMS vibration) then linear decay to 0
     (`Eq. 48`).
5. Feeds windows of raw sensor readings into `RULWindowDataset` (or a
   variant of it) exactly like the synthetic path.

Then set `PCBNN(n_features=..., window=..., ...)` and `fit(...)` using the
per-dataset hyperparameters in the paper's Table 2 (learning rate, decay
schedule, epochs, batch size, PDE derivative order, segment window size).

## Design notes / deviations worth knowing about

- **Which weights are Bayesian.** The paper states network weights are
  "treated as variational parameters" broadly; here the `BayesianLinear`
  treatment is applied to the HGRR (gates, recurrence, regressor head) —
  the component that directly parameterizes the Weibull likelihood and
  the piece the paper's Fig. 2 draws the prior/posterior/ELBO box around.
  The segmented Bi-LSTM encoder is a deterministic feature extractor. If
  you want a fully Bayesian network, swap `nn.LSTM`/`nn.Linear` inside
  `segmented_bilstm.py` for Bayesian equivalents too — `bayesian_layers.py`
  gives you the primitive.
- **HGRU recurrence.** Eqs. (17)–(19) sketch a generic real-valued gated
  update, and Eqs. (20)–(22) then respecify the forget gate as a complex
  rotation. The paper doesn't give a single fully consistent equation set
  for both together, so `hgrr.py` implements the self-consistent version
  described by (20)–(22): `h_t = eps_t·exp(iθ_t)·h_{t-1} + (1-eps_t)·c_t`,
  with `eps_t` bounded to `[κ, 1)` by a learnable lower bound κ, matching
  the text ("adjusted by a learned lower bound κ").
- **Physics residual loss.** Eq. (31) writes `L_residual = mean(f(h_BL,t))`;
  since `f` is a signed residual, the sensible regularizer (and what makes
  gradients well-behaved) is the **mean squared** residual, which is what
  `deep_hpm.py`'s `residual_loss()` computes.
- **Beta floor.** `Var[Weibull] ∝ Γ(1+2/β)`, which diverges as `β → 0`. An
  un-floored shape parameter makes early/undertrained predictions blow up
  numerically. `hgrr.py` floors `β` at `0.1` (a standard safeguard); this
  has negligible effect once training pushes `β` into a sensible range.
- **KL weighting.** Eq. (34)'s ELBO sums a per-batch KL against a
  dataset-summed NLL; in practice the KL term needs a small multiplicative
  weight (`kl_weight` in `losses.py`/`train.py`) so it doesn't dominate the
  likelihood term on small mini-batches — a standard Bayes-by-Backprop
  adjustment, tune per dataset size.

## Reproducing Table 3 / Table 6 style comparisons

`predict.py`'s `predict_with_uncertainty()` returns everything needed for
the paper's evaluation metrics:
- **RMSE / Score (Eqs. 45–46):** compute directly from `y_pred` vs. true RUL.
- **PICP (Eq. 47):** fraction of test points where `ci_lower <= y_true <= ci_upper`.
- **PDF / survival plots (Figs. 11–13, 16–18):** use `alpha_samples`,
  `beta_samples` with `predict.weibull_pdf` and `predict.survival_probability`.
# pcbnn_for_rul
