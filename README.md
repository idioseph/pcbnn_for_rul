# PCBNN: Physics-Constrained Bayesian Neural Network

A TensorFlow reconstruction of the Physics-Constrained Bayesian Neural Network (PCBNN) framework for machinery Remaining Useful Life (RUL) prediction and uncertainty quantification.

This implementation strictly adheres to the architecture and methodologies described in:

> _E. Wang et al., "A physics-constrained Bayesian neural network for machinery remaining useful life prediction and uncertainty quantification", DOI: 10.1016/j.ress.2025.111778._

## Architecture Overview

The PCBNN framework addresses the limitations of standard data-driven prognostics by integrating probabilistic reasoning and physical laws into a unified, end-to-end architecture[cite: 1]:

1. **Dual-Stage Neural Network (DSNN):**
   - **Segmented Bi-LSTM:** Partitions time-series sensor data to capture bidirectional, localized degradation features efficiently[cite: 1].
   - **Bayesian Hierarchical Gated Recurrent Regressor (HGRR):** Utilizes complex-valued temporal dynamics to model phase-coupled and frequency-dependent system behaviors[cite: 1].
2. **Weibull Likelihood Variational Inference:** Outputs the scale ($\alpha$) and shape ($\beta$) parameters of a Weibull distribution to accurately model the time-to-failure probability density, capturing both epistemic and aleatoric uncertainties[cite: 1].
3. **Deep Hidden Physics Model (deep-HPM):** Extracts underlying degradation mechanics by incorporating a data-driven partial differential equation (PDE) as a residual penalty in the loss function, enforcing physical consistency[cite: 1].

## Requirements

- Python 3.8+
- TensorFlow 2.x
- NumPy
- Pandas

Install dependencies via pip or `uv`:

```bash
pip install tensorflow numpy pandas
# or using uv:
uv add tensorflow numpy pandas

```

## Dataset Preparation

This repository includes preprocessing pipelines tuned for the **C-MAPSS** turbofan engine degradation dataset.

1. Download the C-MAPSS dataset (e.g., from the NASA Prognostics Data Repository or Kaggle's NASA Turbofan dataset mirror).
2. Ensure the text files (`train_FD001.txt`, `test_FD001.txt`, `RUL_FD001.txt`, etc.) are located within a designated directory (e.g., `./CMAPSSData/CMaps/`).

Note: The preprocessing script automatically drops uncorrelated sensors (1, 5, 10, 16, 18, 19) and applies a maximum RUL cap of 125, matching the literature's specifications.

## Execution

### Running Locally

To train and evaluate the model on the FD003 dataset using Case II configurations:

```bash
python main.py --data-root ./CMAPSSData --dataset FD003 --case II --mc-train 10 --mc-test 50 --batch-size 128

```

### Running on Kaggle (Recommended for GPU Acceleration)

Because the second-order partial derivative computations in `DeepHPM` are computationally intensive, execution on a free Kaggle GPU (P100 or T4) with background commits is ideal:

1. Create a Kaggle Notebook, set the accelerator to **GPU P100**, and enable **Internet**.
2. Attach the NASA C-MAPSS dataset.
3. Run the pipeline inside a notebook cell:

```bash
git clone [https://github.com/idioseph/pcbnn_for_rul](https://github.com/idioseph/pcbnn_for_rul)
%cd pcbnn_for_rul

curl -LsSf [https://astral.sh/uv/install.sh](https://astral.sh/uv/install.sh) | sh
export PATH="/root/.local/bin:$PATH"

uv run --with tensorflow main.py \
  --data-root /kaggle/input/datasets/behrad3d/nasa-cmaps/CMaps \
  --dataset FD003 \
  --case II \
  --mc-train 10 \
  --mc-test 50 \
  --batch-size 128 \
  --results-csv /kaggle/working/results_FD003.csv

```

### Command-Line Arguments

| Argument      | Description                                                             | Default      |
| ------------- | ----------------------------------------------------------------------- | ------------ |
| `--data-root` | Path to the directory containing C-MAPSS data files.                    | **Required** |
| `--dataset`   | Target subset to process (`FD001`, `FD002`, `FD003`, `FD004`).          | `FD001`      |
| `--case`      | Hyperparameter configurations (`I` for bearing data, `II` for C-MAPSS). |

| `II` |
| `--seed` | Random seed for reproducibility across TF and NumPy. | `0` |
| `--val-fraction` | Proportion of training trajectories reserved for validation. | `0.1` |
| `--stride` | Step size for the sliding window across engine histories. | `5` |
| `--mc-train` | Number of Monte Carlo samples used during Bayesian training.

| `10` |
| `--mc-test` | Number of Monte Carlo samples used for testing and UI bounds.

| `50` |
| `--results-csv` | Output file path for saving detailed prediction metrics. | `results_<dataset>.csv` |

## Output

Upon completion, the script outputs a comprehensive console report detailing the Real vs. Predicted RUL for every engine in the test set. It calculates predictive bounds (95% CI), decomposes variance (Aleatoric vs. Epistemic), and returns aggregate metrics (RMSE, MAE, Score, PICP). A detailed CSV of row-by-row engine metrics is saved to the specified path.
