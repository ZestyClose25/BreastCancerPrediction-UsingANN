# ANN from Scratch — Breast Cancer Classification

Binary classification on the **Breast Cancer Wisconsin** dataset using a neural network built entirely from scratch in NumPy, with a PyTorch implementation for comparison.

---

## Repository Structure

```
├── data_preprocessing.ipynb     # Load, clean, scale, split → export CSVs
├── ann_from_scratch.ipynb       # ANN built in pure NumPy (no frameworks)
├── ann_pytorch.ipynb            # Same ANN implemented using PyTorch
├── XTrain.csv                   # Training features  (455 × 30)
├── YTrain.csv                   # Training labels    (455 × 1)
├── XTest.csv                    # Test features      (114 × 30)
└── YTest.csv                    # Test labels        (114 × 1)
```

---

## Dataset

**Breast Cancer Wisconsin** — available via `sklearn.datasets.load_breast_cancer`

| Property | Value |
|---|---|
| Samples | 569 |
| Features | 30 (cell nucleus measurements) |
| Classes | 2 — Malignant (0), Benign (1) |
| Train / Test split | 80% / 20% |

---

## Notebooks

### 1. `data_preprocessing.ipynb`

Prepares the raw dataset for training.

- Loads the Breast Cancer Wisconsin dataset
- Separates features `X` and labels `y`
- Removes null / missing values
- Applies `StandardScaler` — normalizes features to mean = 0, std = 1
- Splits into train (80%) and test (20%) sets
- Exports four CSV files: `XTrain.csv`, `YTrain.csv`, `XTest.csv`, `YTest.csv`

> Run this notebook first before either of the model notebooks.

---

### 2. `ann_from_scratch.ipynb`

A fully hand-coded neural network — **no PyTorch, no Keras, no autograd**. Every forward pass, backward pass, and weight update is written explicitly in NumPy.

**Architecture**

```
Input (30) → Hidden (32, ReLU) → Output (1, Sigmoid)
```

**What's implemented from scratch**

- `Linear` layer — forward pass (`X @ W + b`) and backward pass (`dW`, `db`, `dX`)
- `ReLU` activation — forward and backward with mask caching
- `Sigmoid` activation — forward and backward
- Binary Cross-Entropy loss and its gradient
- Xavier weight initialization
- Mini-batch Stochastic Gradient Descent (batch size = 32)

**Key concepts demonstrated**

- Why weights are shaped `(in_features, out_features)`
- Why `self._X` is cached during forward for use in backward
- How `dW = X.T @ dout`, `db = dout.sum(axis=0)`, `dX = dout @ W.T` are derived from chain rule
- Why gradients always have the same shape as their parameters
- How the training loop: shuffle → slice batches → forward → loss → backward → update

**Results**

| Metric | Value |
|---|---|
| Test Accuracy | ~96.5% |
| Precision | ~0.97 |
| Recall | ~0.99 |
| F1 Score | ~0.98 |

---

### 3. `ann_pytorch.ipynb`

The same network architecture implemented using **PyTorch** — `nn.Linear`, `nn.ReLU`, `nn.Sigmoid`, `BCELoss`, and `torch.optim.SGD`.

Useful for comparing:
- How much PyTorch abstracts away vs the from-scratch version
- That the results are equivalent — same architecture, same accuracy

---

## How to Run

**1. Install dependencies**

```bash
pip install numpy pandas scikit-learn matplotlib torch jupyter
```

**2. Run in order**

```
1. data_preprocessing.ipynb   → generates the four CSV files
2. ann_from_scratch.ipynb     → trains the NumPy ANN
3. ann_pytorch.ipynb          → trains the PyTorch ANN
```

> Make sure all notebooks and CSV files are in the same directory.

---

## Architecture Details

```
Input layer     : 30 neurons  (one per feature)
Hidden layer    : 32 neurons  + ReLU activation
Output layer    :  1 neuron   + Sigmoid activation → probability ∈ (0, 1)

Loss function   : Binary Cross-Entropy
Optimizer       : Mini-batch SGD  (lr = 0.01, batch size = 32)
Epochs          : 500
Weight init     : Xavier (scale = sqrt(2 / in_features))
```

---

## What Makes This Different from a Framework

When you use PyTorch or Keras, the following happen automatically. In `ann_from_scratch.ipynb`, every one of these is written explicitly:

| What happens | PyTorch | From scratch |
|---|---|---|
| Forward pass | `model(X)` | `z1 = X @ W1 + b1` → `a1 = relu(z1)` → ... |
| Gradient computation | `loss.backward()` | `dW = X.T @ dout`, `db = dout.sum(0)`, ... |
| Weight update | `optimizer.step()` | `W -= lr * dW` |
| Gradient zeroing | `optimizer.zero_grad()` | not needed (overwritten each batch) |

---

## Dependencies

```
numpy
pandas
scikit-learn
matplotlib
torch
jupyter
```
