# GPS_LSTM: Lightweight Deep Learning for GPS Position Prediction under Signal Loss

## Overview

This repository contains the code and experiments for our research on predicting the GPS position of a boat in scenarios where GPS signals are lost, such as when passing under a bridge. The goal is to develop a lightweight, highly accurate model capable of inferring the next GPS position with minimal error, even in challenging conditions.

## Motivation

Accurate GPS prediction is critical for applications where even small errors can have significant consequences. Signal loss, especially under bridges, poses a unique challenge. Our work addresses this by designing and evaluating deep learning models that can robustly predict the next position based on historical data.

## Models

We implement and compare several sequence models:

- **AttentionLSTM**: Our proposed model, combining LSTM layers with temporal attention and instance normalization (RevIN) for improved accuracy and robustness.
- **LSTM**: Standard LSTM-based sequence model, serving as a baseline.
- **GRU**: Standard GRU-based sequence model, serving as a baseline.

All models are designed to be lightweight for real-time or resource-constrained deployment.

## Evaluation Protocol

To ensure a fair and realistic assessment, we introduce a **Leave-One-River-Out** evaluation strategy:
- The dataset consists of GPS trajectories from three different rivers.
- In each experiment, one river is used as the test set, while the other two are used for training and validation.
- This process is repeated for each river, providing a robust estimate of model generalization to unseen environments.

## Results

Our experiments show that the proposed AttentionLSTM model consistently outperforms both LSTM and GRU baselines, achieving the highest accuracy in next-step GPS prediction across all evaluation splits.

## Repository Structure

```
models/         # Model definitions (AttentionLSTM, LSTM, GRU, RevIN)
exp/            # Experiment scripts for training and evaluation
data_provider/  # Data loading utilities
utils/          # Metrics and data preprocessing
scripts/        # Shell scripts for running experiments
run.py          # Main entry point (if applicable)
README.md       # Project documentation
```

## Getting Started

1. **Install dependencies**  
   Ensure you have Python 3.8+ and install required packages:
   ```powershell
   pip install torch numpy pandas
   ```

2. **Prepare data**  
   Place your river GPS datasets in the expected directory structure as referenced in `utils/read_data.py`.

3. **Run experiments**  
   Use the scripts in the `scripts/` directory or run experiment files directly, e.g.:
   ```powershell
   python run.py
   ```


