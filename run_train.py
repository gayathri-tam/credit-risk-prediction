#!/usr/bin/env python3
"""
Entry point for model training.
Run this script explicitly to train the model. Training is NEVER invoked by the Streamlit app.
"""
from ml.train_model import train

if __name__ == "__main__":
    train(sample_frac=0.3, model_type="logistic")
