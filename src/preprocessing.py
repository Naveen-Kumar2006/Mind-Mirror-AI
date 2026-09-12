from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


# ==========================================
# Paths
# ==========================================

ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT_DIR / "data" / "fer2013.csv"


# ==========================================
# Load Dataset
# ==========================================

def load_data():

    df = pd.read_csv(DATA_PATH)

    X = []

    for pixels in df["pixels"]:

        image = np.array(
            pixels.split(),
            dtype="float32"
        ).reshape(48, 48)

        X.append(image)

    X = np.array(X)

    y = df["emotion"].values

    print(f"Image Shape : {X.shape}")
    print(f"Label Shape : {y.shape}")

    return X, y


# ==========================================
# Preprocess Images
# ==========================================

def preprocess_images(X):

    X = X / 255.0

    X = X.reshape(-1, 48, 48, 1)

    print(f"Processed Image Shape : {X.shape}")

    return X


# ==========================================
# Split Dataset
# ==========================================

def split_data(X, y):

    # ==========================================
    # First Split
    # ==========================================


    X_temp, X_test, y_temp, y_test = train_test_split(
        X,
        y,
        test_size=0.10,
        random_state=42,
        stratify=y
    )


    # ==========================================
    # Second Split
    # ==========================================

    X_train, X_val, y_train, y_val = train_test_split(
        X_temp,
        y_temp,
        test_size=0.1111,
        random_state=42,
        stratify=y_temp
    )


    print("\nDataset Split:")

    print(f"Training Images   : {X_train.shape}")
    print(f"Validation Images : {X_val.shape}")
    print(f"Testing Images    : {X_test.shape}")


    return (
        X_train,
        X_val,
        X_test,
        y_train,
        y_val,
        y_test
    )


# ==========================================
# Encode Labels
# ==========================================

def encode_labels(y_train, y_val, y_test):
    y_train = np.eye(7)[y_train]

    y_val = np.eye(7)[y_val]

    y_test = np.eye(7)[y_test]


    return (
        y_train,
        y_val,
        y_test
    )