import os
import sys
import json
import time
from pathlib import Path
import cv2
import numpy as np

# Ensure stdout flushes immediately
sys.stdout.reconfigure(line_buffering=True)

BASE_DIR = Path(__file__).resolve().parent.parent
DATASET_DIR = BASE_DIR / "dataset"
MODELS_DIR = BASE_DIR / "models"
MODEL_FILE = MODELS_DIR / "ir_led_model.json"

FEATURE_NAMES = [
    "peak_brightness",
    "core_purple_excess",
    "core_spectral_ratio",
    "halo_purple_excess",
    "halo_spectral_ratio",
    "blue_green_diff",
    "red_green_diff",
    "halo_b_g_diff",
    "halo_r_g_diff",
    "core_to_halo_ratio",
    "aspect_ratio",
    "circularity"
]


def extract_features_from_image(img_bgr: np.ndarray) -> np.ndarray:
    """
    Extracts the 12-dimensional spectral, chromatic, and morphological feature vector
    from an image or candidate patch.
    """
    if img_bgr is None or img_bgr.size == 0:
        return None

    h, w = img_bgr.shape[:2]
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    peak = float(np.max(gray))

    # Core region: top 15% intensity
    core_thresh = max(40.0, peak * 0.82)
    core_mask = gray >= core_thresh
    halo_mask = (gray >= peak * 0.25) & (~core_mask)

    if np.sum(core_mask) == 0:
        core_mask = gray >= (peak * 0.5)

    b = img_bgr[:, :, 0].astype(np.float32)
    g = img_bgr[:, :, 1].astype(np.float32)
    r = img_bgr[:, :, 2].astype(np.float32)

    # Core chromaticity
    c_b = float(np.mean(b[core_mask]))
    c_g = float(np.mean(g[core_mask]))
    c_r = float(np.mean(r[core_mask]))

    c_excess = (c_b - c_g) + (c_r - c_g)
    c_ratio = (c_b + c_r) / (2.0 * max(c_g, 1.0))
    bg_diff = c_b - c_g
    rg_diff = c_r - c_g

    # Halo chromaticity (epoxy body glow)
    if np.sum(halo_mask) > 0:
        h_b = float(np.mean(b[halo_mask]))
        h_g = float(np.mean(g[halo_mask]))
        h_r = float(np.mean(r[halo_mask]))
        h_int = float(np.mean(gray[halo_mask]))
    else:
        h_b, h_g, h_r = c_b, c_g, c_r
        h_int = c_g

    h_excess = (h_b - h_g) + (h_r - h_g)
    h_ratio = (h_b + h_r) / (2.0 * max(h_g, 1.0))
    h_bg_diff = h_b - h_g
    h_rg_diff = h_r - h_g

    c_int = float(np.mean(gray[core_mask]))
    core_to_halo = (c_int + 1.0) / (h_int + 1.0)

    # Morphology: shape of the emission blob
    _, bin_mask = cv2.threshold(gray, int(peak * 0.45), 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(bin_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        c_largest = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(c_largest)
        peri = cv2.arcLength(c_largest, True)
        circ = (4.0 * np.pi * area) / (peri * peri) if peri > 0 else 0.5
        _, _, bw, bh = cv2.boundingRect(c_largest)
        ar = float(bw) / max(float(bh), 1.0)
    else:
        circ = 0.5
        ar = 1.0

    return np.array([
        peak,
        c_excess,
        c_ratio,
        h_excess,
        h_ratio,
        bg_diff,
        rg_diff,
        h_bg_diff,
        h_rg_diff,
        core_to_halo,
        ar,
        circ
    ], dtype=np.float32)


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -30.0, 30.0)))


def train_logistic_regression(X: np.ndarray, y: np.ndarray, lr: float = 0.05, epochs: int = 1500, reg: float = 0.01):
    """
    Trains a Logistic Regression model with L2 regularization and balanced class weights.
    """
    n_samples, n_features = X.shape

    # Compute class weights to handle any imbalance
    n_pos = np.sum(y == 1)
    n_neg = np.sum(y == 0)
    w_pos = (n_samples) / (2.0 * max(n_pos, 1))
    w_neg = (n_samples) / (2.0 * max(n_neg, 1))
    sample_weights = np.where(y == 1, w_pos, w_neg)

    weights = np.zeros(n_features, dtype=np.float32)
    bias = 0.0

    for epoch in range(epochs):
        linear = np.dot(X, weights) + bias
        preds = sigmoid(linear)

        errors = (preds - y) * sample_weights
        dw = (np.dot(X.T, errors) / n_samples) + (reg * weights)
        db = np.sum(errors) / n_samples

        weights -= lr * dw
        bias -= lr * db

    return weights, bias


def build_and_train():
    print("=" * 60)
    print("SMART NOTEBOOK — TRAIN IR LED DETECTION MODEL")
    print("=" * 60)

    # 1. Collect features from dataset
    X_list = []
    y_list = []
    names = []

    # Directories to search (both core dataset and captured camera dataset)
    captured_dir = BASE_DIR / "dataset_captured"

    def collect_files(cat_name):
        seen = set()
        out = []
        for parent in [DATASET_DIR / cat_name, captured_dir / cat_name]:
            if parent.exists():
                for f in list(parent.glob("*.jpg")) + list(parent.glob("*.png")):
                    if f.name not in seen:
                        seen.add(f.name)
                        out.append(f)
        return out

    # Positive: Active IR LED bulb emitting purple IR light
    pos_files = collect_files("ir_led_emitting")
    print(f"[Dataset] Loading & processing {len(pos_files)} Emitting IR LED images (Class 1)...")
    for idx, f in enumerate(pos_files):
        im = cv2.imread(str(f))
        feat = extract_features_from_image(im)
        if feat is not None:
            X_list.append(feat)
            y_list.append(1.0)
            names.append(f.name)
        if (idx + 1) % 150 == 0:
            print(f"  Processed {idx + 1}/{len(pos_files)} emitting frames...")

    # Negative 1: Unlit 5mm LED bulb (Class 0)
    off_files = collect_files("ir_bulb_off")
    print(f"[Dataset] Loading & processing {len(off_files)} Unlit 5mm LED bulb images (Class 0)...")
    for idx, f in enumerate(off_files):
        im = cv2.imread(str(f))
        feat = extract_features_from_image(im)
        if feat is not None:
            X_list.append(feat)
            y_list.append(0.0)
            names.append(f.name)
        if (idx + 1) % 150 == 0:
            print(f"  Processed {idx + 1}/{len(off_files)} unlit bulb frames...")

    # Negative 2: Other light sources / whiteboard drawings / lamps (Class 0)
    other_files = collect_files("other_lights")
    print(f"[Dataset] Loading & processing {len(other_files)} Non-IR Light images (Class 0)...")
    for idx, f in enumerate(other_files):
        im = cv2.imread(str(f))
        feat = extract_features_from_image(im)
        if feat is not None:
            X_list.append(feat)
            y_list.append(0.0)
            names.append(f.name)
        if (idx + 1) % 150 == 0:
            print(f"  Processed {idx + 1}/{len(other_files)} other light frames...")

    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.float32)

    total_samples = len(X)
    pos_count = int(np.sum(y == 1))
    neg_count = int(np.sum(y == 0))
    print(f"\n[Training Data Summary] Total Samples: {total_samples} (Positive: {pos_count}, Negative: {neg_count})")

    # 2. Compute Mean & Std for standardization
    mean = np.mean(X, axis=0)
    std = np.std(X, axis=0)
    std = np.where(std < 1e-5, 1.0, std)  # Avoid division by zero

    X_scaled = (X - mean) / std

    # 3. Train Model
    weights, bias = train_logistic_regression(X_scaled, y, lr=0.08, epochs=2500, reg=0.01)

    # 4. Evaluate on Training Data
    linear = np.dot(X_scaled, weights) + bias
    probabilities = sigmoid(linear)
    predictions = (probabilities >= 0.50).astype(np.float32)

    accuracy = float(np.mean(predictions == y) * 100.0)
    tp = int(np.sum((predictions == 1) & (y == 1)))
    fp = int(np.sum((predictions == 1) & (y == 0)))
    tn = int(np.sum((predictions == 0) & (y == 0)))
    fn = int(np.sum((predictions == 0) & (y == 1)))

    precision = float(tp / (tp + fp)) * 100.0 if (tp + fp) > 0 else 0.0
    recall = float(tp / (tp + fn)) * 100.0 if (tp + fn) > 0 else 0.0
    f1 = float(2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

    print("\n" + "=" * 60)
    print("TRAINING EVALUATION METRICS:")
    print(f"  - Accuracy  : {accuracy:.2f}%")
    print(f"  - Precision : {precision:.2f}% (False Positive Rate: {fp}/{neg_count})")
    print(f"  - Recall    : {recall:.2f}% (False Negative Rate: {fn}/{pos_count})")
    print(f"  - F1 Score  : {f1:.2f}%")
    print(f"  - Confusion Matrix : TP={tp}, FP={fp}, TN={tn}, FN={fn}")
    print("=" * 60)

    # Top feature weights
    print("\n[Feature Importance / Weights]:")
    feat_importance = sorted(zip(FEATURE_NAMES, weights), key=lambda x: abs(x[1]), reverse=True)
    for name, w_val in feat_importance:
        sign = "+" if w_val >= 0 else "-"
        print(f"  {sign} {name:22s} : {w_val:+.4f}")

    # 5. Export Model to JSON
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    model_data = {
        "model_name": "IR_LED_Bulb_Classifier",
        "version": "1.0.0",
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "total_training_samples": total_samples,
        "positive_samples": pos_count,
        "negative_samples": neg_count,
        "accuracy_pct": accuracy,
        "precision_pct": precision,
        "recall_pct": recall,
        "f1_pct": f1,
        "threshold": 0.50,
        "features": FEATURE_NAMES,
        "mean": mean.tolist(),
        "std": std.tolist(),
        "weights": weights.tolist(),
        "bias": float(bias)
    }

    with open(MODEL_FILE, "w", encoding="utf-8") as f:
        json.dump(model_data, f, indent=2)

    print(f"\n[Model Exported] Successfully saved trained IR LED model to: {MODEL_FILE}")


if __name__ == "__main__":
    build_and_train()
