#!/usr/bin/env python3

import os
import re
import joblib
import pandas as pd
from sklearn.metrics import accuracy_score
import logging
import traceback

VAL_DATA_PATH = "examples/saves/val_data.joblib"
MODEL_DIR = "examples/saves"
CHUNK_PATTERN = re.compile(r"heuristic_chunk_(\d+)\.joblib")

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)

def load_validation_data():
    logging.info(f"🔍 Loading validation data from: {VAL_DATA_PATH}")
    val = joblib.load(VAL_DATA_PATH)
    X_val = val["X_val"]
    y_val = val["y_val"]
    logging.info(f"✅ Loaded validation set with {len(X_val)} rows")
    return X_val, y_val

def load_models():
    logging.info(f"🔍 Scanning directory: {MODEL_DIR}")
    models = []
    for fname in sorted(os.listdir(MODEL_DIR)):
        match = CHUNK_PATTERN.match(fname)
        if match:
            chunk_id = int(match.group(1))
            path = os.path.join(MODEL_DIR, fname)
            logging.info(f"🧠 Found model chunk: {fname} (chunk_id={chunk_id})")
            models.append((chunk_id, path))
    if not models:
        logging.warning("⚠️ No chunk models found in the saves directory.")
    return sorted(models)

def get_model_metadata(model):
    return {
        "prompt_len": getattr(model, "prompt_length", None)
    }

def main():
    logging.info("🚀 Starting chunk validation")

    X_val, y_val = load_validation_data()
    models = load_models()

    results = []

    for chunk_id, path in models:
        logging.info(f"\n📦 Loading model chunk {chunk_id} from {path}")
        try:
            model = joblib.load(path)
            logging.info(f"✅ Model loaded successfully (type: {type(model).__name__})")

            metadata = get_model_metadata(model)
            for k, v in metadata.items():
                if v is not None:
                    logging.info(f"📊 {k}: {v}")

            logging.info("🧪 Predicting on validation set...")
            y_pred = model.predict(X_val)
            acc = accuracy_score(y_val, y_pred)
            logging.info(f"✅ Accuracy for chunk {chunk_id}: {acc:.4f}")

            result = {"chunk": chunk_id, "accuracy": acc, **metadata}

        except Exception as e:
            logging.error(f"❌ Failed to validate chunk {chunk_id}: {e}")
            traceback.print_exc()
            result = {
                "chunk": chunk_id,
                "accuracy": None,
                "error": str(e)
            }

        results.append(result)

    df = pd.DataFrame(results).sort_values("chunk")

    # Pretty output
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", None)

    print("\n📊 Chunk Validation Summary:\n")
    print(df.round(4).to_string(index=False))

if __name__ == "__main__":
    main()
