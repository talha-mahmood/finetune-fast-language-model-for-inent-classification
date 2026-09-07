
import json
import re
import unicodedata

import fasttext
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix,
)


# ============================================================
# 1. Configuration
# ============================================================

MODEL_PATH = "intent_model_v2.bin"
CONFIG_PATH = "model_v2_config.json"

# Your fellow developer's dataset
DATASET_PATH = "tabinda_prediction_logs_truth_fixed.csv"

# Output dataset
OUTPUT_PATH = "tabinda_model_predictions_v2.csv"


# ============================================================
# 2. Load model
# ============================================================

print("=" * 60)
print("Loading model...")
print("=" * 60)

model = fasttext.load_model(MODEL_PATH)

print(f"Model loaded: {MODEL_PATH}")


# ============================================================
# 3. Load model configuration
# ============================================================

try:
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        config = json.load(f)

    CONFIDENCE_THRESHOLD = float(
        config.get("confidence_threshold", 0.5)
    )

    LABELS = config.get("labels", [])

except FileNotFoundError:
    print(
        f"WARNING: {CONFIG_PATH} not found."
    )

    CONFIDENCE_THRESHOLD = 0.5
    LABELS = []


print(f"Confidence threshold: {CONFIDENCE_THRESHOLD}")

if LABELS:
    print(f"Number of intents: {len(LABELS)}")


# ============================================================
# 4. Same normalization used during training
# ============================================================

AR_DIACRITICS = re.compile(
    r"[\u0610-\u061A\u064B-\u065F\u0670\u06D6-\u06ED]"
)

ZERO_WIDTH = re.compile(
    r"[\u200B-\u200F\u202A-\u202E\u2066-\u2069\uFEFF]"
)

TATWEEL = "\u0640"

DIGIT_MAP = {}

for i in range(10):
    DIGIT_MAP[ord("\u0660") + i] = str(i)
    DIGIT_MAP[ord("\u06F0") + i] = str(i)


VARIANT_MAP = str.maketrans({
    "\u0623": "\u0627",
    "\u0625": "\u0627",
    "\u0622": "\u0627",
    "\u064A": "\u06CC",
    "\u06D2": "\u06CC",
    "\u0643": "\u06A9",
    "\u0647": "\u06C1",
    "\u06C3": "\u06C1",
    "\u0629": "\u06C1",
})


KEEP_CHARS = set("'&")


def normalize(text: str) -> str:
    """
    Apply the SAME normalization used during model training.
    """

    t = unicodedata.normalize(
        "NFKC",
        str(text)
    ).strip().lower()

    # Remove zero-width characters
    t = ZERO_WIDTH.sub("", t)

    # Remove Arabic/Urdu diacritics
    t = AR_DIACRITICS.sub("", t)

    # Remove tatweel
    t = t.replace(TATWEEL, "")

    # Convert Arabic/Persian digits to normal digits
    t = t.translate(DIGIT_MAP)

    # Normalize Arabic/Urdu character variants
    t = t.translate(VARIANT_MAP)

    # Keep letters, numbers and selected characters
    t = "".join(
        c
        if (
            unicodedata.category(c)[0] in "LMN"
            or c in KEEP_CHARS
        )
        else " "
        for c in t
    )

    # Remove extra spaces
    return re.sub(r"\s+", " ", t).strip()


# ============================================================
# 5. Load evaluation dataset
# ============================================================

print()
print("=" * 60)
print("Loading evaluation dataset...")
print("=" * 60)

df = pd.read_csv(DATASET_PATH)

print(f"Dataset: {DATASET_PATH}")
print(f"Total rows: {len(df)}")
print(f"Columns: {list(df.columns)}")


# ============================================================
# 6. Validate dataset
# ============================================================

required_columns = ["Query", "Ground_Truth"]

for column in required_columns:
    if column not in df.columns:
        raise ValueError(
            f"Required column '{column}' not found in dataset."
        )


# Remove rows with missing values
df = df.dropna(
    subset=["Query", "Ground_Truth"]
).copy()

df["Query"] = df["Query"].astype(str).str.strip()
df["Ground_Truth"] = (
    df["Ground_Truth"]
    .astype(str)
    .str.strip()
)

df = df[
    (df["Query"] != "")
    & (df["Ground_Truth"] != "")
].reset_index(drop=True)


print(f"Rows after cleaning: {len(df)}")


# ============================================================
# 7. Normalize queries
# ============================================================

print()
print("Normalizing queries...")

df["Normalized_Query"] = df["Query"].apply(normalize)


# ============================================================
# 8. Predict intents
# ============================================================

print()
print("=" * 60)
print("Running predictions...")
print("=" * 60)

predicted_intents = []
confidence_scores = []
statuses = []

for query in df["Normalized_Query"]:

    # Handle empty normalized query
    if not query:
        predicted_intents.append(None)
        confidence_scores.append(0.0)
        statuses.append("invalid")
        continue

    labels, probabilities = model.predict(
        query,
        k=1
    )

    intent = labels[0].replace(
        "__label__",
        ""
    )

    confidence = float(probabilities[0])

    # Apply confidence threshold
    if confidence >= CONFIDENCE_THRESHOLD:
        final_intent = intent
        status = "confident"
    else:
        final_intent = None
        status = "uncertain"

    predicted_intents.append(final_intent)
    confidence_scores.append(confidence)
    statuses.append(status)


# ============================================================
# 9. Add predictions to dataframe
# ============================================================

df["Predicted_Intent"] = predicted_intents
df["Confidence"] = confidence_scores
df["Status"] = statuses


# ============================================================
# 10. Determine whether prediction is correct
# ============================================================

df["Correct"] = (
    df["Predicted_Intent"]
    == df["Ground_Truth"]
)


# ============================================================
# 11. Calculate accuracy
# ============================================================

y_true = df["Ground_Truth"]

# For accuracy, an uncertain prediction is treated
# as incorrect.
y_pred = df["Predicted_Intent"].fillna(
    "__UNCERTAIN__"
)

accuracy = accuracy_score(
    y_true,
    y_pred
)


# ============================================================
# 12. Calculate Macro F1
# ============================================================

macro_f1 = f1_score(
    y_true,
    y_pred,
    average="macro",
    zero_division=0
)


# ============================================================
# 13. Print overall results
# ============================================================

print()
print("=" * 60)
print("EVALUATION RESULTS")
print("=" * 60)

print(f"Total queries       : {len(df)}")
print(
    f"Correct predictions : {df['Correct'].sum()}"
)
print(
    f"Incorrect predictions: {(~df['Correct']).sum()}"
)

print(
    f"Accuracy            : {accuracy:.4f}"
)

print(
    f"Accuracy (%)        : {accuracy * 100:.2f}%"
)

print(
    f"Macro F1            : {macro_f1:.4f}"
)


# ============================================================
# 14. Confidence statistics
# ============================================================

print()
print("=" * 60)
print("CONFIDENCE STATISTICS")
print("=" * 60)

print(
    f"Average confidence  : "
    f"{df['Confidence'].mean():.4f}"
)

print(
    f"Minimum confidence  : "
    f"{df['Confidence'].min():.4f}"
)

print(
    f"Maximum confidence  : "
    f"{df['Confidence'].max():.4f}"
)

uncertain_count = (
    df["Status"] == "uncertain"
).sum()

print(
    f"Uncertain predictions: {uncertain_count}"
)


# ============================================================
# 15. Classification report
# ============================================================

print()
print("=" * 60)
print("CLASSIFICATION REPORT")
print("=" * 60)

print(
    classification_report(
        y_true,
        y_pred,
        digits=4,
        zero_division=0
    )
)


# ============================================================
# 16. Per-intent accuracy
# ============================================================

print()
print("=" * 60)
print("PER-INTENT ACCURACY")
print("=" * 60)

per_intent = (
    df.groupby("Ground_Truth")
    .agg(
        total_queries=("Ground_Truth", "size"),
        correct=("Correct", "sum")
    )
)

per_intent["accuracy"] = (
    per_intent["correct"]
    / per_intent["total_queries"]
)

per_intent = per_intent.sort_values(
    "accuracy"
)

print(per_intent.to_string())


# ============================================================
# 17. Save detailed prediction dataset
# ============================================================

output_df = df[
    [
        "Query",
        "Predicted_Intent",
        "Confidence",
        "Ground_Truth",
        "Correct",
        "Status",
    ]
].copy()

output_df.to_csv(
    OUTPUT_PATH,
    index=False,
    encoding="utf-8-sig"
)


print()
print("=" * 60)
print("OUTPUT SAVED")
print("=" * 60)

print(
    f"Prediction dataset saved to:"
)
print(
    OUTPUT_PATH
)


# ============================================================
# 18. Show incorrect predictions
# ============================================================

incorrect = output_df[
    output_df["Correct"] == False
]

print()
print("=" * 60)
print(
    f"INCORRECT PREDICTIONS ({len(incorrect)})"
)
print("=" * 60)

if len(incorrect) > 0:

    print(
        incorrect[
            [
                "Query",
                "Predicted_Intent",
                "Confidence",
                "Ground_Truth",
            ]
        ].to_string(index=False)
    )

else:
    print("No incorrect predictions!")

