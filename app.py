
import json
import re
import unicodedata

import fasttext
from fastapi import FastAPI
from pydantic import BaseModel


# ============================================================
# 1. Configuration
# ============================================================

MODEL_PATH = "intent_model_v2.bin"
CONFIG_PATH = "model_v2_config.json"


# ============================================================
# 2. Load model and configuration
# ============================================================

print("Loading intent classification model...")

model = fasttext.load_model(MODEL_PATH)

with open(CONFIG_PATH, "r", encoding="utf-8") as f:
    config = json.load(f)

CONFIDENCE_THRESHOLD = float(
    config.get("confidence_threshold", 0.5)
)

LABELS = config.get("labels", [])

print("Model loaded successfully.")
print(f"Number of intents: {len(LABELS)}")
print(f"Confidence threshold: {CONFIDENCE_THRESHOLD}")


# ============================================================
# 3. Text normalization
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
    Apply exactly the same normalization used during training.
    """

    t = unicodedata.normalize("NFKC", str(text)).strip().lower()

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
        if (unicodedata.category(c)[0] in "LMN" or c in KEEP_CHARS)
        else " "
        for c in t
    )

    # Remove extra spaces
    return re.sub(r"\s+", " ", t).strip()


# ============================================================
# 4. FastAPI application
# ============================================================

app = FastAPI(
    title="Multilingual Banking Intent Classifier",
    description="API for predicting banking intents from user messages",
    version="1.0.0",
)


# ============================================================
# 5. Request schema
# ============================================================

class PredictionRequest(BaseModel):
    text: str


# ============================================================
# 6. Health check
# ============================================================

@app.get("/")
def root():
    return {
        "message": "Multilingual Intent Classification API is running",
        "model": MODEL_PATH,
        "number_of_intents": len(LABELS),
        "confidence_threshold": CONFIDENCE_THRESHOLD,
    }


@app.get("/health")
def health():
    return {
        "status": "healthy"
    }


# ============================================================
# 7. Intent prediction
# ============================================================

@app.post("/predict")
def predict(request: PredictionRequest):

    original_text = request.text

    # Validate input
    if not original_text or not original_text.strip():
        return {
            "success": False,
            "error": "Text cannot be empty"
        }

    # Apply same normalization used during training
    normalized_text = normalize(original_text)

    if not normalized_text:
        return {
            "success": False,
            "error": "Text contains no valid characters"
        }

    # Get prediction
    labels, probabilities = model.predict(
        normalized_text,
        k=1
    )

    # Remove __label__ prefix
    intent = labels[0].replace("__label__", "")

    confidence = float(probabilities[0])

    # Apply confidence threshold
    if confidence >= CONFIDENCE_THRESHOLD:
        final_intent = intent
        status = "confident"
    else:
        final_intent = None
        status = "uncertain"

    return {
        "success": True,
        "text": original_text,
        "normalized_text": normalized_text,
        "intent": final_intent,
        "confidence": round(confidence, 4),
        "status": status
    }


# ============================================================
# 8. Run directly with:
#
#     python app.py
#
# ============================================================

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=8000,
        reload=False
    )

