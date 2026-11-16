import os
from contextlib import asynccontextmanager
import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from transformers import AutoTokenizer, AutoModelForSequenceClassification

MODEL_PATH = os.environ.get("MODEL_PATH", "./my_fine_tuned_dnabert")

model_state = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    if not os.path.isdir(MODEL_PATH):
        raise RuntimeError(f"Model directory not found: {MODEL_PATH}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_PATH, trust_remote_code=True
    )
    model.eval()
    model_state["tokenizer"] = tokenizer
    model_state["model"] = model
    yield
    model_state.clear()


app = FastAPI(
    title="DNABERT-2 & Quantum-LoRA Promoter API",
    description="Predicts whether a DNA sequence is a Promoter or Non-Promoter.",
    version="1.0.0",
    lifespan=lifespan,
)


class DNASequenceRequest(BaseModel):
    sequence: str = Field(
        ...,
        min_length=6,
        description="DNA sequence consisting of A, C, G, T, N characters.",
        examples=["AGCTCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATC"],
    )


class PredictionResponse(BaseModel):
    label: str
    predicted_class: int
    promoter_probability: float
    non_promoter_probability: float


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": "model" in model_state}


@app.post("/predict", response_model=PredictionResponse)
def predict(req: DNASequenceRequest):
    clean_seq = req.sequence.strip().upper()
    if not set(clean_seq).issubset({"A", "C", "G", "T", "N"}):
        raise HTTPException(
            status_code=400,
            detail="Sequence must only contain valid DNA bases (A, C, G, T, N).",
        )

    tokenizer = model_state["tokenizer"]
    model = model_state["model"]

    inputs = tokenizer(
        clean_seq,
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=251,
    )

    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.softmax(outputs.logits, dim=-1)[0].tolist()
        pred_cls = int(torch.argmax(outputs.logits, dim=-1).item())

    return PredictionResponse(
        label="Promoter" if pred_cls == 1 else "Non-Promoter",
        predicted_class=pred_cls,
        non_promoter_probability=round(probs[0], 6),
        promoter_probability=round(probs[1], 6),
    )
