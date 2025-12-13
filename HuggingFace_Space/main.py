import re
import emoji
import torch
import torch.nn.functional as F
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import RobertaTokenizer, RobertaModel
from torch.nn import Module, Linear
from huggingface_hub import hf_hub_download
import os

# ==============================================================================
# ⚠️ CONFIGURATION - YOU MUST EDIT THIS ⚠️
# ==============================================================================
# 1. Your Hugging Face Model Repository path (e.g., "jdoe/EmoBerta-Custom")
HF_MODEL_REPO = "SRKR6115/EmoBerta" 
MODEL_FILE_NAME = "emoberta_model.bin"

# 2. Your list of emotion classes (verify this is correct from your training code)
EMOTION_CLASSES = ['anger', 'annoyance', 'approval', 'caring', 'curiosity', 'fear', 'gratitude', 'joy', 'love', 'neutral', 'sadness', 'surprise']
# ==============================================================================

# --- Define the request body for the API ---
class Tweet(BaseModel):
    text: str

# --- Preprocessing Function (must be identical to the one used for training) ---
def preprocess_text(text: str):
    if not isinstance(text, str): return ""
    text = emoji.demojize(text)
    text = text.lower()
    text = re.sub(r'@\w+', '', text)
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    text = re.sub(r'[^a-z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# --- Re-define the Model Architecture (must be identical to your trained model) ---
class EmoRoBERTa(Module):
    def __init__(self, n_emotions):
        super(EmoRoBERTa, self).__init__()
        self.roberta = RobertaModel.from_pretrained('roberta-base')
        self.emotion_classifier = Linear(self.roberta.config.hidden_size, n_emotions)
        self.intensity_regressor = Linear(self.roberta.config.hidden_size, n_emotions)
        self.sarcasm_detector = Linear(self.roberta.config.hidden_size, 1)

    def forward(self, input_ids, attention_mask):
        outputs = self.roberta(input_ids=input_ids, attention_mask=attention_mask).pooler_output
        return (self.emotion_classifier(outputs), self.intensity_regressor(outputs), self.sarcasm_detector(outputs))

# --- Load Model, Tokenizer, and Configuration ---
model = None
try:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
    model = EmoRoBERTa(n_emotions=len(EMOTION_CLASSES)).to(device)

    # ** Download the model weights from the Hugging Face Hub **
    print(f"Attempting to download {MODEL_FILE_NAME} from {HF_MODEL_REPO}...")
    model_path = hf_hub_download(repo_id=HF_MODEL_REPO, filename=MODEL_FILE_NAME)

    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    print(f"✅ Model loaded successfully from {model_path}!")

except Exception as e:
    print(f"--- ❌ An error occurred while loading the model: {e} ---")
    model = None

# --- Initialize the FastAPI app ---
app = FastAPI()

@app.get("/")
def read_root():
    return {"status": "EmoSense API is running", "model_loaded": bool(model)}

@app.post("/analyze")
def analyze_emotion(tweet: Tweet):
    if not model:
        return {"error": "Model is not loaded. Please check the API logs for configuration errors."}, 500

    text = tweet.text
    cleaned_text = preprocess_text(text)
    
    # Check if text is valid after cleaning
    if not cleaned_text:
        return {"error": "Input text is empty or only contained unusable characters."}, 400

    encoding = tokenizer.encode_plus(
        cleaned_text, add_special_tokens=True, max_length=128, return_token_type_ids=False,
        padding='max_length', truncation=True, return_attention_mask=True, return_tensors='pt'
    )
    
    input_ids = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)
    
    with torch.no_grad():
        emotion_logits, intensity_scores, sarcasm_logits = model(input_ids, attention_mask)
        
        # Apply sigmoid to logits for probabilities/confidence scores
        emotion_probs = F.sigmoid(emotion_logits).flatten().cpu().numpy()
        sarcasm_prob = F.sigmoid(sarcasm_logits).flatten().cpu().numpy()[0]
        
        # Intensity scores are typically linear outputs for regression (no sigmoid needed)
        intensity_scores = intensity_scores.flatten().cpu().numpy()
        
    return {
        "text": text,
        "emotions_confidence": {emotion: float(prob) for emotion, prob in zip(EMOTION_CLASSES, emotion_probs)},
        "predicted_intensity": {emotion: float(score) for emotion, score in zip(EMOTION_CLASSES, intensity_scores)},
        "sarcasm_score": float(sarcasm_prob)
    }

