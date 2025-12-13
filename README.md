# 🧠 EmoSense: The Unified Emotion Monitor

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://srkreddy-emosense.streamlit.app/)
[![Hugging Face Spaces](https://img.shields.io/badge/🤗%20Hugging%20Face-Spaces-blue)](https://huggingface.co/spaces/SRKR6115/EmoSense)
[![Python](https://img.shields.io/badge/Python-3.9%2B-blue?logo=python&logoColor=white)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.68%2B-009688?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![License](https://img.shields.io/badge/license-MIT-green)](./LICENSE)

**EmoSense** is a state-of-the-art NLP pipeline designed to decode the complexity of human emotion in text. Unlike traditional sentiment analysis (which only detects "Positive" or "Negative"), EmoSense utilizes a fine-tuned **RoBERTa** model to simultaneously predict:
1.  **Emotion Category** (12 classes including Joy, Fear, Anger, Gratitude, etc.)
2.  **Emotion Intensity** (A regression score of how strong the feeling is)
3.  **Sarcasm Detection** (Identifying if the literal meaning contradicts the sentiment)

![EmoSense UI](assets/ui_screenshot.png)
*(Screenshot of the EmoSense Live Dashboard)*

---

## 🚀 Live Demo

Experience the power of the Unified Emotion Monitor live on Streamlit Cloud:

### 👉 [Click Here to Launch App](https://srkreddy-emosense.streamlit.app/)

---

## 🌟 Key Features

* **Unified Multi-Task Learning:** A single model pass handles classification (Emotion), regression (Intensity), and binary classification (Sarcasm).
* **Real-Time X (Twitter) Analysis:** Integrated with the X API v2 to fetch live tweets on any trending topic and analyze public sentiment instantly.
* **Deep Learning Backend:** Powered by a custom `RoBERTa-base` architecture hosted on Hugging Face Spaces for 24/7 availability.
* **Interactive Visualizations:** Rich, interactive charts built with Plotly to visualize confidence scores and intensity distributions.
* **Explainable Metrics:** Provides detailed breakdowns of confidence scores for all 12 supported emotions.

---

## 🏗️ Architecture

The project follows a decoupled Microservices architecture to ensure scalability and ease of deployment.

```mermaid
graph LR
    A[User / Streamlit UI] -- "1. Input Text / Topic" --> B[Hugging Face API (FastAPI)]
    B -- "2. Tokenization" --> C[RoBERTa Tokenizer]
    C -- "3. Input IDs" --> D[EmoRoBERTa Model]
    D -- "4. Raw Logits" --> B
    B -- "5. JSON Response\n(Emotion, Intensity, Sarcasm)" --> A
    A -- "6. Visualization" --> E[Plotly Charts]

