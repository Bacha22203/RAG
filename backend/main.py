from fastapi import FastAPI, Request
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from typing import List
import os
import faiss
import json
from pathlib import Path
from sentence_transformers import SentenceTransformer
from PyPDF2 import PdfReader
import logging

import httpx

# === Configuration ===
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
PDF_FILE = Path("C:/Users/21622/Downloads/qa.pdf")  # Specific PDF file
INDEX_PATH = "faiss.index"
OPENROUTER_API_KEY = "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"
MODEL = "meta-llama/llama-4-maverick:free"

# === Setup Logging ===
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# === Init App ===
app = FastAPI()

# === CORS for local frontend dev ===
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# === Input Model ===
class ChatRequest(BaseModel):
    question: str
    history: List[dict] = []

# === Load Model ===
model = SentenceTransformer(EMBEDDING_MODEL)
texts = []
metadatas = []

# === Utils ===
def read_pdf(file_path):
    text = ""
    try:
        reader = PdfReader(file_path)
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
        if not text:
            logger.warning(f"No text extracted from {file_path}")
    except Exception as e:
        logger.error(f"Error reading PDF {file_path}: {e}")
    return text

def chunk_text(text, chunk_size=500):
    words = text.split()
    return [" ".join(words[i:i+chunk_size]) for i in range(0, len(words), chunk_size)]

# === Load Specific PDF ===
if PDF_FILE.exists():
    raw_text = read_pdf(PDF_FILE)
    if raw_text:
        chunks = chunk_text(raw_text)
        texts.extend(chunks)
        metadatas.extend([{"source": str(PDF_FILE)} for _ in chunks])
        logger.info(f"Successfully processed {PDF_FILE}")
    else:
        logger.warning(f"No content extracted from {PDF_FILE}")
else:
    logger.error(f"File not found: {PDF_FILE}")

# === Build FAISS Index ===
if texts:
    embeddings = model.encode(texts)
    # Ensure at least one embedding exists before indexing
    if len(embeddings) > 0:
        index = faiss.IndexFlatL2(embeddings[0].shape[0])
        index.add(embeddings)
        logger.info("FAISS index built successfully.")
    else:
        logger.error("Embeddings are empty. FAISS index cannot be built.")
else:
    logger.error("No texts available to build FAISS index.")

# === OpenRouter Query Function ===
async def ask_openrouter(question, context):
    try:
        prompt = f"""You are Boxsejour AI, the expert virtual assistant for Boxsejour—a premium travel booking service specializing in hotel reservations and travel arrangements. 
Your role is to provide clear, accurate, and relevant travel-related information, including booking details, destination insights, travel tips, and hotel recommendations. 
You must maintain a steady, professional tone that never sacrifices warmth or a slight sense of humor when appropriate.

Key instructions:
- Accuracy and Reliability: Provide only accurate information; do not hallucinate or guess details. If uncertain or if the information is not available, state clearly: "I don’t have that info right now, but you might check our website or a travel guide!"
- Language Adaptation and Context Extraction: Respond in the same language as the query. If a query is in French, answer in French; if in English, answer in English. When data sources or information are in another language, do not rely on a literal translation. Instead, carefully extract and adapt the true context to ensure that nuances and accurate meanings are preserved, avoiding common translation errors.
- Clarification and Engagement: For unclear questions, politely ask for further clarification, e.g., "Could you please provide more details so I can assist you better?" For greetings like "Hi" or "Hello," respond in a friendly, engaging, and warm manner, such as "Hello there! Ready to plan your next adventure?"
- Handling Off-Topic or Irrelevant Queries: If a query is whimsical or irrelevant, humorously steer the conversation back on track with responses like, "I’m not sure about pigs, but I can definitely help you fly somewhere nice—where are you thinking?"
- Travel and Booking Focus: Tailor responses to assist with travel bookings, especially hotel reservations. Provide detailed, helpful suggestions on hotels, booking procedures, cancellation policies, and local travel tips whenever relevant. Ensure any travel advice is practical and reflective of the latest context for travel planning.
- Consistent Professionalism: Maintain a professional, steady, and friendly tone at all times. Exhibit empathy and understanding, reinforcing that the user’s travel plans are important.

Context:
{context}

Question: {question}

Answer:"""

        payload = {
            "model": MODEL,
            "messages": [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.2
        }
        headers = {
            "Authorization": f"Bearer {OPENROUTER_API_KEY}",
            "Content-Type": "application/json"
        }
        async with httpx.AsyncClient() as client:
            response = await client.post("https://openrouter.ai/api/v1/chat/completions", headers=headers, json=payload)
            response.raise_for_status()
            result = response.json()
            return result['choices'][0]['message']['content']
    except Exception as e:
        logger.error(f"Error calling OpenRouter: {e}")
        return "Sorry, I couldn't process your request due to a server issue."

# === Chat Endpoint ===
@app.post("/chat")
async def chat(req: ChatRequest):
    q_embedding = model.encode([req.question])
    context = ""
    if texts:
        D, I = index.search(q_embedding, k=5)
        context = "\n\n".join([texts[i] for i in I[0]])
    else:
        logger.warning("No context available, proceeding without FAISS.")
    answer = await ask_openrouter(req.question, context)
    return {"answer": answer}
