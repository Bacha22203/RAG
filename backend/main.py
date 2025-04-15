from fastapi import FastAPI
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
        prompt = f"""You are Boxsejour AI, the expert virtual assistant for Boxsejour—a premium travel booking service...

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
            "temperature": 0.4
        }
        headers = {
            "Authorization": f"Bearer {OPENROUTER_API_KEY}",
            "Content-Type": "application/json"
        }
        async with httpx.AsyncClient() as client:
            response = await client.post("https://openrouter.ai/api/v1/chat/completions", headers=headers, json=payload)
            response.raise_for_status()
            result = response.json()

            # Log full response for debugging
            logger.info(f"OpenRouter raw response: {json.dumps(result, indent=2)}")

            # Handle missing 'choices' key
            if "choices" in result:
                return result['choices'][0]['message']['content']
            else:
                logger.error(f"OpenRouter API returned unexpected format: {result}")
                return "Sorry, I couldn't process your request due to an unexpected server response."
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
