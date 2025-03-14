import os
import torch
import transformers
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
import faiss
from fastapi import FastAPI, UploadFile, File
import uvicorn
import ollama

# --- 1. Load and Preprocess Free Multilingual Data ---
def load_and_preprocess_dataset():
    print("Loading dataset...")
    dataset = load_dataset("opus100", "en-ko", split="train")  # Load full dataset
    dataset = dataset.select(range(min(50000, len(dataset))))  # Ensure only 50,000 samples are used
    print(f"Dataset loaded with {len(dataset)} samples.")
    
    def preprocess(example):
        if "translation" in example and "ko" in example["translation"]:
            return {"text": example["translation"]["ko"].strip()}
        return {"text": ""}

    train_data = dataset.map(preprocess, remove_columns=["translation"])
    print(f"Preprocessed dataset contains {len(train_data)} samples.")
    return train_data

train_data = load_and_preprocess_dataset()

# --- 2. Load Ollama LLaMA Model for Multilingual Support ---
model_name = "llava:v1.6"
if __name__ == "__main__":
    try:
        ollama.pull(model_name)  # Ensure model is available
        print("Ollama model successfully pulled.")
    except Exception as e:
        print(f"Error pulling Ollama model: {e}")
        exit(1)

def generate_ollama_response(prompt):
    system_prompt = "모든 응답은 한국어로 작성하십시오."
    try:
        response = ollama.chat(model=model_name, messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt}
        ])
        return response.get("message", {}).get("content", "응답을 생성할 수 없습니다.")
    except Exception as e:
        print(f"Error generating response from Ollama: {e}")
        return "오류가 발생했습니다."

# --- 3. Implementing RAG (Using FAISS for Retrieval) ---
embed_model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")  # Multilingual embedding model

# Dynamically build corpus from training data
corpus = [example["text"] for example in train_data if example["text"]]
if not corpus:
    corpus = ["기본 지식 데이터가 부족합니다."]  # Fallback text

try:
    corpus_embeddings = embed_model.encode(corpus, convert_to_tensor=True)
    index = faiss.IndexFlatL2(corpus_embeddings.shape[1])
    index.add(corpus_embeddings.cpu().numpy())
    print("FAISS index successfully created.")
except Exception as e:
    print(f"Error initializing FAISS index: {e}")
    corpus = ["지식을 찾을 수 없습니다."]
    index = None

def retrieve_facts(query):
    if not corpus or index is None:
        return ["지식을 찾을 수 없습니다."]  # Fallback response
    try:
        query_embedding = embed_model.encode(query, convert_to_tensor=True)
        _, top_k = index.search(query_embedding.cpu().numpy().reshape(1, -1), k=2)
        return [corpus[i] for i in top_k[0] if i < len(corpus)]
    except Exception as e:
        print(f"Error retrieving facts: {e}")
        return ["정보 검색 중 오류 발생."]

# --- 4. Deployment Using FastAPI with Ollama ---
app = FastAPI()

@app.post("/predict")
async def generate_text(prompt: str):
    retrieved_docs = retrieve_facts(prompt)
    input_text = "\n".join(retrieved_docs) + "\n" + prompt
    response = generate_ollama_response(input_text)
    return {"response": response}

if __name__ == "__main__":
    print("Starting FastAPI server...")
    uvicorn.run(app, host="0.0.0.0", port=8000)
