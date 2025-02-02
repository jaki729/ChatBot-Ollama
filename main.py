from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
import requests
import json
import os
import faiss
import numpy as np
import ollama
import pickle  # For saving/loading FAISS index
import uvicorn
from bs4 import BeautifulSoup
import re

# File paths for saved FAISS index and metadata
FAISS_INDEX_FILE = "faiss_index.pkl"
CHUNKS_FILE = "chunks.json"

# Initialize FAISS index (None initially)
index = None
chunks = []
chunk_metadata = {}

# FastAPI app initialization
app = FastAPI()

# Set up static files and templates
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# Function to check if the paragraph is relevant
def is_relevant_paragraph(p):
    text = p.text.strip()
    # Skip text that looks like "Read more" or similar
    if len(text) < 30 or re.search(r"(read more|continue reading|next)", text, re.IGNORECASE):
        return False
    return True

# Function to load FAISS index and chunks if available
def load_index():
    global index, chunks, chunk_metadata
    
    # Load FAISS index if file exists
    if os.path.exists(FAISS_INDEX_FILE) and os.path.exists(CHUNKS_FILE):
        with open(CHUNKS_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
            chunks.extend(data["chunks"])
            chunk_metadata.update(data["metadata"])

        with open(FAISS_INDEX_FILE, "rb") as f:
            index = pickle.load(f)

        print("FAISS index and chunks loaded successfully!")
    else:
        print("No existing FAISS index found. Starting fresh.")

# Function to save FAISS index and chunks
def save_index():
    if index is not None:
        with open(FAISS_INDEX_FILE, "wb") as f:
            pickle.dump(index, f)

        with open(CHUNKS_FILE, "w", encoding="utf-8") as f:
            json.dump({"chunks": chunks, "metadata": chunk_metadata}, f)

        print("FAISS index saved successfully!")

# Function to get text embeddings using Ollama's model
def get_embedding(text):
    response = ollama.embeddings(model="nomic-embed-text", prompt=text)
    embedding = np.array(response["embedding"], dtype=np.float32)
    return embedding

# Function to scrape and index website content with pagination handling
def scrape_and_index(url, max_pages=3):
    global index  # Ensure FAISS index is updated globally
    headers = {"User-Agent": "Mozilla/5.0"}
    all_text = ""
    page_count = 0

    while url and page_count < max_pages:
        print(f"Scraping page {page_count + 1} of {url}")
        response = requests.get(url, headers=headers)
        if response.status_code != 200:
            print(f"Failed to fetch {url}")
            break
        
        soup = BeautifulSoup(response.text, "html.parser")
        
        # Extract text from paragraphs, filtering out irrelevant content
        paragraphs = [p.text.strip() for p in soup.find_all("p") if is_relevant_paragraph(p)]
        all_text += " ".join(paragraphs) + "\n\n"

        # Try to find the "Next" page link
        next_page = None
        next_button = soup.find("a", string=re.compile("Next", re.IGNORECASE))  # Looking for "Next" text
        if next_button:
            next_page = next_button.get("href")
            if next_page and not next_page.startswith("http"):
                next_page = requests.compat.urljoin(url, next_page)  # Handle relative URLs

        # Update the URL for the next iteration
        url = next_page
        page_count += 1

    if all_text:
        # Chunk and store embeddings
        chunk_size = 500
        for i in range(0, len(all_text), chunk_size):
            chunk = all_text[i:i+chunk_size]
            if chunk:
                embedding = get_embedding(chunk)

                # Initialize FAISS index with correct dimension
                if index is None:
                    d = embedding.shape[0]
                    index = faiss.IndexFlatL2(d)

                # Verify embedding matches FAISS dimension
                if embedding.shape[0] == index.d:
                    index.add(np.array([embedding]))  # Add to FAISS
                    chunk_id = len(chunks)
                    chunks.append(chunk)
                    chunk_metadata[chunk_id] = url
                else:
                    print(f"Embedding dimension mismatch: Expected {index.d}, got {embedding.shape[0]}")

        # Save FAISS index after updating
        save_index()
        print("Website indexed successfully with pagination!")

# Function to retrieve most relevant chunk
def retrieve_context(query, k=2):
    query_embedding = get_embedding(query).reshape(1, -1)
    _, indices = index.search(query_embedding, k)
    return "\n".join([chunks[i] for i in indices[0]])

# Function to chat using retrieved context
def chat_with_llama2(prompt):
    context = retrieve_context(prompt)
    full_prompt = f"Use the following website information to answer queries:\n\n{context}\n\nUser: {prompt}\n\nAnswer:"
    response = ollama.chat(model="llama2", messages=[{"role": "user", "content": full_prompt}])
    return response["message"]["content"]

# Pydantic models for the incoming requests
class WebsiteRequest(BaseModel):
    url: str

class ChatRequest(BaseModel):
    prompt: str

# FastAPI Routes
@app.on_event("startup")
async def startup():
    load_index()  # Load FAISS index at startup

@app.get("/", response_class=HTMLResponse)
async def read_root():
    return templates.TemplateResponse("index.html", {"request": {}})

@app.post("/add_website/")
async def add_website(request: WebsiteRequest):
    try:
        scrape_and_index(request.url)
        return {"message": "Website indexed successfully!"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/chat/")
async def chat(request: ChatRequest):
    try:
        response = chat_with_llama2(request.prompt)
        return {"response": response}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
