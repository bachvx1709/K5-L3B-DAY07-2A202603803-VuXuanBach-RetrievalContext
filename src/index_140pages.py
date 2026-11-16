import os
import sys
import json
import re
from typing import List, Dict, Any
from dotenv import load_dotenv
import pypdf
import chromadb
from openai import OpenAI

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)
load_dotenv(os.path.join(BASE_DIR, ".env"))

PDF_PATH = os.path.join(BASE_DIR, "data", "college-success_-_WEB.pdf")
CHROMA_PATH = os.path.join(BASE_DIR, "chroma_db")
ARTIFACTS_DIR = os.path.join(BASE_DIR, "artifacts")
os.makedirs(ARTIFACTS_DIR, exist_ok=True)

def extract_and_chunk_140_pages(num_pages: int = 140) -> List[Dict[str, Any]]:
    print(f"Reading first {num_pages} pages from PDF...")
    reader = pypdf.PdfReader(PDF_PATH)
    chunks = []
    chunk_counter = 1
    
    for page_idx in range(num_pages):
        pdf_page = page_idx + 1
        page = reader.pages[page_idx]
        text = page.extract_text() or ""
        
        # Clean text
        text = re.sub(r"[ \t]+", " ", text).strip()
        if not text:
            continue
            
        # Detect Tables / Figures on this page
        has_table = "Table " in text
        has_figure = "Figure " in text
        
        # Split by double newline or natural paragraph
        paragraphs = [p.strip() for p in text.split("\n\n") if len(p.strip()) > 30]
        if not paragraphs:
            paragraphs = [text]
            
        current_chunk = ""
        for p in paragraphs:
            if len(current_chunk) + len(p) < 900:
                current_chunk += ("\n\n" + p) if current_chunk else p
            else:
                cid = f"ADV_chunk_{chunk_counter:04d}"
                chunks.append({
                    "chunk_id": cid,
                    "pdf_page": pdf_page,
                    "text": current_chunk.strip(),
                    "has_table": "Table " in current_chunk,
                    "has_figure": "Figure " in current_chunk
                })
                chunk_counter += 1
                current_chunk = p
                
        if current_chunk:
            cid = f"ADV_chunk_{chunk_counter:04d}"
            chunks.append({
                "chunk_id": cid,
                "pdf_page": pdf_page,
                "text": current_chunk.strip(),
                "has_table": "Table " in current_chunk,
                "has_figure": "Figure " in current_chunk
            })
            chunk_counter += 1
            
    print(f"Extracted {len(chunks)} chunks across {num_pages} pages.")
    return chunks

def build_advanced_vector_index(chunks: List[Dict[str, Any]]):
    client = chromadb.PersistentClient(path=CHROMA_PATH)
    col_name = "advanced_140pages"
    
    # Reset collection if exists
    try:
        client.delete_collection(col_name)
    except Exception:
        pass
        
    collection = client.create_collection(
        name=col_name,
        metadata={"hnsw:space": "cosine"}
    )
    
    openai_client = OpenAI(
        api_key=os.getenv("OPENAI_API_KEY"),
        base_url=os.getenv("OPENAI_BASE_URL", "https://api.shopaikey.com/v1")
    )
    
    print("Generating embeddings for 140 pages in batches...")
    batch_size = 20
    for i in range(0, len(chunks), batch_size):

        batch = [c for c in chunks[i:i+batch_size] if len(c["text"].strip()) > 10]
        if not batch:
            continue

        texts = [c["text"] for c in batch]
        ids = [c["chunk_id"] for c in batch]
        metadatas = [{
            "pdf_page": c["pdf_page"],
            "has_table": c["has_table"],
            "has_figure": c["has_figure"]
        } for c in batch]
        
        resp = openai_client.embeddings.create(
            model="gemini-embedding-001",
            input=texts
        )
        embeddings = [item.embedding for item in resp.data]
        
        collection.add(
            ids=ids,
            embeddings=embeddings,
            metadatas=metadatas,
            documents=texts
        )
        print(f"Indexed [{min(i+batch_size, len(chunks))}/{len(chunks)}] chunks...")
        
    print(f"Collection '{col_name}' successfully built with {collection.count()} vectors.")

def main():
    chunks = extract_and_chunk_140_pages(140)
    with open(os.path.join(ARTIFACTS_DIR, "chunks_advanced_140p.json"), "w", encoding="utf-8") as f:
        json.dump(chunks, f, indent=2)
    build_advanced_vector_index(chunks)

if __name__ == "__main__":
    main()
