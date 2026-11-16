"""
Retrievers Module for Day 7 - Agentic RAG
Supports 5 Distinct Retrieval Methods:
1. Dense Retrieval (ChromaDB + gemini-embedding-001)
2. Lexical Retrieval (BM25Okapi)
3. Hybrid Retrieval (RRF: Dense + BM25)
4. Reranked Retrieval (Candidate Generation + Local Ultra-lightweight FlashRank ms-marco-TinyBERT-L-2-v2)
5. Query Transformation Retrieval (Query Expansion / Rewriting + Hybrid Search)
"""

import os
import re
import json
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv
import chromadb
from rank_bm25 import BM25Okapi
from flashrank import Ranker, RerankRequest
from openai import OpenAI

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ENV_PATH = os.path.join(BASE_DIR, ".env")
load_dotenv(ENV_PATH)

CHUNKS_PATH = os.path.join(BASE_DIR, "artifacts", "chunks_strategy_b.json")
CHROMA_PATH = os.path.join(BASE_DIR, "chroma_db")

class RetrievalEngine:
    def __init__(self, api_key: Optional[str] = None, base_url: Optional[str] = None):
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.base_url = base_url or os.getenv("OPENAI_BASE_URL", "https://api.shopaikey.com/v1")
        
        self.client = OpenAI(api_key=self.api_key, base_url=self.base_url) if self.api_key else None
        
        # Load 162 chunks
        with open(CHUNKS_PATH, "r", encoding="utf-8") as f:
            self.chunks = json.load(f)
        self.chunk_map = {c["chunk_id"]: c for c in self.chunks}
        
        # 1. Init ChromaDB (Dense)
        self.chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)
        self.collection = self.chroma_client.get_collection("strategy_b")
        
        # 2. Init BM25 (Lexical)
        self.tokenized_corpus = [self._tokenize(c["text"]) for c in self.chunks]
        self.bm25 = BM25Okapi(self.tokenized_corpus)

        # 3. Init Local Lightweight Reranker (FlashRank ~4MB onnx)
        self.local_ranker = Ranker(model_name="ms-marco-MiniLM-L-12-v2", cache_dir=os.path.join(BASE_DIR, "artifacts", "flashrank_cache"))

    def _tokenize(self, text: str) -> List[str]:
        return re.findall(r"\w+", text.lower())

    def _get_embedding(self, text: str) -> List[float]:
        if not self.client:
            raise ValueError("API Key not configured for embedding generation.")
        res = self.client.embeddings.create(
            model="gemini-embedding-001",
            input=text
        )
        return res.data[0].embedding

    # ==========================================
    # METHOD 1: Dense Retrieval
    # ==========================================
    def dense_search(self, query: str, top_k: int = 5, where: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        query_vector = self._get_embedding(query)
        results = self.collection.query(
            query_embeddings=[query_vector],
            n_results=top_k,
            where=where
        )
        
        candidates = []
        if results and "ids" in results and results["ids"]:
            ids = results["ids"][0]
            distances = results["distances"][0] if "distances" in results and results["distances"] else [0.0] * len(ids)
            for rank, (cid, dist) in enumerate(zip(ids, distances), start=1):
                c = self.chunk_map.get(cid, {})
                score = round(1.0 - dist, 4)
                candidates.append({
                    "chunk_id": cid,
                    "rank": rank,
                    "score": score,
                    "text": c.get("text", ""),
                    "pdf_page": c.get("pdf_page"),
                    "section": c.get("section"),
                    "method": "dense"
                })
        return candidates

    # ==========================================
    # METHOD 2: Lexical Retrieval (BM25)
    # ==========================================
    def lexical_search(self, query: str, top_k: int = 5, section_filter: Optional[str] = None) -> List[Dict[str, Any]]:
        query_tokens = self._tokenize(query)
        scores = self.bm25.get_scores(query_tokens)
        
        ranked_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        candidates = []
        rank = 1
        for idx in ranked_indices:
            c = self.chunks[idx]
            if section_filter and c.get("section") != section_filter:
                continue
            if scores[idx] <= 0 and len(candidates) >= top_k:
                break
            candidates.append({
                "chunk_id": c["chunk_id"],
                "rank": rank,
                "score": round(float(scores[idx]), 4),
                "text": c["text"],
                "pdf_page": c["pdf_page"],
                "section": c["section"],
                "method": "bm25"
            })
            rank += 1
            if len(candidates) >= top_k:
                break
        return candidates

    # ==========================================
    # METHOD 3: Hybrid Retrieval (RRF)
    # ==========================================
    def hybrid_search(self, query: str, top_k: int = 5, rrf_k: int = 60, where: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        fetch_k = max(top_k * 3, 15)
        section = where.get("section") if where else None
        
        dense_results = self.dense_search(query, top_k=fetch_k, where=where)
        bm25_results = self.lexical_search(query, top_k=fetch_k, section_filter=section)
        
        rrf_scores: Dict[str, float] = {}
        for r, item in enumerate(dense_results, start=1):
            cid = item["chunk_id"]
            rrf_scores[cid] = rrf_scores.get(cid, 0.0) + (1.0 / (rrf_k + r))
            
        for r, item in enumerate(bm25_results, start=1):
            cid = item["chunk_id"]
            rrf_scores[cid] = rrf_scores.get(cid, 0.0) + (1.0 / (rrf_k + r))
            
        sorted_cids = sorted(rrf_scores.keys(), key=lambda cid: rrf_scores[cid], reverse=True)[:top_k]
        
        candidates = []
        for rank, cid in enumerate(sorted_cids, start=1):
            c = self.chunk_map[cid]
            candidates.append({
                "chunk_id": cid,
                "rank": rank,
                "score": round(rrf_scores[cid], 5),
                "text": c["text"],
                "pdf_page": c["pdf_page"],
                "section": c["section"],
                "method": "hybrid_rrf"
            })
        return candidates

    # ==========================================
    # METHOD 4: Retrieval + Local Reranker (FlashRank ms-marco-TinyBERT)
    # ==========================================
    def reranked_search(self, query: str, top_k: int = 5, initial_candidates_k: int = 15) -> List[Dict[str, Any]]:
        initial_candidates = self.hybrid_search(query, top_k=initial_candidates_k)
        if len(initial_candidates) <= top_k:
            return initial_candidates
            
        passages = [
            {"id": c["chunk_id"], "text": c["text"]}
            for c in initial_candidates
        ]
        
        try:
            req = RerankRequest(query=query, passages=passages)
            reranked_results = self.local_ranker.rerank(req)
            
            # Map back to full metadata
            candidates = []
            for rank, r in enumerate(reranked_results[:top_k], start=1):
                cid = r["id"]
                c = self.chunk_map[cid]
                candidates.append({
                    "chunk_id": cid,
                    "rank": rank,
                    "score": round(float(r["score"]), 4),
                    "text": c["text"],
                    "pdf_page": c["pdf_page"],
                    "section": c["section"],
                    "method": "reranked_local_minilm"
                })
            return candidates
        except Exception:
            for r, c in enumerate(initial_candidates[:top_k], start=1):
                c["rank"] = r
                c["method"] = "reranked_fallback"
            return initial_candidates[:top_k]

    # ==========================================
    # METHOD 5: Query Transformation (gemini-3.5-flash-lite)
    # ==========================================
    def query_transformed_search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        if not self.client:
            return self.hybrid_search(query, top_k=top_k)
            
        prompt = f"""Rewrite and expand the following query into a concise, keyword-rich search query to retrieve college textbook information.

Query: "{query}"

Output ONLY the expanded query string, no quotation marks or explanations.
"""
        try:
            resp = self.client.chat.completions.create(
                model="gemini-3.5-flash-lite",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0
            )
            expanded_query = resp.choices[0].message.content.strip().replace('"', '')
        except Exception:
            expanded_query = query
            
        results = self.hybrid_search(expanded_query, top_k=top_k)
        for c in results:
            c["method"] = "query_transformation"
            c["transformed_query"] = expanded_query
        return results

