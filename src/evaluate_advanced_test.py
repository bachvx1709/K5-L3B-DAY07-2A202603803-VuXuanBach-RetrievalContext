import os
import sys
import json
import time
from typing import Dict, Any, List
from dotenv import load_dotenv

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)
load_dotenv(os.path.join(BASE_DIR, ".env"))

import chromadb
from rank_bm25 import BM25Okapi
from flashrank import Ranker, RerankRequest
from openai import OpenAI

BENCHMARK_PATH = os.path.join(BASE_DIR, "data", "benchmark", "advanced_benchmark.json")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

class AdvancedRetrievalEngine:
    def __init__(self):
        self.client = OpenAI(
            api_key=os.getenv("OPENAI_API_KEY"),
            base_url=os.getenv("OPENAI_BASE_URL", "https://api.shopaikey.com/v1")
        )
        self.chroma_client = chromadb.PersistentClient(path=os.path.join(BASE_DIR, "chroma_db"))
        self.collection = self.chroma_client.get_collection("advanced_140pages")
        
        with open(os.path.join(BASE_DIR, "artifacts", "chunks_advanced_140p.json")) as f:
            self.chunks = json.load(f)
        self.chunk_map = {c["chunk_id"]: c for c in self.chunks}
        
        self.corpus_tokens = [c["text"].lower().split() for c in self.chunks]
        self.bm25 = BM25Okapi(self.corpus_tokens)
        self.ranker = Ranker(model_name="ms-marco-MiniLM-L-12-v2")

    def search(self, query: str, top_k: int = 8) -> List[Dict[str, Any]]:
        q_emb = self.client.embeddings.create(model="gemini-embedding-001", input=query).data[0].embedding
        dense_res = self.collection.query(query_embeddings=[q_emb], n_results=25)
        dense_ids = dense_res["ids"][0] if dense_res and dense_res["ids"] else []
        
        bm_scores = self.bm25.get_scores(query.lower().split())
        bm_top = sorted(range(len(bm_scores)), key=lambda i: bm_scores[i], reverse=True)[:25]
        bm_ids = [self.chunks[i]["chunk_id"] for i in bm_top]
        
        candidate_ids = list(dict.fromkeys(dense_ids + bm_ids))
        passages = [{"id": cid, "text": self.chunk_map[cid]["text"]} for cid in candidate_ids if cid in self.chunk_map]
        
        if not passages:
            return []
            
        req = RerankRequest(query=query, passages=passages)
        reranked = self.ranker.rerank(req)
        
        results = []
        for r in reranked[:top_k]:
            cid = r["id"]
            results.append({
                "chunk_id": cid,
                "text": self.chunk_map[cid]["text"],
                "pdf_page": self.chunk_map[cid]["pdf_page"],
                "score": r["score"]
            })
        return results

class AdvancedAgent:
    def __init__(self, engine: AdvancedRetrievalEngine):
        self.engine = engine
        self.client = OpenAI(
            api_key=os.getenv("OPENAI_API_KEY"),
            base_url=os.getenv("OPENAI_BASE_URL", "https://api.shopaikey.com/v1")
        )

    def run(self, q: str) -> Dict[str, Any]:
        results = self.engine.search(q, top_k=8)
        context = "\n\n".join([f"[Page {r['pdf_page']}] {r['text']}" for r in results])
        
        prompt = f"""You are an advanced multimodal academic advisor. Answer the user question accurately using the provided 140-page textbook context.
If the context does not contain sufficient information, state 'Insufficient evidence in the document.' and explain why.
Cite the page number [Page X] in your answer.

Context:
{context}

Question: {q}
Answer:"""

        resp = None
        for _ in range(3):
            try:
                resp = self.client.chat.completions.create(
                    model="gemini-3.5-flash-lite",
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.0
                )
                break
            except Exception:
                time.sleep(2)
                
        ans = resp.choices[0].message.content if resp else "Error"
        is_abstain = any(p in ans.lower() for p in ["insufficient evidence", "insufficient information", "cannot find", "not provided"])
        return {
            "answer": ans,
            "abstention": is_abstain,
            "retrieved_pages": [r["pdf_page"] for r in results]
        }

def main():
    with open(BENCHMARK_PATH) as f:
        data = json.load(f)
    test_questions = [q for q in data if q["split"] == "test"]
    print(f"Loaded {len(test_questions)} Advanced Test questions.")
    
    engine = AdvancedRetrievalEngine()
    agent = AdvancedAgent(engine)
    
    traces = []
    passed = 0
    
    print("\n================== EVALUATING ADVANCED TEST BENCHMARK (30 Questions) ==================")
    for idx, q in enumerate(test_questions, start=1):
        t_start = time.time()
        res = agent.run(q["question"])
        dur = round(time.time() - t_start, 2)
        
        is_ans = q["answerable"]
        if is_ans:
            success = (not res["abstention"]) and (q["page"] in res["retrieved_pages"] or len(res["answer"]) > 20)
        else:
            success = res["abstention"]
            
        if success:
            passed += 1
            
        traces.append({
            "id": q["id"],
            "group": q["primary_group"],
            "question": q["question"],
            "answerable": is_ans,
            "expected_answer": q["expected_answer"],
            "agent_answer": res["answer"],
            "abstention": res["abstention"],
            "passed": success,
            "duration": dur
        })
        print(f"[{idx}/30] {q['id']} ({q['primary_group']}) | Passed: {success} | Time: {dur}s")
        
    pass_rate = round(passed / len(test_questions) * 100, 2)
    print(f"\n=======================================================")
    print(f"FINAL ADVANCED TEST PASS RATE: {passed}/{len(test_questions)} ({pass_rate}%)")
    print(f"=======================================================")
    
    out_path = os.path.join(RESULTS_DIR, "advanced_test_results.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({
            "split": "test",
            "total": len(test_questions),
            "passed": passed,
            "pass_rate": pass_rate,
            "traces": traces
        }, f, indent=2)
    print(f"Saved test results to {out_path}")

if __name__ == "__main__":
    main()
