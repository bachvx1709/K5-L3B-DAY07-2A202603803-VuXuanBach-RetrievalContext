import os
import sys
import json
import time
from typing import Dict, Any, List
from dotenv import load_dotenv

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)
load_dotenv(os.path.join(BASE_DIR, ".env"))

from src.retrievers import RetrievalEngine
from src.agent import RAGAgent

BENCHMARK_PATH = os.path.join(BASE_DIR, "data", "benchmark", "benchmark.json")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

def load_dev_questions():
    with open(BENCHMARK_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    return [q for q in data if q.get("split") == "dev"]

def evaluate_retrieval(engine: RetrievalEngine, dev_questions: List[Dict[str, Any]], method_name: str, top_k: int = 3):
    total_ans = 0
    hit1 = 0
    hit3 = 0
    records = []
    
    t_start = time.time()
    for q in dev_questions:
        if not q.get("answerable"):
            continue
        total_ans += 1
        query = q["question"]
        gold_cids = [ev["chunk_id"] for ev in q.get("evidence", [])]
        filt = q.get("filter")
        where = {"section": filt["section"]} if filt and "section" in filt else None
        
        # Execute method
        if method_name == "dense":
            res = engine.dense_search(query, top_k=top_k, where=where)
        elif method_name == "bm25":
            res = engine.lexical_search(query, top_k=top_k, section_filter=filt.get("section") if filt else None)
        elif method_name == "hybrid":
            res = engine.hybrid_search(query, top_k=top_k, where=where)
        elif method_name == "rerank_local":
            res = engine.reranked_search(query, top_k=top_k)
        elif method_name == "query_transform":
            res = engine.query_transformed_search(query, top_k=top_k)
        else:
            res = engine.hybrid_search(query, top_k=top_k, where=where)
            
        retrieved_cids = [r["chunk_id"] for r in res]
        is_hit1 = any(cid in gold_cids for cid in retrieved_cids[:1])
        is_hit3 = any(cid in gold_cids for cid in retrieved_cids[:3])
        
        if is_hit1:
            hit1 += 1
        if is_hit3:
            hit3 += 1
            
        records.append({
            "id": q["id"],
            "question": query,
            "gold_chunks": gold_cids,
            "retrieved_chunks": retrieved_cids,
            "hit@1": is_hit1,
            "hit@3": is_hit3
        })
        
    dur = time.time() - t_start
    return {
        "method": method_name,
        "total_answerable": total_ans,
        "hit1_count": hit1,
        "hit1_rate": round(hit1 / total_ans * 100, 2),
        "hit3_count": hit3,
        "hit3_rate": round(hit3 / total_ans * 100, 2),
        "duration_sec": round(dur, 2),
        "records": records
    }

def evaluate_agent(agent: RAGAgent, dev_questions: List[Dict[str, Any]], method_name: str):
    ans_count = 0
    unans_count = 0
    correct_abstentions = 0
    context_has_evidence = 0
    grounded_scores = []
    total_tokens = 0
    latencies = []
    detailed_traces = []
    
    print(f"\n--- Running Full Agent Evaluation for method: {method_name} ---")
    for idx, q in enumerate(dev_questions, start=1):
        is_ans = q.get("answerable", True)
        gold_cids = [ev["chunk_id"] for ev in q.get("evidence", [])]
        filt = q.get("filter")
        
        # Run agent
        trace = agent.run(q["question"], default_method=method_name, force_filter=filt)
        latencies.append(trace["latency_seconds"])
        total_tokens += trace["total_tokens"]
        
        context_cids = [c["chunk_id"] for c in trace["final_context"]]
        has_ev = any(cid in gold_cids for cid in context_cids)
        
        if is_ans:
            ans_count += 1
            if has_ev:
                context_has_evidence += 1
                
            cited_in_gold = any(c in gold_cids for c in trace["citations"])
            if trace["abstention"]:
                score = 0
            elif cited_in_gold:
                score = 2
            elif has_ev:
                score = 1
            else:
                score = 0
            grounded_scores.append(score)
        else:
            unans_count += 1
            if trace["abstention"]:
                correct_abstentions += 1
                
        detailed_traces.append({
            "id": q["id"],
            "question": q["question"],
            "answerable": is_ans,
            "gold_evidence": gold_cids,
            "retrieved_chunks": trace["retrieved_chunks"],
            "agent_answer": trace["answer"],
            "citations": trace["citations"],
            "abstention": trace["abstention"],
            "latency": trace["latency_seconds"],
            "tokens": trace["total_tokens"]
        })
        print(f"[{idx}/30] {q['id']} | Ans: {is_ans} | Abstain: {trace['abstention']} | Time: {trace['latency_seconds']}s")
        
    avg_grounded = round(sum(grounded_scores) / len(grounded_scores), 2) if grounded_scores else 0.0
    abstention_rate = round(correct_abstentions / unans_count * 100, 2) if unans_count else 0.0
    context_ev_rate = round(context_has_evidence / ans_count * 100, 2) if ans_count else 0.0
    
    return {
        "method": method_name,
        "total_questions": len(dev_questions),
        "context_evidence_rate": f"{context_has_evidence}/{ans_count} ({context_ev_rate}%)",
        "grounded_score_avg": f"{avg_grounded}/2.0",
        "grounded_distribution": {
            "2": grounded_scores.count(2),
            "1": grounded_scores.count(1),
            "0": grounded_scores.count(0)
        },
        "abstention_rate": f"{correct_abstentions}/{unans_count} ({abstention_rate}%)",
        "avg_latency": round(sum(latencies) / len(latencies), 2),
        "total_tokens": total_tokens,
        "traces": detailed_traces
    }

def main():
    dev_q = load_dev_questions()
    print(f"Loaded {len(dev_q)} Dev questions ({sum(1 for q in dev_q if q['answerable'])} answerable, {sum(1 for q in dev_q if not q['answerable'])} unanswerable).")
    
    engine = RetrievalEngine()
    
    # STEP 1: RETRIEVAL METRICS (Top-3) across ALL 5 METHODS
    print("\n================== STEP 1: RETRIEVAL METRICS FOR ALL 5 METHODS (Top-3) ==================")
    configs = ["dense", "bm25", "hybrid", "rerank_local", "query_transform"]
    retrieval_summaries = {}
    for cfg in configs:
        res = evaluate_retrieval(engine, dev_q, cfg, top_k=3)
        retrieval_summaries[cfg] = res
        print(f"Config: {cfg:<16} | Hit@1: {res['hit1_count']}/25 ({res['hit1_rate']}%) | Hit@3: {res['hit3_count']}/25 ({res['hit3_rate']}%) | Time: {res['duration_sec']}s")
        
    with open(os.path.join(RESULTS_DIR, "retrieval_dev_summary.json"), "w", encoding="utf-8") as f:
        json.dump(retrieval_summaries, f, indent=2)
        
    # STEP 2: END-TO-END AGENT BENCHMARK
    print("\n================== STEP 2: END-TO-END AGENT BENCHMARK ==================")
    agent = RAGAgent(retrieval_engine=engine, model="gemini-3.5-flash-lite")
    
    # 1. Baseline: Dense
    dense_agent_summary = evaluate_agent(agent, dev_q, "dense")
    with open(os.path.join(RESULTS_DIR, "agent_dev_baseline_dense.json"), "w", encoding="utf-8") as f:
        json.dump(dense_agent_summary, f, indent=2)
        
    # 2. Selected Best Configuration: Rerank Local FlashRank
    rerank_agent_summary = evaluate_agent(agent, dev_q, "rerank_local")
    with open(os.path.join(RESULTS_DIR, "agent_dev_improved_rerank.json"), "w", encoding="utf-8") as f:
        json.dump(rerank_agent_summary, f, indent=2)
        
    print("\nALL DEV BENCHMARK EXPERIMENTS COMPLETED!")

if __name__ == "__main__":
    main()
