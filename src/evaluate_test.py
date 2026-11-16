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

def load_test_questions():
    with open(BENCHMARK_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    return [q for q in data if q.get("split") == "test"]

def evaluate_agent_on_test(agent: RAGAgent, test_questions: List[Dict[str, Any]], method_name: str):
    ans_count = 0
    unans_count = 0
    correct_abstentions = 0
    context_has_evidence = 0
    grounded_scores = []
    total_tokens = 0
    latencies = []
    detailed_traces = []
    
    print(f"\n================== EVALUATING ON TEST SPLIT (10 Questions): {method_name} ==================")
    for idx, q in enumerate(test_questions, start=1):
        is_ans = q.get("answerable", True)
        gold_cids = [ev["chunk_id"] for ev in q.get("evidence", [])]
        filt = q.get("filter")
        
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
            "tokens": trace["total_tokens"],
            "grounded_score": score if is_ans else None
        })
        print(f"[{idx}/10] {q['id']} | Ans: {is_ans} | Abstain: {trace['abstention']} | Time: {trace['latency_seconds']}s")
        
    avg_grounded = round(sum(grounded_scores) / len(grounded_scores), 2) if grounded_scores else 0.0
    abstention_rate = round(correct_abstentions / unans_count * 100, 2) if unans_count else 0.0
    context_ev_rate = round(context_has_evidence / ans_count * 100, 2) if ans_count else 0.0
    
    return {
        "method": method_name,
        "split": "test",
        "total_questions": len(test_questions),
        "total_answerable": ans_count,
        "total_unanswerable": unans_count,
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
    test_q = load_test_questions()
    print(f"Loaded {len(test_q)} Test questions ({sum(1 for q in test_q if q['answerable'])} answerable, {sum(1 for q in test_q if not q['answerable'])} unanswerable).")
    
    engine = RetrievalEngine()
    agent = RAGAgent(retrieval_engine=engine, model="gemini-3.5-flash-lite")
    
    # 1. Evaluate Baseline (Dense) on Test
    dense_test = evaluate_agent_on_test(agent, test_q, "dense")
    with open(os.path.join(RESULTS_DIR, "agent_test_baseline_dense.json"), "w", encoding="utf-8") as f:
        json.dump(dense_test, f, indent=2)
        
    # 2. Evaluate Selected Best Configuration (MiniLM Rerank) on Test
    rerank_test = evaluate_agent_on_test(agent, test_q, "rerank_local")
    with open(os.path.join(RESULTS_DIR, "agent_test_improved_rerank.json"), "w", encoding="utf-8") as f:
        json.dump(rerank_test, f, indent=2)
        
    print("\nALL TEST BENCHMARK EVALUATIONS COMPLETED!")

if __name__ == "__main__":
    main()
