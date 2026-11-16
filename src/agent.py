"""
Agentic RAG Module for Day 7
Uses gemini-3.5-flash-lite via ShopAIKey with Tool Calling.
Features:
- Configurable Retrieval Tools (Dense, BM25, Hybrid, Rerank, Query Transformation)
- Multi-turn Tool Calling with Stopping Conditions
- Context Budget Management & Deduplication
- Citation Grounding & Abstention Handling
- Full Tracing (queries, tool calls, latencies, tokens)
"""

import os
import time
import json
import re
from typing import Dict, Any, List, Optional
from dotenv import load_dotenv
from openai import OpenAI

from src.retrievers import RetrievalEngine

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
load_dotenv(os.path.join(BASE_DIR, ".env"))

class RAGAgent:
    def __init__(
        self,
        retrieval_engine: Optional[RetrievalEngine] = None,
        model: str = "gemini-3.5-flash-lite",
        max_turns: int = 3,
        context_budget: int = 3500
    ):
        self.engine = retrieval_engine or RetrievalEngine()
        self.model = model
        self.max_turns = max_turns
        self.context_budget = context_budget
        
        self.client = OpenAI(
            api_key=os.getenv("OPENAI_API_KEY"),
            base_url=os.getenv("OPENAI_BASE_URL", "https://api.shopaikey.com/v1")
        )
        
        # Tools definitions for OpenAI function calling
        self.tools = [
            {
                "type": "function",
                "function": {
                    "name": "search_knowledge_base",
                    "description": "Search the college textbook for information using various retrieval methods.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "The specific search query, keyword, or concept to search for."
                            },
                            "method": {
                                "type": "string",
                                "enum": ["dense", "bm25", "hybrid", "rerank", "query_transform"],
                                "description": "The retrieval method to use: 'dense' (semantic), 'bm25' (keyword/numbers), 'hybrid' (combined), 'rerank' (high precision reranked), 'query_transform' (expanded)."
                            },
                            "section_filter": {
                                "type": "string",
                                "description": "Optional section name to filter by (e.g. '3.3 Procrastination: The Enemy Within', '3.7 Enhanced Strategies for Time and Task Management')."
                            },
                            "top_k": {
                                "type": "integer",
                                "description": "Number of candidate chunks to retrieve (default 3, max 8)."
                            }
                        },
                        "required": ["query"]
                    }
                }
            }
        ]

    def _execute_tool(self, name: str, args: Dict[str, Any]) -> List[Dict[str, Any]]:
        if name != "search_knowledge_base":
            return []
            
        query = args.get("query", "")
        method = args.get("method", "hybrid")
        top_k = min(args.get("top_k", 3), 8)
        section = args.get("section_filter")
        where = {"section": section} if section else None
        
        if method == "dense":
            return self.engine.dense_search(query, top_k=top_k, where=where)
        elif method == "bm25":
            return self.engine.lexical_search(query, top_k=top_k, section_filter=section)
        elif method == "rerank":
            return self.engine.reranked_search(query, top_k=top_k)
        elif method == "query_transform":
            return self.engine.query_transformed_search(query, top_k=top_k)
        else: # default hybrid
            return self.engine.hybrid_search(query, top_k=top_k, where=where)

    def run(self, question: str, default_method: str = "hybrid", force_filter: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        start_time = time.time()
        
        system_prompt = """You are an accurate, grounded College Academic Advisor assistant.
Your task is to answer questions using ONLY the evidence retrieved from the textbook Chapter 3 (Managing Your Time and Priorities).

RULES:
1. Always call the search tool to find evidence before answering.
2. In your answer, cite the exact chunk_id in brackets (e.g., [B_chunk_0016]) whenever stating facts, figures, or definitions.
3. If the retrieved evidence DOES NOT contain sufficient information to answer the question, you MUST explicitly state that there is insufficient evidence in the document and abstain from answering (DO NOT hallucinate or guess facts).
4. Provide concise, clear, and direct answers.
"""
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": question}
        ]
        
        trace = {
            "question": question,
            "turns": 0,
            "tool_calls": [],
            "retrieved_chunks": {},
            "final_context": [],
            "answer": "",
            "citations": [],
            "abstention": False,
            "latency_seconds": 0.0,
            "total_tokens": 0
        }
        
        accumulated_chunks = {}
        
        for turn in range(self.max_turns):
            trace["turns"] += 1
            
            resp = None
            for retry in range(4):
                try:
                    resp = self.client.chat.completions.create(
                        model=self.model,
                        messages=messages,
                        tools=self.tools,
                        tool_choice="auto",
                        temperature=0.0
                    )
                    break
                except Exception as e:
                    if retry == 3:
                        raise e
                    time.sleep(2)
            
            choice = resp.choices[0]
            msg = choice.message
            
            if resp.usage:
                trace["total_tokens"] += resp.usage.total_tokens
                
            # If model wants to call tools
            if msg.tool_calls:
                messages.append(msg)
                
                for tc in msg.tool_calls:
                    fn_name = tc.function.name
                    raw_args = tc.function.arguments
                    try:
                        args = json.loads(raw_args) if isinstance(raw_args, str) else raw_args
                    except Exception:
                        args = {"query": question}
                        
                    # Apply forced method or filter if specified by benchmark config
                    if default_method:
                        args["method"] = default_method
                    if force_filter and "section" in force_filter:
                        args["section_filter"] = force_filter["section"]
                        
                    t_call_start = time.time()
                    results = self._execute_tool(fn_name, args)
                    t_call_dur = time.time() - t_call_start
                    
                    trace["tool_calls"].append({
                        "turn": turn + 1,
                        "tool": fn_name,
                        "args": args,
                        "duration_sec": round(t_call_dur, 3),
                        "results_count": len(results),
                        "result_chunk_ids": [r["chunk_id"] for r in results]
                    })
                    
                    # Store chunks for context deduplication
                    for r in results:
                        cid = r["chunk_id"]
                        if cid not in accumulated_chunks:
                            accumulated_chunks[cid] = r
                            
                    # Provide tool response back to LLM
                    tool_output = "\n".join([
                        f"[{r['chunk_id']}] (p.{r['pdf_page']}, {r['section']}): {r['text']}"
                        for r in results
                    ])
                    
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": tool_output
                    })
            else:
                # Model finalized answer
                trace["answer"] = msg.content
                break
                
        # If loop exited without answer
        if not trace["answer"]:
            # Final completion prompt
            final_resp = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=0.0
            )
            trace["answer"] = final_resp.choices[0].message.content
            if final_resp.usage:
                trace["total_tokens"] += final_resp.usage.total_tokens
                
        # Post-process context budget and citations
        total_len = 0
        final_context = []
        for cid, chunk in accumulated_chunks.items():
            clen = len(chunk["text"])
            if total_len + clen <= self.context_budget:
                final_context.append(chunk)
                total_len += clen
            else:
                break
        trace["final_context"] = final_context
        trace["retrieved_chunks"] = list(accumulated_chunks.keys())
        
        # Extract citations from answer e.g. [B_chunk_0016]
        citations = re.findall(r"\[(B_chunk_\d+)\]", trace["answer"])
        trace["citations"] = list(dict.fromkeys(citations))
        
        # Check abstention
        answer_lower = trace["answer"].lower()
        if any(phrase in answer_lower for phrase in ["insufficient evidence", "cannot find", "not mentioned", "not provided", "does not contain"]):
            trace["abstention"] = True
            
        trace["latency_seconds"] = round(time.time() - start_time, 2)
        return trace

