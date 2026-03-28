# LangChain RAG Evaluation

Builds a RAG pipeline over two datasets and evaluates it using both a manual LLM-as-judge approach and the RAGAS framework.

→ [lab-langchain-evaluation.ipynb](lab-langchain-evaluation.ipynb)

## What it demonstrates

Getting a RAG pipeline to return answers is easy. Knowing whether those answers are actually good is harder. This lab addresses the core evaluation problem: traditional string matching fails the moment an LLM answers correctly but with different wording than the reference. The lab works through two evaluation strategies — a custom LLM judge scored on a 0–1 scale, and RAGAS's structured metrics (faithfulness, answer relevancy, context precision, context recall) — and shows how the results differ in what they surface.

## Key things worth noting

**Context recall is the bottleneck, not the LLM.** Across both datasets the weakest metric was context recall (0.63–0.75), meaning the retriever often failed to surface the relevant chunk — not that the model hallucinated. Improving retrieval (larger top-k, stronger embeddings, a reranker) would likely lift final answer quality more than swapping the LLM.

**A retrieval failure cascades visibly into answer quality.** The clearest example: the question "Which borough has the highest population?" should have a direct answer in the source document (Brooklyn), but the retriever missed the relevant chunk. The model hedged with "The context does not explicitly state...". RAGAS correctly scored this 0.00 on answer relevancy — which is exactly right, since a non-answer is irrelevant to the question. This chain of failure (recall → grounding → answer) is the main thing to take away from this lab.

**The LLM judge handles verbosity better than exact match.** A predicted answer of "The Ultra-Lofty 850 Stretch Down Hooded Jacket is from the DownTek collection" scored 0.9, not 1.0, against the reference "The DownTek collection" — semantically correct but not a perfect match. Exact string matching would have failed this entirely. The 0.9 vs 1.0 distinction is meaningful: the judge is distinguishing verbatim precision from semantic equivalence.

**The high/low demos validate that the metrics are real signals.** Deliberately swapping a real grounded answer for a fabricated one produced faithfulness 1.0 → 0.0. Swapping real retrieved docs for off-topic ones (Eiffel Tower, Python language) produced context recall 1.0 → 0.0. This rules out metrics that produce noise — these are measuring something actionable.

**Groq free-tier rate limits are a real constraint for evaluation pipelines.** Running 5 questions × 4 RAGAS metrics in a single session consumed the 100k daily token budget mid-evaluation, producing NaN faithfulness scores. This is not a code issue; it's an operational one. Evaluation pipelines need either a paid tier or session spacing built in.

## Stack

| Component | Choice |
|---|---|
| LLM | Groq `llama-3.3-70b-versatile` |
| Embeddings | `all-MiniLM-L6-v2` via HuggingFace (local, CPU) |
| Vector store | LangChain `InMemoryVectorStore` |
| Evaluation | Custom LLM judge + RAGAS (`faithfulness`, `answer_relevancy`, `context_precision`, `context_recall`) |
| Datasets | Outdoor Clothing Catalog (CSV), NYC Wikipedia excerpt (TXT) |

## How to run

1. Copy `.env` and fill in your keys:
   ```
   GROQ_API_KEY=
   HUGGINGFACEHUB_API_TOKEN=
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Open `lab-langchain-evaluation.ipynb` and run top to bottom. The notebook is split into two independent examples — Example 1 (clothing catalog, manual LLM judge) and Example 2 (NYC text, RAGAS). They share imports but use separate retrievers and chains.

> **Note:** RAGAS evaluation makes many LLM calls per question. On Groq's free tier (100k tokens/day), running all four metrics across five questions in one session may hit the rate limit. Run the evaluation cells in separate sessions or use a paid tier if you need complete results.
