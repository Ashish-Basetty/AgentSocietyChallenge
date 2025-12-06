Title: Simulating Yelp User Reviews with Multi-Candidate Generation, Memory, and Reranking

Abstract

This paper describes a system for simulating Yelp user reviews that combines task decomposition, episodic memory, multi-candidate language-model generation, and candidate selection strategies. The simulator aims to produce review text and star ratings that closely match real users in sentiment, topicality, emotional tone, and rating behavior. We evaluate different reranking strategies (embedding-based, hybrid heuristics, and LLM-based scoring) and memory architectures (verbatim episodic memory, generative memory, plan-based memory, and summarization memory). Experiments show that multi-candidate generation with effective reranking improves overall simulation fidelity when the candidate pool is sufficiently large, and that verbatim episodic memory preserves stylistic and topical cues that materially benefit review generation fidelity.

1. Introduction

1.1 Motivation

Accurately simulating user reviews is valuable for system evaluation, dataset augmentation, and behavior-driven testing. The complexity of human reviews—combining sentiment, emotion, content details, and idiosyncratic style—makes simple single-shot generation inadequate for high-fidelity simulation. We propose a modular approach that decomposes the simulation task, gathers context via memory and tools, generates multiple candidate reviews using a language model, and selects the best candidate with a reranker that encodes stylistic, topical, and quality signals.

1.2 Contributions

- A structured simulation pipeline tailored to user review generation that integrates planning, episodic memory, and multi-candidate reasoning.
- A comparison of reranking strategies showing trade-offs among embedding similarity, hybrid heuristics, and LLM-based scoring.
- Empirical evidence that verbatim episodic memory improves both rating and textual similarity metrics.
- Practical recommendations for balancing cost, latency, and fidelity in real-world deployments.

2. Method / System Design

2.1 Task Definition and Objectives

The simulation objective is to generate, for each user-item pair, a compact review (2–4 sentences) and a discrete star rating (1–5). The output should reflect the user’s historical style and preferences, be consistent with the item’s characteristics, and align with norms observed in real review data.

Evaluation uses two complementary axes:

- Preference estimation: how closely predicted star ratings match ground truth (normalized absolute star error).
- Review generation quality: an aggregate of sentiment alignment, emotion distribution similarity, and topical similarity between simulated and real reviews.

The combined overall quality metric is the mean of preference estimation and review generation.

2.2 Modular Workflow

The simulator follows a pipeline of four phases:

- Planning: Decompose the top-level task into subtasks that explicitly instruct which context to retrieve (e.g., fetch user profile, fetch item details, pull prior reviews).
- Tool-based Information Gathering: Execute subtasks to collect user history and item context, used as context for generation.
- Reasoning & Candidate Generation: Ask a reasoning model to produce N candidate reviews (each candidate includes a numeric rating and text).
- Reranking & Selection: Score candidates using one of several strategies and choose the top candidate to return.

2.3 Candidate Generation

Multi-candidate generation increases the chance that at least one generated output matches the combination of user style, sentiment, and topical detail necessary for high-fidelity simulation. Candidate outputs adhere to a concise structured format (rating + review text) to facilitate parsing and scoring. The number of candidates, N, is a tunable parameter influencing diversity, computation, and selection potential.

2.4 Memory Architectures

We evaluated four memory designs for serving user history to the reasoning module:

- Verbatim Episodic Memory (DILU): Store full prior task trajectories and return the single most similar stored trajectory verbatim.
- Generative Memory: Retrieve several candidate past trajectories, then use the model to score their relevance and return the highest-scoring memory or a synthesized exemplar.
- Plan‑Transform Memory (TP): Use retrieved memories to generate new strategic plans or condensed actions to influence current reasoning.
- Summarization Memory (Voyager): Summarize trajectories into concise descriptions to economize retrieval bandwidth.

Key design trade-offs:

- Verbatim memory preserves maximal detail and style but scales less efficiently without indexing optimizations.
- Generative and plan-based memories can generalize but risk information loss and increased stochasticity due to extra model calls.
- Summarization reduces retrieval size but may lose fine-grained cues critical for style matching.

2.5 Reranking Strategies

We implemented and compared three reranking strategies:

- Embedding similarity:
  - Compute semantic embeddings for candidates and user historical texts;
  - Score candidates by average cosine similarity to the user’s prior reviews.
  - Strengths: fast, deterministic, strongly favors topical/style alignment.
  - Weaknesses: less sensitive to fluency, sentence-level coherence, or consistency with rated stars.

- Hybrid Heuristic:
  - Normalize embedding similarity and a candidate quality heuristic to 0..1, then combine them with a weight that can vary with N.
  - The heuristic includes length suitability, sentence count appropriateness, and polarity consistency between text sentiment and numeric rating.
  - Strengths: balances semantic alignment with surface-level quality features; more robust to outliers.

- LLM-based Scorer:
  - Prompt a language model to score a candidate on a numeric scale for usefulness, clarity, and style consistency.
  - Strengths: captures nuanced judgments not directly encoded in embeddings or heuristics.
  - Weaknesses: cost/latency scales with N, scoring can be miscalibrated, and responses may require robust parsing.

2.6 Evaluation Pipeline

The evaluation aggregates per-task errors into dataset-level metrics. Sentiment alignment is measured via a standard polarity analyzer; emotion similarity via a transformer-based classifier producing distributional vectors; and topical similarity via sentence embeddings and cosine distance. These three components are combined into a review-generation error metric and merged with preference estimation to form overall quality.

3. Experiment Results

3.1 Experimental Setup

We performed controlled experiments sweeping candidate pool sizes N ∈ {1, 3, 5, 10, 20} to assess how selection quality scales with candidate diversity. To enable rapid iteration and reproducibility, many experiments used deterministic substitutes for language models and embeddings; these offline experiments are deterministic proxies that allow comparisons of reranker behavior and memory effects independent of external API variability.

3.2 Key Findings

- Baseline (N=1): Provided a stable baseline for comparison.
- Hybrid reranker: Showed steady improvements with increasing N, particularly up to N=10, where gains in both preference estimation and review generation were observed.
- LLM-based reranker: Achieved the largest single improvement in overall quality at N=10 in offline experiments, demonstrating the capacity to pick superior candidates from larger pools; performance at lower N could be inconsistent depending on scoring calibration.
- Memory effects: The verbatim episodic memory consistently outperformed alternative memory approaches in reproduction of rating behavior and fidelity of generated text.

3.3 Representative Metrics

Representative aggregate values from controlled runs indicate baseline overall quality around 0.66 for N=1. With effective reranking and N=10, overall quality rose to approximately 0.69 in our experiments. Decomposition shows improvements distributed across both rating fidelity and textual similarity components.

4. Experiment Analysis

4.1 Where Gains Come From

- Candidate pools increase the probability that at least one candidate captures user-specific cues. Embedding and hybrid rerankers can select candidates that are topically aligned and conform to expected sentence structure and polarity, improving both numerical rating and textual measures.
- Verbatim memory supplies concrete, stylistic cues directly usable by the reasoning model; this direct reuse lowers the chance that the model drifts stylistically or semantically from the target user.

4.2 Reranker Trade-offs

- Embedding-only rerankers are computationally efficient and favor topical consistency, but can be blind to fluency or rating-text polarity mismatch.
- Hybrid rerankers mitigate the blind spots of embeddings by explicitly penalizing or rewarding surface-level quality features and polarity consistency.
- LLM-based rerankers capture nuanced qualities but carry variance from prompt formulation and model behavior. They can be miscalibrated across models or prompt contexts and incur significant cost when scoring many candidates individually.

4.3 Memory Behavior

- Verbatim episodic retrieval tends to produce deterministic, high-fidelity context that improves generated outputs without adding model-induced noise.
- Generative and plan-based memories introduce secondary model calls that can both generalize and introduce variance; they can be helpful for generalization tasks but are less reliable for reproducing precise stylistic markers.

4.4 Robustness and Failure Modes

- Small candidate sets may not provide enough diversity to benefit from reranking; in such cases, naive selection or fallback heuristics perform as well or better.
- Misparsing of candidate outputs (e.g., missing numeric ratings or malformed templates) can trigger fallback heuristics that reduce selection quality.
- Scoring calibration and prompt sensitivity for LLM scorers can lead to inconsistent selection across model versions or prompts.

5. Limitations and Future Work

5.1 Limitations of Current Study

- Offline experiments use deterministic proxies that simplify comparative analysis but do not substitute for full evaluations with production LLMs and embedding models.
- The evaluation metrics, while multi-faceted, are still automated proxies and do not fully capture human judgment of helpfulness, clarity, or trustworthiness.
- Experiments were run with moderate sample sizes; scaling to broader samples and multiple runs is needed for robust statistical claims.

5.2 Proposed Extensions

- Real-model validation: Re-run candidate generation and reranking with production-caliber LLMs and embedding models to assess whether gains generalize beyond the mock setup.
- Single-call ranking: Implement and evaluate a single LLM call that accepts all N candidates and returns an explicit ranking or scores in a single response, reducing cost and latency.
- Learned reranker: Collect labeled comparisons and train a reranker (e.g., a cross-encoder or lightweight classifier) to replace or complement LLM scoring, improving calibration and reducing per-candidate cost.
- Hybrid memory: Combine DILU's verbatim retrieval with a lightweight summarizer that triggers only when similarity is low, balancing fidelity and compactness.
- Human evaluation: Complement automatic metrics with human judgments on helpfulness, naturalness, and fidelity to the user's style.

6. Conclusion

This study demonstrates that multi-candidate generation combined with an effective selection mechanism can enhance the fidelity of simulated user reviews, particularly when the candidate pool is large and selection uses informative signals. We find that deterministic, verbatim episodic memory provides consistent benefits by preserving detailed stylistic and topical cues. Hybrid rerankers that combine semantic similarity with heuristic quality signals offer stable improvements and lower variance than per-candidate LLM scoring in practice, although LLM-based scorers can capture nuanced qualitative judgments if calibrated carefully.

Practical recommendations for practitioners include:

- Employ multi-candidate generation with reranking when fidelity matters; tune N to balance diversity and compute cost.
- Favor hybrid rerankers or single-call LLM ranking to minimize cost while maintaining selection quality.
- Use verbatim episodic memory where precise historical cues are important; consider hybrid approaches for scalability.
- Validate improvements in real-mode with production LLMs and incorporate human evaluation to align automated metrics with subjective quality.

Acknowledgements

The system and experiments were developed through iterative design, rigorous offline testing, and extensive evaluation. Appreciation is due to all contributors who implemented modular planning, memory, reasoning, and evaluation components, and to reviewers who validated experiment design.

References

A curated set of references on user simulation, reranking/learning-to-rank, semantic embeddings, sentiment and emotion analysis, and LLM-based evaluation tools would be listed in a final draft.
