"""Compare single-candidate vs N-candidate+rerank modes using OptimizedSimulationAgent.

This script runs a small number of tasks (configurable) for both modes and
prints & saves evaluation results to `results/`.

Usage:
    python compare_candidate_modes.py --num-tasks 5

Ensure you have dataset in `dataset/` and GEMINI_API_KEY in your environment.
"""
import os
import json
import argparse
from websocietysimulator import Simulator
from websocietysimulator.llm import GeminiLLM
from websocietysimulator.utils import LLMLogger
from dotenv import load_dotenv
from optimized_simulation_agent import OptimizedSimulationAgent

load_dotenv()

parser = argparse.ArgumentParser()
parser.add_argument('--num-tasks', type=int, default=5)
parser.add_argument('--output-dir', type=str, default='results')
parser.add_argument('--mock', action='store_true', help='Run in mock mode without external LLM or heavy deps')
args = parser.parse_args()

os.makedirs(args.output_dir, exist_ok=True)

logger = LLMLogger.get_instance(log_file_path=os.path.join(args.output_dir, 'compare_llm_logs.jsonl'), enabled=True)

modes = [
    {'name': 'single', 'n_candidates': 1, 'use_rerank': False},
    {'name': 'n3_rerank', 'n_candidates': 3, 'use_rerank': True}
]

# Added mode to test 10 candidates with reranking
modes.append({'name': 'n10_rerank', 'n_candidates': 10, 'use_rerank': True})

results = {}
for mode in modes:
    print(f"Running mode: {mode['name']}")
    simulator = Simulator(data_dir='dataset', device='cpu', cache=False)
    simulator.set_task_and_groundtruth(
        task_dir='example/track1/yelp/tasks',
        groundtruth_dir='example/track1/yelp/groundtruth'
    )
    # Create a small agent subclass to configure candidates/rerank
    class ModeAgent(OptimizedSimulationAgent):
        def __init__(self, llm):
            super().__init__(llm=llm, n_candidates=mode['n_candidates'], use_rerank=mode['use_rerank'])

    simulator.set_agent(ModeAgent)

    # Support mock mode for offline testing (no external LLM, chroma, transformers needed)
    if args.mock:
        class MockEmbeddingProvider:
            def __init__(self, dim=64):
                self.dim = dim
            def _vec(self, text):
                v = [0.0] * self.dim
                for tok in str(text).split():
                    idx = abs(hash(tok)) % self.dim
                    v[idx] += 1.0
                return v
            def embed_documents(self, texts):
                return [self._vec(t) for t in texts]
            def embed_query(self, text):
                return self._vec(text)

        class MockLLM:
            def __init__(self):
                self._emb = MockEmbeddingProvider()
            def __call__(self, messages=None, temperature=0.0, max_tokens=500, n=1, **kwargs):
                # produce deterministic mock candidates based on prompt length
                base = "Mock review candidate"
                if n == 1:
                    return f"stars: 4\nreview: {base} single"
                else:
                    outs = []
                    for i in range(n):
                        stars = 3 + (i % 3)
                        outs.append(f"stars: {stars}\nreview: {base} {i+1}")
                    return outs
            def get_embedding_model(self):
                return self._emb

        llm = MockLLM()
    else:
        llm = GeminiLLM(api_key=os.getenv('GEMINI_API_KEY'), logger=logger)
    simulator.set_llm(llm)
    simulator.logger = logger

    agent_outputs = simulator.run_simulation(number_of_tasks=args.num_tasks, enable_threading=False)
    evaluation_results = simulator.evaluate()
    out_file = os.path.join(args.output_dir, f"{mode['name']}_results.json")
    with open(out_file, 'w') as f:
        json.dump(evaluation_results, f, indent=2)
    results[mode['name']] = evaluation_results
    print(f"Saved results to {out_file}")

# Save comparison summary
with open(os.path.join(args.output_dir, 'comparison_summary.json'), 'w') as f:
    json.dump(results, f, indent=2)

print('Done. Results saved to', args.output_dir)
