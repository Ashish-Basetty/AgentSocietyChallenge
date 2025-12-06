"""Run a mock-mode sweep over multiple n-candidate values and save results.

Usage:
    python mock_sweep_n_values.py --num-tasks 50 --output-dir results/sweep

This script uses the same MockLLM/MockEmbeddingProvider as
`compare_candidate_modes.py` to run offline experiments without external deps.
"""
import os
import json
import argparse
from websocietysimulator import Simulator

parser = argparse.ArgumentParser()
parser.add_argument('--num-tasks', type=int, default=50)
parser.add_argument('--output-dir', type=str, default='results/sweep')
args = parser.parse_args()

os.makedirs(args.output_dir, exist_ok=True)

# Define the mock LLM and embedding provider (same behavior as compare script)
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

def run_mode(n, use_rerank, num_tasks, out_dir):
    simulator = Simulator(data_dir='dataset', device='cpu', cache=False)
    simulator.set_task_and_groundtruth(
        task_dir='example/track1/yelp/tasks',
        groundtruth_dir='example/track1/yelp/groundtruth'
    )

    # Create inline agent class with configured n and rerank
    from optimized_simulation_agent import OptimizedSimulationAgent
    class ModeAgent(OptimizedSimulationAgent):
        def __init__(self, llm):
            super().__init__(llm=llm, n_candidates=n, use_rerank=use_rerank)

    simulator.set_agent(ModeAgent)

    # Use MockLLM for offline run
    llm = MockLLM()
    simulator.set_llm(llm)

    agent_outputs = simulator.run_simulation(number_of_tasks=num_tasks, enable_threading=False)
    evaluation_results = simulator.evaluate()

    out_file = os.path.join(out_dir, f"n{n}{'_rerank' if use_rerank else ''}_results.json")
    with open(out_file, 'w') as f:
        json.dump(evaluation_results, f, indent=2)
    return evaluation_results

def main():
    ns = [1, 3, 5, 10, 20]
    summary = {}
    for n in ns:
        print(f"Running n={n} (rerank=True for n>1)")
        use_rerank = (n > 1)
        res = run_mode(n=n, use_rerank=use_rerank, num_tasks=args.num_tasks, out_dir=args.output_dir)
        summary[f"n{n}"] = res

    # Save summary
    with open(os.path.join(args.output_dir, 'sweep_summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)
    print('Sweep complete. Results saved to', args.output_dir)

if __name__ == '__main__':
    main()
