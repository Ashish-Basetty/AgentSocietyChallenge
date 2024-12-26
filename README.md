<div style="text-align: center; display: flex; align-items: center; justify-content: center; background-color: white; padding: 20px; border-radius: 30px;">
  <img src="./static/ASC.jpg" alt="AgentSociety Challenge Logo" width="100" style="margin-right: 20px; border-radius: 10%;">
  <h1 style="color: black; margin: 0; font-size: 2em;">WWW'25 AgentSociety Challenge: WebSocietySimulator</h1>
</div>

# 🚀 AgentSociety Challenge
![License](https://img.shields.io/badge/license-MIT-green) &ensp;
[![Competition Link](https://img.shields.io/badge/competition-link-orange)](https://www.codabench.org/competitions/4574/)

Welcome to the **WWW'25 AgentSociety Challenge**! This repository provides the tools and framework needed to participate in a competition that focuses on building **LLM Agents** for **user behavior simulation** and **recommendation systems** based on open source datasets.

Participants are tasked with developing intelligent agents that interact with a simulated environment and perform specific tasks in two competition tracks:
1. **User Behavior Simulation Track**: Agents simulate user behavior, including generating reviews and ratings.
2. **Recommendation Track**: Agents generate recommendations based on provided contextual data.

This repository includes:
- The core library `websocietysimulator` for environment simulation.
- Scripts for dataset processing and analysis.
- Example usage for creating and evaluating agents.

---

## Directory Structure

### 1. **`websocietysimulator/`**  
This is the core library containing all source code required for the competition.

- **`agents/`**: Contains base agent classes (`SimulationAgent`, `RecommendationAgent`) and their abstractions. Participants must extend these classes for their implementations.
- **`task/`**: Defines task structures for each track (`SimulationTask`, `RecommendationTask`).
- **`llm/`**: Contains base LLM client classes (`DeepseekLLM`, `OpenAILLM`).
- **`tools/`**: Includes utility tools:
  - `InteractionTool`: A utility for interacting with the Yelp dataset during simulations.
  - `EvaluationTool`: Provides comprehensive metrics for both recommendation (HR@1/3/5) and simulation tasks (RMSE, sentiment analysis).
- **`simulator.py`**: The main simulation framework, which handles task and groundtruth setting, evaluation and agent execution.

### 2. **`example/`**  
Contains usage examples of the `websocietysimulator` library. Includes sample agents and scripts to demonstrate how to load scenarios, set agents, and evaluate them.

### 3. **`data_process.py`**  
A script to process the raw Yelp dataset into the required format for use with the `websocietysimulator` library. This script ensures the dataset is cleaned and structured correctly for simulations.

---

## Quick Start

### 1. Install the Library

The repository is organized using [Python Poetry](https://python-poetry.org/). Follow these steps to install the library:

1. Clone the repository:
   ```bash
   git clone <this_repo>
   cd websocietysimulator
   ```

2. Install dependencies using Poetry:
   ```bash
   poetry install
   ```

3. Activate the virtual environment:
   ```bash
   poetry shell
   ```

---

### 2. Data Preparation

1. Download the raw dataset from the Yelp[1], Amazon[2] or Goodreads[3].
2. Run the `data_process.py` script to process the dataset:
   ```bash
   python data_process.py --input <path_to_raw_dataset> --output <path_to_processed_dataset>
   ```
- Check out the [Data Preparation Guide](./tutorials/data_preparation.md) for more information.

---

### 3. Organize Your Data

Ensure the dataset is organized in a directory structure similar to this:

```
<your_dataset_directory>/
├── item.json
├── review.json
├── user.json
```

You can name the dataset directory whatever you prefer (e.g., `dataset/`).

---

### 4. Develop Your Agent

Create a custom agent by extending either `SimulationAgent` or `RecommendationAgent`. Refer to the examples in the `example/` directory. Here's a quick template:

```python
from yelpsimulator.agents.simulation_agent import SimulationAgent

class MySimulationAgent(SimulationAgent):
    def forward(self):
        # Implement your logic here
        star = 4.0
        review_text = "Great experience!"
        behavior_metrics = (10, 2, 1)
        return star, review_text, behavior_metrics
```

- Check out the [Tutorial](./tutorials/agent_development.md) for Agent Development.
- Check out the [Baseline User Behavior Simulation Agent](./example/userBehaviorSimulation.py) for the baseline implementation (Track 1).
- Check out the [Baseline Recommendation Agent](./example/recommendation.py) for the baseline implementation (Track 2).
---

### 5. Evaluation your agent with training data

Run the simulation using the provided `Simulator` class:

```python
from websocietysimulator import Simulator
from my_agent import MySimulationAgent

# Initialize Simulator
simulator = Simulator(data_dir="path/to/your/dataset")

# Load scenarios
simulator.set_task_and_groundtruth(task_dir="path/to/task_directory", groundtruth_dir="path/to/groundtruth_directory")

# Set your custom agent
simulator.set_agent(MySimulationAgent)

# Set LLM client
simulator.set_llm(DeepseekLLM(api_key="Your API Key"))

# Run evaluation
agent_outputs = simulator.run_simulation()

# Evaluate the agent
evaluation_results = simulator.evaluate()
```

---

## Introduction to the `InteractionTool`

The `InteractionTool` is the core utility for interacting with the raw dataset. It provides an interface for querying user, business, review, tip, and check-in data within the context of a task.

### Functions

- **Get User Information**:
  Retrieve user data by user ID or current scenario context.
  ```python
  user_info = interaction_tool.get_user(user_id="example_user_id")
  ```

- **Get Item Information**:
  Retrieve item data by item ID or current scenario context.
  ```python
  item_info = interaction_tool.get_item(item_id="example_item_id")
  ```

- **Get Reviews**:
  Fetch reviews related to a specific item or user, filtered by time.
  ```python
  reviews = interaction_tool.get_reviews(item_id="example_item_id")
  ```


## License

This project is licensed under the MIT License. See the `LICENSE` file for details.