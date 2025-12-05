# Agent Society Challange - Yelp User Behavior Simulation

Our project is for simulating Yelp user behavior using LLM agents. This project uses the Yelp dataset to train and evaluate agents that can generate realistic restaurant reviews and ratings based on user profiles and business information.

## Overview

Our project implements an intelligent agent system that simulates how Yelp users would rate and review restaurants. The system uses large language models (LLMs) to generate contextually appropriate reviews and ratings by analyzing:

- User profiles and review history
- Business information and characteristics
- Existing reviews for context

Our project uses the `websocietysimulator` framework and implements a Tree of Thoughts (TOT) based simulation agent that generates realistic user behavior patterns.

## Features

- **User Behavior Simulation**: Generates realistic restaurant reviews and ratings based on user profiles
- **TOT-based Agent**: Uses Tree of Thoughts reasoning for improved decision-making
- **Yelp Dataset Integration**: Works with processed Yelp dataset for realistic simulations
- **Comprehensive Evaluation**: Includes metrics for preference estimation and review generation quality

## Project Structure

```
.
├── websocietysimulator/     # Core simulation framework
│   ├── agent/               # Agent implementations (TOTSimulationAgent, etc.)
│   ├── llm/                 # LLM client implementations (GeminiLLM)
│   ├── tasks/               # Task definitions
│   ├── tools/               # Interaction and evaluation tools
│   └── simulator.py         # Main simulation framework
├── dataset/                 # Processed Yelp dataset
│   ├── item.json           # Business/item data
│   ├── review.json         # Review data
│   └── user.json           # User data
├── example/                 # Example tasks and ground truth data
│   └── track1/yelp/        # Yelp simulation tasks
├── simulator_script.py      # Main script to run simulations
├── data_process.py          # Script to process raw Yelp dataset
└── tutorials/               # Documentation and guides
```

## Installation

### Prerequisites

- Python 3.10 or higher
- At least 16GB RAM (for dataset processing)
- Gemini API key

### Setup

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd AgentSocietyChallenge
   ```

2. Install dependencies:
   
   **Option 1: Using Poetry (Recommended)**
   ```bash
   poetry install && poetry shell
   ```
   
   **Option 2: Using pip**
   ```bash
   pip install -r requirements.txt && pip install .
   ```
   
   **Option 3: Using conda**
   ```bash
   conda create -n websocietysimulator python=3.11
   conda activate websocietysimulator
   pip install -r requirements.txt && pip install .
   ```

3. Set up environment variables:
   
   Create a `.env` file in the root directory:
   ```bash
   GEMINI_API_KEY=your_gemini_api_key_here
   ```

4. Verify installation:
   ```python
   import websocietysimulator
   ```

## Data Preparation

1. Download the Yelp dataset from [Yelp Dataset](https://www.yelp.com/dataset)

2. Process the dataset:
   ```bash
   python data_process.py --input <path_to_raw_dataset> --output dataset/
   ```
   
   The processed dataset should have the following structure:
   ```
   dataset/
   ├── item.json
   ├── review.json
   └── user.json
   ```
   
   **Note:** Dataset processing requires at least 16GB RAM.

   For more details, see the [Data Preparation Guide](./tutorials/data_preparation.md).

## Running Simulations

### Quick Start

Run the simulation using the provided script:

```bash
python simulator_script.py --output results/
```

This will:
- Load the dataset from `dataset/` directory
- Run 30 simulation tasks using the TOTSimulationAgent
- Save evaluation results to `results/evaluation_results.json`
- Save evaluation history to `results/evaluation_history.json`
- Save LLM logs to `results/llm_logs.jsonl` (unless `--disable-logging` is used)

### Command-line Options

- `--output`: Directory or file prefix for saving results (default: current directory)
- `--disable-logging`: Disable LLM call and diagnostic logging

### Examples

```bash
# Run with default settings (saves to current directory)
python simulator_script.py

# Run and save results to a specific directory
python simulator_script.py --output my_results/

# Run without logging
python simulator_script.py --output results/ --disable-logging
```

**Note:** Make sure your `.env` file contains `GEMINI_API_KEY` before running the script.

### Programmatic Usage

You can also run simulations programmatically:

```python
from websocietysimulator import Simulator
from websocietysimulator.agent import TOTSimulationAgent
from websocietysimulator.llm import GeminiLLM
import os
from dotenv import load_dotenv

load_dotenv()

# Initialize Simulator
simulator = Simulator(data_dir="dataset", device="auto", cache=False)

# Load scenarios
simulator.set_task_and_groundtruth(
    task_dir="example/track1/yelp/tasks",
    groundtruth_dir="example/track1/yelp/groundtruth"
)

# Set agent
simulator.set_agent(TOTSimulationAgent)

# Set LLM client
llm = GeminiLLM(api_key=os.getenv("GEMINI_API_KEY"))
simulator.set_llm(llm)

# Run simulation
agent_outputs = simulator.run_simulation(
    number_of_tasks=30, 
    enable_threading=True, 
    max_workers=10
)

# Evaluate results
evaluation_results = simulator.evaluate()
print(evaluation_results)
```

## Evaluation Metrics

The simulation evaluates agent performance using:

- **Preference Estimation**: Measures how well the agent predicts user ratings (RMSE-based)
- **Review Generation**: Evaluates the quality of generated reviews (sentiment analysis and similarity)
- **Overall Quality**: Combined metric for overall performance

Results are saved in JSON format with detailed metrics for each task.

## Agent Architecture

The project uses `TOTSimulationAgent`, which implements:

- **Tree of Thoughts (TOT) Reasoning**: Generates multiple reasoning paths and selects the best one
- **Planning Module**: Breaks down tasks into structured subtasks
- **Memory Module**: Maintains context across interactions
- **Interaction Tool**: Accesses user, business, and review data from the dataset

For more details on agent development, see the [Agent Development Guide](./tutorials/agent_development.md).

## Dataset

This project uses the Yelp Open Dataset, which includes:

- Business information (restaurants, locations, categories)
- User profiles and review history
- Historical reviews and ratings

The dataset is processed to extract relevant features for simulation tasks. For dataset details and download links, see the [Data Preparation Guide](./tutorials/data_preparation.md).

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.

## References

- Yelp Dataset: https://www.yelp.com/dataset
