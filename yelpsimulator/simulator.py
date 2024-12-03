import os
import json
from typing import List, Type, Dict, Any
from tools.interaction_tool import InteractionTool
from agents.simulation_agent import SimulationAgent
from agents.recommendation_agent import RecommendationAgent
from scenarios.simulation_scenario import SimulationScenario
from scenarios.recommendation_scenario import RecommendationScenario


class Simulator:
    def __init__(self, data_dir: str):
        """
        Initialize the Simulator.
        Args:
            data_dir: Path to the directory containing Yelp dataset files.
        """
        self.data_dir = data_dir
        self.interaction_tool = InteractionTool(data_dir)
        self.scenarios = []  # List to store scenarios
        self.agent_class = None  # The agent class used for simulation

    def set_scenario(self, scenario_dir: str):
        """
        Load scenarios from a directory.
        Args:
            scenario_dir: Directory containing scenario files.
        """
        self.scenarios = []  # Clear previous scenarios

        for file_name in os.listdir(scenario_dir):
            file_path = os.path.join(scenario_dir, file_name)
            with open(file_path, 'r') as f:
                scenario_data = json.load(f)
                scenario_type = scenario_data.get('type')

                # Determine scenario type and create corresponding object
                if scenario_type == 'user_behavior_simulation':
                    scenario = SimulationScenario(
                        time=scenario_data['time'],
                        user=scenario_data['user'],
                        business=scenario_data['business']
                    )
                elif scenario_type == 'recommendation':
                    scenario = RecommendationScenario(
                        time=scenario_data['time'],
                        context=scenario_data['context'],
                        candidate_poi=scenario_data['candidate_poi']
                    )
                else:
                    raise ValueError(f"Unsupported scenario type: {scenario_type}")
                
                self.scenarios.append(scenario)

    def set_agent(self, agent_class: Type):
        """
        Set the agent class to be used for the simulation.
        Args:
            agent_class: A class inheriting from the abstract Agent class.
        """
        if not issubclass(agent_class, (SimulationAgent, RecommendationAgent)):
            raise ValueError("Agent class must inherit from SimulationAgent or RecommendationAgent.")
        self.agent_class = agent_class

    def run_simulation(self) -> List[Any]:
        """
        Run the simulation.
        Creates agents, invokes their forward methods, and collects outputs.
        Returns:
            List of outputs from agents for each scenario.
        """
        if not self.agent_class:
            raise RuntimeError("Agent class is not set. Use set_agent() to set it.")

        outputs = []
        for scenario in self.scenarios:
            # Initialize the agent
            agent = self.agent_class(self.data_dir)
            
            # Set the scenario in the agent
            agent.insert_scenario(scenario)
            
            # Invoke the forward method and collect output
            try:
                output = agent.forward()
                outputs.append({
                    "scenario": scenario.to_dict(),
                    "output": output
                })
            except NotImplementedError:
                outputs.append({
                    "scenario": scenario.to_dict(),
                    "error": "Forward method not implemented by participant."
                })
        
        return outputs