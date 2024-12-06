import os
import json
from typing import List, Type, Dict, Any, Optional
from .tools.interaction_tool import InteractionTool
from .tools.evaluation_tool import RecommendationEvaluator, SimulationEvaluator
from .agents.simulation_agent import SimulationAgent
from .agents.recommendation_agent import RecommendationAgent
from .scenarios.simulation_scenario import SimulationScenario
from .scenarios.recommendation_scenario import RecommendationScenario
import numpy as np


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
        
        # 添加评估器
        self.recommendation_evaluator = RecommendationEvaluator()
        self.simulation_evaluator = SimulationEvaluator()
        self.simulation_outputs = []  # 存储模拟输出
        self.evaluation_results = []  # 存储评估结果

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

        self.simulation_outputs = []
        for scenario in self.scenarios:
            # Initialize the agent
            agent = self.agent_class(self.data_dir)
            
            # Set the scenario in the agent
            agent.insert_scenario(scenario)
            
            # Invoke the forward method and collect output
            try:
                output = agent.forward()
                result = {
                    "scenario": scenario.to_dict(),
                    "output": output
                }
                self.simulation_outputs.append(result)
            except NotImplementedError:
                result = {
                    "scenario": scenario.to_dict(),
                    "error": "Forward method not implemented by participant."
                }
                self.simulation_outputs.append(result)
        
        return self.simulation_outputs

    def evaluate(self, ground_truth: Optional[List[Dict]] = None) -> Dict[str, Any]:
        """
        Evaluate the simulation results.
        Args:
            ground_truth: Optional ground truth data for evaluation
        Returns:
            Dictionary containing evaluation metrics
        """
        if not self.simulation_outputs:
            raise RuntimeError("No simulation outputs to evaluate. Run simulation first.")

        evaluation_results = {}
        
        # 根据agent类型选择评估方法
        if issubclass(self.agent_class, RecommendationAgent):
            evaluation_results = self._evaluate_recommendation(ground_truth)
        elif issubclass(self.agent_class, SimulationAgent):
            evaluation_results = self._evaluate_simulation(ground_truth)
        
        self.evaluation_results.append(evaluation_results)
        return evaluation_results

    def _evaluate_recommendation(self, ground_truth: List[Dict]) -> Dict[str, Any]:
        """
        Evaluate recommendation results
        """
        if not ground_truth:
            raise ValueError("Ground truth data is required for recommendation evaluation")

        # 准备评估数据
        gt_pois = [item['poi_id'] for item in ground_truth]
        pred_pois = [
            output['output']['recommended_pois'] 
            for output in self.simulation_outputs
            if 'output' in output and 'recommended_pois' in output['output']
        ]

        # 计算评估指标
        metrics = self.recommendation_evaluator.calculate_hr_at_n(
            ground_truth=gt_pois,
            predictions=pred_pois,
            n=10  # 可以通过参数配置
        )

        return {
            'type': 'recommendation',
            'metrics': metrics.__dict__,
            'timestamp': self.scenarios[0].time if self.scenarios else None
        }

    def _evaluate_simulation(self, ground_truth: List[Dict]) -> Dict[str, Any]:
        """
        Evaluate simulation results
        """
        if not ground_truth:
            raise ValueError("Ground truth data is required for simulation evaluation")

        all_metrics = []
        for sim_output, gt_data in zip(self.simulation_outputs, ground_truth):
            if 'error' in sim_output:
                continue

            # 准备评估数据
            simulated_data = sim_output['output']
            metrics = self.simulation_evaluator.calculate_metrics(
                simulated_data=simulated_data,
                real_data=gt_data
            )
            all_metrics.append(metrics)

        # 计算平均指标
        avg_metrics = {
            'star_rmse': np.mean([m.star_rmse for m in all_metrics]),
            'sentiment_rmse': np.mean([m.sentiment_rmse for m in all_metrics]),
            'useful_rmse': np.mean([m.useful_rmse for m in all_metrics]),
            'cool_rmse': np.mean([m.cool_rmse for m in all_metrics]),
            'funny_rmse': np.mean([m.funny_rmse for m in all_metrics]),
            'overall_rmse': np.mean([m.overall_rmse for m in all_metrics])
        }

        return {
            'type': 'simulation',
            'metrics': avg_metrics,
            'detailed_metrics': [m.__dict__ for m in all_metrics],
            'timestamp': self.scenarios[0].time if self.scenarios else None
        }

    def get_evaluation_history(self) -> List[Dict[str, Any]]:
        """
        Get the history of evaluation results
        Returns:
            List of evaluation results
        """
        return self.evaluation_results