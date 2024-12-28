import logging
import os
import json
from typing import List, Type, Dict, Any, Optional
from .tools.interaction_tool import InteractionTool
from .tools.evaluation_tool import RecommendationEvaluator, SimulationEvaluator
from .agent.simulation_agent import SimulationAgent
from .llm import LLMBase
from .agent.recommendation_agent import RecommendationAgent
from .tasks.simulation_task import SimulationTask
from .tasks.recommendation_task import RecommendationTask
import numpy as np

class Simulator:
    def __init__(self, data_dir: str = None, device: str = "auto"):
        """
        Initialize the Simulator.
        Args:
            data_dir: Path to the directory containing Yelp dataset files.
            device: Device to use for evaluation. "auto" (default) will use GPU if available, otherwise CPU. Available options: "gpu", "cpu", "auto".
        """
        logging.info("Start initializing Simulator")
        self.data_dir = data_dir
        if data_dir is None:
            self.interaction_tool = None
        else:
            self.interaction_tool = InteractionTool(data_dir)
        
        self.tasks = []  # List to store tasks
        self.groundtruth_data = []  # List to store groundtruth data
        self.agent_class = None
        self.llm = None
        self.recommendation_evaluator = RecommendationEvaluator()
        self.simulation_evaluator = SimulationEvaluator(device)
        self.simulation_outputs = []
        self.evaluation_results = []
        logging.info("Simulator initialized")

    def set_interaction_tool(self, interaction_tool: InteractionTool):
        self.interaction_tool = interaction_tool

    def set_task_and_groundtruth(self, task_dir: str, groundtruth_dir: str):
        """
        Load tasks from a directory.
        Args:
            task_dir: Directory containing task files.
            groundtruth_dir: Directory containing groundtruth files.
        """
        self.tasks = []  # Clear previous tasks

        for file_name in os.listdir(task_dir):
            file_path = os.path.join(task_dir, file_name)
            with open(file_path, 'r') as f:
                task_data = json.load(f)
                task_type = task_data.get('type')

                # Determine scenario type and create corresponding object
                if task_type == 'user_behavior_simulation':
                    task = SimulationTask(
                        user_id=task_data['user_id'],
                        business_id=task_data['business_id']
                    )
                elif task_type == 'recommendation':
                    task = RecommendationTask(
                        user_id=task_data['user_id'],
                        candidate_category=task_data['candidate_category'],
                        candidate_list=task_data['candidate_id_list'],
                        loc=task_data['loc']
                    )
                else:
                    raise ValueError(f"Unsupported task type: {task_type}")
                
                self.tasks.append(task)
        logging.info("Tasks loaded")

        self.groundtruth_data = []
        for file_name in os.listdir(groundtruth_dir):
            file_path = os.path.join(groundtruth_dir, file_name)
            with open(file_path, 'r') as f:
                groundtruth_data = json.load(f)
                self.groundtruth_data.append(groundtruth_data)
        logging.info("Groundtruth data loaded")

    def set_agent(self, agent_class: Type):
        """
        Set the agent class to be used for the simulation.
        Args:
            agent_class: A class inheriting from the abstract Agent class.
        """
        if not issubclass(agent_class, (SimulationAgent, RecommendationAgent)):
            raise ValueError("Agent class must inherit from SimulationAgent or RecommendationAgent.")
        self.agent_class = agent_class
        logging.info("Agent class set")
    def set_llm(self, llm: LLMBase):
        """
        Set the LLM to be used for the simulation.
        Args:
            llm: A class inheriting from the abstract LLM class.
        """
        self.llm = llm
        logging.info("LLM set")
    def run_simulation(self) -> List[Any]:
        """
        Run the simulation.
        Creates agents, invokes their forward methods, and collects outputs.
        Returns:
            List of outputs from agents for each scenario.
        """
        logging.info("Running simulation")
        if not self.agent_class:
            raise RuntimeError("Agent class is not set. Use set_agent() to set it.")

        self.simulation_outputs = []
        for task in self.tasks:
            # Initialize the agent
            agent = self.agent_class(llm=self.llm)
            agent.set_interaction_tool(self.interaction_tool)
            
            # Set the scenario in the agent
            agent.insert_task(task)
            
            # Invoke the forward method and collect output
            try:
                output = agent.forward()
                result = {
                    "task": task.to_dict(),
                    "output": output
                }
                self.simulation_outputs.append(result)
            except NotImplementedError:
                result = {
                    "task": task.to_dict(),
                    "error": "Forward method not implemented by participant."
                }
                self.simulation_outputs.append(result)
        logging.info("Simulation finished")
        return self.simulation_outputs

    def evaluate(self) -> Dict[str, Any]:
        """
        Evaluate the simulation results using the loaded groundtruth data.
        Returns:
            Dictionary containing evaluation metrics
        """
        logging.info("Evaluating simulation results")
        if not self.simulation_outputs:
            raise RuntimeError("No simulation outputs to evaluate. Run simulation first.")
        
        # 检查数据条目数量
        sim_count = len(self.simulation_outputs)
        gt_count = len(self.groundtruth_data)
        
        if sim_count != gt_count:
            logging.warning(f"Warning: Number of simulation outputs ({sim_count}) does not match ground truth data ({gt_count})")
            # 使用较小的数量
            eval_count = min(sim_count, gt_count)
            groundtruth_data = self.groundtruth_data[:eval_count]
            self.simulation_outputs = self.simulation_outputs[:eval_count]
        else:
            groundtruth_data = self.groundtruth_data
        
        evaluation_results = {}
        
        # 根据agent类型选择评估方法
        if issubclass(self.agent_class, RecommendationAgent):
            evaluation_results = self._evaluate_recommendation(groundtruth_data)
        elif issubclass(self.agent_class, SimulationAgent):
            evaluation_results = self._evaluate_simulation(groundtruth_data)
        
        # 添加数据条目信息到评估结果中
        evaluation_results['data_info'] = {
            'evaluated_count': eval_count if sim_count != gt_count else sim_count,
            'original_simulation_count': sim_count,
            'original_ground_truth_count': gt_count
        }
        
        self.evaluation_results.append(evaluation_results)
        logging.info("Evaluation finished")
        return evaluation_results

    def _evaluate_recommendation(self, ground_truth_data: List[Dict]) -> Dict[str, Any]:
        """
        Evaluate recommendation results using groundtruth
        """
        # 从ground truth数据中提取真实POI
        gt_pois = [item['ground truth'] for item in ground_truth_data]
        
        pred_pois = [
            output['output']
            for output in self.simulation_outputs
            if 'output' in output
        ]

        # 计算评估指标
        metrics = self.recommendation_evaluator.calculate_hr_at_n(
            ground_truth=gt_pois,
            predictions=pred_pois,
        )

        return {
            'type': 'recommendation',
            'metrics': metrics.__dict__,
        }

    def _evaluate_simulation(self, ground_truth_data: List[Dict]) -> Dict[str, Any]:
        """
        Evaluate simulation results
        """
        all_metrics = []
        for sim_output, gt_data in zip(self.simulation_outputs, ground_truth_data):
            if 'error' in sim_output:
                continue
            
            # 准备评估数据：从ground truth中提取所需字段
            gt_info = {
                'stars': gt_data['stars'],
                'useful': gt_data['useful'],
                'funny': gt_data['funny'],
                'cool': gt_data['cool'],
                'review': gt_data['review']
            }
            
            simulated_data = sim_output['output']
            metrics = self.simulation_evaluator.calculate_metrics(
                simulated_data=simulated_data,
                real_data=gt_info
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
        }

    def get_evaluation_history(self) -> List[Dict[str, Any]]:
        """
        Get the history of evaluation results
        Returns:
            List of evaluation results
        """
        return self.evaluation_results