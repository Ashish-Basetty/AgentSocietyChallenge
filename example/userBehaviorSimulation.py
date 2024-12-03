from yelpsimulator import Simulator
from yelpsimulator.agents import SimulationAgent


class MySimulationAgent(SimulationAgent):
    """
    Participant's implementation of SimulationAgent.
    """

    def forward(self):
        """
        Simulate user behavior.
        Returns:
            tuple: (star (float), review_text (str), behavior_metrics (tuple))
        """
        user = self.interaction_tool.get_user()
        business = self.interaction_tool.get_business()
        reviews = self.interaction_tool.get_reviews()

        # Example implementation (participants can customize this logic)
        star = 4.0  # Simulated rating
        review_text = "Great place with excellent food!"  # Simulated review text
        behavior_metrics = (len(reviews), 0, 0)  # Simulated behavior (e.g., total reviews, tips, etc.)

        return star, review_text, behavior_metrics


# Participant's main entry point
if __name__ == "__main__":
    # Initialize the simulator
    simulator = Simulator(data_dir="path/to/dataset")

    # Set scenarios (participants may receive predefined scenarios from the organizers)
    simulator.set_scenario(scenario_dir="user_behavior_simulation_scenarios")

    # Use the participant's custom agent
    simulator.set_agent(MySimulationAgent)

    # Run the simulation
    results = simulator.run_simulation()

    # Print results
    for result in results:
        scenario_info = result.get("scenario")
        output = result.get("output", result.get("error"))
        print(f"Scenario: {scenario_info}, Output: {output}")