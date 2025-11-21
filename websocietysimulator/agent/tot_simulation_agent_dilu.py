from . import SimulationAgent
import json
from ..llm import LLMBase
from .modules.planning_modules import PlanningBase 
from .modules.reasoning_modules import ReasoningTOT
from .modules.memory_modules import MemoryDILU
import logging
logging.basicConfig(level=logging.INFO)

class PlanningBaseline(PlanningBase):
    """Inherit from PlanningBase"""
    
    def __init__(self, llm, logger=None):
        """Initialize the planning module"""
        super().__init__(llm=llm, logger=logger)
    
    def __call__(self, task_description):
        """Override the parent class's __call__ method"""
        self.plan = [
            {
                'description': 'First I need to find user information',
                'reasoning instruction': 'None', 
                'tool use instruction': {task_description['user_id']}
            },
            {
                'description': 'Next, I need to find business information',
                'reasoning instruction': 'None',
                'tool use instruction': {task_description['item_id']}
            }
        ]
        return self.plan


class TOTSimulationAgentDILU(SimulationAgent):
    """TOT Simulation Agent with DILU memory."""
    
    def __init__(self, llm: LLMBase):
        """Initialize TOTSimulationAgentDILU"""
        super().__init__(llm=llm)
        logger = getattr(llm, 'logger', None)
        self.memory = MemoryDILU(llm=self.llm, logger=logger)
        self.planning = PlanningBaseline(llm=self.llm, logger=logger)
        self.reasoning = ReasoningTOT(profile_type_prompt='', memory=self.memory, llm=self.llm, logger=logger)
        
    def workflow(self):
        """
        Simulate user behavior
        Returns:
            tuple: (star (float), useful (float), funny (float), cool (float), review_text (str))
        """
        try:
            plan = self.planning(task_description=self.task)

            user = None
            business = None
            for sub_task in plan:
                if 'user' in sub_task['description']:
                    user_data = self.interaction_tool.get_user(user_id=self.task['user_id'])
                    user = str(user_data) if user_data else "No user data available"
                elif 'business' in sub_task['description']:
                    item_data = self.interaction_tool.get_item(item_id=self.task['item_id'])
                    business = str(item_data) if item_data else "No business data available"
            
            # Ensure we have user and business data
            if not user:
                user = str(self.interaction_tool.get_user(user_id=self.task['user_id']) or "No user data available")
            if not business:
                business = str(self.interaction_tool.get_item(item_id=self.task['item_id']) or "No business data available")
            reviews_item = self.interaction_tool.get_reviews(item_id=self.task['item_id'])
            for review in reviews_item:
                review_text = review['text']
                self.memory(f'review: {review_text}')
            reviews_user = self.interaction_tool.get_reviews(user_id=self.task['user_id'])
            # Handle case where user has no reviews
            if reviews_user and len(reviews_user) > 0:
                review_similar = self.memory(f'{reviews_user[0]["text"]}')
            else:
                review_similar = ""
            task_description = f'''
            You are a real human user on Yelp, a platform for crowd-sourced business reviews. Here is your Yelp profile and review history: {user}

            You need to write a review for this business: {business}

            Others have reviewed this business before: {review_similar}

            Please analyze the following aspects carefully:
            1. Based on your user profile and review style, what rating would you give this business? Remember that many users give 5-star ratings for excellent experiences that exceed expectations, and 1-star ratings for very poor experiences that fail to meet basic standards.
            2. Given the business details and your past experiences, what specific aspects would you comment on? Focus on the positive aspects that make this business stand out or negative aspects that severely impact the experience.
            3. Consider how other users might engage with your review in terms of:
            - Useful: How informative and helpful is your review?
            - Funny: Does your review have any humorous or entertaining elements?
            - Cool: Is your review particularly insightful or praiseworthy?

            Requirements:
            - Star rating must be one of: 1.0, 2.0, 3.0, 4.0, 5.0
            - If the business meets or exceeds expectations in key areas, consider giving a 5-star rating
            - If the business fails significantly in key areas, consider giving a 1-star rating
            - Review text should be 2-4 sentences, focusing on your personal experience and emotional response
            - Useful/funny/cool counts should be non-negative integers that reflect likely user engagement
            - Maintain consistency with your historical review style and rating patterns
            - Focus on specific details about the business rather than generic comments
            - Be generous with ratings when businesses deliver quality service and products
            - Be critical when businesses fail to meet basic standards

            Format your response exactly as follows:
            stars: [your rating]
            review: [your review]
            '''
            result = self.reasoning(task_description)
            
            # Parse the result, handling errors gracefully
            stars_line = None
            review_line = None
            try:
                if result and isinstance(result, str):
                    stars_lines = [line for line in result.split('\n') if 'stars:' in line.lower()]
                    review_lines = [line for line in result.split('\n') if 'review:' in line.lower()]
                    if stars_lines:
                        stars_line = stars_lines[0]
                    if review_lines:
                        review_line = review_lines[0]
            except Exception as e:
                print(f'Error parsing result: {e}')
                print(f'Result was: {result}')

            # Default values if parsing failed
            if not stars_line:
                stars = 3.0  # Default rating
            else:
                try:
                    parts = stars_line.split(':')
                    if len(parts) > 1:
                        stars = float(parts[1].strip())
                    else:
                        stars = 3.0
                except (ValueError, IndexError, AttributeError):
                    stars = 3.0
            
            if not review_line:
                review_text = "No review generated."
            else:
                try:
                    parts = review_line.split(':')
                    if len(parts) > 1:
                        review_text = parts[1].strip()
                    else:
                        review_text = "No review generated."
                except (ValueError, IndexError, AttributeError):
                    review_text = "No review generated."

            if len(review_text) > 512:
                review_text = review_text[:512]
                
            return {
                "stars": stars,
                "review": review_text
            }
        except Exception as e:
            print(f"Error in workflow: {e}")
            return {
                "stars": 0,
                "review": ""
            }

