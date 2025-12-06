from websocietysimulator.agent import SimulationAgent
from websocietysimulator.llm import LLMBase
from websocietysimulator.agent.modules.planning_modules import (
    PlanningHUGGINGGPT, 
    PlanningTD, 
    PlanningOPENAGI,
    PlanningBase
)
from websocietysimulator.agent.modules.reasoning_modules import ReasoningBase
from websocietysimulator.agent.modules.memory_modules import MemoryDILU
import logging

logging.basicConfig(level=logging.INFO)


class OptimizedPlanningWrapper(PlanningBase):
    """
    Wrapper that adapts standard planning modules to work with task_description dict
    and provides a good few-shot example for Yelp user simulation.
    """
    
    def __init__(self, planning_module, llm, logger=None):
        """
        Initialize with a specific planning module.
        
        Args:
            planning_module: The planning module class (e.g., PlanningHUGGINGGPT)
            llm: LLM instance
            logger: Optional logger
        """
        super().__init__(llm=llm, logger=logger)
        self.planning_module = planning_module(llm=llm, logger=logger)
        
        # Few-shot example for Yelp user simulation
        self.few_shot_example = '''Task: Simulate a Yelp user reviewing a restaurant
[
    {
        "description": "Retrieve the user's profile and review history to understand their preferences and writing style",
        "reasoning instruction": "Analyze the user's past reviews to identify rating patterns, common themes, writing style, and preferences",
        "tool use instruction": {"action": "get_user", "user_id": "{user_id}"}
    },
    {
        "description": "Retrieve detailed information about the restaurant/business being reviewed",
        "reasoning instruction": "Examine business details including cuisine type, price range, location, and amenities",
        "tool use instruction": {"action": "get_item", "item_id": "{item_id}"}
    },
    {
        "description": "Retrieve existing reviews for this business to understand context and common themes",
        "reasoning instruction": "Review what other users have said to understand the restaurant's strengths and weaknesses",
        "tool use instruction": {"action": "get_reviews", "item_id": "{item_id}"}
    },
    {
        "description": "Generate a rating and review that matches the user's historical patterns and style",
        "reasoning instruction": "Based on user preferences, business characteristics, and review context, determine appropriate rating and write review in user's style",
        "tool use instruction": {"action": "generate_review"}
    }
]'''

    def __call__(self, task_description):
        """
        Call the planning module with proper format.
        
        Args:
            task_description: Dict with 'user_id' and 'item_id', or dict format from task
        """
        # Convert task_description to string format if it's a dict
        if isinstance(task_description, dict):
            if 'description' in task_description:
                # Task dict format from SimulationTask.to_dict()
                task_str = f"Simulate a Yelp user with ID {task_description.get('user_id', '')} writing a review for business with ID {task_description.get('item_id', '')}"
            else:
                # Direct dict with user_id and item_id
                task_str = f"Simulate a Yelp user with ID {task_description.get('user_id', '')} writing a review for business with ID {task_description.get('item_id', '')}"
        else:
            task_str = str(task_description)
        
        # Call the planning module with proper parameters
        plan = self.planning_module(
            task_type='user_behavior_simulation',
            task_description=task_str,
            feedback='',
            few_shot=self.few_shot_example
        )
        
        return plan


class OptimizedReasoning(ReasoningBase):
    """Optimized reasoning module for Yelp review generation."""
    
    def __init__(self, profile_type_prompt, llm, logger=None):
        super().__init__(profile_type_prompt=profile_type_prompt, memory=None, llm=llm, logger=logger)
        
    def __call__(self, task_description: str, n: int = 1):
        """
        Get reasoning result(s) from the underlying LLM.

        Args:
            task_description: prompt string
            n: number of candidates to return (default 1)

        Returns:
            If n==1: a string. If n>1: a list of strings.
        """
        prompt = f'''{task_description}'''
        messages = [{"role": "user", "content": prompt}]
        reasoning_result = self.llm(
            messages=messages,
            temperature=0.3,  # Slightly higher for more creative reviews
            max_tokens=4000,
            n=n
        )
        return reasoning_result


from websocietysimulator.tools.reranker import rerank_by_user_similarity, rerank_hybrid, rerank_by_llm


class OptimizedSimulationAgent(SimulationAgent):
    """
    Optimized Simulation Agent using PlanningHUGGINGGPT.
    
    This agent uses the PlanningHUGGINGGPT module which:
    1. Explicitly considers dependencies and order
    2. Minimizes tasks while ensuring completeness
    3. Thinks step-by-step about all required tasks
    """
    
    def __init__(self, llm: LLMBase, planning_module=PlanningHUGGINGGPT, n_candidates: int = 1, use_rerank: bool = False):
        """
        Initialize the optimized simulation agent.
        
        Args:
            llm: LLM instance
            planning_module: Planning module class to use (default: PlanningHUGGINGGPT)
        """
        super().__init__(llm=llm)
        logger = getattr(llm, 'logger', None)
        
        # Use PlanningHUGGINGGPT wrapped for proper interface
        self.planning = OptimizedPlanningWrapper(planning_module, llm=self.llm, logger=logger)
        self.reasoning = OptimizedReasoning(profile_type_prompt='', llm=self.llm, logger=logger)
        self.memory = MemoryDILU(llm=self.llm, logger=logger)
        # Candidate and rerank configuration
        self.n_candidates = max(1, int(n_candidates))
        self.use_rerank = bool(use_rerank)
        
    def workflow(self):
        """
        Simulate user behavior using optimized planning.
        Returns:
            dict: {"stars": float, "review": str}
        """
        try:
            # Generate plan using optimized planning module
            plan = self.planning(task_description=self.task)
            
            # Execute the plan
            user = None
            business = None
            reviews_item = []
            
            for sub_task in plan:
                description = sub_task.get('description', '').lower()
                
                # Extract tool use instruction
                tool_instruction = sub_task.get('tool use instruction', {})
                
                if 'user' in description or 'profile' in description or 'history' in description:
                    user_data = self.interaction_tool.get_user(user_id=self.task['user_id'])
                    user = str(user_data) if user_data else "No user data available"
                    
                elif 'business' in description or 'restaurant' in description or 'item' in description:
                    item_data = self.interaction_tool.get_item(item_id=self.task['item_id'])
                    business = str(item_data) if item_data else "No business data available"
                    
                elif 'review' in description and 'existing' in description:
                    reviews_item = self.interaction_tool.get_reviews(item_id=self.task['item_id'])
                    for review in reviews_item:
                        review_text = review.get('text', '')
                        if review_text:
                            self.memory(f'review: {review_text}')
            
            # Fallback: ensure we have user and business data
            if not user:
                user_data = self.interaction_tool.get_user(user_id=self.task['user_id'])
                user = str(user_data) if user_data else "No user data available"
            if not business:
                item_data = self.interaction_tool.get_item(item_id=self.task['item_id'])
                business = str(item_data) if item_data else "No business data available"
            if not reviews_item:
                reviews_item = self.interaction_tool.get_reviews(item_id=self.task['item_id'])
                for review in reviews_item[:5]:  # Limit to avoid too much memory
                    review_text = review.get('text', '')
                    if review_text:
                        self.memory(f'review: {review_text}')
            
            # Get user's past reviews for style reference
            reviews_user = self.interaction_tool.get_reviews(user_id=self.task['user_id'])
            if reviews_user and len(reviews_user) > 0:
                review_similar = self.memory(f'{reviews_user[0]["text"]}')
            else:
                review_similar = ""
            
            # Build comprehensive task description
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

            # Get N candidates from reasoning module
            reasoning_output = self.reasoning(task_description, n=self.n_candidates)

            # Normalize to list of strings
            if isinstance(reasoning_output, list):
                candidates_raw = reasoning_output
            else:
                candidates_raw = [reasoning_output]

            parsed_candidates = []
            for cand in candidates_raw:
                try:
                    stars_lines = [line for line in cand.split('\n') if 'stars:' in line.lower()]
                    review_lines = [line for line in cand.split('\n') if 'review:' in line.lower()]
                    if stars_lines:
                        stars_line = stars_lines[0]
                    else:
                        raise ValueError("No stars line found")
                    if review_lines:
                        review_line = review_lines[0]
                    else:
                        raise ValueError("No review line found")
                    stars_val = float(stars_line.split(':')[1].strip())
                    review_text = review_line.split(':', 1)[1].strip()
                except Exception:
                    # Fallback: try to extract numbers and the rest
                    try:
                        import re
                        m = re.search(r'stars:\s*([0-9]+)', cand, re.IGNORECASE)
                        stars_val = float(m.group(1)) if m else 3.0
                        m2 = re.search(r'review:\s*(.*)', cand, re.IGNORECASE | re.DOTALL)
                        review_text = m2.group(1).strip() if m2 else cand.strip()
                    except Exception:
                        stars_val = 3.0
                        review_text = cand.strip()

                if len(review_text) > 512:
                    review_text = review_text[:512]

                parsed_candidates.append({
                    'stars': float(max(1.0, min(5.0, round(stars_val)))),
                    'review': review_text,
                    'raw': cand
                })

            # If reranking enabled and multiple candidates, pick best via LLM scorer first
            selected_idx = 0
            if self.use_rerank and len(parsed_candidates) > 1:
                candidate_texts = [c['review'] for c in parsed_candidates]
                user_texts = [r.get('text', '') for r in reviews_user] if reviews_user else []
                embedding_provider = None
                if hasattr(self.llm, 'get_embedding_model'):
                    try:
                        embedding_provider = self.llm.get_embedding_model()
                    except Exception:
                        embedding_provider = None

                # Try LLM-based reranker first (this will use the MockLLM deterministically in mock mode)
                try:
                    selected_idx = rerank_by_llm(self.llm, candidate_texts, user_texts, parsed_candidates=parsed_candidates, n=self.n_candidates)
                except Exception:
                    # Fallback to embedding-based hybrid reranker or similarity-only reranker
                    try:
                        if embedding_provider is not None:
                            selected_idx = rerank_hybrid(candidate_texts, user_texts, embedding_provider=embedding_provider, parsed_candidates=parsed_candidates, n=self.n_candidates)
                        else:
                            selected_idx = rerank_by_user_similarity(candidate_texts, user_texts, embedding_provider=None)
                    except Exception:
                        selected_idx = 0

            chosen = parsed_candidates[selected_idx]

            return {
                "stars": float(chosen['stars']),
                "review": chosen['review']
            }

        except Exception as e:
            print(f"Error in workflow: {e}")
            import traceback
            traceback.print_exc()
            return {
                "stars": 0.0,
                "review": ""
            }

