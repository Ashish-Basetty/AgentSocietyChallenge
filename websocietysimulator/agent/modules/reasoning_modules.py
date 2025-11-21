from collections import Counter
import re

class ReasoningBase:
    def __init__(self, profile_type_prompt, memory, llm, logger=None):
        """
        Initialize the reasoning base class
        
        Args:
            profile_type_prompt: Profile type prompt
            memory: Memory module
            llm: LLM instance used to generate reasoning
            logger: Optional logger instance for diagnostics
        """
        self.profile_type_prompt = profile_type_prompt
        self.memory = memory
        self.llm = llm
        self.logger = logger
    
    def process_task_description(self, task_description):
        examples = []
        example_1_description = '''
        stars: 1.0
        review: I had high hopes for the Masters Inn Fairgrounds, but my experience was a major letdown. The room was infested with roaches, and the furniture was old and falling apart. The staff was unhelpful and seemed disinterested in addressing the issues. The overall cleanliness and maintenance were poor, which made the stay very uncomfortable. Given my preference for well-maintained and clean environments, this place did not meet any of my standards. I would not recommend it to anyone, especially families or those looking for a pleasant stay.
        '''
        examples.append(example_1_description)
        example_2_description = '''
        stars: 3.0
        review: I visited Arizona Bug Doctor for pest control services, and my experience was mixed. On the positive side, the technicians were knowledgeable and thorough, which is important when dealing with pests. However, scheduling the appointment was a bit of a challenge, and the office staff could be more responsive. The limited hours of operation were also inconvenient. Overall, the service was effective, but there's room for improvement in customer service and flexibility.
        '''
        examples.append(example_2_description)
        return examples, task_description

class ReasoningIO(ReasoningBase):
    def __call__(self, task_description: str, feedback :str= ''):
        examples, task_description = self.process_task_description(task_description)
        prompt = '''Your instructions must follow the examples.
Here are some examples.
{examples}
Here is the task:
{task_description}'''
        prompt = prompt.format(task_description=task_description, examples=examples)
        messages = [{"role": "user", "content": prompt}]
        
        if self.logger:
            self.logger.log_module_diagnostic(
                module_name="reasoning",
                function_name="__call__",
                event_type="reasoning_started",
                data={
                    "reasoning_type": "IO",
                    "task_length": len(task_description),
                    "has_feedback": bool(feedback),
                    "examples_count": len(examples)
                }
            )
        
        reasoning_result = self.llm(
            messages=messages,
            temperature=0.1,
        )
        
        if self.logger:
            self.logger.log_module_diagnostic(
                module_name="reasoning",
                function_name="__call__",
                event_type="reasoning_completed",
                data={
                    "reasoning_type": "IO",
                    "result_length": len(reasoning_result) if isinstance(reasoning_result, str) else len(str(reasoning_result))
                }
            )
        
        return reasoning_result
    
class ReasoningCOT(ReasoningBase):
    def __call__(self, task_description: str, feedback :str= ''):
        examples, task_description = self.process_task_description(task_description)
        prompt = '''Solve the task step by step. Your instructions must follow the examples.
Here are some examples.
{examples}
Here is the task:
{task_description}'''
        prompt = prompt.format(task_description=task_description, examples=examples)
        messages = [{"role": "user", "content": prompt}]
        reasoning_result = self.llm(
            messages=messages,
            temperature=0.1,
        )
        return reasoning_result

class ReasoningCOTSC(ReasoningBase):
    def __call__(self, task_description: str, feedback :str= ''):
        examples, task_description = self.process_task_description(task_description)
        prompt = '''Solve the task step by step. Your instructions must follow the examples.
Here are some examples.
{examples}
Here is the task:
{task_description}'''
        prompt = prompt.format(task_description=task_description, examples=examples)
        messages = [{"role": "user", "content": prompt}]
        reasoning_results = self.llm(
            messages=messages,
            temperature=0.1,
            n=5
        )
        string_counts = Counter(reasoning_results)
        most_common = string_counts.most_common(1)
        if most_common and len(most_common) > 0:
            reasoning_result = most_common[0][0]
        else:
            reasoning_result = reasoning_results[0] if reasoning_results else "stars: 3.0\nreview: Unable to generate review."
        return reasoning_result
    
class ReasoningTOT(ReasoningBase):
    def __call__(self, task_description: str, feedback :str= ''):
        examples, task_description = self.process_task_description(task_description)
        prompt = '''Solve the task step by step. Your instructions must follow the examples.
Here are some examples.
{examples}
Here is the task:
{task_description}'''
        prompt = prompt.format(task_description=task_description, examples=examples)
        messages = [{"role": "user", "content": prompt}]
        
        if self.logger:
            self.logger.log_module_diagnostic(
                module_name="reasoning",
                function_name="__call__",
                event_type="reasoning_started",
                data={
                    "reasoning_type": "TOT",
                    "task_length": len(task_description),
                    "has_feedback": bool(feedback),
                    "examples_count": len(examples),
                    "n_candidates": 3
                }
            )
        
        reasoning_results = self.llm(
            messages=messages,
            temperature=0.1,
            n=3
        )
        
        # Handle case where reasoning_results might be empty or None
        if not reasoning_results:
            reasoning_results = ["stars: 3.0\nreview: Unable to generate review."]
        elif not isinstance(reasoning_results, list):
            reasoning_results = [reasoning_results]
        elif len(reasoning_results) == 0:
            reasoning_results = ["stars: 3.0\nreview: Unable to generate review."]
        
        if self.logger:
            self.logger.log_module_diagnostic(
                module_name="reasoning",
                function_name="__call__",
                event_type="reasoning_candidates_generated",
                data={
                    "reasoning_type": "TOT",
                    "candidates_count": len(reasoning_results) if isinstance(reasoning_results, list) else 1
                }
            )
        
        reasoning_result = self.get_votes(task_description, reasoning_results, examples)
        
        if self.logger:
            self.logger.log_module_diagnostic(
                module_name="reasoning",
                function_name="__call__",
                event_type="reasoning_completed",
                data={
                    "reasoning_type": "TOT",
                    "result_length": len(reasoning_result) if isinstance(reasoning_result, str) else len(str(reasoning_result))
                }
            )
        
        return reasoning_result
    def get_votes(self, task_description, reasoning_results, examples):
        # Handle empty results
        if not reasoning_results or len(reasoning_results) == 0:
            return "stars: 3.0\nreview: Unable to generate review."
        
        if reasoning_results[0] and 'think' in reasoning_results[0].lower():
            return reasoning_results[0]
        prompt = '''Given the reasoning process for two completed tasks and one ongoing task, and several answers for the next step, decide which answer best follows the reasoning process for example command format. Output "The best answer is {{s}}", where s is the integer id chosen.
Here are some examples.
{examples}
Here is the task:
{task_description}

'''     
        prompt = prompt.format(task_description=task_description, examples=examples)
        messages = [{"role": "user", "content": prompt}]
        for i, y in enumerate(reasoning_results, 1):
            prompt += f'Answer {i}:\n{y}\n'
        vote_outputs = self.llm(
            messages=messages,
            temperature=0.7,
            n=5
        )
        vote_results = [0] * len(reasoning_results)
        # Handle case where vote_outputs might not be a list
        if not isinstance(vote_outputs, list):
            vote_outputs = [vote_outputs] if vote_outputs else []
        for vote_output in vote_outputs:
            if not vote_output:
                continue
            pattern = r".*best answer is .*(\d+).*"
            match = re.match(pattern, vote_output, re.DOTALL)
            if match:
                vote = int(match.groups()[0]) - 1
                if vote in range(len(reasoning_results)):
                    vote_results[vote] += 1
            else:
                print(f'vote no match: {[vote_output]}')
        ids = list(range(len(reasoning_results)))
        if len(ids) == 0:
            return "stars: 3.0\nreview: Unable to generate review."
        select_id = sorted(ids, key=lambda x: vote_results[x], reverse=True)[0]
        if select_id >= len(reasoning_results):
            return reasoning_results[0] if reasoning_results else "stars: 3.0\nreview: Unable to generate review."
        
        if self.logger:
            self.logger.log_module_diagnostic(
                module_name="reasoning",
                function_name="get_votes",
                event_type="vote_selection",
                data={
                    "reasoning_type": "TOT",
                    "vote_results": vote_results,
                    "selected_id": select_id,
                    "votes_count": len(vote_outputs)
                }
            )
        
        return reasoning_results[select_id]

class ReasoningDILU(ReasoningBase):
    def __call__(self, task_description: str, feedback :str= ''):
        examples, task_description = self.process_task_description(task_description)
        messages = [
            {
                "role": "system",
                "content": '''You are ChatGPT, a large language model trained by OpenAI. Now you act as a real human user on Yelp. You will be given a detailed description of the scenario of current frame along with your history of previous decisions. 
'''
            },
            {
                "role": "user",
                "content": f'''Above messages are some examples of how you make a step successfully in the past. Those scenarios are similar to the current scenario. You should refer to those examples to make a step for the current scenario. Your instructions must follow the examples.
Here are two examples.
{examples}
Here is the task:
{task_description}'''
            }
        ]
        reasoning_result = self.llm(
            messages=messages,
            temperature=0.1,
        )
        return reasoning_result

class ReasoningSelfRefine(ReasoningBase):
    def __call__(self, task_description: str, feedback :str= ''):
        examples, task_description = self.process_task_description(task_description)
        prompt = '''Solve the task step by step. Your instructions must follow the examples.
Here are some examples.
{examples}
Here is the task:
{task_description}'''
        prompt = prompt.format(task_description=task_description, examples=examples)
        messages = [{"role": "user", "content": prompt}]
        reasoning_result = self.llm(
            messages=messages,
            temperature=0.1,
        )
        reasoning_result = self.refine(reasoning_result)
        return reasoning_result
    def refine(self, reasoning_result):
        prompt = f'''Reflect on the reasoning process and identify any potential errors or areas for improvement. Provide a revised version of the reasoning if necessary.
Here is the original reasoning:
{reasoning_result}
'''     
        messages = [{"role": "user", "content": prompt}]
        feedback_result = self.llm(
            messages=messages,
            temperature=0.0
        )
        return feedback_result
        
class ReasoningStepBack(ReasoningBase):
    def __call__(self, task_description: str, feedback :str= ''):
        examples, task_description = self.process_task_description(task_description)
        self.principle = self.stepback(task_description)
            
        prompt = f'''Solve the task step by step. Your instructions must follow the examples.
Here are some examples.
{examples}{self.principle}
Here is the task:
{task_description}'''
        messages = [{"role": "user", "content": prompt}]
        reasoning_result = self.llm(
            messages=messages,
            temperature=0.1,
        )
        return reasoning_result
    def stepback(self, task_description):
        stepback_prompt = f'''What common sense, instruction structure is involved in solving this task?
{task_description}'''
        messages = [{"role": "user", "content": stepback_prompt}]
        principle = self.llm(
            messages=messages,
            temperature=0.1,
        )
        return principle
    

