import os
import re
from langchain_chroma import Chroma
from langchain_core.documents import Document
import shutil
import uuid
import inspect

class MemoryBase:
    def __init__(self, memory_type: str, llm, logger=None) -> None:
        """
        Initialize the memory base class
        
        Args:
            memory_type: Type of memory
            llm: LLM instance used to generate memory-related text
            logger: Optional logger instance for diagnostics
        """
        self.llm = llm
        self.logger = logger
        self.memory_type = memory_type
        self.embedding = self.llm.get_embedding_model()
        db_path = os.path.join('./db', memory_type, f'{str(uuid.uuid4())}')
        if os.path.exists(db_path):
            shutil.rmtree(db_path)
        self.scenario_memory = Chroma(
            embedding_function=self.embedding,
            persist_directory=db_path
        )
        
        if self.logger:
            self.logger.log_module_diagnostic(
                module_name="memory",
                function_name="__init__",
                event_type="memory_initialized",
                data={
                    "memory_type": memory_type,
                    "db_path": db_path
                }
            )

    def __call__(self, current_situation: str = ''):
        if 'review:' in current_situation:
            self.addMemory(current_situation.replace('review:', ''))
        else:
            return self.retriveMemory(current_situation)

    def retriveMemory(self, query_scenario: str):
        raise NotImplementedError("This method should be implemented by subclasses.")

    def addMemory(self, current_situation: str):
        raise NotImplementedError("This method should be implemented by subclasses.")

class MemoryDILU(MemoryBase):
    def __init__(self, llm, logger=None):
        super().__init__(memory_type='dilu', llm=llm, logger=logger)

    def retriveMemory(self, query_scenario: str):
        # Extract task name from query scenario
        task_name = query_scenario
        
        # Return empty string if memory is empty
        memory_count = self.scenario_memory._collection.count()
        if memory_count == 0:
            if self.logger:
                self.logger.log_module_diagnostic(
                    module_name="memory",
                    function_name="retriveMemory",
                    event_type="memory_retrieval_empty",
                    data={
                        "memory_type": self.memory_type,
                        "query": query_scenario[:100]  # Truncate for logging
                    }
                )
            return ''
            
        # Find most similar memory
        similarity_results = self.scenario_memory.similarity_search_with_score(
            task_name, k=1)
        
        # Extract task trajectories from results
        task_trajectories = [
            result[0].metadata['task_trajectory'] for result in similarity_results
            if result and len(result) > 0 and result[0].metadata.get('task_trajectory')
        ]
        
        result_text = '\n'.join(task_trajectories)
        
        if self.logger:
            self.logger.log_module_diagnostic(
                module_name="memory",
                function_name="retriveMemory",
                event_type="memory_retrieved",
                data={
                    "memory_type": self.memory_type,
                    "query": query_scenario[:100],
                    "memory_count": memory_count,
                    "similarity_score": similarity_results[0][1] if similarity_results and len(similarity_results) > 0 and len(similarity_results[0]) > 1 else None,
                    "result_length": len(result_text)
                }
            )
        
        # Join trajectories with newlines and return
        return result_text

    def addMemory(self, current_situation: str):
        # Extract task description
        task_name = current_situation
        
        # Create document with metadata
        memory_doc = Document(
            page_content=task_name,
            metadata={
                "task_name": task_name,
                "task_trajectory": current_situation
            }
        )
        
        # Add to memory store
        self.scenario_memory.add_documents([memory_doc])
        
        if self.logger:
            self.logger.log_module_diagnostic(
                module_name="memory",
                function_name="addMemory",
                event_type="memory_added",
                data={
                    "memory_type": self.memory_type,
                    "content_length": len(current_situation),
                    "memory_count_after": self.scenario_memory._collection.count()
                }
            )

class MemoryGenerative(MemoryBase):
    def __init__(self, llm, logger=None):
        super().__init__(memory_type='generative', llm=llm, logger=logger)

    def retriveMemory(self, query_scenario: str):
        # Extract task name from query
        task_name = query_scenario
        
        # Return empty if no memories exist
        if self.scenario_memory._collection.count() == 0:
            return ''
            
        # Get top 3 similar memories
        similarity_results = self.scenario_memory.similarity_search_with_score(
            task_name, k=3)
            
        fewshot_results = []
        importance_scores = []

        # Score each memory's relevance
        for result in similarity_results:
            if not result or len(result) == 0 or not result[0].metadata.get('task_trajectory'):
                continue
            trajectory = result[0].metadata['task_trajectory']
            fewshot_results.append(trajectory)
            
            # Generate prompt to evaluate importance
            prompt = f'''You will be given a successful case where you successfully complete the task. Then you will be given an ongoing task. Do not summarize these two cases, but rather evaluate how relevant and helpful the successful case is for the ongoing task, on a scale of 1-10.
Success Case:
{trajectory}
Ongoing task:
{query_scenario}
Your output format should be:
Score: '''

            # Get importance score
            response = self.llm(messages=[{"role": "user", "content": prompt}], temperature=0.1, stop_strs=['\n'])
            score = int(re.search(r'\d+', response).group()) if re.search(r'\d+', response) else 0
            importance_scores.append(score)
            
            if self.logger:
                self.logger.log_module_diagnostic(
                    module_name="memory",
                    function_name="retriveMemory",
                    event_type="importance_scored",
                    data={
                        "memory_type": self.memory_type,
                        "score": score,
                        "trajectory_index": len(importance_scores) - 1
                    }
                )

        # Return trajectory with highest importance score
        if not importance_scores or len(importance_scores) == 0:
            return ""
        if not similarity_results or len(similarity_results) == 0:
            return ""
        max_score_idx = importance_scores.index(max(importance_scores))
        if max_score_idx >= len(similarity_results) or not similarity_results[max_score_idx] or len(similarity_results[max_score_idx]) == 0:
            return ""
        result = similarity_results[max_score_idx][0].metadata.get('task_trajectory', '')
        
        if self.logger:
            self.logger.log_module_diagnostic(
                module_name="memory",
                function_name="retriveMemory",
                event_type="memory_selected",
                data={
                    "memory_type": self.memory_type,
                    "selected_index": max_score_idx,
                    "max_score": max(importance_scores),
                    "all_scores": importance_scores
                }
            )
        
        return result
    
    def addMemory(self, current_situation: str):
        # Extract task description
        task_name = current_situation
        
        # Create document with metadata
        memory_doc = Document(
            page_content=task_name,
            metadata={
                "task_name": task_name,
                "task_trajectory": current_situation
            }
        )
        
        # Add to memory store
        self.scenario_memory.add_documents([memory_doc])
        
        if self.logger:
            self.logger.log_module_diagnostic(
                module_name="memory",
                function_name="addMemory",
                event_type="memory_added",
                data={
                    "memory_type": self.memory_type,
                    "content_length": len(current_situation),
                    "memory_count_after": self.scenario_memory._collection.count()
                }
            )

class MemoryTP(MemoryBase):
    def __init__(self, llm, logger=None):
        super().__init__(memory_type='tp', llm=llm, logger=logger)

    def retriveMemory(self, query_scenario: str):
        # Extract task name from scenario
        task_name = query_scenario
        
        # Return empty if no memories exist
        if self.scenario_memory._collection.count() == 0:
            return ''
            
        # Find most similar memory
        similarity_results = self.scenario_memory.similarity_search_with_score(
            task_name, k=1)
            
        # Generate plans based on similar experiences
        experience_plans = []
        task_description = query_scenario
        
        for result in similarity_results:
            if not result or len(result) == 0 or not result[0].metadata.get('task_trajectory'):
                continue
            prompt = f"""You will be given a successful case where you successfully complete the task. Then you will be given an ongoing task. Do not summarize these two cases, but rather use the successful case to think about the strategy and path you took to attempt to complete the task in the ongoing task. Devise a concise, new plan of action that accounts for your task with reference to specific actions that you should have taken. You will need this later to solve the task. Give your plan after "Plan".
Success Case:
{result[0].metadata['task_trajectory']}
Ongoing task:
{task_description}
Plan:
"""
            experience_plans.append(self.llm(messages=[{"role": "user", "content": prompt}], temperature=0.1))
            
        return 'Plan from successful attempt in similar task:\n' + '\n'.join(experience_plans)

    def addMemory(self, current_situation: str):
        # Extract task name
        task_name = current_situation
        
        # Create document with metadata
        memory_doc = Document(
            page_content=task_name,
            metadata={
                "task_name": task_name,
                "task_trajectory": current_situation
            }
        )
        
        # Add to memory store
        self.scenario_memory.add_documents([memory_doc])
        
        if self.logger:
            self.logger.log_module_diagnostic(
                module_name="memory",
                function_name="addMemory",
                event_type="memory_added",
                data={
                    "memory_type": self.memory_type,
                    "content_length": len(current_situation),
                    "memory_count_after": self.scenario_memory._collection.count()
                }
            )

class MemoryVoyager(MemoryBase):
    def __init__(self, llm, logger=None):
        super().__init__(memory_type='voyager', llm=llm, logger=logger)

    def retriveMemory(self, query_scenario: str):
        # Extract task name from query
        task_name = query_scenario
        
        # Return empty if no memories exist
        if self.scenario_memory._collection.count() == 0:
            return ''
            
        # Find most similar memories
        similarity_results = self.scenario_memory.similarity_search_with_score(task_name, k=1)
        
        # Extract trajectories from results
        memory_trajectories = [result[0].metadata['task_trajectory'] 
                             for result in similarity_results]
                             
        return '\n'.join(memory_trajectories)

    def addMemory(self, current_situation: str):
        # Prompt template for summarizing trajectory
        voyager_prompt = '''You are a helpful assistant that writes a description of the task resolution trajectory.

        1) Try to summarize the trajectory in no more than 6 sentences.
        2) Your response should be a single line of text.

        For example:

Please fill in this part yourself

        Trajectory:
        '''
        
        # Generate summarized trajectory
        prompt = voyager_prompt + current_situation
        trajectory_summary = self.llm(messages=[{"role": "user", "content": prompt}], temperature=0.1)
        
        # Create document with metadata
        doc = Document(
            page_content=trajectory_summary,
            metadata={
                "task_description": trajectory_summary,
                "task_trajectory": current_situation
            }
        )
        
        # Add to memory store
        self.scenario_memory.add_documents([doc])
        
        if self.logger:
            self.logger.log_module_diagnostic(
                module_name="memory",
                function_name="addMemory",
                event_type="memory_added",
                data={
                    "memory_type": self.memory_type,
                    "content_length": len(current_situation),
                    "summary_length": len(trajectory_summary),
                    "memory_count_after": self.scenario_memory._collection.count()
                }
            )

