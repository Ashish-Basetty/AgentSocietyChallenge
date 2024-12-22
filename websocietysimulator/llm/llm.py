from typing import Dict, List, Optional
from openai import OpenAI

class LLMBase:
    def __init__(self, model: str = "deepseek-chat"):
        """
        Initialize LLM base class
        
        Args:
            model: Model name, defaults to deepseek-chat
        """
        self.model = model
        
    def __call__(self, messages: List[Dict[str, str]], model: Optional[str] = None, temperature: float = 0.0, max_tokens: int = 500, stop_strs: Optional[List[str]] = None, n: int = 1) -> str:
        """
        Call LLM to get response
        
        Args:
            messages: List of input messages, each message is a dict containing role and content
            model: Optional model override
            temperature: Sampling temperature, defaults to 0.0
            max_tokens: Maximum tokens in response, defaults to 500
            stop_strs: Optional list of stop strings
            n: Number of responses to generate, defaults to 1
            
        Returns:
            str: Response text from LLM
        """
        raise NotImplementedError("Subclasses need to implement this method")

class DeepseekLLM(LLMBase):
    def __init__(self, api_key: str, model: str = "deepseek-chat"):
        """
        Initialize Deepseek LLM
        
        Args:
            api_key: Deepseek API key
            model: Model name, defaults to deepseek-chat
        """
        super().__init__(model)
        self.client = OpenAI(
            api_key=api_key,
            base_url="https://api.deepseek.com/v1"
        )
        
    def __call__(self, messages: List[Dict[str, str]], model: Optional[str] = None, temperature: float = 0.0, max_tokens: int = 500, stop_strs: Optional[List[str]] = None, n: int = 1) -> str:
        """
        Call Deepseek API to get response
        
        Args:
            messages: List of input messages, each message is a dict containing role and content
            model: Optional model override
            temperature: Sampling temperature, defaults to 0.0
            max_tokens: Maximum tokens in response, defaults to 500
            stop_strs: Optional list of stop strings
            n: Number of responses to generate, defaults to 1
            
        Returns:
            str: Response text from LLM
        """
        response = self.client.chat.completions.create(
            model=model or self.model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            stop=stop_strs,
            n=n
        )
        
        return response.choices[0].message.content

class OpenAILLM(LLMBase):
    def __init__(self, api_key: str, model: str = "gpt-3.5-turbo"):
        """
        Initialize OpenAI LLM
        
        Args:
            api_key: OpenAI API key
            model: Model name, defaults to gpt-3.5-turbo
        """
        super().__init__(model)
        self.client = OpenAI(api_key=api_key)
        
    def __call__(self, messages: List[Dict[str, str]], model: Optional[str] = None, temperature: float = 0.0, max_tokens: int = 500, stop_strs: Optional[List[str]] = None, n: int = 1) -> str:
        """
        Call OpenAI API to get response
        
        Args:
            messages: List of input messages, each message is a dict containing role and content
            model: Optional model override
            temperature: Sampling temperature, defaults to 0.0
            max_tokens: Maximum tokens in response, defaults to 500
            stop_strs: Optional list of stop strings
            n: Number of responses to generate, defaults to 1
            
        Returns:
            str: Response text from LLM
        """
        response = self.client.chat.completions.create(
            model=model or self.model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            stop=stop_strs,
            n=n
        )
        
        return response.choices[0].message.content
