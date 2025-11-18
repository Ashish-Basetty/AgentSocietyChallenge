from typing import Dict, List, Optional, Union
from google import genai
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
import logging
logger = logging.getLogger("websocietysimulator")

class LLMBase:
    def __init__(self, model: str = "qwen2.5-72b-instruct"):
        """
        Initialize LLM base class
        
        Args:
            model: Model name, defaults to deepseek-chat
        """
        self.model = model
        
    def __call__(self, messages: List[Dict[str, str]], model: Optional[str] = None, temperature: float = 0.0, max_tokens: int = 500, stop_strs: Optional[List[str]] = None, n: int = 1) -> Union[str, List[str]]:
        """
        Call LLM to get response
        
        Args:
            messages: List of input messages, each message is a dict containing role and content
            model: Optional model override
            max_tokens: Maximum tokens in response, defaults to 500
            stop_strs: Optional list of stop strings
            n: Number of responses to generate, defaults to 1
            
        Returns:
            Union[str, List[str]]: Response text from LLM, either a single string or list of strings
        """
        raise NotImplementedError("Subclasses need to implement this method")
    
    def get_embedding_model(self):
        """
        Get the embedding model for text embeddings
        
        Returns:
            OpenAIEmbeddings: An instance of OpenAI's text embedding model
        """
        raise NotImplementedError("Subclasses need to implement this method")

class GeminiLLM(LLMBase):
    def __init__(self, api_key: str, model: str = "gemini-2.5-flash"):
        """
        Initialize Gemini LLM
        
        Args:
            api_key: Google Gemini API key
            model: Model name, defaults to gemini-2.5-flash
        """
        self.model = model
        self.client = genai.Client(api_key=api_key)
    
    def __call__(
        self,
        messages: List[Dict[str, str]],
        model: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: int = 500,
        n: int = 1
    ) -> Union[str, List[str]]:
        """
        Call Gemini API to get response
        
        Args:
            messages: List of input messages, each a dict with 'role' and 'content'
            model: Optional model override
            temperature: Temperature for generation
            max_tokens: Max tokens for response
            n: Number of responses to generate
            
        Returns:
            Union[str, List[str]]: Response text from LLM
        """
        # Prepare the messages for Gemini
        contents = []
        for msg in messages:
            parts = [{"text": msg["content"]}]
            contents.append({"role": msg["role"], "parts": parts})
        
        response = self.client.models.generate_content(
            model=model or self.model,
            contents=contents,
            temperature=temperature,
            max_output_tokens=max_tokens
        )
        
        # Gemini returns candidates; handle n > 1
        if n == 1:
            return response.candidates[0].content.parts[0].text
        else:
            return [c.content.parts[0].text for c in response.candidates[:n]]
