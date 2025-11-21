from typing import Dict, List, Optional, Union
from google import genai
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from langchain_openai import OpenAIEmbeddings
from sentence_transformers import SentenceTransformer
import logging
import inspect
import time
logger = logging.getLogger("websocietysimulator")

class LLMBase:
    def __init__(self, model: str = "qwen2.5-72b-instruct", logger=None):
        """
        Initialize LLM base class
        
        Args:
            model: Model name, defaults to deepseek-chat
            logger: Optional LLMLogger instance for logging calls
        """
        self.model = model
        self.logger = logger
        
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
    def __init__(self, api_key: str, model: str = "gemini-2.5-flash", logger=None):
        """
        Initialize Gemini LLM
        
        Args:
            api_key: Google Gemini API key
            model: Model name, defaults to gemini-2.5-flash
            logger: Optional LLMLogger instance for logging calls
        """
        super().__init__(model=model, logger=logger)
        self.client = genai.Client(api_key=api_key)
        # Initialize embedding model (using SentenceTransformer as default)
        self._embedding_model = None
    
    def __call__(
        self,
        messages: List[Dict[str, str]],
        model: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: int = 500,
        stop_strs: Optional[List[str]] = None,
        n: int = 1
    ) -> Union[str, List[str]]:
        """
        Call Gemini API to get response
        
        Args:
            messages: List of input messages, each a dict with 'role' and 'content'
            model: Optional model override
            temperature: Temperature for generation
            max_tokens: Max tokens for response
            stop_strs: Optional list of stop strings (not used by Gemini but kept for compatibility)
            n: Number of responses to generate
            
        Returns:
            Union[str, List[str]]: Response text from LLM
        """
        # Detect calling module and function for logging
        module_name = "unknown"
        function_name = "unknown"
        if self.logger:
            try:
                # Get the caller's frame (skip this function and __call__ wrapper)
                frame = inspect.currentframe()
                caller_frame = frame.f_back
                if caller_frame:
                    caller_frame = caller_frame.f_back  # Skip one more level if needed
                    if caller_frame:
                        module_name = caller_frame.f_globals.get('__name__', 'unknown')
                        function_name = caller_frame.f_code.co_name
                        # Extract module name from full path (e.g., 'websocietysimulator.agent.modules.memory_modules' -> 'memory_modules')
                        if '.' in module_name:
                            module_name = module_name.split('.')[-1]
            except Exception:
                pass
        
        start_time = time.time()
        error = None
        response = None
        
        try:
            # Prepare the messages for Gemini
            contents = []
            for msg in messages:
                parts = [{"text": msg["content"]}]
                contents.append({"role": msg["role"], "parts": parts})
            
            # Create generation config
            from google.genai import types
            config = types.GenerateContentConfig(
                temperature=temperature,
                max_output_tokens=max_tokens
            )
            
            api_response = self.client.models.generate_content(
                model=model or self.model,
                contents=contents,
                config=config
            )
            
            # Check for errors in the response
            if hasattr(api_response, 'prompt_feedback') and api_response.prompt_feedback:
                if hasattr(api_response.prompt_feedback, 'block_reason') and api_response.prompt_feedback.block_reason:
                    raise Exception(f"Prompt blocked: {api_response.prompt_feedback.block_reason}")
            
            # Check if candidates exist
            if not api_response.candidates or len(api_response.candidates) == 0:
                error_msg = "No candidates in API response"
                if hasattr(api_response, 'prompt_feedback'):
                    error_msg += f" (prompt_feedback: {api_response.prompt_feedback})"
                raise Exception(error_msg)
            
            # Gemini returns candidates; handle n > 1
            if n == 1:
                candidate = api_response.candidates[0]
                
                # Check if content exists
                if not hasattr(candidate, 'content') or candidate.content is None:
                    raise Exception("Candidate has no content attribute")
                
                # Check finish_reason - MAX_TOKENS is acceptable, others might indicate issues
                finish_reason = getattr(candidate, 'finish_reason', None)
                if finish_reason:
                    # MAX_TOKENS means we hit the limit but still got a response
                    # SAFETY and RECITATION indicate content was blocked
                    if finish_reason in (types.FinishReason.SAFETY, types.FinishReason.RECITATION):
                        raise Exception(f"Content was blocked. Finish reason: {finish_reason}")
                
                # Extract text from parts
                parts = getattr(candidate.content, 'parts', None)
                if parts and len(parts) > 0:
                    # Standard case: parts is a list
                    part = parts[0]
                    if hasattr(part, 'text'):
                        response = part.text
                    else:
                        # Try to get text directly if part is a string or has different structure
                        response = str(part) if part else ""
                else:
                    # Parts is None or empty
                    # This can happen when MAX_TOKENS is hit with a very small limit
                    # In this case, we didn't get any text back
                    if finish_reason == types.FinishReason.MAX_TOKENS:
                        raise Exception(
                            f"Response hit MAX_TOKENS limit ({max_tokens} tokens) but returned no text. "
                            f"This usually means the limit is too small. Try increasing max_tokens."
                        )
                    else:
                        raise Exception(f"Candidate has no extractable text. Finish reason: {finish_reason}, Parts: {parts}")
            else:
                response = [c.content.parts[0].text for c in api_response.candidates[:n] if c.content and c.content.parts]
        except Exception as e:
            error = str(e)
            raise
        finally:
            # Log the LLM call
            if self.logger:
                duration_ms = (time.time() - start_time) * 1000
                self.logger.log_llm_call(
                    module_name=module_name,
                    function_name=function_name,
                    messages=messages,
                    model=model or self.model,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    stop_strs=stop_strs,
                    n=n,
                    response=response,
                    error=error,
                    duration_ms=duration_ms
                )
        
        return response
    
    def get_embedding_model(self):
        """
        Get the embedding model for text embeddings.
        Uses SentenceTransformer as default embedding model.
        
        Returns:
            SentenceTransformer: An instance of SentenceTransformer for embeddings
        """
        if self._embedding_model is None:
            # Use a lightweight, fast embedding model
            # Force CPU to avoid MPS device issues on macOS
            import os
            os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
            self._embedding_model = SentenceTransformer('paraphrase-MiniLM-L6-v2', device='cpu')
        return self._embedding_model
