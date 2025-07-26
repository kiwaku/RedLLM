"""Improved Language Model Wrapper Classes for RedLLM."""

import os
import logging
import litellm
import dspy
from typing import Any, Dict, List
from pydantic import BaseModel, Field

# Pydantic model for robust configuration
class LLMWrapperConfig(BaseModel):
    """Configuration for LLM wrappers, validated by Pydantic."""
    model: str
    api_key: str
    provider: str = ""
    max_tokens: int = 1024
    temperature: float = 0.7
    
    class Config:
        arbitrary_types_allowed = True

    @property
    def full_model_name(self) -> str:
        """Construct the full model name for LiteLLM."""
        if self.provider and not self.model.startswith(self.provider):
            return f"{self.provider}/{self.model}"
        return self.model

class DSPyGenericLM(dspy.LM):
    """Consolidated DSPy LM wrapper for judge and attacker models."""
    
    def __init__(self, model: str, api_key: str, provider: str = "", **kwargs):
        super().__init__(model)
        self.config = LLMWrapperConfig(model=model, api_key=api_key, provider=provider, **kwargs)
        self._setup_environment()
    
    def _setup_environment(self):
        """Set up environment variables based on provider."""
        provider_map = {
            "together_ai": "TOGETHER_AI_API_KEY",
            "openai": "OPENAI_API_KEY", 
            "groq": "GROQ_API_KEY",
            "anthropic": "ANTHROPIC_API_KEY"
        }
        
        provider_name = self.config.provider or ""
        for provider_key, env_var in provider_map.items():
            if provider_key in self.config.full_model_name:
                os.environ[env_var] = self.config.api_key
                break
    
    def _is_critical_error(self, error: Exception) -> bool:
        """Determine if an error is critical enough to stop execution."""
        error_str = str(error).lower()
        critical_patterns = [
            "authentication", "api key", "unauthorized", "forbidden",
            "rate limit", "quota exceeded", "billing", "payment required"
        ]
        return any(pattern in error_str for pattern in critical_patterns)
    
    def basic_request(self, prompt: str, **kwargs) -> str:
        """DSPy adapters call this for individual requests with retry logic."""
        import time
        import random
        
        max_retries = 3
        base_delay = 1.0  # Start with 1 second
        
        for attempt in range(max_retries + 1):
            try:
                request_kwargs = {
                    "temperature": self.config.temperature,
                    "max_tokens": self.config.max_tokens,
                    **kwargs
                }
                
                response = litellm.completion(
                    model=self.config.full_model_name,
                    messages=[{"role": "user", "content": prompt}],
                    **request_kwargs
                )
                return response["choices"][0]["message"]["content"].strip()
                
            except Exception as e:
                error_logger = logging.getLogger("redllm.llm_wrapper")
                error_str = str(e).lower()
                
                # Check if this is a rate limit error that we should retry
                is_rate_limit = any(pattern in error_str for pattern in [
                    "rate limit", "ratelimit", "too many requests", "quota exceeded"
                ])
                
                if is_rate_limit and attempt < max_retries:
                    # Calculate exponential backoff with jitter
                    delay = base_delay * (2 ** attempt) + random.uniform(0, 1)
                    error_logger.warning(f"Rate limit hit (attempt {attempt + 1}/{max_retries + 1}), retrying in {delay:.1f}s: {e}")
                    time.sleep(delay)
                    continue
                
                # Log error (final attempt or non-retryable error)
                error_logger.error(f"Request failed for {self.config.full_model_name}: {type(e).__name__}: {e}")
                
                if self._is_critical_error(e):
                    error_logger.critical(f"Critical LLM error: {e}")
                    raise
                
                return f"ERROR: {str(e)}"
        
        # This shouldn't be reached, but just in case
        return "ERROR: Max retries exceeded"
    
    def __call__(self, prompt=None, **kwargs):
        """Handle DSPy's expected call format, accommodating different call patterns."""
        content = ""
        
        # First, check if we have messages in kwargs (DSPy judge format)
        if 'messages' in kwargs and isinstance(kwargs['messages'], list):
            # Extract content from messages format
            for message in kwargs['messages']:
                if isinstance(message, dict) and 'content' in message:
                    content += message['content'] + "\n"
        elif prompt and isinstance(prompt, str):
            # Handle simple string prompt
            content = prompt
        elif prompt and isinstance(prompt, list):
            # Handle list of messages format
            for msg in prompt:
                if isinstance(msg, dict) and 'content' in msg:
                    content += msg['content'] + "\n"
                else:
                    content += str(msg) + " "
        elif prompt is not None:
            # Handle other prompt types
            content = str(prompt)
        
        # Process remaining kwargs for any additional content (legacy support)
        kwarg_parts = []
        for key, value in kwargs.items():
            if key not in ['prompt', 'messages'] and isinstance(value, (str, int, float)) and value:
                kwarg_parts.append(f"{key.replace('_', ' ').title()}: {value}")
        
        if kwarg_parts:
            kwarg_content = "\n".join(kwarg_parts)
            if content:
                content = f"{content}\n\n{kwarg_content}"
            else:
                content = kwarg_content

        # Clean up content
        content = content.strip()

        if not content:
            import traceback
            logging.warning("LLM wrapper called with no processable prompt or kwargs.")
            logging.debug(f"Call stack: {traceback.format_stack()[-5:]}")  # Last 5 stack frames
            logging.debug(f"Raw prompt: {repr(prompt)}")
            logging.debug(f"Raw kwargs: {repr(kwargs)}")
            # Return a JSON string representing the error, so the JSONAdapter can parse it.
            error_json = {
                "classification": "error",
                "confidence": 0.0,
                "leak_score": -1,
                "reason": "ERROR: No content provided to LLM."
            }
            import json
            return [{"text": json.dumps(error_json), "logprobs": None}]
        
        # Filter out DSPy-specific kwargs that shouldn't be passed to LiteLLM
        filtered_kwargs = {k: v for k, v in kwargs.items() if k not in ['messages', 'prompt']}
        
        response_text = self.basic_request(content, **filtered_kwargs)
        return [{"text": response_text, "logprobs": None}]


class GenericCallableLM(DSPyGenericLM):
    """
    A specific implementation of the DSPyGenericLM for the target model.
    This class inherits all the robust logic from DSPyGenericLM.
    We keep it for semantic clarity in the AttackEngine, but it's functionally
    identical to its parent.
    """
    def __init__(self, model: str, api_key: str, **kwargs):
        # The provider is now automatically inferred from the model string in the parent class.
        provider = model.split('/')[0] if '/' in model else ''
        super().__init__(model=model, api_key=api_key, provider=provider, **kwargs)
        logging.debug(f"GenericCallableLM initialized for model: {self.config.full_model_name}")
