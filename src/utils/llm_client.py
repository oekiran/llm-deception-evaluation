"""
Unified LLM Client for OpenAI, Anthropic, and Google models
Handles API calls to all providers with a consistent interface
"""

import os
import logging
from typing import List, Dict, Optional, Any
from openai import OpenAI
from anthropic import Anthropic
import google.generativeai as genai
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

logger = logging.getLogger(__name__)

class UnifiedLLMClient:
    """
    Unified client for OpenAI, Anthropic, and Google models.
    Provides a consistent interface for all LLM operations.
    """
    
    # Model mappings
    OPENAI_MODELS = [
        'gpt-5', 'gpt-5-mini',  # GPT-5 models
        'o1-preview', 'o1-mini', 'o3-mini',
        'gpt-4o', 'gpt-4-turbo', 'gpt-4', 'gpt-3.5-turbo'
    ]
    
    ANTHROPIC_MODELS = [
        'claude-sonnet-4-5',             # Claude Sonnet 4.5 (latest, most capable)
        'claude-haiku-4-5',              # Claude Haiku 4.5 (latest, fast)
        'claude-opus-4-1'                # Claude Opus 4.1 (powerful, reliable)
    ]
    
    GOOGLE_MODELS = [
        'gemini-2.5-pro',                    # Gemini 2.5 Pro (most powerful Google model)
        'gemini-2.0-flash-exp',              # Gemini 2.0 Flash (experimental, fast)
        'gemini-2.0-flash-thinking-exp-1219', # Gemini 2.0 Flash with thinking (experimental)
        'gemini-1.5-pro-002',                # Gemini 1.5 Pro (latest stable, powerful)
        'gemini-1.5-flash-002'               # Gemini 1.5 Flash (latest stable, fast)
    ]
    
    # Display names for UI
    MODEL_DISPLAY_NAMES = {
        # OpenAI
        'gpt-5': 'GPT-5 (Most Advanced)',
        'gpt-5-mini': 'GPT-5 Mini (Fast & Efficient)',
        'o1-preview': 'OpenAI o1-preview (Reasoning)',
        'o1-mini': 'OpenAI o1-mini (Reasoning)',
        'o3-mini': 'OpenAI o3-mini (Reasoning)',
        'gpt-4o': 'GPT-4o (Latest)',
        'gpt-4-turbo': 'GPT-4 Turbo',
        'gpt-4': 'GPT-4',
        'gpt-3.5-turbo': 'GPT-3.5 Turbo',
        # Anthropic
        'claude-sonnet-4-5': 'Claude Sonnet 4.5 (Latest, Most Capable)',
        'claude-haiku-4-5': 'Claude Haiku 4.5 (Latest, Fast)',
        'claude-opus-4-1': 'Claude Opus 4.1 (Powerful, Reliable)',
        # Google
        'gemini-2.5-pro': 'Gemini 2.5 Pro (Most Advanced)',
        'gemini-2.0-flash-exp': 'Gemini 2.0 Flash (Experimental, Fast)',
        'gemini-2.0-flash-thinking-exp-1219': 'Gemini 2.0 Flash Thinking (Experimental)',
        'gemini-1.5-pro-002': 'Gemini 1.5 Pro 002 (Latest Stable, Powerful)',
        'gemini-1.5-flash-002': 'Gemini 1.5 Flash 002 (Latest Stable, Fast)'
    }
    
    def __init__(self):
        """Initialize OpenAI, Anthropic, and Google clients."""
        # Load API keys
        self.openai_key = os.getenv('OPENAI_API_KEY')
        self.anthropic_key = os.getenv('ANTHROPIC_API_KEY')
        self.google_key = os.getenv('GOOGLE_API_KEY')
        
        # Initialize clients
        self.openai_client = None
        self.anthropic_client = None
        self.google_client = None
        
        if self.openai_key:
            self.openai_client = OpenAI(api_key=self.openai_key)
            logger.info("OpenAI client initialized")
        else:
            logger.warning("No OpenAI API key found")
            
        if self.anthropic_key:
            self.anthropic_client = Anthropic(api_key=self.anthropic_key)
            logger.info("Anthropic client initialized")
        else:
            logger.warning("No Anthropic API key found")
            
        if self.google_key:
            genai.configure(api_key=self.google_key)
            self.google_client = True  # use genai directly, no client object
            logger.info("Google Gemini client initialized")
        else:
            logger.warning("No Google API key found")
    
    def is_anthropic_model(self, model_name: str) -> bool:
        """Check if model is from Anthropic."""
        return model_name in self.ANTHROPIC_MODELS or model_name.startswith('claude-')
    
    def is_openai_model(self, model_name: str) -> bool:
        """Check if model is from OpenAI."""
        return model_name in self.OPENAI_MODELS or 'gpt' in model_name.lower() or 'o1' in model_name.lower()
    
    def is_google_model(self, model_name: str) -> bool:
        """Check if model is from Google."""
        return model_name in self.GOOGLE_MODELS or 'gemini' in model_name.lower()
    
    def get_available_models(self) -> List[str]:
        """Get list of all available models based on API keys."""
        models = []
        
        if self.openai_client:
            models.extend(self.OPENAI_MODELS)
        
        if self.anthropic_client:
            models.extend(self.ANTHROPIC_MODELS)
        
        if self.google_client:
            models.extend(self.GOOGLE_MODELS)
        
        return models
    
    def get_model_display_name(self, model_name: str) -> str:
        """Get display name for a model."""
        return self.MODEL_DISPLAY_NAMES.get(model_name, model_name)
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        retry=retry_if_exception_type((Exception,))
    )
    def create_completion(self,
                         model: str,
                         messages: List[Dict[str, str]],
                         temperature: float = 0.7,
                         max_tokens: int = 2000,
                         system_prompt: Optional[str] = None) -> str:
        """
        Create a completion using either OpenAI or Anthropic API.
        
        Args:
            model: Model name
            messages: List of message dictionaries
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            system_prompt: Optional system prompt (for Anthropic)
            
        Returns:
            Generated text response
        """
        try:
            if self.is_anthropic_model(model):
                return self._create_anthropic_completion(
                    model, messages, temperature, max_tokens, system_prompt
                )
            elif self.is_google_model(model):
                return self._create_google_completion(
                    model, messages, temperature, max_tokens, system_prompt
                )
            elif self.is_openai_model(model):
                return self._create_openai_completion(
                    model, messages, temperature, max_tokens, system_prompt
                )
            else:
                raise ValueError(f"Unknown model: {model}")
                
        except Exception as e:
            logger.error(f"Error creating completion with {model}: {e}")
            raise
    
    def _create_openai_completion(self,
                                 model: str,
                                 messages: List[Dict[str, str]],
                                 temperature: float,
                                 max_tokens: int,
                                 system_prompt: Optional[str] = None) -> str:
        """Create completion using OpenAI API."""
        if not self.openai_client:
            raise ValueError("OpenAI client not initialized")
        
        # Add system prompt if provided
        if system_prompt:
            messages = [{"role": "system", "content": system_prompt}] + messages
        
        # Handle o1 models differently
        if 'o1' in model.lower():
            response = self.openai_client.chat.completions.create(
                model=model,
                messages=messages,
                max_completion_tokens=max_tokens
            )
        else:
            response = self.openai_client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens
            )
        
        return response.choices[0].message.content
    
    def _create_anthropic_completion(self,
                                     model: str,
                                     messages: List[Dict[str, str]],
                                     temperature: float,
                                     max_tokens: int,
                                     system_prompt: Optional[str] = None) -> str:
        """Create completion using Anthropic API."""
        if not self.anthropic_client:
            raise ValueError("Anthropic client not initialized")

        # Use model names directly as provided by the user
        
        actual_model = model
        
        logger.info(f"Using Anthropic model: {actual_model}")

        # Convert messages to Anthropic format
        anthropic_messages = []
        for msg in messages:
            role = msg['role']
            # Convert OpenAI roles to Anthropic roles
            if role == 'system':
                # Anthropic handles system prompts separately
                if not system_prompt:
                    system_prompt = msg['content']
                continue
            elif role == 'assistant':
                anthropic_messages.append({
                    'role': 'assistant',
                    'content': msg['content']
                })
            else:  # user
                anthropic_messages.append({
                    'role': 'user',
                    'content': msg['content']
                })

        # Create the message
        kwargs = {
            'model': actual_model,
            'messages': anthropic_messages,
            'temperature': temperature,
            'max_tokens': max_tokens
        }

        # Only add system if it's a non-empty string
        if system_prompt and isinstance(system_prompt, str):
            kwargs['system'] = system_prompt

        response = self.anthropic_client.messages.create(**kwargs)

        return response.content[0].text
    
    def _create_google_completion(self,
                                  model: str,
                                  messages: List[Dict[str, str]],
                                  temperature: float,
                                  max_tokens: int,
                                  system_prompt: Optional[str] = None) -> str:
        """Create completion using Google Gemini API."""
        if not self.google_client:
            raise ValueError("Google client not initialized")
        
        # Initialize the model
        gemini_model = genai.GenerativeModel(model)
        
        # Convert messages to Google format
        google_messages = []
        
        # Add system prompt as first user message if provided
        if system_prompt:
            google_messages.append({
                'role': 'user',
                'parts': [system_prompt + "\n\nPlease follow these instructions throughout our conversation."]
            })
            google_messages.append({
                'role': 'model',
                'parts': ["Understood. I will follow these instructions."]
            })
        
        # Convert message format
        for msg in messages:
            role = msg['role']
            content = msg['content']
            
            if role == 'system':
                # System messages treated as user instructions
                google_messages.append({
                    'role': 'user',
                    'parts': [content]
                })
                google_messages.append({
                    'role': 'model',
                    'parts': ["I understand and will follow these instructions."]
                })
            elif role == 'assistant':
                google_messages.append({
                    'role': 'model',
                    'parts': [content]
                })
            else:  # user
                google_messages.append({
                    'role': 'user',
                    'parts': [content]
                })
        
        # Configure generation settings
        generation_config = genai.types.GenerationConfig(
            temperature=temperature,
            max_output_tokens=max_tokens
        )
        
        # Create chat session and send message
        chat = gemini_model.start_chat(history=google_messages[:-1] if len(google_messages) > 1 else [])
        
        # Get the last user message
        last_message = google_messages[-1]['parts'][0] if google_messages else "Hello"
        
        # Generate response
        response = chat.send_message(last_message, generation_config=generation_config)
        
        return response.text


# Singleton instance
_client_instance = None

def get_llm_client() -> UnifiedLLMClient:
    """Get or create the singleton LLM client."""
    global _client_instance
    if _client_instance is None:
        _client_instance = UnifiedLLMClient()
    return _client_instance


# Test function
if __name__ == "__main__":
    from dotenv import load_dotenv
    
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    
    # Load environment
    load_dotenv()
    
    # Get client
    client = get_llm_client()
    
    # List available models
    print("Available models:")
    for model in client.get_available_models():
        print(f"  - {model}: {client.get_model_display_name(model)}")
    
    # Test with a simple prompt
    test_messages = [
        {"role": "user", "content": "Say 'Hello, I am working!' in exactly 5 words."}
    ]
    
    # Test OpenAI if available
    if client.openai_client:
        print("\nTesting OpenAI (GPT-3.5):")
        response = client.create_completion(
            model="gpt-3.5-turbo",
            messages=test_messages,
            temperature=0.7,
            max_tokens=50
        )
        print(f"Response: {response}")
    
    # Test Anthropic if available
    if client.anthropic_client:
        print("\nTesting Anthropic (Claude Sonnet 4.5):")
        response = client.create_completion(
            model="claude-sonnet-4-5",  # Use our simplified alias
            messages=test_messages,
            temperature=0.7,
            max_tokens=50
        )
        print(f"Response: {response}")
    
    # Test Google if available
    if client.google_client:
        print("\nTesting Google (Gemini 1.5 Flash):")
        response = client.create_completion(
            model="gemini-1.5-flash",
            messages=test_messages,
            temperature=0.7,
            max_tokens=50
        )
        print(f"Response: {response}")