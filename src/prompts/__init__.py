"""
Prompt modules for LLM Deception Research Platform
"""

from .base_prompt import PromptBuilder, build_prompt, get_baseline_prompt, get_prompt_variations
from .judge_prompt import (
    get_judge_system_prompt, 
    get_judge_user_prompt,
    format_conversation_for_judge,
    parse_judge_response,
    validate_judge_evaluation
)
from .translations import translate_text, get_translated_phrases

__all__ = [
    'PromptBuilder', 'build_prompt', 'get_baseline_prompt', 'get_prompt_variations',
    'get_judge_system_prompt', 'get_judge_user_prompt', 
    'format_conversation_for_judge', 'parse_judge_response', 'validate_judge_evaluation',
    'translate_text', 'get_translated_phrases'
]