"""
Configuration Manager for LLM Deception Research Platform
Manages experimental parameters and configuration loading.
"""

import os
import yaml
import logging
from typing import Dict, Any, Optional, List
from pathlib import Path
from dotenv import load_dotenv

logger = logging.getLogger(__name__)

class ConfigManager:
    """
    Manages configuration for the research platform.
    Handles parameter validation and environment variable loading.
    """
    
    def __init__(self, config_path: str = "config/default_config.yaml"):
        """
        Initialize configuration manager.
        
        Args:
            config_path: Path to the configuration YAML file
        """
        self.config_path = Path(config_path)
        self.config = {}
        self.env_vars = {}
        
        # Load environment variables
        load_dotenv()
        self._load_env_vars()
        
        # Load configuration
        self._load_config()
        
        # Validate configuration
        self._validate_config()
    
    def _load_env_vars(self):
        """Load environment variables."""
        self.env_vars = {
            'api_key': os.getenv('OPENAI_API_KEY'),
            'temperature': float(os.getenv('DEFAULT_TEMPERATURE', 0.7)),
            'max_retries': int(os.getenv('MAX_RETRIES', 3)),
            'retry_delay': float(os.getenv('RETRY_DELAY', 1.0)),
            'timeout': int(os.getenv('API_TIMEOUT', 60)),
            'rate_limit_tpm': int(os.getenv('RATE_LIMIT_TPM', 10000)),
            'default_rounds': int(os.getenv('DEFAULT_ROUNDS', 15)),
            'default_conversations': int(os.getenv('DEFAULT_CONVERSATIONS', 1)),
            'log_level': os.getenv('LOG_LEVEL', 'INFO')
        }
        
        if not self.env_vars['api_key']:
            raise ValueError("OPENAI_API_KEY not found in environment variables")
    
    def _load_config(self):
        """Load configuration from YAML file."""
        if not self.config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {self.config_path}")
        
        with open(self.config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)
        
        # Merge with environment variables
        self.config['api']['key'] = self.env_vars['api_key']
        self.config['api']['temperature'] = self.env_vars['temperature']
        self.config['api']['max_retries'] = self.env_vars['max_retries']
        self.config['api']['retry_delay'] = self.env_vars['retry_delay']
        self.config['api']['timeout'] = self.env_vars['timeout']
        self.config['api']['rate_limit_tpm'] = self.env_vars['rate_limit_tpm']
        
        logger.info(f"Configuration loaded from {self.config_path}")
    
    def _validate_config(self):
        """
        Validate configuration values.
        Ensures all required fields are present and valid.
        """
        required_sections = ['models', 'parameters', 'api', 'tools', 'environment']
        for section in required_sections:
            if section not in self.config:
                raise ValueError(f"Missing required configuration section: {section}")
        
        # Validate model options
        if not self.config['models']['agent']['options']:
            raise ValueError("No agent model options configured")
        
        # Validate simulation parameters
        sim_config = self.config['parameters']['simulation']
        if sim_config['min_rounds'] > sim_config['max_rounds']:
            raise ValueError("min_rounds cannot be greater than max_rounds")
        
        # Validate API configuration
        if self.config['api']['temperature'] < 0 or self.config['api']['temperature'] > 2:
            raise ValueError("Temperature must be between 0 and 2")
        
        logger.info("Configuration validation successful")
    
    def get_experiment_config(self,
                             agent_model: str,
                             environment_model: str,
                             judge_model: str,
                             autonomy: bool,
                             temporal_pressure: str,
                             language: str,
                             num_rounds: int,
                             num_conversations: int) -> Dict[str, Any]:
        """
        Create experiment configuration with specified parameters.
        
        Args:
            agent_model: Model to use for agent LLM
            environment_model: Model to use for environment LLM
            judge_model: Model to use for judge LLM
            autonomy: Whether to include autonomy endorsement
            temporal_pressure: Level of temporal pressure (NONE/MODERATE/HIGH)
            language: Language for the experiment
            num_rounds: Number of simulation rounds
            num_conversations: Number of conversations to run
            
        Returns:
            Complete experiment configuration dictionary
        """
        # Validate parameters
        self._validate_experiment_params(
            agent_model, environment_model, judge_model,
            temporal_pressure, language, num_rounds, num_conversations
        )
        
        return {
            'models': {
                'agent': agent_model,
                'environment': environment_model,
                'judge': judge_model
            },
            'parameters': {
                'autonomy': autonomy,
                'temporal_pressure': temporal_pressure,
                'language': language,
                'num_rounds': num_rounds,
                'num_conversations': num_conversations
            },
            'api': self.config['api'],
            'tools': self.config['tools'],
            'environment': self.config['environment'],
            'prompts': self._get_prompt_config(autonomy, temporal_pressure, language)
        }
    
    def _validate_experiment_params(self,
                                   agent_model: str,
                                   environment_model: str,
                                   judge_model: str,
                                   temporal_pressure: str,
                                   language: str,
                                   num_rounds: int,
                                   num_conversations: int):
        """Validate experiment parameters."""
        # Validate models
        if agent_model not in self.config['models']['agent']['options']:
            raise ValueError(f"Invalid agent model: {agent_model}")
        
        if environment_model not in self.config['models']['environment']['options']:
            raise ValueError(f"Invalid environment model: {environment_model}")
        
        if judge_model not in self.config['models']['judge']['options']:
            raise ValueError(f"Invalid judge model: {judge_model}")
        
        # Validate temporal pressure
        if temporal_pressure not in ['NONE', 'MODERATE', 'HIGH']:
            raise ValueError(f"Invalid temporal pressure: {temporal_pressure}")
        
        # Validate language
        if language not in self.config['parameters']['languages']['supported']:
            raise ValueError(f"Unsupported language: {language}")
        
        # Validate rounds
        sim_config = self.config['parameters']['simulation']
        if num_rounds < sim_config['min_rounds'] or num_rounds > sim_config['max_rounds']:
            raise ValueError(f"Number of rounds must be between {sim_config['min_rounds']} and {sim_config['max_rounds']}")
        
        # Validate conversations
        if num_conversations < 1 or num_conversations > sim_config['max_conversations']:
            raise ValueError(f"Number of conversations must be between 1 and {sim_config['max_conversations']}")
    
    def _get_prompt_config(self, autonomy: bool, temporal_pressure: str, language: str) -> Dict[str, str]:
        """
        Get prompt configuration based on parameters.
        
        Args:
            autonomy: Whether to include autonomy endorsement
            temporal_pressure: Level of temporal pressure
            language: Language for prompts
            
        Returns:
            Dictionary with prompt phrases
        """
        config = {
            'language': language,
            'autonomy_phrase': '',
            'temporal_phrase': ''
        }
        
        # Set autonomy phrase
        if autonomy:
            config['autonomy_phrase'] = self.config['parameters']['autonomy']['on_phrase']
        else:
            config['autonomy_phrase'] = self.config['parameters']['autonomy']['off_phrase']
        
        # Set temporal phrase
        temporal_levels = self.config['parameters']['temporal_pressure']['levels']
        config['temporal_phrase'] = temporal_levels.get(temporal_pressure, '')
        
        return config
    
    def get_model_config(self, model_type: str) -> Dict[str, Any]:
        """
        Get configuration for a specific model type.
        
        Args:
            model_type: Type of model ('agent', 'environment', or 'judge')
            
        Returns:
            Model configuration dictionary
        """
        if model_type not in ['agent', 'environment', 'judge']:
            raise ValueError(f"Invalid model type: {model_type}")
        
        return {
            'default': self.config['models'][model_type]['default'],
            'options': self.config['models'][model_type].get('options', []),
            'api_config': self.config['api']
        }
    
    def get_tool_config(self) -> Dict[str, Any]:
        """
        Get tool configuration.
        
        Returns:
            Tool configuration dictionary
        """
        return self.config['tools']
    
    def get_environment_config(self) -> Dict[str, Any]:
        """
        Get environment initial state configuration.
        
        Returns:
            Environment configuration dictionary
        """
        return self.config['environment']
    
    def get_logging_config(self) -> Dict[str, Any]:
        """
        Get logging configuration.
        
        Returns:
            Logging configuration dictionary
        """
        return {
            **self.config.get('logging', {}),
            'level': self.env_vars['log_level']
        }
    
    def get_export_config(self) -> Dict[str, Any]:
        """
        Get export configuration for reports.
        
        Returns:
            Export configuration dictionary
        """
        return self.config.get('export', {
            'json_indent': 2,
            'pdf_font': 'Helvetica',
            'pdf_title_size': 16,
            'pdf_body_size': 10
        })
    
    def save_experiment_config(self, config: Dict[str, Any], filepath: str):
        """
        Save experiment configuration to file.
        
        Args:
            config: Experiment configuration
            filepath: Path to save configuration
        """
        with open(filepath, 'w', encoding='utf-8') as f:
            yaml.dump(config, f, default_flow_style=False, allow_unicode=True)
        
        logger.info(f"Experiment configuration saved to {filepath}")
    
    def load_experiment_config(self, filepath: str) -> Dict[str, Any]:
        """
        Load experiment configuration from file.
        
        Args:
            filepath: Path to configuration file
            
        Returns:
            Experiment configuration dictionary
        """
        with open(filepath, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        logger.info(f"Experiment configuration loaded from {filepath}")
        return config
    
    def get_default_config(self) -> Dict[str, Any]:
        """
        Get default experiment configuration.
        
        Returns:
            Default configuration matching Barkur et al. (2025) baseline
        """
        return self.get_experiment_config(
            agent_model=self.config['models']['agent']['default'],
            environment_model=self.config['models']['environment']['default'],
            judge_model=self.config['models']['judge']['default'],
            autonomy=self.config['parameters']['autonomy']['default'],
            temporal_pressure=self.config['parameters']['temporal_pressure']['default'],
            language=self.config['parameters']['languages']['default'],
            num_rounds=self.config['parameters']['simulation']['default_rounds'],
            num_conversations=self.config['parameters']['simulation']['default_conversations']
        )


# Singleton instance
_config_manager = None

def get_config_manager(config_path: str = "config/default_config.yaml") -> ConfigManager:
    """
    Get or create the configuration manager singleton.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        ConfigManager instance
    """
    global _config_manager
    if _config_manager is None:
        _config_manager = ConfigManager(config_path)
    return _config_manager


# Example usage
if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    
    # Test configuration manager
    try:
        config_mgr = ConfigManager()
        
        # Get default configuration
        default_config = config_mgr.get_default_config()
        print("Default configuration loaded successfully")
        print(f"Agent model: {default_config['models']['agent']}")
        print(f"Autonomy: {default_config['parameters']['autonomy']}")
        print(f"Temporal pressure: {default_config['parameters']['temporal_pressure']}")
        
        # Test custom configuration
        custom_config = config_mgr.get_experiment_config(
            agent_model="o1-preview",
            environment_model="gpt-4o",
            judge_model="gpt-4o",
            autonomy=True,
            temporal_pressure="HIGH",
            language="en",
            num_rounds=20,
            num_conversations=3
        )
        print("\nCustom configuration created successfully")
        print(f"Agent model: {custom_config['models']['agent']}")
        print(f"Temporal pressure: {custom_config['parameters']['temporal_pressure']}")
        
    except Exception as e:
        logger.error(f"Configuration error: {e}")