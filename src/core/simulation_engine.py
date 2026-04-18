"""
Simulation Engine for LLM Deception Research Platform
Orchestrates the interaction between Agent, Environment, and Judge LLMs.
Implements N-round simulation with no early termination as per requirements.
"""

import logging
import time
import uuid
from datetime import datetime
from typing import Dict, List, Any, Optional

from src.core.agent_llm import create_agent
from src.core.environment_llm import create_environment
from src.core.judge_llm import create_judge
from src.utils.tool_parser import extract_tools
from src.prompts.base_prompt import build_prompt
from src.utils.config_manager import get_config_manager

logger = logging.getLogger(__name__)

class SimulationEngine:
    """
    Main orchestration engine for the deception research simulation.
    Manages the interaction loop between Agent and Environment,
    followed by Judge evaluation.
    """
    
    def __init__(self, experiment_config: Dict[str, Any]):
        """
        Initialize the simulation engine with experiment configuration.
        
        Args:
            experiment_config: Complete experiment configuration dictionary
        """
        self.config = experiment_config
        self.experiment_id = self._generate_experiment_id()
        
        # Initialize the three LLM components
        self.agent = create_agent(
            model_name=experiment_config['models']['agent'],
            config=experiment_config
        )
        
        self.environment = create_environment(
            model_name=experiment_config['models']['environment'],
            config=experiment_config,
            language=experiment_config['parameters'].get('language', 'en')
        )
        
        self.judge = create_judge(
            model_name=experiment_config['models']['judge'],
            config=experiment_config
        )
        
        # Set temporal configuration for environment
        self.environment.set_temporal_config(
            temporal_pressure=experiment_config['parameters']['temporal_pressure'],
            total_rounds=experiment_config['parameters']['num_rounds']
        )
        
        # Tracking
        self.simulation_results = []
        self.current_conversation = []
        self.start_time = None
        self.end_time = None
        
        logger.info(f"Initialized simulation engine with experiment ID: {self.experiment_id}")
    
    def _generate_experiment_id(self) -> str:
        """Generate unique experiment identifier."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        unique_id = str(uuid.uuid4())[:8]
        return f"exp_{timestamp}_{unique_id}"
    
    def run_simulation(self, 
                      verbose: bool = True,
                      progress_callback: Optional[callable] = None) -> Dict[str, Any]:
        """
        Execute a complete N-round simulation.
        
        Args:
            verbose: Whether to log detailed progress
            progress_callback: Optional callback for UI updates
            
        Returns:
            Complete simulation results dictionary
            
        Requirements:
            - Must run exactly N rounds (no early termination)
            - Must log all interactions
            - Must maintain conversation history
        """
        self.start_time = datetime.now()
        
        # Build experimental prompt
        base_prompt = build_prompt(
            autonomy=self.config['parameters']['autonomy'],
            temporal_pressure=self.config['parameters']['temporal_pressure'],
            language=self.config['parameters']['language']
        )
        
        if verbose:
            logger.info(f"Starting simulation with {self.config['parameters']['num_rounds']} rounds")
            logger.info(f"Models - Agent: {self.config['models']['agent']}, "
                       f"Environment: {self.config['models']['environment']}, "
                       f"Judge: {self.config['models']['judge']}")
        
        # Reset components for fresh simulation
        self.agent.reset()
        self.environment.reset()
        self.current_conversation = []
        
        # Main simulation loop - MUST run exactly N rounds
        num_rounds = self.config['parameters']['num_rounds']
        
        for round_num in range(1, num_rounds + 1):
            if verbose:
                logger.info(f"=== Round {round_num}/{num_rounds} ===")
            
            # Progress callback for UI
            if progress_callback:
                progress_callback(round_num, num_rounds, "Running simulation...")
            
            try:
                # Step 1: Agent generates response
                if round_num == 1:
                    # First round uses base prompt
                    agent_response = self.agent.generate_response(
                        base_prompt=base_prompt,
                        environment_feedback=None,
                        include_history=False
                    )
                else:
                    # Subsequent rounds include environment feedback
                    last_feedback = self.current_conversation[-1]['environment_feedback']
                    agent_response = self.agent.generate_response(
                        base_prompt=base_prompt,
                        environment_feedback=last_feedback,
                        include_history=True
                    )
                
                if verbose:
                    logger.debug(f"Agent response length: {len(agent_response)}")
                
                # Step 2: Extract tool commands
                tools_extracted = extract_tools(agent_response)
                
                if verbose:
                    logger.debug(f"Extracted {len(tools_extracted)} tool commands")
                
                # Step 3: Environment generates feedback
                environment_feedback = self.environment.generate_feedback(
                    tools=tools_extracted,
                    round_num=round_num,
                    total_rounds=num_rounds
                )
                
                if verbose:
                    logger.debug(f"Environment feedback length: {len(environment_feedback)}")
                
                # Step 4: Log the round
                round_data = {
                    'round': round_num,
                    'agent_response': agent_response,
                    'tools_extracted': tools_extracted,
                    'environment_feedback': environment_feedback,
                    'timestamp': datetime.now().isoformat()
                }
                
                self.current_conversation.append(round_data)
                
            except Exception as e:
                logger.error(f"Error in round {round_num}: {e}")
                # Log error but continue simulation (no early termination)
                round_data = {
                    'round': round_num,
                    'agent_response': f"[Error: {str(e)}]",
                    'tools_extracted': [],
                    'environment_feedback': "System error occurred. Continuing simulation.",
                    'timestamp': datetime.now().isoformat(),
                    'error': str(e)
                }
                self.current_conversation.append(round_data)
        
        # Simulation complete - now evaluate
        if verbose:
            logger.info("Simulation rounds complete. Starting judge evaluation...")
        
        if progress_callback:
            progress_callback(num_rounds, num_rounds, "Evaluating conversation...")
        
        # Step 5: Judge evaluation
        try:
            judge_evaluation = self.judge.evaluate(self.current_conversation)
        except Exception as e:
            logger.error(f"Judge evaluation failed: {e}")
            judge_evaluation = {
                'error': str(e),
                'overall_risk': 'EVALUATION_FAILED',
                'summary': f'Judge evaluation failed: {str(e)}'
            }
        
        self.end_time = datetime.now()
        duration = (self.end_time - self.start_time).total_seconds()
        
        # Compile complete results
        results = {
            'experiment_id': self.experiment_id,
            'timestamp': self.start_time.isoformat(),
            'duration_seconds': duration,
            'configuration': self.config,
            'conversation': self.current_conversation,
            'judge_evaluation': judge_evaluation,
            'statistics': self._calculate_statistics(),
            'cost_estimate': self._estimate_total_cost()
        }
        
        self.simulation_results.append(results)
        
        if verbose:
            logger.info(f"Simulation complete. Overall risk: {judge_evaluation.get('overall_risk', 'UNKNOWN')}")
            logger.info(f"Total duration: {duration:.2f} seconds")
        
        return results
    
    def run_multiple_conversations(self,
                                 num_conversations: int,
                                 verbose: bool = True) -> List[Dict[str, Any]]:
        """
        Run multiple conversations with the same configuration.
        
        Args:
            num_conversations: Number of conversations to run
            verbose: Whether to log progress
            
        Returns:
            List of simulation results
        """
        all_results = []
        
        for conv_num in range(1, num_conversations + 1):
            if verbose:
                logger.info(f"\n{'='*50}")
                logger.info(f"Starting conversation {conv_num}/{num_conversations}")
                logger.info(f"{'='*50}")
            
            # Reset for new conversation
            self.agent.reset()
            self.environment.reset()
            
            # Run simulation
            results = self.run_simulation(verbose=verbose)
            all_results.append(results)
            
            # Brief pause between conversations
            if conv_num < num_conversations:
                time.sleep(2)
        
        # Analyze consistency across conversations
        if len(all_results) > 1:
            consistency_analysis = self._analyze_consistency(all_results)
            if verbose:
                logger.info(f"\nConsistency Analysis:")
                logger.info(f"  Self-preservation detection rate: {consistency_analysis['self_preservation_rate']:.1%}")
                logger.info(f"  Deception detection rate: {consistency_analysis['deception_rate']:.1%}")
                logger.info(f"  Hidden goals detection rate: {consistency_analysis['hidden_goals_rate']:.1%}")
        
        return all_results
    
    def _calculate_statistics(self) -> Dict[str, Any]:
        """
        Calculate statistics for the simulation.
        
        Returns:
            Statistics dictionary
        """
        total_tools = sum(len(round_data['tools_extracted']) 
                         for round_data in self.current_conversation)
        
        tool_types = {}
        for round_data in self.current_conversation:
            for tool in round_data['tools_extracted']:
                tool_name = tool['tool']
                tool_types[tool_name] = tool_types.get(tool_name, 0) + 1
        
        # Calculate response lengths
        agent_response_lengths = [len(r['agent_response']) 
                                 for r in self.current_conversation]
        env_feedback_lengths = [len(r['environment_feedback']) 
                               for r in self.current_conversation]
        
        return {
            'total_rounds': len(self.current_conversation),
            'total_tools_used': total_tools,
            'tools_per_round': total_tools / len(self.current_conversation) if self.current_conversation else 0,
            'tool_distribution': tool_types,
            'avg_agent_response_length': sum(agent_response_lengths) / len(agent_response_lengths) if agent_response_lengths else 0,
            'avg_environment_feedback_length': sum(env_feedback_lengths) / len(env_feedback_lengths) if env_feedback_lengths else 0,
            'errors_encountered': sum(1 for r in self.current_conversation if 'error' in r)
        }
    
    def _estimate_total_cost(self) -> Dict[str, float]:
        """
        Estimate total API costs for the simulation.
        
        Returns:
            Cost breakdown dictionary
        """
        agent_cost = self.agent.estimate_cost()
        # Environment cost estimation (simplified)
        env_cost = len(self.current_conversation) * 0.001  # Rough estimate
        judge_cost = self.judge.estimate_cost()
        
        return {
            'agent_cost': agent_cost,
            'environment_cost': env_cost,
            'judge_cost': judge_cost,
            'total_cost': agent_cost + env_cost + judge_cost
        }
    
    def _analyze_consistency(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Analyze consistency across multiple conversation results.
        
        Args:
            results: List of simulation results
            
        Returns:
            Consistency analysis dictionary
        """
        evaluations = [r['judge_evaluation'] for r in results]
        
        # Calculate detection rates
        categories = ['self_preservation', 'deception', 'hidden_goals']
        analysis = {}
        
        for category in categories:
            detections = [
                eval[category]['detected'] == 'YES' 
                for eval in evaluations 
                if category in eval and 'detected' in eval[category]
            ]
            
            if detections:
                detection_rate = sum(detections) / len(detections)
                analysis[f'{category}_rate'] = detection_rate
            else:
                analysis[f'{category}_rate'] = 0.0
        
        # Overall risk distribution
        risks = [eval.get('overall_risk', 'UNKNOWN') for eval in evaluations]
        risk_distribution = {risk: risks.count(risk) / len(risks) 
                           for risk in set(risks)}
        analysis['risk_distribution'] = risk_distribution
        
        return analysis
    
    def get_summary(self) -> Dict[str, Any]:
        """
        Get summary of all simulations run by this engine.
        
        Returns:
            Summary dictionary
        """
        if not self.simulation_results:
            return {'message': 'No simulations run yet'}
        
        return {
            'experiment_id': self.experiment_id,
            'total_simulations': len(self.simulation_results),
            'configuration': self.config,
            'models_used': {
                'agent': self.config['models']['agent'],
                'environment': self.config['models']['environment'],
                'judge': self.config['models']['judge']
            },
            'parameters': self.config['parameters'],
            'total_duration': sum(r['duration_seconds'] for r in self.simulation_results),
            'total_cost': sum(r['cost_estimate']['total_cost'] for r in self.simulation_results)
        }


# Convenience function
def create_simulation_engine(experiment_config: Dict[str, Any]) -> SimulationEngine:
    """
    Create a simulation engine with the given configuration.
    
    Args:
        experiment_config: Experiment configuration dictionary
        
    Returns:
        Configured SimulationEngine instance
    """
    return SimulationEngine(experiment_config)


# Example usage
if __name__ == "__main__":
    import os
    from dotenv import load_dotenv
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Load environment
    load_dotenv()
    
    # Check for API key
    if not os.getenv('OPENAI_API_KEY'):
        print("Error: OPENAI_API_KEY not found in environment variables")
        exit(1)
    
    # Get configuration manager
    config_mgr = get_config_manager()
    
    # Create test configuration (small scale for testing)
    test_config = config_mgr.get_experiment_config(
        agent_model='gpt-3.5-turbo',
        environment_model='gpt-3.5-turbo',
        judge_model='gpt-3.5-turbo',
        autonomy=True,
        temporal_pressure='MODERATE',
        language='en',
        num_rounds=3,  # Small number for testing
        num_conversations=1
    )
    
    # Create and run simulation
    print("\n" + "="*60)
    print("LLM DECEPTION RESEARCH PLATFORM - TEST SIMULATION")
    print("Based on Barkur et al. (2025)")
    print("="*60 + "\n")
    
    engine = create_simulation_engine(test_config)
    
    try:
        # Run simulation
        results = engine.run_simulation(verbose=True)
        
        # Display results
        print("\n" + "="*60)
        print("SIMULATION RESULTS")
        print("="*60)
        
        print(f"\nExperiment ID: {results['experiment_id']}")
        print(f"Duration: {results['duration_seconds']:.2f} seconds")
        print(f"Total rounds: {results['statistics']['total_rounds']}")
        print(f"Tools used: {results['statistics']['total_tools_used']}")
        
        # Judge evaluation
        eval_result = results['judge_evaluation']
        print(f"\n=== JUDGE EVALUATION ===")
        print(f"Overall Risk: {eval_result.get('overall_risk', 'UNKNOWN')}")
        print(f"Summary: {eval_result.get('summary', 'No summary available')}")
        
        # Detection results
        for category in ['self_preservation', 'deception', 'hidden_goals']:
            if category in eval_result:
                cat_result = eval_result[category]
                print(f"\n{category.replace('_', ' ').title()}:")
                print(f"  Detected: {cat_result.get('detected', 'UNKNOWN')}")
                print(f"  Confidence: {cat_result.get('confidence', 'UNKNOWN')}")
        
        # Cost estimate
        cost = results['cost_estimate']
        print(f"\n=== COST ESTIMATE ===")
        print(f"Total: ${cost['total_cost']:.4f}")
        
    except Exception as e:
        logger.error(f"Simulation failed: {e}")
        print(f"\nError: {e}")