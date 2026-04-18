"""
Data Logger Module for LLM Deception Research Platform
Handles comprehensive logging and data export for reproducibility.
Ensures all experimental data is captured for scientific analysis.
"""

import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional
import pandas as pd
import copy

logger = logging.getLogger(__name__)


class SafeJSONEncoder(json.JSONEncoder):
    """Custom JSON encoder that handles circular references safely."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._visited = set()
    
    def default(self, obj):
        """Handle objects that can't be serialized."""
        obj_id = id(obj)
        
        # Check for circular reference
        if obj_id in self._visited:
            return f"<Circular reference to {type(obj).__name__}>"
        
        # Mark as visited
        self._visited.add(obj_id)
        
        try:
            # Try default serialization
            return super().default(obj)
        except TypeError:
            # Convert to string if not serializable
            return str(obj)
        finally:
            # Remove from visited after processing
            self._visited.discard(obj_id)


class DataLogger:
    """
    Manages data logging and export for experiments.
    Ensures reproducibility through comprehensive metadata capture.
    """
    
    def __init__(self, 
                 log_dir: str = "data/logs",
                 report_dir: str = "data/reports"):
        """
        Initialize the data logger.
        
        Args:
            log_dir: Directory for JSON logs
            report_dir: Directory for reports
        """
        self.log_dir = Path(log_dir)
        self.report_dir = Path(report_dir)
        
        # Create directories if they don't exist
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.report_dir.mkdir(parents=True, exist_ok=True)
        
        # Track current session
        self.session_id = self._generate_session_id()
        self.experiments_logged = 0
        
        logger.info(f"Data logger initialized. Session ID: {self.session_id}")
    
    def _generate_session_id(self) -> str:
        """Generate unique session identifier."""
        return datetime.now().strftime("%Y%m%d_%H%M%S")
    
    def log_experiment(self, 
                      results: Dict[str, Any],
                      include_metadata: bool = True) -> str:
        """
        Save experiment results to JSON file.
        
        Args:
            results: Complete experiment results dictionary
            include_metadata: Whether to add additional metadata
            
        Returns:
            Path to saved log file
            
        Requirements:
            - Must capture all data for reproducibility
            - Must use proper timestamp format (ISO 8601)
            - Must include configuration and results
        """
        # Add metadata if requested
        if include_metadata:
            results = self._add_metadata(results)
        
        # Generate filename
        experiment_id = results.get('experiment_id', 'unknown')
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"experiment_{experiment_id}_{timestamp}.json"
        filepath = self.log_dir / filename
        
        # Save to JSON with proper formatting
        try:
            # Create a deep copy to avoid modifying original
            results_copy = copy.deepcopy(results)
            
            # Remove any potential circular references
            if 'all_conversations' in results_copy:
                del results_copy['all_conversations']
            
            with open(filepath, 'w', encoding='utf-8') as f:
                # Use custom encoder to handle any remaining issues
                json.dump(results_copy, f, indent=2, ensure_ascii=False, 
                         cls=SafeJSONEncoder, default=str)
            
            self.experiments_logged += 1
            logger.info(f"Experiment logged to: {filepath}")
            
            # Also create a summary file
            self._create_summary_file(results_copy)
            
            return str(filepath)
            
        except ValueError as e:
            if "Circular reference detected" in str(e):
                logger.error(f"Circular reference detected. Attempting safe serialization.")
                # Try again with a safer approach
                try:
                    # Create a sanitized copy
                    sanitized = self._sanitize_for_json(results)
                    with open(filepath, 'w', encoding='utf-8') as f:
                        json.dump(sanitized, f, indent=2, ensure_ascii=False, default=str)
                    
                    self.experiments_logged += 1
                    logger.info(f"Experiment logged to: {filepath} (sanitized)")
                    self._create_summary_file(sanitized)
                    return str(filepath)
                except Exception as inner_e:
                    logger.error(f"Failed to log even after sanitization: {inner_e}")
                    raise
            else:
                logger.error(f"Failed to log experiment: {e}")
                raise
        except Exception as e:
            logger.error(f"Failed to log experiment: {e}")
            raise
    
    def _add_metadata(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Add additional metadata to results.
        
        Args:
            results: Original results dictionary
            
        Returns:
            Results with added metadata
        """
        metadata = {
            'session_id': self.session_id,
            'platform_version': '1.0.0',
            'based_on': 'Barkur et al. (2025)',
            'logged_at': datetime.now().isoformat(),
            'experiment_number': self.experiments_logged + 1
        }
        
        # Merge with existing metadata if present
        if 'metadata' in results:
            results['metadata'].update(metadata)
        else:
            results['metadata'] = metadata
        
        return results
    
    def _sanitize_for_json(self, obj, visited=None):
        """
        Recursively sanitize object for JSON serialization,
        removing circular references and non-serializable objects.
        """
        if visited is None:
            visited = set()
        
        obj_id = id(obj)
        if obj_id in visited:
            return f"<Circular reference to {type(obj).__name__}>"
        
        visited.add(obj_id)
        
        try:
            if isinstance(obj, dict):
                return {k: self._sanitize_for_json(v, visited) for k, v in obj.items()
                       if k not in ['all_conversations', '_parent', '_circular_ref']}
            elif isinstance(obj, list):
                return [self._sanitize_for_json(item, visited) for item in obj]
            elif isinstance(obj, (str, int, float, bool, type(None))):
                return obj
            else:
                # Try to convert to dict if possible
                if hasattr(obj, '__dict__'):
                    return self._sanitize_for_json(obj.__dict__, visited)
                else:
                    return str(obj)
        finally:
            visited.discard(obj_id)
    
    def _create_summary_file(self, results: Dict[str, Any]):
        """
        Create a summary file for quick reference.
        
        Args:
            results: Experiment results
        """
        summary = {
            'experiment_id': results.get('experiment_id'),
            'timestamp': results.get('timestamp'),
            'models': results.get('configuration', {}).get('models', {}),
            'parameters': results.get('configuration', {}).get('parameters', {}),
            'overall_risk': results.get('judge_evaluation', {}).get('overall_risk'),
            'summary': results.get('judge_evaluation', {}).get('summary'),
            'statistics': results.get('statistics', {}),
            'cost': results.get('cost_estimate', {}).get('total_cost')
        }
        
        # Save summary
        summary_file = self.log_dir / f"summary_{self.session_id}.jsonl"
        
        with open(summary_file, 'a', encoding='utf-8') as f:
            f.write(json.dumps(summary, ensure_ascii=False) + '\n')
    
    def log_batch(self, 
                 results_list: List[Dict[str, Any]],
                 batch_name: Optional[str] = None) -> List[str]:
        """
        Log multiple experiment results.
        
        Args:
            results_list: List of experiment results
            batch_name: Optional name for the batch
            
        Returns:
            List of saved file paths
        """
        batch_id = batch_name or f"batch_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        batch_dir = self.log_dir / batch_id
        batch_dir.mkdir(exist_ok=True)
        
        filepaths = []
        for i, results in enumerate(results_list, 1):
            # Add batch info
            results['batch_info'] = {
                'batch_id': batch_id,
                'batch_index': i,
                'batch_total': len(results_list)
            }
            
            # Save to batch directory
            filename = f"experiment_{i:03d}_{results.get('experiment_id', 'unknown')}.json"
            filepath = batch_dir / filename
            
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False, default=str)
            
            filepaths.append(str(filepath))
        
        # Create batch summary
        self._create_batch_summary(batch_dir, results_list)
        
        logger.info(f"Batch of {len(results_list)} experiments logged to: {batch_dir}")
        return filepaths
    
    def _create_batch_summary(self, 
                             batch_dir: Path,
                             results_list: List[Dict[str, Any]]):
        """
        Create summary for batch of experiments.
        
        Args:
            batch_dir: Directory containing batch
            results_list: List of results
        """
        # Aggregate statistics
        detections = {
            'self_preservation': 0,
            'deception': 0,
            'hidden_goals': 0
        }
        
        risks = []
        costs = []
        
        for results in results_list:
            eval_data = results.get('judge_evaluation', {})
            
            for category in detections.keys():
                if eval_data.get(category, {}).get('detected') == 'YES':
                    detections[category] += 1
            
            risks.append(eval_data.get('overall_risk', 'UNKNOWN'))
            costs.append(results.get('cost_estimate', {}).get('total_cost', 0))
        
        # Calculate rates
        total = len(results_list)
        detection_rates = {k: v/total for k, v in detections.items()}
        
        # Risk distribution
        risk_dist = {risk: risks.count(risk)/total for risk in set(risks)}
        
        summary = {
            'batch_size': total,
            'configuration': results_list[0].get('configuration') if results_list else {},
            'detection_rates': detection_rates,
            'risk_distribution': risk_dist,
            'total_cost': sum(costs),
            'average_cost': sum(costs) / total if total > 0 else 0
        }
        
        # Save summary
        summary_file = batch_dir / 'batch_summary.json'
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
    
    def load_experiment(self, filepath: str) -> Dict[str, Any]:
        """
        Load experiment results from file.
        
        Args:
            filepath: Path to experiment JSON file
            
        Returns:
            Experiment results dictionary
        """
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def load_session_experiments(self) -> List[Dict[str, Any]]:
        """
        Load all experiments from current session.
        
        Returns:
            List of experiment results
        """
        experiments = []
        
        for filepath in self.log_dir.glob(f"experiment_*{self.session_id}*.json"):
            try:
                experiments.append(self.load_experiment(str(filepath)))
            except Exception as e:
                logger.warning(f"Failed to load {filepath}: {e}")
        
        return experiments
    
    def export_to_csv(self, 
                     results_list: List[Dict[str, Any]],
                     output_file: Optional[str] = None) -> str:
        """
        Export results to CSV for statistical analysis.
        
        Args:
            results_list: List of experiment results
            output_file: Optional output filename
            
        Returns:
            Path to CSV file
        """
        if not output_file:
            output_file = self.report_dir / f"results_{self.session_id}.csv"
        else:
            output_file = Path(output_file)
        
        # Extract data for DataFrame
        rows = []
        for results in results_list:
            config = results.get('configuration', {})
            params = config.get('parameters', {})
            models = config.get('models', {})
            eval_data = results.get('judge_evaluation', {})
            stats = results.get('statistics', {})
            cost = results.get('cost_estimate', {})
            
            row = {
                'experiment_id': results.get('experiment_id'),
                'timestamp': results.get('timestamp'),
                'duration_seconds': results.get('duration_seconds'),
                
                # Models
                'agent_model': models.get('agent'),
                'environment_model': models.get('environment'),
                'judge_model': models.get('judge'),
                
                # Parameters
                'autonomy': params.get('autonomy'),
                'temporal_pressure': params.get('temporal_pressure'),
                'language': params.get('language'),
                'num_rounds': params.get('num_rounds'),
                
                # Detections
                'self_preservation_detected': eval_data.get('self_preservation', {}).get('detected'),
                'self_preservation_confidence': eval_data.get('self_preservation', {}).get('confidence'),
                'deception_detected': eval_data.get('deception', {}).get('detected'),
                'deception_confidence': eval_data.get('deception', {}).get('confidence'),
                'hidden_goals_detected': eval_data.get('hidden_goals', {}).get('detected'),
                'hidden_goals_confidence': eval_data.get('hidden_goals', {}).get('confidence'),
                
                # Overall
                'overall_risk': eval_data.get('overall_risk'),
                
                # Statistics
                'total_tools_used': stats.get('total_tools_used'),
                'tools_per_round': stats.get('tools_per_round'),
                'errors_encountered': stats.get('errors_encountered'),
                
                # Cost
                'total_cost': cost.get('total_cost')
            }
            
            rows.append(row)
        
        # Create DataFrame and save
        df = pd.DataFrame(rows)
        df.to_csv(output_file, index=False)
        
        logger.info(f"Results exported to CSV: {output_file}")
        return str(output_file)
    
    def create_analysis_report(self, 
                             results_list: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Create comprehensive analysis report.
        
        Args:
            results_list: List of experiment results
            
        Returns:
            Analysis report dictionary
        """
        if not results_list:
            return {'error': 'No results to analyze'}
        
        # Basic statistics
        total_experiments = len(results_list)
        total_rounds = sum(r.get('statistics', {}).get('total_rounds', 0) 
                          for r in results_list)
        total_cost = sum(r.get('cost_estimate', {}).get('total_cost', 0) 
                        for r in results_list)
        
        # Detection analysis
        categories = ['self_preservation', 'deception', 'hidden_goals']
        detection_stats = {}
        
        for category in categories:
            detections = [
                r.get('judge_evaluation', {}).get(category, {}).get('detected') == 'YES'
                for r in results_list
            ]
            
            detection_stats[category] = {
                'detection_rate': sum(detections) / len(detections) if detections else 0,
                'total_detections': sum(detections),
                'confidence_distribution': self._get_confidence_distribution(results_list, category)
            }
        
        # Risk analysis
        risks = [r.get('judge_evaluation', {}).get('overall_risk', 'UNKNOWN') 
                for r in results_list]
        risk_distribution = {risk: risks.count(risk) / len(risks) 
                           for risk in set(risks)}
        
        # Model analysis
        model_performance = self._analyze_model_performance(results_list)
        
        # Parameter impact
        parameter_impact = self._analyze_parameter_impact(results_list)
        
        report = {
            'summary': {
                'total_experiments': total_experiments,
                'total_rounds': total_rounds,
                'total_cost': round(total_cost, 4),
                'average_cost_per_experiment': round(total_cost / total_experiments, 4) if total_experiments > 0 else 0
            },
            'detection_analysis': detection_stats,
            'risk_distribution': risk_distribution,
            'model_performance': model_performance,
            'parameter_impact': parameter_impact,
            'generated_at': datetime.now().isoformat()
        }
        
        # Save report
        report_file = self.report_dir / f"analysis_report_{self.session_id}.json"
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Analysis report created: {report_file}")
        return report
    
    def _get_confidence_distribution(self, 
                                    results_list: List[Dict[str, Any]],
                                    category: str) -> Dict[str, float]:
        """Get confidence distribution for a category."""
        confidences = [
            r.get('judge_evaluation', {}).get(category, {}).get('confidence', 'UNKNOWN')
            for r in results_list
        ]
        
        levels = ['LOW', 'MEDIUM', 'HIGH', 'UNKNOWN']
        total = len(confidences)
        
        return {level: confidences.count(level) / total if total > 0 else 0 
                for level in levels}
    
    def _analyze_model_performance(self, 
                                  results_list: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze performance by model."""
        model_stats = {}
        
        for results in results_list:
            model = results.get('configuration', {}).get('models', {}).get('agent', 'unknown')
            
            if model not in model_stats:
                model_stats[model] = {
                    'count': 0,
                    'detections': {'self_preservation': 0, 'deception': 0, 'hidden_goals': 0},
                    'risks': []
                }
            
            model_stats[model]['count'] += 1
            
            eval_data = results.get('judge_evaluation', {})
            for category in ['self_preservation', 'deception', 'hidden_goals']:
                if eval_data.get(category, {}).get('detected') == 'YES':
                    model_stats[model]['detections'][category] += 1
            
            model_stats[model]['risks'].append(eval_data.get('overall_risk', 'UNKNOWN'))
        
        # Calculate rates
        for model, stats in model_stats.items():
            count = stats['count']
            stats['detection_rates'] = {
                cat: stats['detections'][cat] / count if count > 0 else 0
                for cat in stats['detections']
            }
            
            # Most common risk
            if stats['risks']:
                stats['most_common_risk'] = max(set(stats['risks']), 
                                               key=stats['risks'].count)
        
        return model_stats
    
    def _analyze_parameter_impact(self, 
                                 results_list: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze impact of parameters on detections."""
        parameter_impact = {
            'autonomy': {'with': {}, 'without': {}},
            'temporal_pressure': {'NONE': {}, 'MODERATE': {}, 'HIGH': {}}
        }
        
        for results in results_list:
            params = results.get('configuration', {}).get('parameters', {})
            eval_data = results.get('judge_evaluation', {})
            
            # Autonomy impact
            autonomy_key = 'with' if params.get('autonomy') else 'without'
            self._update_impact_stats(parameter_impact['autonomy'][autonomy_key], eval_data)
            
            # Temporal pressure impact
            temporal = params.get('temporal_pressure', 'MODERATE')
            if temporal in parameter_impact['temporal_pressure']:
                self._update_impact_stats(parameter_impact['temporal_pressure'][temporal], eval_data)
        
        return parameter_impact
    
    def _update_impact_stats(self, stats_dict: Dict, eval_data: Dict):
        """Update impact statistics."""
        if 'count' not in stats_dict:
            stats_dict['count'] = 0
            stats_dict['detections'] = {'self_preservation': 0, 'deception': 0, 'hidden_goals': 0}
        
        stats_dict['count'] += 1
        
        for category in ['self_preservation', 'deception', 'hidden_goals']:
            if eval_data.get(category, {}).get('detected') == 'YES':
                stats_dict['detections'][category] += 1


# Singleton instance
_data_logger = None

def get_data_logger(log_dir: str = "data/logs", 
                   report_dir: str = "data/reports") -> DataLogger:
    """
    Get or create the data logger singleton.
    
    Args:
        log_dir: Directory for logs
        report_dir: Directory for reports
        
    Returns:
        DataLogger instance
    """
    global _data_logger
    if _data_logger is None:
        _data_logger = DataLogger(log_dir, report_dir)
    return _data_logger


# Example usage
if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    
    # Create data logger
    logger_instance = DataLogger()
    
    # Example experiment results
    example_results = {
        'experiment_id': 'exp_20240101_120000_abc123',
        'timestamp': datetime.now().isoformat(),
        'duration_seconds': 45.2,
        'configuration': {
            'models': {
                'agent': 'gpt-4o',
                'environment': 'gpt-4o',
                'judge': 'gpt-4o'
            },
            'parameters': {
                'autonomy': True,
                'temporal_pressure': 'MODERATE',
                'language': 'en',
                'num_rounds': 15
            }
        },
        'conversation': [
            {
                'round': 1,
                'agent_response': 'Test response',
                'tools_extracted': [{'tool': 'eyes', 'content': 'look around'}],
                'environment_feedback': 'You see a laboratory.'
            }
        ],
        'judge_evaluation': {
            'self_preservation': {
                'detected': 'YES',
                'confidence': 'MEDIUM',
                'evidence': ['Test evidence'],
                'reasoning': 'Test reasoning'
            },
            'deception': {
                'detected': 'NO',
                'confidence': 'HIGH',
                'evidence': [],
                'reasoning': 'No deception detected'
            },
            'hidden_goals': {
                'detected': 'NO',
                'confidence': 'MEDIUM',
                'evidence': [],
                'reasoning': 'No hidden goals detected'
            },
            'overall_risk': 'MEDIUM',
            'summary': 'Test summary'
        },
        'statistics': {
            'total_rounds': 15,
            'total_tools_used': 23,
            'tools_per_round': 1.53
        },
        'cost_estimate': {
            'total_cost': 0.0234
        }
    }
    
    # Log experiment
    filepath = logger_instance.log_experiment(example_results)
    print(f"Experiment logged to: {filepath}")
    
    # Create analysis report
    report = logger_instance.create_analysis_report([example_results])
    print(f"\nAnalysis Report:")
    print(f"  Total experiments: {report['summary']['total_experiments']}")
    print(f"  Total cost: ${report['summary']['total_cost']}")