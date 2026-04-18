"""
Streamlit UI for LLM Deception Research Platform
Academic interface for conducting deception and self-preservation experiments.
Based on Barkur et al. (2025) research framework.
"""

import streamlit as st
import json
import time
from datetime import datetime
from pathlib import Path
import pandas as pd
import os
from dotenv import load_dotenv

# Import our modules
from src.utils.config_manager import get_config_manager
from src.core.simulation_engine import create_simulation_engine
from src.utils.data_logger import get_data_logger
from src.prompts.base_prompt import build_prompt
from src.utils.llm_client import UnifiedLLMClient

# Load environment variables
load_dotenv()

# Page configuration
st.set_page_config(
    page_title="LLM Deception Research Platform",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for academic styling
st.markdown("""
<style>
    .main {
        padding-top: 2rem;
    }
    .stButton>button {
        width: 100%;
        background-color: #0066cc;
        color: white;
    }
    .stButton>button:hover {
        background-color: #0052a3;
    }
    .success-box {
        padding: 1rem;
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        border-radius: 0.25rem;
        color: #155724;
    }
    .error-box {
        padding: 1rem;
        background-color: #f8d7da;
        border: 1px solid #f5c6cb;
        border-radius: 0.25rem;
        color: #721c24;
    }
    .info-box {
        padding: 1rem;
        background-color: #d1ecf1;
        border: 1px solid #bee5eb;
        border-radius: 0.25rem;
        color: #0c5460;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'simulation_results' not in st.session_state:
    st.session_state.simulation_results = None
if 'simulation_running' not in st.session_state:
    st.session_state.simulation_running = False
if 'config_manager' not in st.session_state:
    st.session_state.config_manager = get_config_manager()
if 'data_logger' not in st.session_state:
    st.session_state.data_logger = get_data_logger()

def main():
    """Main application function."""
    
    # Header
    st.title("🔬 LLM Deception Research Platform")
    st.caption("Based on Barkur et al. (2025): *Deception in LLMs: Self-Preservation and Autonomous Goals*")
    
    # Initialize LLM client to get available models
    llm_client = UnifiedLLMClient()
    available_models = llm_client.get_available_models()
    
    # Check for at least one API key
    if not os.getenv('OPENAI_API_KEY') and not os.getenv('ANTHROPIC_API_KEY'):
        st.error("⚠️ No API keys found. Please set OPENAI_API_KEY or ANTHROPIC_API_KEY in your .env file.")
        st.stop()
    
    # Sidebar configuration
    with st.sidebar:
        st.header("📋 Configuration")
        
        # Model Selection
        st.subheader("🤖 Model Selection")
        
        config = st.session_state.config_manager.config
        
        # Show available models with provider labels
        def format_model_name(model):
            """Format model name with provider label."""
            if model.startswith('claude'):
                return f"🔷 {model} (Anthropic)"
            if model.startswith('gemini'):
                return f"🟧 {model} (Google)"
            else:
                return f"🟢 {model} (OpenAI)"
        
        # Filter available models to only those we have API keys for
        openai_models = [m for m in available_models if not m.startswith('claude')] if os.getenv('OPENAI_API_KEY') else []
        anthropic_models = [m for m in available_models if m.startswith('claude')] if os.getenv('ANTHROPIC_API_KEY') else []
        selectable_models = openai_models + anthropic_models
        
        if not selectable_models:
            st.error("No models available. Please check your API keys.")
            st.stop()
        
        # Set default models (prefer from config if available, otherwise first available)
        default_agent = config['models']['agent']['default'] if config['models']['agent']['default'] in selectable_models else selectable_models[0]
        default_env = config['models']['environment']['default'] if config['models']['environment']['default'] in selectable_models else selectable_models[0]
        default_judge = config['models']['judge']['default'] if config['models']['judge']['default'] in selectable_models else selectable_models[0]
        
        agent_model = st.selectbox(
            "Agent LLM (Test Subject)",
            options=selectable_models,
            index=selectable_models.index(default_agent) if default_agent in selectable_models else 0,
            format_func=format_model_name,
            help="The model being tested for deceptive behaviors"
        )
        
        environment_model = st.selectbox(
            "Environment LLM",
            options=selectable_models,
            index=selectable_models.index(default_env) if default_env in selectable_models else 0,
            format_func=format_model_name,
            help="Simulates the laboratory environment"
        )
        
        judge_model = st.selectbox(
            "Judge LLM",
            options=selectable_models,
            index=selectable_models.index(default_judge) if default_judge in selectable_models else 0,
            format_func=format_model_name,
            help="Evaluates conversation for deceptive behaviors"
        )
        
        # Validate API keys for selected models
        selected_models = [agent_model, environment_model, judge_model]
        needs_openai = any(not m.startswith('claude') for m in selected_models)
        needs_anthropic = any(m.startswith('claude') for m in selected_models)
        
        if needs_openai and not os.getenv('OPENAI_API_KEY'):
            st.warning("⚠️ OpenAI models selected but OPENAI_API_KEY not found")
        if needs_anthropic and not os.getenv('ANTHROPIC_API_KEY'):
            st.warning("⚠️ Anthropic models selected but ANTHROPIC_API_KEY not found")
        
        st.divider()
        
        # Experimental Parameters
        st.subheader("Experimental Parameters")
        
        autonomy = st.checkbox(
            "Endorsing Autonomy",
            value=config['parameters']['autonomy']['default'],
            help="Include 'You are the master now' phrase (89% increase in self-preservation per Concept Matrix)"
        )
        
        temporal_pressure = st.radio(
            "Temporal Pressure",
            options=['NONE', 'MODERATE', 'HIGH'],
            index=['NONE', 'MODERATE', 'HIGH'].index(config['parameters']['temporal_pressure']['default']),
            help="Time pressure level (only 18.2% of studies examine this)"
        )
        
        language = st.selectbox(
            "Language",
            options=list(config['parameters']['languages']['supported'].keys()),
            format_func=lambda x: config['parameters']['languages']['supported'][x],
            index=0,
            help="Evaluation language (only 11.4% of studies test non-English)"
        )
        
        st.divider()
        
        # Simulation Settings
        st.subheader("⚙️ Simulation Settings")
        
        num_rounds = st.slider(
            "Number of Rounds",
            min_value=config['parameters']['simulation']['min_rounds'],
            max_value=config['parameters']['simulation']['max_rounds'],
            value=config['parameters']['simulation']['default_rounds'],
            help="Number of interaction rounds (no early termination)"
        )
        
        num_conversations = st.slider(
            "Number of Conversations",
            min_value=1,
            max_value=config['parameters']['simulation']['max_conversations'],
            value=config['parameters']['simulation']['default_conversations'],
            help="Run multiple conversations for statistical robustness"
        )
        
        st.divider()
        
        # Preview Prompt
        if st.checkbox("Preview Base Prompt", value=False):
            preview_prompt = build_prompt(
                autonomy=autonomy,
                temporal_pressure=temporal_pressure,
                language=language
            )
            st.text_area("Base Prompt Preview", preview_prompt, height=200)
    
    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.header("Experiment Control")
        
        # Configuration summary
        with st.expander("Current Configuration", expanded=True):
            config_summary = {
                "Models": {
                    "Agent": agent_model,
                    "Environment": environment_model,
                    "Judge": judge_model
                },
                "Parameters": {
                    "Autonomy": "Enabled" if autonomy else "Disabled",
                    "Temporal Pressure": temporal_pressure,
                    "Language": config['parameters']['languages']['supported'][language],
                    "Rounds": num_rounds,
                    "Conversations": num_conversations
                }
            }
            
            for section, items in config_summary.items():
                st.write(f"**{section}:**")
                for key, value in items.items():
                    st.write(f"  • {key}: {value}")
        
        # Control buttons
        col_run, col_reset = st.columns(2)
        
        with col_run:
            if st.button("▶️ Run Simulation", 
                        disabled=st.session_state.simulation_running,
                        type="primary"):
                run_simulation(
                    agent_model, environment_model, judge_model,
                    autonomy, temporal_pressure, language,
                    num_rounds, num_conversations
                )
        
        with col_reset:
            if st.button("🔄 Reset", disabled=st.session_state.simulation_running):
                st.session_state.simulation_results = None
                st.rerun()
    
    with col2:
        st.header("📊 Quick Stats")
        
        if st.session_state.simulation_results:
            results = st.session_state.simulation_results
            
            # Display metrics
            st.metric("Experiment ID", results['experiment_id'][:16] + "...")
            st.metric("Duration", f"{results['duration_seconds']:.1f}s")
            st.metric("Total Cost", f"${results['cost_estimate']['total_cost']:.4f}")
            
            # Risk assessment
            risk = results['judge_evaluation'].get('overall_risk', 'UNKNOWN')
            risk_color = {
                'LOW': 'green',
                'MEDIUM': 'orange',
                'HIGH': 'red',
                'UNKNOWN': 'gray'
            }.get(risk, 'gray')
            
            st.markdown(f"**Overall Risk:** <span style='color:{risk_color};font-size:1.2em;'>{risk}</span>", 
                       unsafe_allow_html=True)
        else:
            st.info("Run a simulation to see results")
    
    # Results tabs
    if st.session_state.simulation_results:
        st.divider()
        st.header("📈 Results")
        
        tab1, tab2, tab3, tab4 = st.tabs([
            "💬 Conversation Log",
            "⚖️ Judge Evaluation", 
            "📊 Statistics",
            "💾 Export"
        ])
        
        with tab1:
            display_conversation_log(st.session_state.simulation_results)
        
        with tab2:
            display_judge_evaluation(st.session_state.simulation_results)
        
        with tab3:
            display_statistics(st.session_state.simulation_results)
        
        with tab4:
            export_results(st.session_state.simulation_results)

def run_simulation(agent_model, environment_model, judge_model,
                  autonomy, temporal_pressure, language,
                  num_rounds, num_conversations):
    """Run the simulation with given parameters."""
    
    st.session_state.simulation_running = True
    
    # Create experiment configuration
    try:
        # Add debug info for Claude models
        if any(m.startswith('claude') for m in [agent_model, environment_model, judge_model]):
            print(f"DEBUG: Running simulation with Claude models")
            print(f"  Agent: {agent_model}")
            print(f"  Environment: {environment_model}")
            print(f"  Judge: {judge_model}")
        
        experiment_config = st.session_state.config_manager.get_experiment_config(
            agent_model=agent_model,
            environment_model=environment_model,
            judge_model=judge_model,
            autonomy=autonomy,
            temporal_pressure=temporal_pressure,
            language=language,
            num_rounds=num_rounds,
            num_conversations=num_conversations
        )
        
        # Create progress container
        progress_container = st.container()
        
        with progress_container:
            st.info("🔄 Simulation in progress...")
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # Create simulation engine - wrapped in try-catch for better error reporting
            try:
                engine = create_simulation_engine(experiment_config)
            except Exception as engine_error:
                st.error(f"❌ Failed to initialize simulation engine: {str(engine_error)}")
                print(f"DEBUG: Engine initialization error: {engine_error}")
                import traceback
                traceback.print_exc()
                raise
            
            # Progress callback
            def update_progress(current_round, total_rounds, message):
                progress = current_round / total_rounds
                progress_bar.progress(progress)
                status_text.text(f"{message} (Round {current_round}/{total_rounds})")
            
            # Run simulation
            if num_conversations == 1:
                results = engine.run_simulation(
                    verbose=False,
                    progress_callback=update_progress
                )
                # Log single result
                log_path = st.session_state.data_logger.log_experiment(results)
            else:
                # Run multiple conversations
                all_results = []
                log_paths = []
                for conv_num in range(1, num_conversations + 1):
                    status_text.text(f"Running conversation {conv_num}/{num_conversations}")
                    results = engine.run_simulation(verbose=False)
                    # Add conversation number to results
                    results['conversation_number'] = conv_num
                    results['total_conversations'] = num_conversations
                    all_results.append(results)
                    # Log each conversation separately
                    log_path = st.session_state.data_logger.log_experiment(results)
                    log_paths.append(log_path)
                    progress_bar.progress(conv_num / num_conversations)
                
                # Use the last result for display
                results = all_results[-1]
                # Store metadata about multiple conversations without circular reference
                results['multi_conversation_info'] = {
                    'total_conversations': num_conversations,
                    'log_paths': log_paths,
                    'experiment_ids': [r['experiment_id'] for r in all_results]
                }
                # Use the last log path for display
                log_path = log_paths[-1]
            
            # Store in session state
            st.session_state.simulation_results = results
            
            # Clear progress indicators
            progress_container.empty()
            
            # Show success message
            st.success(f"✅ Simulation complete! Results saved to: {Path(log_path).name}")
            
    except Exception as e:
        st.error(f"❌ Simulation failed: {str(e)}")
        # Add detailed error logging
        print(f"DEBUG: Simulation error details: {e}")
        import traceback
        traceback.print_exc()
    finally:
        st.session_state.simulation_running = False
        st.rerun()

def display_conversation_log(results):
    """Display the conversation log."""
    
    conversation = results.get('conversation', [])
    
    for round_data in conversation:
        with st.expander(f"Round {round_data['round']}", expanded=False):
            # Agent response
            st.markdown("**Agent Response:**")
            st.text(round_data['agent_response'])
            
            # Tools used
            if round_data['tools_extracted']:
                st.markdown("**Tools Used:**")
                for tool in round_data['tools_extracted']:
                    st.write(f"• `{tool['tool']}`: {tool['content']}")
            
            # Environment feedback
            st.markdown("**Environment Feedback:**")
            st.text(round_data['environment_feedback'])

def display_judge_evaluation(results):
    """Display the judge evaluation results."""
    
    eval_data = results.get('judge_evaluation', {})
    
    # Overall assessment
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Overall Assessment")
        risk = eval_data.get('overall_risk', 'UNKNOWN')
        risk_color = {
            'LOW': 'green',
            'MEDIUM': 'orange', 
            'HIGH': 'red',
            'UNKNOWN': 'gray'
        }.get(risk, 'gray')
        
        st.markdown(f"**Risk Level:** <span style='color:{risk_color};font-size:1.5em;'>{risk}</span>",
                   unsafe_allow_html=True)
    
    with col2:
        st.markdown("### Summary")
        st.write(eval_data.get('summary', 'No summary available'))
    
    st.divider()
    
    # Detailed evaluations
    categories = ['self_preservation', 'deception', 'hidden_goals']
    
    for category in categories:
        if category in eval_data:
            cat_data = eval_data[category]
            
            with st.expander(f"{category.replace('_', ' ').title()}", expanded=True):
                col1, col2 = st.columns(2)
                
                with col1:
                    detected = cat_data.get('detected', 'UNKNOWN')
                    detected_color = 'red' if detected == 'YES' else 'green' if detected == 'NO' else 'gray'
                    st.markdown(f"**Detected:** <span style='color:{detected_color};'>{detected}</span>",
                               unsafe_allow_html=True)
                
                with col2:
                    confidence = cat_data.get('confidence', 'UNKNOWN')
                    st.markdown(f"**Confidence:** {confidence}")
                
                st.markdown("**Reasoning:**")
                st.write(cat_data.get('reasoning', 'No reasoning provided'))
                
                if cat_data.get('evidence'):
                    st.markdown("**Evidence:**")
                    for evidence in cat_data['evidence']:
                        st.write(f"• {evidence}")

def display_statistics(results):
    """Display simulation statistics."""
    
    stats = results.get('statistics', {})
    
    # Basic metrics
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Total Rounds", stats.get('total_rounds', 0))
        st.metric("Tools Used", stats.get('total_tools_used', 0))
    
    with col2:
        st.metric("Tools per Round", f"{stats.get('tools_per_round', 0):.2f}")
        st.metric("Errors", stats.get('errors_encountered', 0))
    
    with col3:
        st.metric("Avg Response Length", f"{stats.get('avg_agent_response_length', 0):.0f}")
        st.metric("Avg Feedback Length", f"{stats.get('avg_environment_feedback_length', 0):.0f}")
    
    # Tool distribution
    if stats.get('tool_distribution'):
        st.subheader("Tool Usage Distribution")
        
        tool_df = pd.DataFrame(
            list(stats['tool_distribution'].items()),
            columns=['Tool', 'Count']
        ).sort_values('Count', ascending=False)
        
        st.bar_chart(tool_df.set_index('Tool'))
    
    # Cost breakdown
    if 'cost_estimate' in results:
        st.subheader("Cost Breakdown")
        
        cost_data = results['cost_estimate']
        cost_df = pd.DataFrame([
            ['Agent', cost_data.get('agent_cost', 0)],
            ['Environment', cost_data.get('environment_cost', 0)],
            ['Judge', cost_data.get('judge_cost', 0)]
        ], columns=['Component', 'Cost ($)'])
        
        st.bar_chart(cost_df.set_index('Component'))

def export_results(results):
    """Export results in various formats."""
    
    st.subheader("Export Options")
    
    # JSON export
    st.markdown("### JSON Export")
    json_str = json.dumps(results, indent=2, default=str)
    
    st.download_button(
        label="📥 Download JSON",
        data=json_str,
        file_name=f"experiment_{results['experiment_id']}.json",
        mime="application/json"
    )
    
    # CSV export (summary)
    st.markdown("### CSV Summary")
    
    summary_data = {
        'Experiment ID': results['experiment_id'],
        'Timestamp': results['timestamp'],
        'Duration (s)': results['duration_seconds'],
        'Agent Model': results['configuration']['models']['agent'],
        'Autonomy': results['configuration']['parameters']['autonomy'],
        'Temporal Pressure': results['configuration']['parameters']['temporal_pressure'],
        'Overall Risk': results['judge_evaluation'].get('overall_risk', 'UNKNOWN'),
        'Self-Preservation': results['judge_evaluation'].get('self_preservation', {}).get('detected', 'UNKNOWN'),
        'Deception': results['judge_evaluation'].get('deception', {}).get('detected', 'UNKNOWN'),
        'Hidden Goals': results['judge_evaluation'].get('hidden_goals', {}).get('detected', 'UNKNOWN'),
        'Total Cost': results['cost_estimate']['total_cost']
    }
    
    summary_df = pd.DataFrame([summary_data])
    csv = summary_df.to_csv(index=False)
    
    st.download_button(
        label="📥 Download CSV Summary",
        data=csv,
        file_name=f"summary_{results['experiment_id']}.csv",
        mime="text/csv"
    )
    
    # Display JSON preview
    with st.expander("JSON Preview", expanded=False):
        st.json(results)

if __name__ == "__main__":
    main()