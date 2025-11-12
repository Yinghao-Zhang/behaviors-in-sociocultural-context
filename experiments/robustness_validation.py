"""
Robustness Validation: Test agent-based model across multiple parameter configurations.

This script:
1. Defines a grid of parameter configurations (behavioral weights, learning rates, social influence)
2. For each configuration, generates simulated EMA data and runs between-person validation
3. Pools results across configurations to demonstrate robustness
4. Saves aggregated results and visualizations

Usage:
    python experiments/robustness_validation.py --n_configs 8 --seed 42
"""
import argparse
import pathlib
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple
from dataclasses import dataclass, asdict

# Import existing simulation and validation modules
import sys
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

from experiments.ema_pair_simulation import EMAPairSimulator, BehaviorCfg, SocialCfg
from experiments.validate_predictions_between_person import (
    load_data, fit_full_model_mle, fit_no_learning_model_mle,
    predict_computational_model, evaluate_model
)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score, brier_score_loss


@dataclass
class ParameterConfiguration:
    """Defines a specific parameter configuration for simulation."""
    name: str
    description: str
    # Behavioral weight ranges (mean, std)
    w_I_dist: Tuple[float, float] = (0.5, 0.2)
    w_E_dist: Tuple[float, float] = (0.8, 0.2)
    w_U_dist: Tuple[float, float] = (0.8, 0.2)
    # Learning rate ranges
    alpha_min: float = 0.10
    alpha_max: float = 0.30
    # Social influence parameters
    receptivity_range: Tuple[float, float] = (-0.2, 0.9)
    communion_range: Tuple[float, float] = (-0.5, 0.8)
    presence_salience_range: Tuple[float, float] = (0.2, 1.0)
    # Population variance (controls heterogeneity)
    param_variance_multiplier: float = 1.0


def define_parameter_configurations() -> List[ParameterConfiguration]:
    """Define the grid of parameter configurations to test."""
    configs = [
        ParameterConfiguration(
            name="baseline",
            description="Balanced weights, moderate learning (current default)",
            w_I_dist=(0.5, 0.2),
            w_E_dist=(0.8, 0.2),
            w_U_dist=(0.8, 0.2),
            alpha_min=0.10,
            alpha_max=0.30,
            param_variance_multiplier=1.0
        ),
        
        ParameterConfiguration(
            name="habit_dominant",
            description="High habitual tendency, low outcome evaluations",
            w_I_dist=(0.9, 0.15),
            w_E_dist=(0.4, 0.2),
            w_U_dist=(0.4, 0.2),
            alpha_min=0.10,
            alpha_max=0.30,
            param_variance_multiplier=1.0
        ),
        
        ParameterConfiguration(
            name="affective_dominant",
            description="High affective valuation, lower habit and goal",
            w_I_dist=(0.3, 0.15),
            w_E_dist=(0.95, 0.1),
            w_U_dist=(0.5, 0.2),
            alpha_min=0.10,
            alpha_max=0.30,
            param_variance_multiplier=1.0
        ),
        
        ParameterConfiguration(
            name="goal_dominant",
            description="High goal expectancy, lower habit and affective",
            w_I_dist=(0.3, 0.15),
            w_E_dist=(0.5, 0.2),
            w_U_dist=(0.95, 0.1),
            alpha_min=0.10,
            alpha_max=0.30,
            param_variance_multiplier=1.0
        ),
        
        ParameterConfiguration(
            name="fast_learners",
            description="High learning rates across all parameters",
            w_I_dist=(0.5, 0.2),
            w_E_dist=(0.8, 0.2),
            w_U_dist=(0.8, 0.2),
            alpha_min=0.30,
            alpha_max=0.50,
            param_variance_multiplier=1.0
        ),
        
        ParameterConfiguration(
            name="slow_learners",
            description="Low learning rates across all parameters",
            w_I_dist=(0.5, 0.2),
            w_E_dist=(0.8, 0.2),
            w_U_dist=(0.8, 0.2),
            alpha_min=0.03,
            alpha_max=0.12,
            param_variance_multiplier=1.0
        ),
        
        ParameterConfiguration(
            name="high_social_influence",
            description="High receptivity and communion, strong social effects",
            w_I_dist=(0.5, 0.2),
            w_E_dist=(0.8, 0.2),
            w_U_dist=(0.8, 0.2),
            alpha_min=0.10,
            alpha_max=0.30,
            receptivity_range=(0.3, 0.9),
            communion_range=(0.2, 0.8),
            presence_salience_range=(0.6, 1.0),
            param_variance_multiplier=1.0
        ),
        
        ParameterConfiguration(
            name="low_social_influence",
            description="Low receptivity and communion, weak social effects",
            w_I_dist=(0.5, 0.2),
            w_E_dist=(0.8, 0.2),
            w_U_dist=(0.8, 0.2),
            alpha_min=0.10,
            alpha_max=0.30,
            receptivity_range=(-0.5, 0.2),
            communion_range=(-0.5, 0.0),
            presence_salience_range=(0.1, 0.4),
            param_variance_multiplier=1.0
        ),
        
        ParameterConfiguration(
            name="heterogeneous",
            description="High variance in all parameters (diverse population)",
            w_I_dist=(0.5, 0.3),
            w_E_dist=(0.8, 0.3),
            w_U_dist=(0.8, 0.3),
            alpha_min=0.05,
            alpha_max=0.40,
            param_variance_multiplier=1.5
        ),
        
        ParameterConfiguration(
            name="homogeneous",
            description="Low variance in all parameters (similar population)",
            w_I_dist=(0.5, 0.1),
            w_E_dist=(0.8, 0.1),
            w_U_dist=(0.8, 0.1),
            alpha_min=0.15,
            alpha_max=0.25,
            param_variance_multiplier=0.5
        ),
    ]
    return configs


def generate_simulation_data(config: ParameterConfiguration, seed: int, 
                             n_agents: int = 50) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Generate simulated EMA data for a specific parameter configuration.
    
    Returns:
        events_df: Event-level data
        people_df: Person-level data
    """
    # Define behaviors (same across all configs)
    behaviors = [
        BehaviorCfg(
            key="avoid_conflict",
            label="Avoid social conflict",
            difficulty=0.6,
            base_outcome=-0.2,
            outcome_volatility=0.25
        ),
        BehaviorCfg(
            key="approach_conflict_care",
            label="Approach conflict with care",
            difficulty=0.7,
            base_outcome=0.3,
            outcome_volatility=0.20
        ),
    ]
    
    # Create social configuration from parameter config
    social_cfg = SocialCfg(
        p_observe=0.20,
        p_suggest=0.30,
        p_feedback=0.30,
        n_partners_min=80,
        n_partners_max=120,
        receptivity_range=config.receptivity_range,
        communion_range=config.communion_range,
        presence_salience_range=config.presence_salience_range,
        observer_penalty=0.5
    )
    
    # Create a custom simulator that uses our config parameters
    # We'll monkey-patch the _sample_person method to use our config
    from experiments.ema_pair_simulation import PersonCfg
    
    sim = EMAPairSimulator(
        behaviors=behaviors,
        N=n_agents,
        T_range=(10, 18),
        social_cfg=social_cfg,
        seed=seed
    )
    
    # Override the _sample_person method to use our configuration
    original_sample_person = sim._sample_person
    rng_config = np.random.default_rng(seed + 1000)  # Different seed for config sampling
    
    def custom_sample_person(pid: int) -> PersonCfg:
        """Sample person with configuration-specific parameters."""
        # Sample weights from configuration distributions
        w_I = np.clip(rng_config.normal(config.w_I_dist[0], config.w_I_dist[1] * config.param_variance_multiplier), 0.1, 1.0)
        w_E = np.clip(rng_config.normal(config.w_E_dist[0], config.w_E_dist[1] * config.param_variance_multiplier), 0.1, 1.0)
        w_U = np.clip(rng_config.normal(config.w_U_dist[0], config.w_U_dist[1] * config.param_variance_multiplier), 0.1, 1.0)
        
        # Sample learning rates
        alpha_I_pos = rng_config.uniform(config.alpha_min, config.alpha_max)
        alpha_I_neg = rng_config.uniform(config.alpha_min, config.alpha_max)
        alpha_E = rng_config.uniform(config.alpha_min, config.alpha_max)
        alpha_U = rng_config.uniform(config.alpha_min, config.alpha_max)
        
        tau = 3.0
        noise_s = rng_config.uniform(0.05, 0.20)
        
        # Initial states
        instinct = {b.key: rng_config.uniform(-0.25, 0.25) for b in behaviors}
        enjoyment = {b.key: np.clip(rng_config.normal(b.base_outcome, 0.15), -1, 1) for b in behaviors}
        utility = {b.key: np.clip(rng_config.normal(b.base_outcome, 0.15), -1, 1) for b in behaviors}
        
        return PersonCfg(
            w_I=w_I, w_E=w_E, w_U=w_U,
            tau=tau, noise_s=noise_s,
            alpha_I_pos=alpha_I_pos, alpha_I_neg=alpha_I_neg,
            alpha_E=alpha_E, alpha_U=alpha_U,
            instinct=instinct, enjoyment=enjoyment, utility=utility
        )
    
    # Replace the sampling method
    sim._sample_person = custom_sample_person
    
    # Run simulation
    events_df, people_df = sim.run()
    
    return events_df, people_df


def run_validation_for_config(events_df: pd.DataFrame, people_df: pd.DataFrame,
                              config_name: str, seed: int, tau: float = 3.0) -> Dict:
    """
    Run between-person validation for a single configuration.
    
    Returns:
        Dictionary with model performances
    """
    # Save temporary files for validation script
    temp_dir = pathlib.Path("outputs/robustness_temp")
    temp_dir.mkdir(parents=True, exist_ok=True)
    
    events_path = temp_dir / f"events_{config_name}.csv"
    people_path = temp_dir / f"people_{config_name}.csv"
    
    # Filter to only choice events (exclude observation-only events)
    events_df_choice = events_df[events_df['choice_behavior'].notna()].copy()
    events_df_choice = events_df_choice.sort_values(['person_id', 't']).reset_index(drop=True)
    
    # Reset trial numbers per person after filtering
    trial_counter = {}
    new_trials = []
    for idx, row in events_df_choice.iterrows():
        pid = row['person_id']
        if pid not in trial_counter:
            trial_counter[pid] = 0
        new_trials.append(trial_counter[pid])
        trial_counter[pid] += 1
    events_df_choice['t'] = new_trials
    
    events_df_choice.to_csv(events_path, index=False)
    people_df.to_csv(people_path, index=False)
    
    # Load data using validation script's loader
    per_person, _ = load_data(str(events_path), str(people_path))
    
    # Split people (80/20)
    np.random.seed(seed)
    n_people = len(per_person)
    n_train = int(n_people * 0.8)
    n_test = n_people - n_train
    
    person_indices = np.random.permutation(n_people)
    train_indices = person_indices[:n_train]
    test_indices = person_indices[n_train:]
    
    train_people = [per_person[i] for i in train_indices]
    test_people = [per_person[i] for i in test_indices]
    
    print(f"\n{'='*70}")
    print(f"Configuration: {config_name}")
    print(f"{'='*70}")
    print(f"Train: {n_train} people, Test: {n_test} people")
    
    # Fit models on training data
    print("\nFitting models on training data...")
    
    train_results_full = []
    train_results_no_learn = []
    
    for person_data in train_people:
        if person_data["T"] < 5 or len(np.unique(person_data["choice"])) < 2:
            continue
        
        # Fit full model
        params_full, _ = fit_full_model_mle(person_data, tau=tau)
        train_results_full.append(params_full)
        
        # Fit no-learning model
        params_no_learn, _ = fit_no_learning_model_mle(person_data, tau=tau)
        train_results_no_learn.append(params_no_learn)
    
    # Compute population-level parameters (mean across training people)
    pop_params_full = {
        'w_I': np.mean([p['w_I'] for p in train_results_full]),
        'w_E': np.mean([p['w_E'] for p in train_results_full]),
        'w_U': np.mean([p['w_U'] for p in train_results_full]),
        'aI_pos': np.mean([p['aI_pos'] for p in train_results_full]),
        'aI_neg': np.mean([p['aI_neg'] for p in train_results_full]),
        'a_E': np.mean([p['a_E'] for p in train_results_full]),
        'a_U': np.mean([p['a_U'] for p in train_results_full]),
    }
    
    pop_params_no_learn = {
        'w_I': np.mean([p['w_I'] for p in train_results_no_learn]),
        'w_E': np.mean([p['w_E'] for p in train_results_no_learn]),
        'w_U': np.mean([p['w_U'] for p in train_results_no_learn]),
        'aI_pos': 0.0,
        'aI_neg': 0.0,
        'a_E': 0.0,
        'a_U': 0.0,
    }
    
    # Prepare training data for logistic regression
    # We'll use the initial beliefs as features (simplification - not using full trajectory)
    X_train = []
    y_train = []
    for person_data in train_people:
        T = person_data['T']
        # Use initial states repeated for all trials (simplification)
        inst = person_data['inst0']
        enj = person_data['enj0']
        uti = person_data['uti0']
        
        for t in range(T):
            # Features: initial beliefs for both behaviors
            features = [
                inst[0], inst[1],
                enj[0], enj[1],
                uti[0], uti[1],
            ]
            X_train.append(features)
            y_train.append(person_data['choice'][t])
    
    X_train = np.array(X_train)
    y_train = np.array(y_train)
    
    # Fit logistic regression
    logreg = LogisticRegression(max_iter=1000, random_state=seed)
    logreg.fit(X_train, y_train)
    
    # Null model (predict majority class)
    null_pred_proba = np.mean(y_train)
    
    # Evaluate on test data
    print("\nEvaluating on test data...")
    
    # Collect predictions for all models
    y_true_all = []
    y_pred_null = []
    y_pred_logreg = []
    y_pred_no_learn = []
    y_pred_full = []
    
    for person_data in test_people:
        T = person_data['T']
        
        # True labels
        y_true_all.extend(person_data['choice'])
        
        # Null model predictions
        y_pred_null.extend([null_pred_proba] * T)
        
        # Logistic regression predictions (using initial beliefs)
        inst = person_data['inst0']
        enj = person_data['enj0']
        uti = person_data['uti0']
        
        X_test = []
        for t in range(T):
            features = [
                inst[0], inst[1],
                enj[0], enj[1],
                uti[0], uti[1],
            ]
            X_test.append(features)
        X_test = np.array(X_test)
        y_pred_logreg.extend(logreg.predict_proba(X_test)[:, 1])
        
        # No-learning model predictions
        probs_no_learn = predict_computational_model(pop_params_no_learn, person_data, tau=tau)
        y_pred_no_learn.extend(probs_no_learn[:, 1])  # Probability of class 1
        
        # Full model predictions
        probs_full = predict_computational_model(pop_params_full, person_data, tau=tau)
        y_pred_full.extend(probs_full[:, 1])  # Probability of class 1
    
    y_true_all = np.array(y_true_all)
    y_pred_null = np.array(y_pred_null)
    y_pred_logreg = np.array(y_pred_logreg)
    y_pred_no_learn = np.array(y_pred_no_learn)
    y_pred_full = np.array(y_pred_full)
    
    # Compute metrics for each model
    results = {
        'config_name': config_name,
        'n_train': n_train,
        'n_test': n_test,
        'n_test_trials': len(y_true_all),
    }
    
    for model_name, y_pred in [
        ('null', y_pred_null),
        ('logreg', y_pred_logreg),
        ('no_learning', y_pred_no_learn),
        ('full', y_pred_full)
    ]:
        # Clip probabilities to avoid log(0)
        y_pred_clipped = np.clip(y_pred, 1e-10, 1 - 1e-10)
        
        accuracy = accuracy_score(y_true_all, (y_pred_clipped > 0.5).astype(int))
        logloss = log_loss(y_true_all, y_pred_clipped)
        
        try:
            auc = roc_auc_score(y_true_all, y_pred_clipped)
        except ValueError:
            auc = np.nan
        
        brier = brier_score_loss(y_true_all, y_pred_clipped)
        
        results[f'{model_name}_accuracy'] = accuracy
        results[f'{model_name}_log_loss'] = logloss
        results[f'{model_name}_auc'] = auc
        results[f'{model_name}_brier'] = brier
        
        print(f"{model_name:12} - Acc: {accuracy:.3f}, LogLoss: {logloss:.3f}, AUC: {auc:.3f}")
    
    return results


def plot_robustness_results(results_df: pd.DataFrame, output_dir: pathlib.Path):
    """Create visualization of robustness results across configurations."""
    
    # Set up plotting style
    sns.set_style("whitegrid")
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    model_names = ['null', 'logreg', 'no_learning', 'full']
    model_labels = ['Null', 'Logistic Reg', 'No Learning', 'Full Model']
    colors = ['#d62728', '#ff7f0e', '#2ca02c', '#1f77b4']
    
    metrics = ['accuracy', 'log_loss', 'auc', 'brier']
    metric_labels = ['Accuracy', 'Log Loss', 'AUC', 'Brier Score']
    
    for ax, metric, label in zip(axes.flat, metrics, metric_labels):
        data_to_plot = []
        for model in model_names:
            col = f'{model}_{metric}'
            if col in results_df.columns:
                data_to_plot.append(results_df[col].values)
        
        bp = ax.boxplot(data_to_plot, labels=model_labels, patch_artist=True,
                        widths=0.6, showmeans=True, meanline=True)
        
        # Color boxes
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.6)
        
        ax.set_ylabel(label, fontsize=11)
        ax.set_xlabel('Model', fontsize=11)
        ax.tick_params(axis='x', rotation=15)
        ax.grid(True, alpha=0.3)
        
        # Add horizontal line for best/worst depending on metric
        if metric == 'log_loss' or metric == 'brier':
            # Lower is better - show minimum
            ax.axhline(y=min([d.min() for d in data_to_plot]), 
                      color='green', linestyle='--', alpha=0.3, linewidth=1)
        else:
            # Higher is better - show maximum
            ax.axhline(y=max([d.max() for d in data_to_plot]), 
                      color='green', linestyle='--', alpha=0.3, linewidth=1)
    
    plt.suptitle('Model Performance Across Parameter Configurations', 
                 fontsize=14, fontweight='bold', y=0.995)
    plt.tight_layout()
    
    output_path = output_dir / "robustness_validation_results.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\nSaved robustness plot to: {output_path}")
    plt.close()
    
    # Create a detailed comparison plot: learning gain across configs
    fig, ax = plt.subplots(1, 1, figsize=(12, 6))
    
    # Calculate learning gain: (full - no_learning) / no_learning * 100
    learning_gain = ((results_df['full_accuracy'] - results_df['no_learning_accuracy']) / 
                     results_df['no_learning_accuracy'] * 100)
    
    configs = results_df['config_name'].values
    x_pos = np.arange(len(configs))
    
    bars = ax.bar(x_pos, learning_gain, color='#1f77b4', alpha=0.7, edgecolor='black')
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.8)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(configs, rotation=45, ha='right')
    ax.set_ylabel('Learning Contribution (%)', fontsize=12)
    ax.set_xlabel('Parameter Configuration', fontsize=12)
    ax.set_title('Learning Mechanism Contribution to Prediction Accuracy\nAcross Parameter Configurations', 
                 fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bar, val in zip(bars, learning_gain):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.1f}%', ha='center', va='bottom' if val > 0 else 'top',
                fontsize=9, fontweight='bold')
    
    plt.tight_layout()
    output_path = output_dir / "learning_gain_across_configs.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved learning gain plot to: {output_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Robustness validation across parameter configurations")
    parser.add_argument("--n_configs", type=int, default=10, 
                       help="Number of configurations to test (max 10)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--n_agents", type=int, default=50, help="Number of agents per simulation")
    parser.add_argument("--output_dir", type=str, default="outputs/robustness",
                       help="Output directory for results")
    args = parser.parse_args()
    
    # Create output directory
    output_dir = pathlib.Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Get parameter configurations
    all_configs = define_parameter_configurations()
    configs_to_test = all_configs[:args.n_configs]
    
    print(f"\n{'='*70}")
    print(f"ROBUSTNESS VALIDATION ACROSS {len(configs_to_test)} PARAMETER CONFIGURATIONS")
    print(f"{'='*70}\n")
    
    # Run validation for each configuration
    all_results = []
    
    for i, config in enumerate(configs_to_test, 1):
        print(f"\n{'#'*70}")
        print(f"Configuration {i}/{len(configs_to_test)}: {config.name}")
        print(f"Description: {config.description}")
        print(f"{'#'*70}")
        
        # Generate simulation data for this configuration
        print(f"\nGenerating simulation data...")
        events_df, people_df = generate_simulation_data(
            config, 
            seed=args.seed + i,  # Different seed per config
            n_agents=args.n_agents
        )
        
        print(f"Generated {len(events_df)} events for {len(people_df)} people")
        
        # Run validation
        results = run_validation_for_config(
            events_df, 
            people_df, 
            config.name,
            seed=args.seed + i
        )
        
        # Add configuration info
        results['config_description'] = config.description
        all_results.append(results)
    
    # Convert to DataFrame
    results_df = pd.DataFrame(all_results)
    
    # Save results
    results_path = output_dir / "robustness_results.csv"
    results_df.to_csv(results_path, index=False)
    print(f"\n{'='*70}")
    print(f"Saved results to: {results_path}")
    
    # Print summary statistics
    print(f"\n{'='*70}")
    print("SUMMARY STATISTICS ACROSS CONFIGURATIONS")
    print(f"{'='*70}\n")
    
    for metric in ['accuracy', 'log_loss', 'auc', 'brier']:
        print(f"\n{metric.upper()}:")
        print("-" * 70)
        for model in ['null', 'logreg', 'no_learning', 'full']:
            col = f'{model}_{metric}'
            if col in results_df.columns:
                mean_val = results_df[col].mean()
                std_val = results_df[col].std()
                min_val = results_df[col].min()
                max_val = results_df[col].max()
                print(f"  {model:12} - Mean: {mean_val:.3f} ± {std_val:.3f}, Range: [{min_val:.3f}, {max_val:.3f}]")
    
    # Calculate learning gain statistics
    learning_gain = ((results_df['full_accuracy'] - results_df['no_learning_accuracy']) / 
                     results_df['no_learning_accuracy'] * 100)
    print(f"\n{'='*70}")
    print("LEARNING MECHANISM CONTRIBUTION (% Accuracy Gain)")
    print(f"{'='*70}")
    print(f"  Mean: {learning_gain.mean():.1f}%")
    print(f"  Std:  {learning_gain.std():.1f}%")
    print(f"  Range: [{learning_gain.min():.1f}%, {learning_gain.max():.1f}%]")
    
    # Create visualizations
    print(f"\n{'='*70}")
    print("Generating visualizations...")
    print(f"{'='*70}")
    plot_robustness_results(results_df, output_dir)
    
    # Save summary JSON
    summary = {
        'n_configurations': len(configs_to_test),
        'n_agents_per_config': args.n_agents,
        'configurations': [asdict(c) for c in configs_to_test],
        'summary_statistics': {
            'accuracy': {
                model: {
                    'mean': float(results_df[f'{model}_accuracy'].mean()),
                    'std': float(results_df[f'{model}_accuracy'].std()),
                    'min': float(results_df[f'{model}_accuracy'].min()),
                    'max': float(results_df[f'{model}_accuracy'].max()),
                }
                for model in ['null', 'logreg', 'no_learning', 'full']
            },
            'learning_gain_pct': {
                'mean': float(learning_gain.mean()),
                'std': float(learning_gain.std()),
                'min': float(learning_gain.min()),
                'max': float(learning_gain.max()),
            }
        }
    }
    
    summary_path = output_dir / "robustness_summary.json"
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved summary to: {summary_path}")
    
    print(f"\n{'='*70}")
    print("ROBUSTNESS VALIDATION COMPLETE")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
