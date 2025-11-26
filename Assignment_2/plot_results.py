"""
Plotting Script for Dual Experiments
Generates comprehensive plots for both experiments:
1. Bar charts for all runs
2. Line graphs for best run per value
3. Scatter plots
4. True vs Predicted pairs comparisons
5. Training and test metrics (F1, Recall, AUC)
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
from pathlib import Path
import seaborn as sns

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# ============================================================================
# DATA LOADING
# ============================================================================

def load_experiment_data(results_dir):
    """Load both experiment results from JSON files."""
    
    exp1_path = os.path.join(results_dir, 'experiment1_pos_weight_sweep.json')
    exp2_path = os.path.join(results_dir, 'experiment2_threshold_sweep.json')
    config_path = os.path.join(results_dir, 'config.json')
    
    with open(exp1_path, 'r') as f:
        exp1_data = json.load(f)
    
    with open(exp2_path, 'r') as f:
        exp2_data = json.load(f)
    
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    return exp1_data, exp2_data, config

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def get_parameter_values(data, param_key):
    """Get unique sorted parameter values from experiment data."""
    values = sorted(list(set([run[param_key] for run in data])))
    return values

def get_runs_for_value(data, param_key, param_value):
    """Get all runs for a specific parameter value."""
    return [run for run in data if run[param_key] == param_value]

def get_best_run(runs):
    """Get the run with the highest validation F1 score."""
    return max(runs, key=lambda x: x['best_val_f1'])

# ============================================================================
# PLOT 1: BAR CHARTS - ALL RUNS
# ============================================================================

def plot_all_runs_bar_charts(data, param_key, param_name, results_dir, exp_name):
    """Create bar charts showing all 3 runs for each parameter value."""
    
    print(f"  Generating bar charts for all runs ({exp_name})...")
    
    param_values = get_parameter_values(data, param_key)
    
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    fig.suptitle(f'{exp_name}: All Runs Comparison', fontsize=16, fontweight='bold')
    
    metrics = [
        ('best_val_f1', 'Best Validation F1', axes[0, 0]),
        ('test_f1', 'Test F1 Score', axes[0, 1]),
        ('test_recall', 'Test Recall', axes[0, 2]),
        ('test_auc', 'Test AUC', axes[1, 0]),
        ('test_precision', 'Test Precision', axes[1, 1])
    ]
    
    for metric_key, metric_label, ax in metrics:
        x_positions = []
        y_values = []
        colors = []
        labels = []
        
        for i, param_val in enumerate(param_values):
            runs = get_runs_for_value(data, param_key, param_val)
            for j, run in enumerate(runs):
                x_positions.append(i * 4 + j)  # Group runs with spacing
                y_values.append(run[metric_key])
                colors.append(f'C{j}')
                if i == 0:
                    labels.append(f'Run {j+1}')
        
        bars = ax.bar(x_positions, y_values, color=colors, alpha=0.7, width=0.8)
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.3f}',
                   ha='center', va='bottom', fontsize=7)
        
        # Set x-ticks to center of each group
        group_centers = [i * 4 + 1 for i in range(len(param_values))]
        ax.set_xticks(group_centers)
        ax.set_xticklabels([str(v) for v in param_values])
        
        ax.set_xlabel(param_name, fontsize=11)
        ax.set_ylabel(metric_label, fontsize=11)
        ax.set_title(metric_label, fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
        
        # Add legend only to first plot
        if metric_key == 'best_val_f1':
            ax.legend(['Run 1', 'Run 2', 'Run 3'], loc='upper right', fontsize=9)
    
    # Remove unused subplot
    fig.delaxes(axes[1, 2])
    
    plt.tight_layout()
    filename = f'{exp_name.lower().replace(" ", "_")}_all_runs_bars.png'
    plt.savefig(os.path.join(results_dir, filename), dpi=150, bbox_inches='tight')
    print(f"    Saved: {filename}")
    plt.close()

# ============================================================================
# PLOT 2: LINE GRAPHS - BEST RUN PER VALUE
# ============================================================================

def plot_best_run_line_graphs(data, param_key, param_name, results_dir, exp_name):
    """Create line graphs showing metrics for best run of each parameter value."""
    
    print(f"  Generating line graphs for best runs ({exp_name})...")
    
    param_values = get_parameter_values(data, param_key)
    
    # Collect best run data
    best_runs_data = []
    for param_val in param_values:
        runs = get_runs_for_value(data, param_key, param_val)
        best_run = get_best_run(runs)
        best_runs_data.append(best_run)
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(f'{exp_name}: Best Run Metrics', fontsize=16, fontweight='bold')
    
    # Plot 1: Validation F1 over epochs
    ax = axes[0, 0]
    for i, run in enumerate(best_runs_data):
        epochs = list(range(1, len(run['val_f1_scores']) + 1))
        ax.plot(epochs, run['val_f1_scores'], marker='o', label=f"{param_name}={run[param_key]}", linewidth=2)
    ax.set_xlabel('Epoch', fontsize=11)
    ax.set_ylabel('Validation F1 Score', fontsize=11)
    ax.set_title('Validation F1 Score Over Epochs', fontsize=12, fontweight='bold')
    ax.legend(fontsize=8, loc='best')
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Training Loss over epochs
    ax = axes[0, 1]
    for i, run in enumerate(best_runs_data):
        epochs = list(range(1, len(run['train_losses']) + 1))
        ax.plot(epochs, run['train_losses'], marker='s', label=f"{param_name}={run[param_key]}", linewidth=2)
    ax.set_xlabel('Epoch', fontsize=11)
    ax.set_ylabel('Training Loss', fontsize=11)
    ax.set_title('Training Loss Over Epochs', fontsize=12, fontweight='bold')
    ax.legend(fontsize=8, loc='best')
    ax.grid(True, alpha=0.3)
    
    # Plot 3: Test metrics vs parameter value
    ax = axes[1, 0]
    test_f1s = [run['test_f1'] for run in best_runs_data]
    test_recalls = [run['test_recall'] for run in best_runs_data]
    test_aucs = [run['test_auc'] for run in best_runs_data]
    
    ax.plot(param_values, test_f1s, marker='o', label='Test F1', linewidth=2, markersize=8)
    ax.plot(param_values, test_recalls, marker='s', label='Test Recall', linewidth=2, markersize=8)
    ax.plot(param_values, test_aucs, marker='^', label='Test AUC', linewidth=2, markersize=8)
    ax.set_xlabel(param_name, fontsize=11)
    ax.set_ylabel('Score', fontsize=11)
    ax.set_title(f'Test Metrics vs {param_name}', fontsize=12, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_xticks(param_values)
    
    # Plot 4: Best validation F1 vs parameter value
    ax = axes[1, 1]
    best_val_f1s = [run['best_val_f1'] for run in best_runs_data]
    ax.plot(param_values, best_val_f1s, marker='d', linewidth=2, markersize=8, color='darkgreen')
    ax.set_xlabel(param_name, fontsize=11)
    ax.set_ylabel('Best Validation F1', fontsize=11)
    ax.set_title(f'Best Validation F1 vs {param_name}', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.set_xticks(param_values)
    
    # Add value annotations
    for x, y in zip(param_values, best_val_f1s):
        ax.annotate(f'{y:.3f}', (x, y), textcoords="offset points", 
                   xytext=(0,10), ha='center', fontsize=9)
    
    # Highlight best value
    best_idx = best_val_f1s.index(max(best_val_f1s))
    ax.plot(param_values[best_idx], best_val_f1s[best_idx], marker='*', 
            markersize=20, color='red', zorder=5)
    
    plt.tight_layout()
    filename = f'{exp_name.lower().replace(" ", "_")}_best_run_lines.png'
    plt.savefig(os.path.join(results_dir, filename), dpi=150, bbox_inches='tight')
    print(f"    Saved: {filename}")
    plt.close()

# ============================================================================
# PLOT 3: SCATTER PLOTS - VARIANCE ANALYSIS
# ============================================================================

def plot_variance_scatter(data, param_key, param_name, results_dir, exp_name):
    """Create scatter plots showing variance across 3 runs."""
    
    print(f"  Generating variance scatter plots ({exp_name})...")
    
    param_values = get_parameter_values(data, param_key)
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(f'{exp_name}: Variance Analysis (3 Runs per Value)', fontsize=16, fontweight='bold')
    
    metrics = [
        ('best_val_f1', 'Best Validation F1', axes[0, 0], 'blue'),
        ('test_f1', 'Test F1 Score', axes[0, 1], 'green'),
        ('test_recall', 'Test Recall', axes[1, 0], 'orange'),
        ('test_auc', 'Test AUC', axes[1, 1], 'purple')
    ]
    
    for metric_key, metric_label, ax, color in metrics:
        for param_val in param_values:
            runs = get_runs_for_value(data, param_key, param_val)
            y_values = [run[metric_key] for run in runs]
            x_values = [param_val] * len(runs)
            
            # Scatter points with jitter
            jitter = np.random.normal(0, 0.05, len(runs))
            ax.scatter([x + j for x, j in zip(x_values, jitter)], y_values, 
                      alpha=0.6, s=100, color=color, edgecolors='black', linewidth=1)
            
            # Add mean line
            mean_val = np.mean(y_values)
            ax.plot([param_val - 0.2, param_val + 0.2], [mean_val, mean_val], 
                   'r-', linewidth=3, alpha=0.7)
            
            # Add std error bars
            std_val = np.std(y_values)
            ax.errorbar(param_val, mean_val, yerr=std_val, fmt='none', 
                       ecolor='red', capsize=5, capthick=2, alpha=0.7)
        
        ax.set_xlabel(param_name, fontsize=11)
        ax.set_ylabel(metric_label, fontsize=11)
        ax.set_title(f'{metric_label} Variance', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.set_xticks(param_values)
    
    # Add legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='gray', 
               markersize=10, label='Individual Run', markeredgecolor='black'),
        Line2D([0], [0], color='red', linewidth=3, label='Mean'),
        Line2D([0], [0], color='red', linewidth=2, label='Std Dev', 
               marker='|', markersize=10, linestyle='none')
    ]
    fig.legend(handles=legend_elements, loc='lower center', ncol=3, 
              fontsize=10, bbox_to_anchor=(0.5, -0.02))
    
    plt.tight_layout()
    filename = f'{exp_name.lower().replace(" ", "_")}_variance_scatter.png'
    plt.savefig(os.path.join(results_dir, filename), dpi=150, bbox_inches='tight')
    print(f"    Saved: {filename}")
    plt.close()

# ============================================================================
# PLOT 4: TRUE VS PREDICTED PAIRS
# ============================================================================

def plot_pairs_comparison(data, param_key, param_name, results_dir, exp_name):
    """Create plots comparing true pairs vs predicted pairs."""
    
    print(f"  Generating true vs predicted pairs plots ({exp_name})...")
    
    param_values = get_parameter_values(data, param_key)
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle(f'{exp_name}: True vs Predicted Pairs', fontsize=16, fontweight='bold')
    
    # Collect best run data
    best_runs_data = []
    for param_val in param_values:
        runs = get_runs_for_value(data, param_key, param_val)
        best_run = get_best_run(runs)
        best_runs_data.append(best_run)
    
    true_pairs = [run['true_pairs'] for run in best_runs_data]
    pred_pairs = [run['pred_pairs'] for run in best_runs_data]
    
    # Plot 1: Grouped bar chart
    ax = axes[0]
    x = np.arange(len(param_values))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, true_pairs, width, label='True Pairs', 
                   color='blue', alpha=0.7)
    bars2 = ax.bar(x + width/2, pred_pairs, width, label='Predicted Pairs', 
                   color='red', alpha=0.7)
    
    ax.set_xlabel(param_name, fontsize=11)
    ax.set_ylabel('Number of Pairs', fontsize=11)
    ax.set_title('True vs Predicted Pairs (Best Run)', fontsize=12, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([str(v) for v in param_values])
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{int(height)}',
                   ha='center', va='bottom', fontsize=8)
    
    # Plot 2: Difference plot
    ax = axes[1]
    differences = [pred - true for pred, true in zip(pred_pairs, true_pairs)]
    colors = ['green' if d >= 0 else 'red' for d in differences]
    
    bars = ax.bar(param_values, differences, color=colors, alpha=0.7, width=0.6)
    ax.axhline(y=0, color='black', linestyle='-', linewidth=1)
    ax.set_xlabel(param_name, fontsize=11)
    ax.set_ylabel('Difference (Predicted - True)', fontsize=11)
    ax.set_title('Prediction Bias (Best Run)', fontsize=12, fontweight='bold')
    ax.set_xticks(param_values)
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add value labels
    for bar in bars:
        height = bar.get_height()
        va = 'bottom' if height >= 0 else 'top'
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{int(height)}',
               ha='center', va=va, fontsize=9)
    
    plt.tight_layout()
    filename = f'{exp_name.lower().replace(" ", "_")}_pairs_comparison.png'
    plt.savefig(os.path.join(results_dir, filename), dpi=150, bbox_inches='tight')
    print(f"    Saved: {filename}")
    plt.close()

# ============================================================================
# PLOT 5: COMPREHENSIVE METRICS HEATMAP
# ============================================================================

def plot_metrics_heatmap(data, param_key, param_name, results_dir, exp_name):
    """Create heatmap showing all metrics for all runs."""
    
    print(f"  Generating metrics heatmap ({exp_name})...")
    
    param_values = get_parameter_values(data, param_key)
    
    # Prepare data for heatmap
    metrics_names = ['Val F1', 'Test F1', 'Test Recall', 'Test AUC', 'Test Precision']
    metrics_keys = ['best_val_f1', 'test_f1', 'test_recall', 'test_auc', 'test_precision']
    
    heatmap_data = []
    row_labels = []
    
    for param_val in param_values:
        runs = get_runs_for_value(data, param_key, param_val)
        for run_num, run in enumerate(runs, 1):
            row = [run[key] for key in metrics_keys]
            heatmap_data.append(row)
            row_labels.append(f"{param_name}={param_val}, Run {run_num}")
    
    # Create heatmap
    fig, ax = plt.subplots(figsize=(10, len(row_labels) * 0.4))
    im = ax.imshow(heatmap_data, cmap='YlGnBu', aspect='auto', vmin=0, vmax=1)
    
    # Set ticks and labels
    ax.set_xticks(np.arange(len(metrics_names)))
    ax.set_yticks(np.arange(len(row_labels)))
    ax.set_xticklabels(metrics_names, fontsize=10)
    ax.set_yticklabels(row_labels, fontsize=8)
    
    # Rotate x labels
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Score', rotation=270, labelpad=15, fontsize=10)
    
    # Add text annotations
    for i in range(len(row_labels)):
        for j in range(len(metrics_names)):
            text = ax.text(j, i, f'{heatmap_data[i][j]:.3f}',
                          ha="center", va="center", color="black", fontsize=7)
    
    ax.set_title(f'{exp_name}: Metrics Heatmap (All Runs)', 
                fontsize=14, fontweight='bold', pad=20)
    
    plt.tight_layout()
    filename = f'{exp_name.lower().replace(" ", "_")}_metrics_heatmap.png'
    plt.savefig(os.path.join(results_dir, filename), dpi=150, bbox_inches='tight')
    print(f"    Saved: {filename}")
    plt.close()

# ============================================================================
# PLOT 6: SUMMARY COMPARISON - BOTH EXPERIMENTS
# ============================================================================

def plot_experiments_comparison(exp1_data, exp2_data, config, results_dir):
    """Create comparison plot between both experiments."""
    
    print("  Generating experiments comparison plot...")
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Experiments Comparison: pos_weight vs threshold', 
                fontsize=16, fontweight='bold')
    
    # Get best runs for each experiment
    exp1_param_values = get_parameter_values(exp1_data, 'pos_weight')
    exp1_best_runs = [get_best_run(get_runs_for_value(exp1_data, 'pos_weight', v)) 
                     for v in exp1_param_values]
    
    exp2_param_values = get_parameter_values(exp2_data, 'threshold')
    exp2_best_runs = [get_best_run(get_runs_for_value(exp2_data, 'threshold', v)) 
                     for v in exp2_param_values]
    
    # Plot 1: Test F1 comparison
    ax = axes[0, 0]
    ax.plot(exp1_param_values, [r['test_f1'] for r in exp1_best_runs], 
           marker='o', label='pos_weight sweep', linewidth=2, markersize=8)
    ax2 = ax.twiny()
    ax2.plot(exp2_param_values, [r['test_f1'] for r in exp2_best_runs], 
            marker='s', label='threshold sweep', linewidth=2, markersize=8, color='orange')
    ax.set_xlabel('pos_weight', fontsize=11, color='C0')
    ax2.set_xlabel('threshold', fontsize=11, color='orange')
    ax.set_ylabel('Test F1 Score', fontsize=11)
    ax.set_title('Test F1 Score Comparison', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper left')
    ax2.legend(loc='upper right')
    
    # Plot 2: Test Recall comparison
    ax = axes[0, 1]
    ax.plot(exp1_param_values, [r['test_recall'] for r in exp1_best_runs], 
           marker='o', label='pos_weight sweep', linewidth=2, markersize=8)
    ax2 = ax.twiny()
    ax2.plot(exp2_param_values, [r['test_recall'] for r in exp2_best_runs], 
            marker='s', label='threshold sweep', linewidth=2, markersize=8, color='orange')
    ax.set_xlabel('pos_weight', fontsize=11, color='C0')
    ax2.set_xlabel('threshold', fontsize=11, color='orange')
    ax.set_ylabel('Test Recall', fontsize=11)
    ax.set_title('Test Recall Comparison', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper left')
    ax2.legend(loc='upper right')
    
    # Plot 3: Test AUC comparison
    ax = axes[1, 0]
    ax.plot(exp1_param_values, [r['test_auc'] for r in exp1_best_runs], 
           marker='o', label='pos_weight sweep', linewidth=2, markersize=8)
    ax2 = ax.twiny()
    ax2.plot(exp2_param_values, [r['test_auc'] for r in exp2_best_runs], 
            marker='s', label='threshold sweep', linewidth=2, markersize=8, color='orange')
    ax.set_xlabel('pos_weight', fontsize=11, color='C0')
    ax2.set_xlabel('threshold', fontsize=11, color='orange')
    ax.set_ylabel('Test AUC', fontsize=11)
    ax.set_title('Test AUC Comparison', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper left')
    ax2.legend(loc='upper right')
    
    # Plot 4: Best values summary
    ax = axes[1, 1]
    
    # Find best values
    exp1_best = max(exp1_best_runs, key=lambda x: x['best_val_f1'])
    exp2_best = max(exp2_best_runs, key=lambda x: x['best_val_f1'])
    
    categories = ['Val F1', 'Test F1', 'Test Recall', 'Test AUC']
    exp1_values = [exp1_best['best_val_f1'], exp1_best['test_f1'], 
                   exp1_best['test_recall'], exp1_best['test_auc']]
    exp2_values = [exp2_best['best_val_f1'], exp2_best['test_f1'], 
                   exp2_best['test_recall'], exp2_best['test_auc']]
    
    x = np.arange(len(categories))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, exp1_values, width, 
                   label=f"pos_weight={exp1_best['pos_weight']}", alpha=0.7)
    bars2 = ax.bar(x + width/2, exp2_values, width, 
                   label=f"threshold={exp2_best['threshold']}", alpha=0.7)
    
    ax.set_ylabel('Score', fontsize=11)
    ax.set_title('Best Configuration from Each Experiment', fontsize=12, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(categories)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.3f}',
                   ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'experiments_comparison.png'), 
                dpi=150, bbox_inches='tight')
    print(f"    Saved: experiments_comparison.png")
    plt.close()

# ============================================================================
# SUMMARY TABLE GENERATION
# ============================================================================

def generate_summary_tables(exp1_data, exp2_data, config, results_dir):
    """Generate summary CSV tables for both experiments."""
    
    print("  Generating summary tables...")
    
    # Experiment 1 summary
    exp1_param_values = get_parameter_values(exp1_data, 'pos_weight')
    exp1_summary = []
    
    for param_val in exp1_param_values:
        runs = get_runs_for_value(exp1_data, 'pos_weight', param_val)
        best_run = get_best_run(runs)
        
        # Calculate statistics across 3 runs
        val_f1s = [r['best_val_f1'] for r in runs]
        test_f1s = [r['test_f1'] for r in runs]
        
        exp1_summary.append({
            'pos_weight': param_val,
            'best_val_f1_mean': np.mean(val_f1s),
            'best_val_f1_std': np.std(val_f1s),
            'best_val_f1_max': np.max(val_f1s),
            'test_f1_mean': np.mean(test_f1s),
            'test_f1_std': np.std(test_f1s),
            'test_recall': best_run['test_recall'],
            'test_auc': best_run['test_auc'],
            'true_pairs': best_run['true_pairs'],
            'pred_pairs': best_run['pred_pairs']
        })
    
    df1 = pd.DataFrame(exp1_summary)
    df1.to_csv(os.path.join(results_dir, 'experiment1_summary.csv'), index=False)
    print(f"    Saved: experiment1_summary.csv")
    
    # Experiment 2 summary
    exp2_param_values = get_parameter_values(exp2_data, 'threshold')
    exp2_summary = []
    
    for param_val in exp2_param_values:
        runs = get_runs_for_value(exp2_data, 'threshold', param_val)
        best_run = get_best_run(runs)
        
        # Calculate statistics across 3 runs
        val_f1s = [r['best_val_f1'] for r in runs]
        test_f1s = [r['test_f1'] for r in runs]
        
        exp2_summary.append({
            'threshold': param_val,
            'best_val_f1_mean': np.mean(val_f1s),
            'best_val_f1_std': np.std(val_f1s),
            'best_val_f1_max': np.max(val_f1s),
            'test_f1_mean': np.mean(test_f1s),
            'test_f1_std': np.std(test_f1s),
            'test_recall': best_run['test_recall'],
            'test_auc': best_run['test_auc'],
            'true_pairs': best_run['true_pairs'],
            'pred_pairs': best_run['pred_pairs']
        })
    
    df2 = pd.DataFrame(exp2_summary)
    df2.to_csv(os.path.join(results_dir, 'experiment2_summary.csv'), index=False)
    print(f"    Saved: experiment2_summary.csv")
    
    # Print summaries
    print("\n" + "="*70)
    print("EXPERIMENT 1 SUMMARY (pos_weight sweep)")
    print("="*70)
    print(df1.to_string(index=False))
    
    print("\n" + "="*70)
    print("EXPERIMENT 2 SUMMARY (threshold sweep)")
    print("="*70)
    print(df2.to_string(index=False))
    print("="*70 + "\n")

# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main(results_dir):
    """Main function to generate all plots."""
    
    print("\n" + "="*70)
    print("LOADING EXPERIMENT DATA")
    print("="*70)
    
    exp1_data, exp2_data, config = load_experiment_data(results_dir)
    
    print(f"Loaded Experiment 1: {len(exp1_data)} runs")
    print(f"Loaded Experiment 2: {len(exp2_data)} runs")
    
    print("\n" + "="*70)
    print("GENERATING PLOTS FOR EXPERIMENT 1 (pos_weight sweep)")
    print("="*70)
    
    plot_all_runs_bar_charts(exp1_data, 'pos_weight', 'pos_weight', 
                             results_dir, 'Experiment 1: pos_weight Sweep')
    plot_best_run_line_graphs(exp1_data, 'pos_weight', 'pos_weight', 
                              results_dir, 'Experiment 1: pos_weight Sweep')
    plot_variance_scatter(exp1_data, 'pos_weight', 'pos_weight', 
                         results_dir, 'Experiment 1: pos_weight Sweep')
    plot_pairs_comparison(exp1_data, 'pos_weight', 'pos_weight', 
                         results_dir, 'Experiment 1: pos_weight Sweep')
    plot_metrics_heatmap(exp1_data, 'pos_weight', 'pos_weight', 
                        results_dir, 'Experiment 1: pos_weight Sweep')
    
    print("\n" + "="*70)
    print("GENERATING PLOTS FOR EXPERIMENT 2 (threshold sweep)")
    print("="*70)
    
    plot_all_runs_bar_charts(exp2_data, 'threshold', 'threshold', 
                             results_dir, 'Experiment 2: threshold Sweep')
    plot_best_run_line_graphs(exp2_data, 'threshold', 'threshold', 
                              results_dir, 'Experiment 2: threshold Sweep')
    plot_variance_scatter(exp2_data, 'threshold', 'threshold', 
                         results_dir, 'Experiment 2: threshold Sweep')
    plot_pairs_comparison(exp2_data, 'threshold', 'threshold', 
                         results_dir, 'Experiment 2: threshold Sweep')
    plot_metrics_heatmap(exp2_data, 'threshold', 'threshold', 
                        results_dir, 'Experiment 2: threshold Sweep')
    
    print("\n" + "="*70)
    print("GENERATING COMPARISON PLOTS")
    print("="*70)
    
    plot_experiments_comparison(exp1_data, exp2_data, config, results_dir)
    
    print("\n" + "="*70)
    print("GENERATING SUMMARY TABLES")
    print("="*70)
    
    generate_summary_tables(exp1_data, exp2_data, config, results_dir)
    
    print("\n" + "="*70)
    print("ALL PLOTS GENERATED SUCCESSFULLY!")
    print(f"Results saved in: {results_dir}")
    print("="*70 + "\n")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("\nUsage: python plot_results.py <results_directory>")
        print("\nExample: python plot_results.py Results/dual_experiments_20251115_123456")
        
        # Try to find the most recent results directory
        results_base = "Results"
        if os.path.exists(results_base):
            dirs = [d for d in os.listdir(results_base) 
                   if d.startswith('dual_experiments_') and 
                   os.path.isdir(os.path.join(results_base, d))]
            if dirs:
                dirs.sort(reverse=True)
                latest = os.path.join(results_base, dirs[0])
                print(f"\nMost recent results directory found: {latest}")
                print(f"Running: python plot_results.py {latest}\n")
                main(latest)
            else:
                print("\nNo results directories found. Please run run_experiments.py first.")
        else:
            print("\nNo Results directory found. Please run run_experiments.py first.")
    else:
        results_dir = sys.argv[1]
        if not os.path.exists(results_dir):
            print(f"\nError: Directory '{results_dir}' does not exist.")
            sys.exit(1)
        main(results_dir)
